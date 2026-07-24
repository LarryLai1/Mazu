"""Low-pass filter global HRES forecasts, then crop to the Taiwan domain (GPU).

For every 2D lat-lon field: a 2D FFT is taken over the FULL GLOBAL grid, spectral
components above a radial cutoff (cycles/degree) are zeroed, the field is
reconstructed, and only THEN is it cropped to the Taiwan lat/lon window. Filtering
globally first means the (globally periodic) transform has no artificial domain
edges -- the crop just selects the region of interest afterwards.

The 2D-FFT convention (fft2 -> fftshift -> fftfreq(d=dlon)) mirrors the reference
in earth2/utils/analysis_tools.py and Mazu/plot_wavenumber.py. Self-contained.

GPU / memory design (global 0.25 deg upper files are ~4.6 GB each):
  * The FFT runs on the GPU (torch), batched over trailing (lat, lon) slices.
  * Each file is processed one variable at a time; each variable is streamed
    through the GPU in bounded chunks, and each chunk is cropped on-device right
    after filtering -- so GPU memory stays ~<1 GB and host RAM holds only one
    source variable plus its (tiny) cropped result.
  * Output is written straight to disk with netCDF4, variable by variable. The
    latitude/longitude dimensions and coordinate variables are the cropped Taiwan
    grid; every other dim/coord/attr is copied verbatim.
  * With several GPU_IDS the files fan out across GPUs (one spawned worker per
    GPU); a single id runs in-process.

Config lives in the __main__ block.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import netCDF4 as nc
import torch


# ---------------------------------------------------------------------------
# GPU 2D-FFT low-pass
# ---------------------------------------------------------------------------
def build_radial_lowpass_mask(lat, lon, cutoff, device):
    """Isotropic 2D low-pass mask on `device`: keep bins with sqrt(fx^2+fy^2) <= cutoff.

    Frequencies are in cycles/degree (fftfreq with d = grid spacing in degrees).
    The mask is symmetric in both axes, so the inverse transform stays real.
    Shape is (lat, lon) over the FULL grid, fftshifted -- matching the spectrum.
    """
    freq_lat = np.fft.fftshift(np.fft.fftfreq(lat.size, d=float(np.mean(np.diff(lat)))))
    freq_lon = np.fft.fftshift(np.fft.fftfreq(lon.size, d=float(np.mean(np.diff(lon)))))
    fx, fy = np.meshgrid(freq_lon, freq_lat)  # (lat, lon)
    mask = (np.sqrt(fx ** 2 + fy ** 2) <= float(cutoff))
    return torch.from_numpy(mask.astype(np.float32)).to(device)


def lowpass_batch(x, mask):
    """Low-pass a batch of 2D fields on the GPU.

    x    : (B, lat, lon) real tensor on the same device as `mask`.
    mask : (lat, lon) float tensor (1.0 keep / 0.0 drop), fftshifted.

    The spatial mean is removed before the transform and added back after, so the
    DC bin (inside the band) and the field mean are preserved exactly.
    """
    mean = x.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fftshift(torch.fft.fft2(x - mean), dim=(-2, -1))
    fft = fft * mask                                   # complex * real -> complex
    out = torch.fft.ifft2(torch.fft.ifftshift(fft, dim=(-2, -1))).real
    return out + mean


def _crop_slice(coord, lo, hi, tol=1e-6):
    """Contiguous index slice of an ascending 1D coord within [lo, hi] (inclusive)."""
    idx = np.where((coord >= lo - tol) & (coord <= hi + tol))[0]
    if idx.size == 0:
        raise ValueError(f"No grid points in [{lo}, {hi}]; coord range "
                         f"[{coord.min()}, {coord.max()}]")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _filter_and_crop(arr, mask, lat_sl, lon_sl, device, chunk):
    """Global low-pass every trailing (lat, lon) slice of `arr`, then crop to
    (lat_sl, lon_sl). Streams through the GPU in blocks of `chunk`; cropping
    happens on-device so only the small Taiwan window returns to host RAM.
    Returns a float32 array with cropped lat/lon dimensions.
    """
    full_h, full_w = mask.shape
    flat = arr.reshape(-1, full_h, full_w)
    n = flat.shape[0]
    ch = lat_sl.stop - lat_sl.start
    cw = lon_sl.stop - lon_sl.start
    out = np.empty((n, ch, cw), dtype=np.float32)
    for i in range(0, n, chunk):
        block = torch.from_numpy(np.ascontiguousarray(flat[i:i + chunk], dtype=np.float32))
        block = block.to(device, non_blocking=True)
        filtered = lowpass_batch(block, mask)          # global filter
        cropped = filtered[:, lat_sl, lon_sl]          # crop AFTER low-pass
        out[i:i + cropped.shape[0]] = cropped.to("cpu", dtype=torch.float32).numpy()
    return out.reshape(*arr.shape[:-2], ch, cw)


# ---------------------------------------------------------------------------
# Per-file processing (streamed netCDF4 write, cropped to Taiwan domain)
# ---------------------------------------------------------------------------
def process_file(in_path, out_path, cutoff, lat_bounds, lon_bounds, device, chunk):
    if os.path.exists(out_path):
        return f"skip (exists): {out_path}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"

    with nc.Dataset(in_path, "r") as src, nc.Dataset(tmp_path, "w") as dst:
        glat = np.asarray(src.variables["latitude"][:], dtype=np.float64)
        glon = np.asarray(src.variables["longitude"][:], dtype=np.float64)
        lat_sl = _crop_slice(glat, *lat_bounds)
        lon_sl = _crop_slice(glon, *lon_bounds)
        mask = build_radial_lowpass_mask(glat, glon, cutoff, device)   # full-grid freq

        crop_len = {"latitude": lat_sl.stop - lat_sl.start,
                    "longitude": lon_sl.stop - lon_sl.start}

        # Dimensions -- latitude/longitude become the cropped Taiwan sizes.
        for name, dim in src.dimensions.items():
            if name in crop_len:
                dst.createDimension(name, crop_len[name])
            else:
                dst.createDimension(name, None if dim.isunlimited() else len(dim))
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

        for name, var in src.variables.items():
            attrs = {k: var.getncattr(k) for k in var.ncattrs()}
            fill = attrs.pop("_FillValue", None)
            out_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill)
            out_var.setncatts(attrs)

            dims = var.dimensions
            if name == "latitude":
                out_var[:] = glat[lat_sl]
            elif name == "longitude":
                out_var[:] = glon[lon_sl]
            elif "latitude" in dims and "longitude" in dims:
                data = np.asarray(var[:])                        # one variable in RAM
                out_var[:] = _filter_and_crop(data, mask, lat_sl, lon_sl, device, chunk)
                del data
            else:
                out_var[:] = var[:]                              # non-spatial: copy

    os.replace(tmp_path, out_path)
    return f"done: {out_path}  [{device}]"


def _worker(task):
    in_path, out_path, cutoff, lat_bounds, lon_bounds, device, chunk = task
    try:
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        return process_file(in_path, out_path, cutoff, lat_bounds, lon_bounds, device, chunk)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return f"FAIL {in_path}: {type(exc).__name__}: {exc}"


def build_tasks(src, dst, subdir, cutoff, lat_bounds, lon_bounds, gpu_ids, chunk):
    in_dir = Path(src) / subdir
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    files = sorted(in_dir.glob("*_sfc.nc")) + sorted(in_dir.glob("*_upper.nc"))
    devices = [f"cuda:{g}" for g in gpu_ids] if gpu_ids else ["cpu"]
    tasks = []
    for idx, in_path in enumerate(files):
        rel = in_path.relative_to(src)
        out_path = str(Path(dst) / rel)
        device = devices[idx % len(devices)]
        tasks.append((str(in_path), out_path, cutoff, lat_bounds, lon_bounds, device, chunk))
    return tasks


def main(src, dst, subdir, cutoff, lat_bounds, lon_bounds, gpu_ids, chunk):
    if gpu_ids and not torch.cuda.is_available():
        print("WARNING: CUDA not available -- falling back to CPU.")
        gpu_ids = []
    tasks = build_tasks(src, dst, subdir, cutoff, lat_bounds, lon_bounds, gpu_ids, chunk)

    print(f"Low-pass + crop {len(tasks)} files from {Path(src) / subdir}")
    print(f"  cutoff={cutoff} cycles/degree (radial 2D low-pass, applied globally)")
    print(f"  crop lat={lat_bounds} lon={lon_bounds}")
    print(f"  devices={sorted({t[5] for t in tasks})}  chunk={chunk}")
    print(f"  output -> {Path(dst) / subdir}")

    if len(gpu_ids) <= 1:
        for task in tasks:
            print(_worker(task), flush=True)
        return

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


# ---------------------------------------------------------------------------
# Post-run sanity check
# ---------------------------------------------------------------------------
def check_outputs_no_nan(dst, subdir):
    """Verify every generated file under dst/subdir has no NaN in any variable.

    Returns the list of (path, var_name) pairs that contain NaNs (empty if clean).
    """
    out_dir = Path(dst) / subdir
    files = sorted(out_dir.glob("*_sfc.nc")) + sorted(out_dir.glob("*_upper.nc"))

    bad = []
    for path in files:
        with nc.Dataset(path, "r") as ds:
            for name, var in ds.variables.items():
                data = np.asarray(var[:])
                if np.issubdtype(data.dtype, np.floating) and np.isnan(data).any():
                    bad.append((str(path), name))

    if bad:
        print(f"NaN CHECK FAILED: {len(bad)} (file, variable) pairs contain NaNs")
        for path, name in bad:
            print(f"  NaN found: {path}  var={name}")
    else:
        print(f"NaN CHECK PASSED: {len(files)} files, no NaNs found")
    return bad


if __name__ == "__main__":
    SRC = "/tmp2/b12902101/hres_tw_forecast_0.25deg_larger"
    DST = "/tmp3/b12902101/hres_tw_forecast_0.25deg_low"
    SUBDIR = "2020/202003"          # March 2020 only
    CUTOFF = 0.2                     # cycles/degree, radial low-pass (Nyquist = 2)
    LAT_BOUNDS = (2.5, 41.25)        # Taiwan domain, matches hres_tw_forecast_0.25deg
    LON_BOUNDS = (97.5, 147.25)
    GPU_IDS = [0, 1, 2, 3]                    # e.g. [0, 1, 2, 3] to fan across GPUs; [] for CPU
    CHUNK = 64                       # 2D slices per GPU batch (bounds GPU memory)

    # main(SRC, DST, SUBDIR, CUTOFF, LAT_BOUNDS, LON_BOUNDS, GPU_IDS, CHUNK)
    check_outputs_no_nan(SRC, SUBDIR)
