#!/usr/bin/env python3
"""
draw_comparison_AuroraTW_era5.py

Compare AuroraTW vs ERA5 (upper.nc + sfc.nc) with:
- side-by-side plots (left Aurora, right ERA5)
- color scale (vmin/vmax) standardized by ERA5 (ground truth)
- automatic var name mapping (Aurora -> ERA5) via JSON or CLI pairs
- optional lat/lon cropping (any order accepted)
- automatic lon range normalization (-180..180 vs 0..360)
- regrid/interp Aurora onto ERA5 grid before plotting
- compute L1 loss (MAE) per var/level and show as centered header
- output format option: png/pdf/svg/jpg/... (PDF/SVG/EPS avoids white-grid via rasterized=True)
- COLORMAPS: 'viridis' for Surface vars, 'plasma' for Atmos/Level vars.

Example:
python draw_comparison_AuroraTW_era5.py \
  --aurora_file Aurora.nc \
  --era5_upper_file upper.nc \
  --era5_sfc_file sfc.nc \
  --output_dir plots_compare \
  --latitude 39.5 5 \
  --longitude 100 144.75 \
  --map_json varmap.json \
  --map_mode aur_to_era \
  --fmt pdf
"""

import os
import json
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

TIME_DIMS = ["valid_time", "time", "history"]
LAT_NAMES = ["latitude", "lat", "Latitude", "LAT"]
LON_NAMES = ["longitude", "lon", "Longitude", "LON"]
LEVEL_DIMS = ["pressure_level", "level", "plev", "isobaricInhPa"]


def _first_existing(name_list, ds_or_da):
    for n in name_list:
        if n in getattr(ds_or_da, "dims", {}) or n in getattr(ds_or_da, "coords", {}):
            return n
    return None


def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Rename lat/lon to latitude/longitude when needed and sort coords ascending."""
    lat = _first_existing(LAT_NAMES, ds)
    lon = _first_existing(LON_NAMES, ds)

    rename = {}
    if lat and lat != "latitude":
        rename[lat] = "latitude"
    if lon and lon != "longitude":
        rename[lon] = "longitude"
    if rename:
        ds = ds.rename(rename)

    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    return ds


def maybe_select_first_time(da: xr.DataArray) -> xr.DataArray:
    """If any known time-like dim exists, select index 0."""
    for tdim in TIME_DIMS:
        if tdim in da.dims:
            da = da.isel({tdim: 0})
    return da


def select_latlon_range(da: xr.DataArray, lat_range, lon_range, debug_name: str = "") -> xr.DataArray:
    """Crop lat/lon; accept either order from user."""
    if lat_range is None or lon_range is None:
        return da

    lat1, lat2 = lat_range
    lon1, lon2 = lon_range

    lat_min, lat_max = (lat1, lat2) if lat1 <= lat2 else (lat2, lat1)
    lon_min, lon_max = (lon1, lon2) if lon1 <= lon2 else (lon2, lon1)

    out = da.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    if out.sizes.get("latitude", 0) == 0 or out.sizes.get("longitude", 0) == 0:
        try:
            print(
                f"[WARN] Empty crop {debug_name or da.name}: "
                f"lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}. "
                f"Available lat [{float(da.latitude.min())}, {float(da.latitude.max())}], "
                f"lon [{float(da.longitude.min())}, {float(da.longitude.max())}]"
            )
        except Exception:
            print(f"[WARN] Empty crop {debug_name or da.name}.")
    return out


def find_level_dim(da: xr.DataArray):
    for d in LEVEL_DIMS:
        if d in da.dims:
            return d
    return None


def finite_minmax(arr: xr.DataArray, robust=False, q=(2, 98)):
    """Compute vmin/vmax from ERA5 only."""
    vals = np.asarray(arr.values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    if robust:
        vmin = float(np.percentile(vals, q[0]))
        vmax = float(np.percentile(vals, q[1]))
    else:
        vmin = float(vals.min())
        vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps
    return vmin, vmax


def mae_l1(da_a: xr.DataArray, da_e: xr.DataArray) -> float:
    """Mean absolute error on overlapping finite values."""
    a = np.asarray(da_a.values)
    e = np.asarray(da_e.values)
    mask = np.isfinite(a) & np.isfinite(e)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(a[mask] - e[mask])))


def normalize_lon_to_match(da_src: xr.DataArray, da_tgt: xr.DataArray) -> xr.DataArray:
    """Shift src longitude range to match target (0..360 vs -180..180)."""
    if "longitude" not in da_src.coords or "longitude" not in da_tgt.coords:
        return da_src

    src_lon = np.asarray(da_src["longitude"].values)
    tgt_lon = np.asarray(da_tgt["longitude"].values)
    if src_lon.size == 0 or tgt_lon.size == 0:
        return da_src

    src_min = float(np.nanmin(src_lon))
    tgt_min = float(np.nanmin(tgt_lon))

    # Target 0..360, src -180..180
    if tgt_min >= 0 and src_min < 0:
        da_src = da_src.assign_coords(longitude=((da_src["longitude"] + 360) % 360))
        da_src = da_src.sortby("longitude")

    # Target -180..180, src 0..360
    if tgt_min < 0 and src_min >= 0:
        da_src = da_src.assign_coords(longitude=(((da_src["longitude"] + 180) % 360) - 180))
        da_src = da_src.sortby("longitude")

    return da_src


def regrid_aurora_to_era5(da_a: xr.DataArray, da_e: xr.DataArray) -> xr.DataArray:
    """Interpolate Aurora field to ERA5 lat/lon grid."""
    da_a = normalize_lon_to_match(da_a, da_e)
    da_ai = da_a.interp(
        latitude=da_e["latitude"],
        longitude=da_e["longitude"],
        method="linear",
        kwargs={"fill_value": np.nan},
    )
    return da_ai


def plot_side_by_side(
    args,
    da_left: xr.DataArray,
    da_right: xr.DataArray,
    title_left: str,
    title_right: str,
    outpath_no_ext: str,
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int,
    fmt: str,
    mae_value: float,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)

    x = da_right["longitude"].values
    y = da_right["latitude"].values

    # Avoid PDF/SVG "white grid" artifact by rasterizing only the mesh for vector outputs
    is_vector = fmt.lower() in ["pdf", "svg", "eps"]
    raster_kw = dict(rasterized=is_vector)

    im0 = axes[0].pcolormesh(
        x, y, da_left.values,
        shading="nearest",
        vmin=vmin, vmax=vmax, cmap=cmap,
        linewidth=0, edgecolors="none", antialiased=False,
        **raster_kw
    )
    im1 = axes[1].pcolormesh(
        x, y, da_right.values,
        shading="nearest",
        vmin=vmin, vmax=vmax, cmap=cmap,
        linewidth=0, edgecolors="none", antialiased=False,
        **raster_kw
    )

    if args.show_var_title:
        axes[0].set_title(title_left, fontsize=16, fontweight='bold')
        axes[1].set_title(title_right, fontsize=16, fontweight='bold')
    
    axes[0].set_xlabel("longitude")
    axes[1].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    axes[1].set_ylabel("latitude")

    mae_str = f"{mae_value:.4g}" if np.isfinite(mae_value) else "NaN"
    # show if not
    if args.show_mae:
        fig.suptitle(f"MAE (L1) = {mae_str}", fontsize=14, y=1.02)

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.12)

    cbar = fig.colorbar(im1, cax=cax)
    # cbar.set_label("value (scale from ERA5)")

    fig.subplots_adjust(top=0.90, wspace=0.15)

    os.makedirs(os.path.dirname(outpath_no_ext), exist_ok=True)
    outpath = f"{outpath_no_ext}.{fmt}"

    raster_fmt = fmt.lower() in ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    plt.savefig(outpath, dpi=dpi if raster_fmt else None, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def open_era5_merged(upper_path: str, sfc_path: str) -> xr.Dataset:
    upper = standardize_latlon(xr.open_dataset(upper_path))
    sfc = standardize_latlon(xr.open_dataset(sfc_path))
    return xr.merge([upper, sfc], compat="no_conflicts", join="outer")


def parse_map_pairs(map_pairs):
    """Parse --map entries like ['surf_2t=t2m', ...] into dict."""
    if not map_pairs:
        return {}
    out = {}
    for p in map_pairs:
        if "=" not in p:
            raise ValueError(f"Bad --map entry '{p}'. Use like --map surf_2t=t2m")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_map_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("--map_json must be a JSON object/dict.")
    return {str(k): str(v) for k, v in obj.items()}


def build_var_pairs(aur: xr.Dataset, era: xr.Dataset, mapping: dict, map_mode: str):
    """
    Returns list of tuples: (aur_var, era_var)
    map_mode:
      - "aur_to_era": mapping keys Aurora, values ERA5
      - "era_to_aur": mapping keys ERA5, values Aurora
    """
    pairs = []
    if mapping:
        if map_mode == "aur_to_era":
            pairs = [(aur_v, era_v) for aur_v, era_v in mapping.items()]
        else:
            pairs = [(aur_v, era_v) for era_v, aur_v in mapping.items()]
        return pairs

    common = [v for v in aur.data_vars if v in era.data_vars]
    return [(v, v) for v in common]


def compare_and_plot(
    args,
    aurora_path: str,
    era5_upper_path: str,
    era5_sfc_path: str,
    output_dir: str,
    mapping: dict,
    map_mode: str,
    lat_range=None,
    lon_range=None,
    cmap="cividis",
    robust=False,
    dpi=300,
    fmt="png",
):
    aur = standardize_latlon(xr.open_dataset(aurora_path))
    era = open_era5_merged(era5_upper_path, era5_sfc_path)

    pairs = build_var_pairs(aur, era, mapping, map_mode)

    print(f"[INFO] Aurora vars: {len(list(aur.data_vars))}")
    print(f"[INFO] ERA5 vars (merged): {len(list(era.data_vars))}")
    print(f"[INFO] Var pairs to plot: {len(pairs)}")
    if not pairs:
        print("[WARN] No variable pairs found. Provide --map/--map_json or ensure names overlap.")
        return

    base_name = (
        f"Aurora_{os.path.splitext(os.path.basename(aurora_path))[0]}"
        f"__ERA5_{os.path.splitext(os.path.basename(era5_upper_path))[0]}+{os.path.splitext(os.path.basename(era5_sfc_path))[0]}"
    )
    out_base = os.path.join(output_dir, base_name)

    for aur_var, era_var in pairs:
        if aur_var not in aur.data_vars:
            print(f"[WARN] Aurora missing '{aur_var}', skip (mapped to ERA5 '{era_var}').")
            continue
        if era_var not in era.data_vars:
            print(f"[WARN] ERA5 missing '{era_var}', skip (mapped from Aurora '{aur_var}').")
            continue

        da_a = maybe_select_first_time(aur[aur_var])
        da_e = maybe_select_first_time(era[era_var])

        # Crop first (cheap), then regrid
        da_a = select_latlon_range(da_a, lat_range, lon_range, debug_name=f"Aurora:{aur_var}")
        da_e = select_latlon_range(da_e, lat_range, lon_range, debug_name=f"ERA5:{era_var}")

        lev_dim_e = find_level_dim(da_e)
        lev_dim_a = find_level_dim(da_a)

        # 2D lat/lon case (Surface Variables -> viridis)
        if set(da_e.dims) == {"latitude", "longitude"} and set(da_a.dims) == {"latitude", "longitude"}:
            da_a_rg = regrid_aurora_to_era5(da_a, da_e)

            vmin, vmax = finite_minmax(da_e, robust=robust)
            if vmin is None:
                print(f"[WARN] {aur_var} vs {era_var}: ERA5 all-NaN, skip.")
                continue

            mae = mae_l1(da_a_rg, da_e)

            outpath_no_ext = os.path.join(out_base, f"{aur_var}__vs__{era_var}")
            plot_side_by_side(
                args,   
                da_left=da_a_rg,
                da_right=da_e,
                # title_left=f"{aur_var} (AuroraTW)",
                # title_right=f"{era_var} (ERA5)",
                title_left=f"Model Prediction",
                title_right=f"Ground Truth",
                outpath_no_ext=outpath_no_ext,
                cmap="viridis",  # Hardcoded for Surface per request
                vmin=vmin,
                vmax=vmax,
                dpi=dpi,
                fmt=fmt,
                mae_value=mae,
            )
            continue

        # Level case (Atmos Variables -> plasma)
        if lev_dim_e is not None:
            levels = da_e[lev_dim_e].values
            for lev in levels:
                try:
                    da_e_lev = da_e.sel({lev_dim_e: lev})
                except Exception:
                    continue

                if set(da_e_lev.dims) != {"latitude", "longitude"}:
                    print(f"[WARN] {aur_var}/{era_var}@{lev}: ERA5 dims {da_e_lev.dims} not 2D, skip.")
                    continue

                if lev_dim_a is None:
                    print(f"[WARN] {aur_var}/{era_var}@{lev}: ERA5 has levels but Aurora doesn't, skip.")
                    continue

                try:
                    da_a_lev = da_a.sel({lev_dim_a: lev})
                except Exception:
                    try:
                        da_a_lev = da_a.sel({lev_dim_a: lev}, method="nearest")
                    except Exception:
                        print(f"[WARN] {aur_var}/{era_var}@{lev}: cannot match Aurora level, skip.")
                        continue

                if set(da_a_lev.dims) != {"latitude", "longitude"}:
                    print(f"[WARN] {aur_var}/{era_var}@{lev}: Aurora dims {da_a_lev.dims} not 2D, skip.")
                    continue

                da_a_lev_rg = regrid_aurora_to_era5(da_a_lev, da_e_lev)

                vmin, vmax = finite_minmax(da_e_lev, robust=robust)
                if vmin is None:
                    print(f"[WARN] {aur_var}/{era_var}@{lev}: ERA5 all-NaN, skip.")
                    continue

                mae = mae_l1(da_a_lev_rg, da_e_lev)

                outpath_no_ext = os.path.join(out_base, f"{aur_var}__vs__{era_var}__{lev_dim_e}_{lev}")
                plot_side_by_side(
                    args,
                    da_left=da_a_lev_rg,
                    da_right=da_e_lev,
                    # title_left=f"{aur_var} ({lev_dim_e}={lev}) AuroraTW",
                    # title_right=f"{era_var} ({lev_dim_e}={lev}) ERA5",
                    title_left=f"Model Prediction",
                    title_right=f"Ground Truth",
                    outpath_no_ext=outpath_no_ext,
                    cmap="plasma",  # Hardcoded for Atmos/Level per request
                    vmin=vmin,
                    vmax=vmax,
                    dpi=dpi,
                    fmt=fmt,
                    mae_value=mae,
                )
            continue

        print(f"[WARN] Skipping {aur_var} vs {era_var}: unsupported dims Aurora{da_a.dims} / ERA5{da_e.dims}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare AuroraTW vs ERA5: side-by-side plots. Surface='viridis', Atmos='plasma'."
    )
    parser.add_argument("--aurora_file", type=str, required=True)
    parser.add_argument("--era5_upper_file", type=str, required=True)
    parser.add_argument("--era5_sfc_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="plots_compare_out")

    parser.add_argument("--show_var_title", action="store_true", help="Show variable names as subplot titles.")
    parser.add_argument("--show_mae", action="store_true", help="Show MAE in the main title.")

    parser.add_argument("--latitude", type=float, nargs=2, metavar=("LAT1", "LAT2"),
                        help="Latitude range (any order OK), e.g. --latitude 39.5 5")
    parser.add_argument("--longitude", type=float, nargs=2, metavar=("LON1", "LON2"),
                        help="Longitude range (any order OK).")

    # This arg is technically ignored now for surf/atmos logic, but kept for compatibility
    parser.add_argument("--cmap", type=str, default="cividis",
                        help="Default cmap (ignored for Surf/Atmos vars which use viridis/plasma).")
    parser.add_argument("--robust", action="store_true",
                        help="Use ERA5 percentile vmin/vmax (2-98%) for color scale.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fmt", type=str, default="png",
                        help="Output format: png, pdf, svg, jpg, etc. (matplotlib-supported).")

    # Mapping options
    parser.add_argument("--map", action="append", default=[],
                        help="Var mapping pair. Example: --map surf_2t=t2m (direction via --map_mode)")
    parser.add_argument("--map_json", type=str, default=None,
                        help="Path to JSON mapping dict. Direction via --map_mode.")
    parser.add_argument("--map_mode", type=str, default="aur_to_era",
                        choices=["aur_to_era", "era_to_aur"],
                        help="Mapping direction. aur_to_era means: AuroraName=ERA5Name.")

    args = parser.parse_args()

    mapping = {}
    if args.map_json:
        mapping.update(load_map_json(args.map_json))
    mapping.update(parse_map_pairs(args.map))

    lat_range = tuple(args.latitude) if args.latitude is not None else None
    lon_range = tuple(args.longitude) if args.longitude is not None else None

    compare_and_plot(
        args,
        aurora_path=args.aurora_file,
        era5_upper_path=args.era5_upper_file,
        era5_sfc_path=args.era5_sfc_file,
        output_dir=args.output_dir,
        mapping=mapping,
        map_mode=args.map_mode,
        lat_range=lat_range,
        lon_range=lon_range,
        cmap=args.cmap,
        robust=args.robust,
        dpi=args.dpi,
        fmt=args.fmt,
    )


if __name__ == "__main__":
    main()