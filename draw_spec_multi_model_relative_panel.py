#!/usr/bin/env python
import os
from pathlib import Path
import argparse
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# ============================================================
# Defaults (override via CLI)
# ============================================================

START = "2023-01-01 01:00"
END   = "2023-12-27 23:00"
LEAD_HOURS = 1

REF_BASE_DIR      = "/work/yunye0121/era5_tw"
REF_UPPER_TMPL    = "{YYYY}/{YYYYMM}/{YYYYMMDD}/{YYYYMMDD}{HH}_upper.nc"
REF_SFC_TMPL      = "{YYYY}/{YYYYMM}/{YYYYMMDD}/{YYYYMMDD}{HH}_sfc.nc"

# Pred files are expected directly under each --pred dir
PRED_FILE_TMPL    = "{YYYYMMDD}_{HH}0000+{LEAD}hr.nc"

SFC_VARS_REF      = ["t2m", "u10", "v10", "msl"]
UPPER_VARS_REF    = ["u", "v", "t", "q", "z"]

SFC_MAP_TO_PRED   = {"t2m": "surf_2t", "u10": "surf_10u", "v10": "surf_10v", "msl": "surf_msl"}
UPPER_MAP_TO_PRED = {"u": "atmos_u", "v": "atmos_v", "t": "atmos_t", "q": "atmos_q", "z": "atmos_z"}

REF_LEVEL_DIM   = "pressure_level"
PRED_LEVEL_DIM  = "level"
TIME_DIM        = "time"
HIST_DIM        = "history"
TIME_INDEX      = 0
HIST_INDEX      = 0

# Crop
USE_LAT_BAND     = True
LAT_MAX, LAT_MIN = 39.75, 5.0
USE_LON_SECTOR   = True
LON_MIN, LON_MAX = 100.0, 145.0

# Spectrum options
COS_LAT_WEIGHTS  = False
USE_WB2          = True
PRED_TIME_MODE   = "valid"   # "valid" uses valid timestamp in filename; "base" uses valid-lead

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Paper-style zonal spectra panel (2xN): abs spectrum (top) and ratio (bottom).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--ref-base-dir", default=REF_BASE_DIR)
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--lead", type=int, default=LEAD_HOURS)

    p.add_argument("--pred", action="append", default=[],
                   help='Repeatable. Format: "NAME=/abs/path/to/pred_dir_with_nc_files"')

    p.add_argument("--pred-time-mode", choices=["base", "valid"], default=PRED_TIME_MODE)

    p.add_argument("--use-wb2", action="store_true", default=USE_WB2)
    p.add_argument("--no-wb2", action="store_false", dest="use_wb2")

    p.add_argument("--coslat", action="store_true", default=COS_LAT_WEIGHTS)
    p.add_argument("--no-coslat", action="store_false", dest="coslat")

    p.add_argument("--use-lat-band", action="store_true", default=USE_LAT_BAND)
    p.add_argument("--no-lat-band", action="store_false", dest="use_lat_band")
    p.add_argument("--lat-min", type=float, default=LAT_MIN)
    p.add_argument("--lat-max", type=float, default=LAT_MAX)

    p.add_argument("--use-lon-sector", action="store_true", default=USE_LON_SECTOR)
    p.add_argument("--no-lon-sector", action="store_false", dest="use_lon_sector")
    p.add_argument("--lon-min", type=float, default=LON_MIN)
    p.add_argument("--lon-max", type=float, default=LON_MAX)

    p.add_argument("--outdir", default=None)
    p.add_argument("--debug", type=int, default=0)

    # Panel spec (repeatable)
    #   sfc:<var>:<label>
    #   upper:<var>:<level_hPa>:<label>
    p.add_argument("--panel", action="append", default=[],
                   help=('Repeatable. '
                         'sfc:<var>:<label> OR upper:<var>:<level_hPa>:<label>. '
                         'Example: --panel "sfc:t2m:2mT" --panel "upper:z:500:Z500"'))

    p.add_argument("--panel-out", default="",
                   help='Panel base filename (no extension needed). Default: PANEL')
    p.add_argument("--panel-title", default="",
                   help="Optional figure suptitle.")

    p.add_argument("--fig-format", choices=["png", "pdf"], default="png",
                   help="Output format for the panel figure.")

    return p.parse_args()

def parse_pred_models(pred_args):
    models = []
    for item in pred_args:
        if "=" not in item:
            raise ValueError(f'Bad --pred "{item}". Expected NAME=/abs/path')
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name:
            raise ValueError(f'Bad --pred "{item}": empty NAME')
        models.append((name, path))

    # dedupe by name
    seen, uniq = set(), []
    for n, pth in models:
        if n in seen:
            continue
        seen.add(n)
        uniq.append((n, pth))
    return uniq

def parse_panels(panel_args):
    """
    Returns list of dicts: {domain, ref, pred, level, label}
    """
    panels = []
    for spec in panel_args:
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError(f"Bad --panel '{spec}'")

        domain = parts[0].strip().lower()

        if domain == "sfc":
            if len(parts) != 3:
                raise ValueError(f"Bad --panel '{spec}'. Use sfc:<var>:<label>")
            var = parts[1].strip()
            label = parts[2].strip()
            if var not in SFC_VARS_REF:
                raise ValueError(f"Panel var '{var}' not in SFC_VARS_REF={SFC_VARS_REF}")
            v_pred = SFC_MAP_TO_PRED.get(var)
            if v_pred is None:
                raise ValueError(f"No mapping for sfc var '{var}'")
            panels.append({"domain":"sfc","ref":var,"pred":v_pred,"level":None,"label":label})

        elif domain == "upper":
            if len(parts) != 4:
                raise ValueError(f"Bad --panel '{spec}'. Use upper:<var>:<level_hPa>:<label>")
            var = parts[1].strip()
            lvl = int(parts[2].strip())
            label = parts[3].strip()
            if var not in UPPER_VARS_REF:
                raise ValueError(f"Panel var '{var}' not in UPPER_VARS_REF={UPPER_VARS_REF}")
            v_pred = UPPER_MAP_TO_PRED.get(var)
            if v_pred is None:
                raise ValueError(f"No mapping for upper var '{var}'")
            panels.append({"domain":"upper","ref":var,"pred":v_pred,"level":lvl,"label":label})

        else:
            raise ValueError(f"Bad --panel '{spec}': domain must be sfc or upper")

    return panels

# ============================================================
# Paths
# ============================================================

def build_ref_paths(ref_base: str, valid_ts: pd.Timestamp):
    YYYY     = valid_ts.strftime("%Y")
    YYYYMM   = valid_ts.strftime("%Y%m")
    YYYYMMDD = valid_ts.strftime("%Y%m%d")
    HH       = valid_ts.strftime("%H")
    upper_rel = REF_UPPER_TMPL.format(YYYY=YYYY, YYYYMM=YYYYMM, YYYYMMDD=YYYYMMDD, HH=HH)
    sfc_rel   = REF_SFC_TMPL  .format(YYYY=YYYY, YYYYMM=YYYYMM, YYYYMMDD=YYYYMMDD, HH=HH)
    return str(Path(ref_base) / upper_rel), str(Path(ref_base) / sfc_rel)

def build_pred_path(pred_dir: str, valid_ts: pd.Timestamp, lead: int, pred_time_mode: str):
    base_ts = valid_ts - pd.Timedelta(hours=lead)
    stamp = valid_ts if pred_time_mode == "valid" else base_ts
    fname = PRED_FILE_TMPL.format(
        YYYYMMDD=stamp.strftime("%Y%m%d"),
        HH=stamp.strftime("%H"),
        LEAD=lead,
    )
    return str(Path(pred_dir) / fname)

# ============================================================
# Data helpers
# ============================================================

def _select_time_hist(da: xr.DataArray) -> xr.DataArray:
    if HIST_DIM in da.dims:
        da = da.isel({HIST_DIM: HIST_INDEX})
    if TIME_DIM in da.dims:
        da = da.isel({TIME_DIM: TIME_INDEX})
    return da

def _crop_latlon(da: xr.DataArray, args) -> xr.DataArray:
    if args.use_lat_band:
        da = da.sel(latitude=slice(args.lat_max, args.lat_min))
    if args.use_lon_sector:
        da = da.sel(longitude=slice(args.lon_min, args.lon_max))
    return da.sortby("longitude")

# ============================================================
# Spectrum
# ============================================================

def zonal_spectrum_fft(field_2d: xr.DataArray, coslat_weights: bool):
    lat = field_2d["latitude"].values
    arr = field_2d.values.astype(float)

    # remove zonal mean per latitude
    arr = arr - np.nanmean(arr, axis=-1, keepdims=True)

    nlon = arr.shape[-1]
    F = np.fft.rfft(arr, axis=-1)

    P = np.abs(F) ** 2
    alpha = np.ones(P.shape[-1], dtype=float)
    if nlon % 2 == 0 and P.shape[-1] >= 2:
        alpha[1:-1] = 2.0
    else:
        alpha[1:] = 2.0
    S = (P * alpha[None, :]) / (nlon ** 2)

    if coslat_weights:
        w = np.cos(np.deg2rad(lat))
        w = np.clip(w, 0.0, None)
        w = w / np.nansum(w)
        E = np.nansum(S * w[:, None], axis=0)
    else:
        E = np.nanmean(S, axis=0)

    m = np.arange(S.shape[-1])
    return m, E

def _wb2_latband_mean(spec_xr, coslat_weights: bool = False):
    import xarray as xr

    if isinstance(spec_xr, xr.DataArray):
        spec_ds = spec_xr.to_dataset(name=spec_xr.name or "spectrum")
    else:
        spec_ds = spec_xr

    spec_var = list(spec_ds.data_vars)[0]

    wavename = None
    for c in spec_ds.coords:
        cl = c.lower()
        if "wavenumber" in cl or cl in ("k", "m") or cl.endswith("_k"):
            wavename = c
            break
    if wavename is None:
        wavename = spec_ds[spec_var].dims[-1]

    latname = None
    for dn in spec_ds[spec_var].dims:
        if dn.lower() in ("lat", "latitude", "y"):
            latname = dn
            break

    if latname is not None:
        if coslat_weights:
            w = np.cos(np.deg2rad(spec_ds[latname].values))
            w = np.clip(w, 0.0, None)
            w = w / np.nansum(w)
            E_mean = (spec_ds[spec_var] * xr.DataArray(w, dims=[latname])).sum(dim=latname)
        else:
            E_mean = spec_ds[spec_var].mean(dim=latname)
    else:
        E_mean = spec_ds[spec_var]

    coord_vals = spec_ds[wavename].values
    if np.issubdtype(coord_vals.dtype, np.number) and np.nanmax(coord_vals) >= 1:
        m = coord_vals
    else:
        m = np.arange(E_mean.sizes[E_mean.dims[-1]])

    return m, E_mean.values

def zonal_spectrum_wb2(field_2d: xr.DataArray, varname_for_wb2: str, coslat_weights: bool = False):
    from weatherbench2.derived_variables import ZonalEnergySpectrum
    ds_in = field_2d.to_dataset(name=varname_for_wb2)
    zes = ZonalEnergySpectrum(variable_name=varname_for_wb2)
    spec = zes.compute(ds_in)
    return _wb2_latband_mean(spec, coslat_weights=coslat_weights)

def compute_spectrum(field_2d: xr.DataArray, varname_for_wb2: str, args):
    if args.use_wb2:
        try:
            return zonal_spectrum_wb2(field_2d, varname_for_wb2=varname_for_wb2, coslat_weights=args.coslat)
        except ImportError:
            print("[warn] weatherbench2 not installed; fallback to FFT.")
        except Exception as e:
            print(f"[warn] WB2 failed ({e}); fallback to FFT.")
    return zonal_spectrum_fft(field_2d, coslat_weights=args.coslat)

# ============================================================
# Accumulation (time-mean spectrum)
# ============================================================

def _mean_over_time_integer_m(m_list, spec_list):
    if not m_list:
        return np.array([]), np.array([])
    L = min(len(m) for m in m_list)
    stack = np.vstack([s[:L] for s in spec_list])
    return np.arange(L), np.nanmean(stack, axis=0)

def accumulate_ref_spectrum_surface(var_ref: str, valid_times, args):
    m_list, s_list = [], []
    n_used = 0

    for ts in valid_times:
        _, ref_sfc_path = build_ref_paths(args.ref_base_dir, ts)
        if not os.path.exists(ref_sfc_path):
            continue
        try:
            ds = xr.open_dataset(ref_sfc_path)
        except Exception:
            continue

        if var_ref not in ds.data_vars:
            ds.close()
            continue

        da = _crop_latlon(_select_time_hist(ds[var_ref]), args)
        ds.close()

        if not {"latitude", "longitude"}.issubset(da.dims):
            continue

        m, E = compute_spectrum(da, varname_for_wb2=var_ref, args=args)
        if m.size:
            m_list.append(m)
            s_list.append(E)
            n_used += 1

    m_mean, E_mean = _mean_over_time_integer_m(m_list, s_list)
    return m_mean, E_mean, n_used

def accumulate_pred_spectrum_surface(var_pred: str, pred_dir: str, valid_times, args):
    m_list, s_list = [], []
    n_used = 0

    for ts in valid_times:
        pth = build_pred_path(pred_dir, ts, args.lead, args.pred_time_mode)
        if not os.path.exists(pth):
            continue
        try:
            ds = xr.open_dataset(pth)
        except Exception:
            continue

        if var_pred not in ds.data_vars:
            ds.close()
            continue

        da = _crop_latlon(_select_time_hist(ds[var_pred]), args)
        ds.close()

        if not {"latitude", "longitude"}.issubset(da.dims):
            continue

        m, E = compute_spectrum(da, varname_for_wb2=var_pred, args=args)
        if m.size:
            m_list.append(m)
            s_list.append(E)
            n_used += 1

    m_mean, E_mean = _mean_over_time_integer_m(m_list, s_list)
    return m_mean, E_mean, n_used

def accumulate_ref_spectrum_upper(var_ref: str, level: int, valid_times, args):
    m_list, s_list = [], []
    n_used = 0

    for ts in valid_times:
        ref_upper_path, _ = build_ref_paths(args.ref_base_dir, ts)
        if not os.path.exists(ref_upper_path):
            continue
        try:
            ds = xr.open_dataset(ref_upper_path)
        except Exception:
            continue

        if var_ref not in ds.data_vars:
            ds.close()
            continue

        da = _select_time_hist(ds[var_ref])
        if REF_LEVEL_DIM not in da.dims or level not in da[REF_LEVEL_DIM].values.tolist():
            ds.close()
            continue

        da = _crop_latlon(da.sel({REF_LEVEL_DIM: level}), args)
        ds.close()

        if not {"latitude", "longitude"}.issubset(da.dims):
            continue

        m, E = compute_spectrum(da, varname_for_wb2=var_ref, args=args)
        if m.size:
            m_list.append(m)
            s_list.append(E)
            n_used += 1

    m_mean, E_mean = _mean_over_time_integer_m(m_list, s_list)
    return m_mean, E_mean, n_used

def accumulate_pred_spectrum_upper(var_pred: str, level: int, pred_dir: str, valid_times, args):
    m_list, s_list = [], []
    n_used = 0

    for ts in valid_times:
        pth = build_pred_path(pred_dir, ts, args.lead, args.pred_time_mode)
        if not os.path.exists(pth):
            continue
        try:
            ds = xr.open_dataset(pth)
        except Exception:
            continue

        if var_pred not in ds.data_vars:
            ds.close()
            continue

        da = _select_time_hist(ds[var_pred])
        if PRED_LEVEL_DIM not in da.dims or level not in da[PRED_LEVEL_DIM].values.tolist():
            ds.close()
            continue

        da = _crop_latlon(da.sel({PRED_LEVEL_DIM: level}), args)
        ds.close()

        if not {"latitude", "longitude"}.issubset(da.dims):
            continue

        m, E = compute_spectrum(da, varname_for_wb2=var_pred, args=args)
        if m.size:
            m_list.append(m)
            s_list.append(E)
            n_used += 1

    m_mean, E_mean = _mean_over_time_integer_m(m_list, s_list)
    return m_mean, E_mean, n_used

# ============================================================
# Panel plotting (2xN) with bottom legend + aligned colors
# ============================================================

# def make_panel_2xN(panels, models, valid_hours, args, outdir, panel_base, fig_format, suptitle=None):
    N = len(panels)
    if N == 0:
        raise ValueError("No panels specified.")

    # ---- Fixed color mapping for models (global) ----
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_colors:
        base_colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    ccycler = cycle(base_colors)

    model_colors = {}
    for model_name, _ in models:
        model_colors[model_name] = next(ccycler)

    era5_color = "black"

    # ---- Figure and axes ----
    fig, axes = plt.subplots(2, N, figsize=(4.3 * N, 7.2), squeeze=False)

    band = f"lat {args.lat_min}–{args.lat_max}°" if args.use_lat_band else "all lat"
    sect = f" lon {args.lon_min}–{args.lon_max}°" if args.use_lon_sector else "all lon"
    region_txt = f"{band};{sect}"

    legend_handles = None
    legend_labels = None

    for j, p in enumerate(panels):
        ax_abs = axes[0, j]
        ax_rat = axes[1, j]

        domain = p["domain"]
        v_ref  = p["ref"]
        v_pred = p["pred"]
        lvl    = p["level"]
        label  = p["label"]

        # --- REF ---
        if domain == "sfc":
            _, E_ref, n_ref = accumulate_ref_spectrum_surface(v_ref, valid_hours, args)
        else:
            _, E_ref, n_ref = accumulate_ref_spectrum_upper(v_ref, lvl, valid_hours, args)

        if n_ref == 0 or E_ref.size == 0:
            ax_abs.text(0.5, 0.5, f"{label}\n(REF missing)", ha="center", va="center")
            ax_abs.axis("off")
            ax_rat.axis("off")
            continue

        m_ref = np.arange(len(E_ref))
        mask = m_ref >= 1

        # Top: ERA5 in fixed color
        ax_abs.loglog(
            m_ref[mask], E_ref[mask],
            label="ERA5",
            linewidth=2.7,
            color=era5_color
        )

        any_model = False

        for model_name, model_dir in models:
            # Always use same color for this model on BOTH axes
            color = model_colors[model_name]

            if domain == "sfc":
                _, E_p, n_p = accumulate_pred_spectrum_surface(v_pred, model_dir, valid_hours, args)
            else:
                _, E_p, n_p = accumulate_pred_spectrum_upper(v_pred, lvl, model_dir, valid_hours, args)

            if n_p == 0 or E_p.size == 0:
                continue

            L = min(len(E_ref), len(E_p))
            m = np.arange(L)
            mm = m[m >= 1]
            Eref = E_ref[:L]
            Ep   = E_p[:L]

            any_model = True

            # Top: abs spectrum
            ax_abs.loglog(
                mm, Ep[m >= 1],
                label=model_name,
                linewidth=1.5,
                color=color
            )

            # Bottom: ratio with SAME color
            ratio = Ep / (Eref + 1e-12)
            ax_rat.semilogx(
                mm, ratio[m >= 1],
                label=model_name,
                linewidth=1.5,
                color=color
            )

        # formatting
        ax_abs.set_title(label)
        ax_abs.grid(True, which="both", ls="--", alpha=0.4)
        ax_abs.set_xlabel("m")
        if j == 0:
            ax_abs.set_ylabel("Zonal spectral power")

        if any_model:
            ax_rat.axhline(1.0, linestyle="--", linewidth=1.0, color="0.3")
            ax_rat.grid(True, which="both", ls="--", alpha=0.4)
            ax_rat.set_xlabel("m")
            if j == 0:
                ax_rat.set_ylabel("E_pred / E_ERA5")
        else:
            ax_rat.text(0.5, 0.5, "No model overlap", ha="center", va="center")
            ax_rat.axis("off")

        # capture handles once (ERA5 + all models) from first working column
        if legend_handles is None:
            h, l = ax_abs.get_legend_handles_labels()
            if h and l:
                legend_handles, legend_labels = h, l

    if suptitle is None:
        suptitle = f"Zonal spectra (top) and ratio (bottom), lead={args.lead}h | {args.start} → {args.end}\nRegion: {region_txt}"
    fig.suptitle(suptitle, y=0.98, fontsize=12)

    # layout leaving room for legend
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])

    # global legend at bottom
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
            fontsize=10,
        )

    outpath = Path(outdir) / f"{panel_base}.{fig_format}"
    if outpath.suffix.lower() == ".pdf":
        fig.savefig(outpath)           # vector
    else:
        fig.savefig(outpath, dpi=300)  # raster
    plt.close(fig)
    print(f"[saved] panel: {outpath}")

# def make_panel_2xN(panels, models, valid_hours, args, outdir, panel_base, fig_format, suptitle=None):
    N = len(panels)
    if N == 0:
        raise ValueError("No panels specified.")

    # ---- Fixed color mapping for models (global) ----
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_colors:
        base_colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    ccycler = cycle(base_colors)

    model_colors = {}
    for model_name, _ in models:
        model_colors[model_name] = next(ccycler)

    era5_color = "black"

    # ---- Figure and axes ----
    fig, axes = plt.subplots(2, N, figsize=(4.3 * N, 7.2), squeeze=False)

    band = f"lat {args.lat_min}–{args.lat_max}°" if args.use_lat_band else "all lat"
    sect = f" lon {args.lon_min}–{args.lon_max}°" if args.use_lon_sector else "all lon"
    region_txt = f"{band};{sect}"

    legend_handles = None
    legend_labels = None

    for j, p in enumerate(panels):
        ax_abs = axes[0, j]
        ax_rat = axes[1, j]

        domain = p["domain"]
        v_ref  = p["ref"]
        v_pred = p["pred"]
        lvl    = p["level"]
        label  = p["label"]

        # --- REF ---
        if domain == "sfc":
            _, E_ref, n_ref = accumulate_ref_spectrum_surface(v_ref, valid_hours, args)
        else:
            _, E_ref, n_ref = accumulate_ref_spectrum_upper(v_ref, lvl, valid_hours, args)

        if n_ref == 0 or E_ref.size == 0:
            ax_abs.text(0.5, 0.5, f"{label}\n(REF missing)", ha="center", va="center")
            ax_abs.axis("off")
            ax_rat.axis("off")
            continue

        m_ref = np.arange(len(E_ref))
        mask = m_ref >= 1

        # Top: ERA5 in fixed color
        ax_abs.loglog(
            m_ref[mask], E_ref[mask],
            label="ERA5",
            linewidth=2.7,
            color=era5_color
        )

        any_model = False

        for model_name, model_dir in models:
            # Always use same color for this model on BOTH axes
            color = model_colors[model_name]

            if domain == "sfc":
                _, E_p, n_p = accumulate_pred_spectrum_surface(v_pred, model_dir, valid_hours, args)
            else:
                _, E_p, n_p = accumulate_pred_spectrum_upper(v_pred, lvl, model_dir, valid_hours, args)

            if n_p == 0 or E_p.size == 0:
                continue

            # =========================================================
            # NEW: Calculate Integrated Error for Label
            # =========================================================
            L = min(len(E_ref), len(E_p))
            m = np.arange(L)
            
            # Slice arrays to common length
            Eref_c = E_ref[:L]
            Ep_c   = E_p[:L]

            # Calculate error on m >= 1 (ignore zonal mean)
            valid_m = m >= 1
            if np.any(valid_m):
                diff = np.abs(Ep_c[valid_m] - Eref_c[valid_m])
                total_ref = np.sum(Eref_c[valid_m])
                if total_ref > 0:
                    err_pct = (np.sum(diff) / total_ref) * 100.0
                    label_str = f"{model_name} ({err_pct:.1f}%)"
                else:
                    label_str = model_name
            else:
                label_str = model_name
            # =========================================================

            any_model = True

            # Top: abs spectrum (use modified label)
            ax_abs.loglog(
                m[valid_m], Ep_c[valid_m],
                label=label_str,
                linewidth=1.5,
                color=color
            )

            # Bottom: ratio with SAME color
            ratio = Ep_c / (Eref_c + 1e-12)
            ax_rat.semilogx(
                m[valid_m], ratio[valid_m],
                label=model_name, # ratio plot doesn't need error in label
                linewidth=1.5,
                color=color
            )

        # formatting
        ax_abs.set_title(label)
        ax_abs.grid(True, which="both", ls="--", alpha=0.4)
        ax_abs.set_xlabel("m")
        if j == 0:
            ax_abs.set_ylabel("Zonal spectral power")

        if any_model:
            ax_rat.axhline(1.0, linestyle="--", linewidth=1.0, color="0.3")
            ax_rat.grid(True, which="both", ls="--", alpha=0.4)
            ax_rat.set_xlabel("m")
            if j == 0:
                ax_rat.set_ylabel("E_pred / E_ERA5")
        else:
            ax_rat.text(0.5, 0.5, "No model overlap", ha="center", va="center")
            ax_rat.axis("off")

        # capture handles once (ERA5 + all models) from first working column
        # We grab handles from ax_abs to ensure we get the "Model (X%)" labels
        if legend_handles is None:
            h, l = ax_abs.get_legend_handles_labels()
            if h and l:
                legend_handles, legend_labels = h, l

    if suptitle is None:
        suptitle = f"Zonal spectra (top) and ratio (bottom), lead={args.lead}h | {args.start} → {args.end}\nRegion: {region_txt}"
    fig.suptitle(suptitle, y=0.98, fontsize=12)

    # layout leaving room for legend
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])

    # global legend at bottom
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
            fontsize=10,
        )

    outpath = Path(outdir) / f"{panel_base}.{fig_format}"
    if outpath.suffix.lower() == ".pdf":
        fig.savefig(outpath)           # vector
    else:
        fig.savefig(outpath, dpi=300)  # raster
    plt.close(fig)
    print(f"[saved] panel: {outpath}")

# def make_panel_2xN(panels, models, valid_hours, args, outdir, panel_base, fig_format, suptitle=None):
    N = len(panels)
    if N == 0:
        raise ValueError("No panels specified.")

    # ---- Fixed color mapping for models (global) ----
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_colors:
        base_colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    ccycler = cycle(base_colors)

    model_colors = {}
    for model_name, _ in models:
        model_colors[model_name] = next(ccycler)

    era5_color = "black"

    # ---- Figure and axes ----
    fig, axes = plt.subplots(2, N, figsize=(4.3 * N, 7.2), squeeze=False)

    band = f"lat {args.lat_min}–{args.lat_max}°" if args.use_lat_band else "all lat"
    sect = f" lon {args.lon_min}–{args.lon_max}°" if args.use_lon_sector else "all lon"
    region_txt = f"{band};{sect}"

    legend_handles = None
    legend_labels = None

    for j, p in enumerate(panels):
        ax_abs = axes[0, j]
        ax_rat = axes[1, j]

        domain = p["domain"]
        v_ref  = p["ref"]
        v_pred = p["pred"]
        lvl    = p["level"]
        label  = p["label"]

        # --- REF ---
        if domain == "sfc":
            _, E_ref, n_ref = accumulate_ref_spectrum_surface(v_ref, valid_hours, args)
        else:
            _, E_ref, n_ref = accumulate_ref_spectrum_upper(v_ref, lvl, valid_hours, args)

        if n_ref == 0 or E_ref.size == 0:
            ax_abs.text(0.5, 0.5, f"{label}\n(REF missing)", ha="center", va="center")
            ax_abs.axis("off")
            ax_rat.axis("off")
            continue

        m_ref = np.arange(len(E_ref))
        mask = m_ref >= 1

        # Plot ERA5 (Reference)
        ax_abs.loglog(
            m_ref[mask], E_ref[mask],
            label="ERA5",
            linewidth=2.7,
            color=era5_color
        )

        any_model = False
        stats_lines = []  # List to hold error strings for this specific panel

        for model_name, model_dir in models:
            color = model_colors[model_name]

            if domain == "sfc":
                _, E_p, n_p = accumulate_pred_spectrum_surface(v_pred, model_dir, valid_hours, args)
            else:
                _, E_p, n_p = accumulate_pred_spectrum_upper(v_pred, lvl, model_dir, valid_hours, args)

            if n_p == 0 or E_p.size == 0:
                continue

            # Align lengths
            L = min(len(E_ref), len(E_p))
            m = np.arange(L)
            
            # Slice arrays
            Eref_c = E_ref[:L]
            Ep_c   = E_p[:L]
            valid_m = m >= 1

            # --- Calculate Variable-Wise Error ---
            if np.any(valid_m):
                diff = np.abs(Ep_c[valid_m] - Eref_c[valid_m])
                total_ref = np.sum(Eref_c[valid_m])
                if total_ref > 0:
                    err_pct = (np.sum(diff) / total_ref) * 100.0
                    # Append to stats list: "GraphCast: 4.2%"
                    stats_lines.append(f"{model_name}: {err_pct:.1f}%")

            any_model = True

            # Top: abs spectrum (Legend remains clean)
            ax_abs.loglog(
                m[valid_m], Ep_c[valid_m],
                label=model_name,
                linewidth=1.5,
                color=color
            )

            # Bottom: ratio
            ratio = Ep_c / (Eref_c + 1e-12)
            ax_rat.semilogx(
                m[valid_m], ratio[valid_m],
                label=model_name,
                linewidth=1.5,
                color=color
            )

        # --- Formatting & Stats Box ---
        ax_abs.set_title(label)
        ax_abs.grid(True, which="both", ls="--", alpha=0.4)
        ax_abs.set_xlabel("m")
        if j == 0:
            ax_abs.set_ylabel("Zonal spectral power")

        # Create the Text Box for this variable
        if stats_lines:
            # Join lines: "Diff w.r.t ERA5:\nModelA: 1.2%\nModelB: 3.4%"
            stats_text = "Diff (NIAE):\n" + "\n".join(stats_lines)
            
            # Place in top-right (usually empty in log-log spectral plots)
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax_abs.text(0.95, 0.95, stats_text, transform=ax_abs.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=props)

        if any_model:
            ax_rat.axhline(1.0, linestyle="--", linewidth=1.0, color="0.3")
            ax_rat.grid(True, which="both", ls="--", alpha=0.4)
            ax_rat.set_xlabel("m")
            if j == 0:
                ax_rat.set_ylabel("E_pred / E_ERA5")
        else:
            ax_rat.text(0.5, 0.5, "No model overlap", ha="center", va="center")
            ax_rat.axis("off")

        # Global Legend Handles (Grab once)
        if legend_handles is None:
            h, l = ax_abs.get_legend_handles_labels()
            if h and l:
                legend_handles, legend_labels = h, l

    if suptitle is None:
        suptitle = f"Zonal spectra (top) and ratio (bottom), lead={args.lead}h | {args.start} → {args.end}\nRegion: {region_txt}"
    fig.suptitle(suptitle, y=0.98, fontsize=12)

    # layout leaving room for legend
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])

    # Global Legend (Bottom)
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
            fontsize=10,
        )

    outpath = Path(outdir) / f"{panel_base}.{fig_format}"
    if outpath.suffix.lower() == ".pdf":
        fig.savefig(outpath)
    else:
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] panel: {outpath}")

# def make_panel_2xN(panels, models, valid_hours, args, outdir, panel_base, fig_format, suptitle=None):
    N = len(panels)
    if N == 0:
        raise ValueError("No panels specified.")

    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_colors:
        base_colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    ccycler = cycle(base_colors)

    model_colors = {}
    for model_name, _ in models:
        model_colors[model_name] = next(ccycler)

    era5_color = "black"

    fig, axes = plt.subplots(2, N, figsize=(4.3 * N, 7.2), squeeze=False)

    band = f"lat {args.lat_min}–{args.lat_max}°" if args.use_lat_band else "all lat"
    sect = f" lon {args.lon_min}–{args.lon_max}°" if args.use_lon_sector else "all lon"
    region_txt = f"{band};{sect}"

    legend_handles = None
    legend_labels = None

    for j, p in enumerate(panels):
        ax_abs = axes[0, j]
        ax_rat = axes[1, j]

        domain = p["domain"]
        v_ref  = p["ref"]
        v_pred = p["pred"]
        lvl    = p["level"]
        label  = p["label"]

        # --- REF ---
        if domain == "sfc":
            _, E_ref, n_ref = accumulate_ref_spectrum_surface(v_ref, valid_hours, args)
        else:
            _, E_ref, n_ref = accumulate_ref_spectrum_upper(v_ref, lvl, valid_hours, args)

        if n_ref == 0 or E_ref.size == 0:
            ax_abs.text(0.5, 0.5, f"{label}\n(REF missing)", ha="center", va="center")
            ax_abs.axis("off")
            ax_rat.axis("off")
            continue

        m_ref = np.arange(len(E_ref))
        mask = m_ref >= 1

        # Plot ERA5
        ax_abs.loglog(
            m_ref[mask], E_ref[mask],
            label="ERA5",
            linewidth=2.7,
            color=era5_color
        )

        any_model = False
        stats_lines = []

        for model_name, model_dir in models:
            color = model_colors[model_name]

            if domain == "sfc":
                _, E_p, n_p = accumulate_pred_spectrum_surface(v_pred, model_dir, valid_hours, args)
            else:
                _, E_p, n_p = accumulate_pred_spectrum_upper(v_pred, lvl, model_dir, valid_hours, args)

            if n_p == 0 or E_p.size == 0:
                continue

            L = min(len(E_ref), len(E_p))
            m = np.arange(L)
            
            Eref_c = E_ref[:L]
            Ep_c   = E_p[:L]
            valid_m = m >= 1

            # =========================================================
            # NEW: Log-Space Error Calculation
            # "How far is the curve on average in log-log space?"
            # =========================================================
            if np.any(valid_m):
                # Add small epsilon to avoid log(0), though spectra should be > 0
                log_ref = np.log10(Eref_c[valid_m] + 1e-20)
                log_pred = np.log10(Ep_c[valid_m] + 1e-20)
                
                # Mean Absolute Error in Log10 space
                # 0.1 means "on average, off by 10^0.1 (factor of ~1.25)"
                # 0.3 means "on average, off by factor of 2 (10^0.3)"
                mae_log = np.mean(np.abs(log_pred - log_ref))
                
                stats_lines.append(f"{model_name}: {mae_log:.3f} (log)")
            # =========================================================

            any_model = True

            ax_abs.loglog(
                m[valid_m], Ep_c[valid_m],
                label=model_name,
                linewidth=1.5,
                color=color
            )

            ratio = Ep_c / (Eref_c + 1e-12)
            ax_rat.semilogx(
                m[valid_m], ratio[valid_m],
                label=model_name,
                linewidth=1.5,
                color=color
            )

        # Formatting
        ax_abs.set_title(label)
        ax_abs.grid(True, which="both", ls="--", alpha=0.4)
        ax_abs.set_xlabel("m")
        if j == 0:
            ax_abs.set_ylabel("Zonal spectral power")

        # --- Stats Box (Log Error) ---
        if stats_lines:
            stats_text = "Log-Diff (Shape):\n" + "\n".join(stats_lines)
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax_abs.text(0.95, 0.95, stats_text, transform=ax_abs.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=props)

        if any_model:
            ax_rat.axhline(1.0, linestyle="--", linewidth=1.0, color="0.3")
            ax_rat.grid(True, which="both", ls="--", alpha=0.4)
            ax_rat.set_xlabel("m")
            if j == 0:
                ax_rat.set_ylabel("E_pred / E_ERA5")
        else:
            ax_rat.text(0.5, 0.5, "No model overlap", ha="center", va="center")
            ax_rat.axis("off")

        if legend_handles is None:
            h, l = ax_abs.get_legend_handles_labels()
            if h and l:
                legend_handles, legend_labels = h, l

    if suptitle is None:
        suptitle = f"Zonal spectra (top) and ratio (bottom), lead={args.lead}h | {args.start} → {args.end}\nRegion: {region_txt}"
    fig.suptitle(suptitle, y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
            fontsize=10,
        )

    outpath = Path(outdir) / f"{panel_base}.{fig_format}"
    if outpath.suffix.lower() == ".pdf":
        fig.savefig(outpath)
    else:
        fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] panel: {outpath}")

def make_panel_2xN(panels, models, valid_hours, args, outdir, panel_base, fig_format, suptitle=None):
    N = len(panels)
    if N == 0:
        raise ValueError("No panels specified.")

    # 1. Setup Colors
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_colors:
        base_colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    ccycler = cycle(base_colors)

    model_colors = {}
    for model_name, _ in models:
        model_colors[model_name] = next(ccycler)
    era5_color = "black"

    # 2. Setup Figure
    fig, axes = plt.subplots(2, N, figsize=(4.3 * N, 7.2), squeeze=False)

    band = f"lat {args.lat_min}–{args.lat_max}°" if args.use_lat_band else "all lat"
    sect = f" lon {args.lon_min}–{args.lon_max}°" if args.use_lon_sector else "all lon"
    region_txt = f"{band};{sect}"

    legend_handles = None
    legend_labels = None

    # Container for saving results to CSV
    # Structure: {'Model': [], 'Variable': [], 'Log_Error': [], 'Energy_Err_Pct': []}
    metrics_data = []

    for j, p in enumerate(panels):
        ax_abs = axes[0, j]
        ax_rat = axes[1, j]

        domain = p["domain"]
        v_ref  = p["ref"]
        v_pred = p["pred"]
        lvl    = p["level"]
        label  = p["label"]

        # --- Load Reference (ERA5) ---
        if domain == "sfc":
            _, E_ref, n_ref = accumulate_ref_spectrum_surface(v_ref, valid_hours, args)
        else:
            _, E_ref, n_ref = accumulate_ref_spectrum_upper(v_ref, lvl, valid_hours, args)

        if n_ref == 0 or E_ref.size == 0:
            ax_abs.text(0.5, 0.5, f"{label}\n(REF missing)", ha="center", va="center")
            ax_abs.axis("off"); ax_rat.axis("off")
            continue

        m_ref = np.arange(len(E_ref))
        mask = m_ref >= 1

        # Plot ERA5
        ax_abs.loglog(m_ref[mask], E_ref[mask], label="ERA5", linewidth=2.7, color=era5_color)

        any_model = False
        stats_lines = []

        # --- Loop Models ---
        for model_name, model_dir in models:
            color = model_colors[model_name]

            if domain == "sfc":
                _, E_p, n_p = accumulate_pred_spectrum_surface(v_pred, model_dir, valid_hours, args)
            else:
                _, E_p, n_p = accumulate_pred_spectrum_upper(v_pred, lvl, model_dir, valid_hours, args)

            if n_p == 0 or E_p.size == 0:
                continue

            # Align Data
            L = min(len(E_ref), len(E_p))
            m = np.arange(L)
            valid_m = m >= 1
            
            Eref_c = E_ref[:L]
            Ep_c   = E_p[:L]

            # --- METRIC CALCULATION (Log & Linear) ---
            mae_log = np.nan
            err_pct = np.nan
            
            if np.any(valid_m):
                # 1. Log-Space Error (Shape) - Best for spectral fairness
                log_ref = np.log10(Eref_c[valid_m] + 1e-20)
                log_pred = np.log10(Ep_c[valid_m] + 1e-20)
                mae_log = np.mean(np.abs(log_pred - log_ref))

                # 2. Linear Energy Error (Percentage) - Good for sanity check
                diff_lin = np.abs(Ep_c[valid_m] - Eref_c[valid_m])
                total_ref = np.sum(Eref_c[valid_m])
                if total_ref > 0:
                    err_pct = (np.sum(diff_lin) / total_ref) * 100.0

                # Record for CSV
                metrics_data.append({
                    "Model": model_name,
                    "Variable": label,
                    "Log_Error": mae_log,
                    "Energy_Err_Pct": err_pct
                })

                # Record for Plot (Log Error)
                stats_lines.append(f"{model_name}: {mae_log:.3f}")

            any_model = True

            # Plot Model
            ax_abs.loglog(m[valid_m], Ep_c[valid_m], label=model_name, linewidth=1.5, color=color)
            ratio = Ep_c / (Eref_c + 1e-12)
            ax_rat.semilogx(m[valid_m], ratio[valid_m], label=model_name, linewidth=1.5, color=color)

        # --- Formatting ---
        ax_abs.set_title(label)
        ax_abs.grid(True, which="both", ls="--", alpha=0.4)
        ax_abs.set_xlabel("m")
        if j == 0: ax_abs.set_ylabel("Zonal spectral power")

        # Stats Box (Log Error only, to keep it clean)
        if stats_lines:
            stats_text = "Log-Diff:\n" + "\n".join(stats_lines)
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax_abs.text(0.95, 0.95, stats_text, transform=ax_abs.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=props)

        if any_model:
            ax_rat.axhline(1.0, linestyle="--", linewidth=1.0, color="0.3")
            ax_rat.grid(True, which="both", ls="--", alpha=0.4)
            ax_rat.set_xlabel("m")
            if j == 0: ax_rat.set_ylabel("Ratio")
        else:
            ax_rat.axis("off")

        if legend_handles is None:
            h, l = ax_abs.get_legend_handles_labels()
            if h and l: legend_handles, legend_labels = h, l

    # 3. Final Figure Layout
    if suptitle is None:
        suptitle = f"Zonal spectra (top) and ratio (bottom), lead={args.lead}h | {args.start} → {args.end}\nRegion: {region_txt}"
    fig.suptitle(suptitle, y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels),
                   frameon=False, bbox_to_anchor=(0.5, 0.02), fontsize=10)

    # 4. Save Figure
    outpath_fig = Path(outdir) / f"{panel_base}.{fig_format}"
    if outpath_fig.suffix.lower() == ".pdf":
        fig.savefig(outpath_fig)
    else:
        fig.savefig(outpath_fig, dpi=300)
    plt.close(fig)
    print(f"[saved] panel figure: {outpath_fig}")

    # 5. Save Metrics to CSV
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        
        # Calculate Global Average per Model
        global_avg = df.groupby("Model")[["Log_Error", "Energy_Err_Pct"]].mean().reset_index()
        global_avg["Variable"] = "GLOBAL_MEAN"
        
        # Combine and Sort
        df_final = pd.concat([df, global_avg], ignore_index=True)
        df_final = df_final.sort_values(by=["Model", "Variable"])
        
        outpath_csv = Path(outdir) / f"{panel_base}_metrics.csv"
        df_final.to_csv(outpath_csv, index=False, float_format="%.4f")
        print(f"[saved] metrics file: {outpath_csv}")
        print("Global Averages:")
        print(global_avg.to_string(index=False))

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    models = parse_pred_models(args.pred)
    if not models:
        raise SystemExit('You must provide at least one --pred "NAME=/path/to/pred_dir"')

    panels = parse_panels(args.panel)
    if not panels:
        raise SystemExit('You must provide at least one --panel ...')

    outdir = args.outdir or f"out_panel_lead{args.lead}"
    os.makedirs(outdir, exist_ok=True)

    valid_hours = pd.date_range(args.start, args.end, freq="H")

    panel_base = args.panel_out.strip()
    if not panel_base:
        panel_base = "PANEL"
    else:
        panel_base = str(Path(panel_base).with_suffix(""))

    panel_title = args.panel_title.strip() or None

    make_panel_2xN(
        panels=panels,
        models=models,
        valid_hours=valid_hours,
        args=args,
        outdir=outdir,
        panel_base=panel_base,
        fig_format=args.fig_format,
        suptitle=panel_title,
    )

if __name__ == "__main__":
    main()
