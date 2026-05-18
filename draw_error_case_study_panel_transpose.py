#!/usr/bin/env python3
"""
draw_grid_compare_models_era5.py

Grid figure: ERA5 (first row) + N model rows x V variables columns.
Optionally insert a diff/error row after each model: (Model - ERA5).

Supports --transpose to swap axes:
  Default: Rows=Models, Cols=Variables
  Transposed: Rows=Variables, Cols=Models (allows side-by-side comparison of specific vars)

Supports --vars tokens with optional pressure level suffix AND optional custom title:
  surf_2t             -> Title: surf_2t
  surf_2t:Temperature -> Title: Temperature
  atmos_t_850         -> Title: atmos_t (850 hPa)
  atmos_z_500:Z500    -> Title: Z500

Mapping is applied to the base name (e.g. atmos_t -> ERA5 t).

Key options:
- --transpose: Swap X/Y axes.
- --add_diff_rows: insert diff rows after each model
- --show_mae_box: show MAE overlay in model panels
- --model_value_scale {era5,model}:
    * era5 (default): model value maps use ERA5 vmin/vmax (best for comparison)
    * model: each model panel uses its own vmin/vmax (qualitative inspection)
"""

import os
import json
import csv
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


def find_level_dim(da: xr.DataArray):
    for d in LEVEL_DIMS:
        if d in da.dims:
            return d
    return None


def parse_var_and_level(varname: str):
    """
    Parse variable name possibly ending with _<level>.
    Examples:
      surf_2t        -> ("surf_2t", None)
      atmos_t_850    -> ("atmos_t", 850.0)
    """
    parts = varname.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        base = "_".join(parts[:-1])
        level = float(parts[-1])
        return base, level
    return varname, None


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
            )
        except Exception:
            pass
    return out


def finite_minmax(arr: xr.DataArray, robust=False, q=(2, 98)):
    """Compute vmin/vmax from a DataArray using finite values."""
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


def symmetric_minmax_from_values(values_1d: np.ndarray, robust=False, q=(2, 98)):
    """Return symmetric (-A, +A) limits from finite 1D values."""
    v = values_1d[np.isfinite(values_1d)]
    if v.size == 0:
        return None, None
    if robust:
        lo = float(np.percentile(v, q[0]))
        hi = float(np.percentile(v, q[1]))
    else:
        lo = float(v.min())
        hi = float(v.max())
    a = max(abs(lo), abs(hi))
    if a == 0:
        a = 1e-6
    return -a, +a


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

    if tgt_min >= 0 and src_min < 0:
        da_src = da_src.assign_coords(longitude=((da_src["longitude"] + 360) % 360))
        da_src = da_src.sortby("longitude")

    if tgt_min < 0 and src_min >= 0:
        da_src = da_src.assign_coords(longitude=(((da_src["longitude"] + 180) % 360) - 180))
        da_src = da_src.sortby("longitude")

    return da_src


def regrid_pred_to_era5(da_p: xr.DataArray, da_e: xr.DataArray) -> xr.DataArray:
    """Interpolate prediction to ERA5 lat/lon grid."""
    da_p = normalize_lon_to_match(da_p, da_e)
    return da_p.interp(
        latitude=da_e["latitude"],
        longitude=da_e["longitude"],
        method="linear",
        kwargs={"fill_value": np.nan},
    )


def open_era5_merged(upper_path: str, sfc_path: str) -> xr.Dataset:
    upper = standardize_latlon(xr.open_dataset(upper_path))
    sfc = standardize_latlon(xr.open_dataset(sfc_path))
    return xr.merge([upper, sfc], compat="no_conflicts", join="outer")


def parse_map_pairs(map_pairs):
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


def resolve_era_var(pred_base_var: str, mapping: dict, map_mode: str) -> str:
    """Return ERA5 var name corresponding to pred base var."""
    if not mapping:
        return pred_base_var
    if map_mode == "aur_to_era":
        return mapping.get(pred_base_var, pred_base_var)
    for era_k, pred_v in mapping.items():
        if pred_v == pred_base_var:
            return era_k
    return pred_base_var


def select_level_if_needed(da: xr.DataArray, level_value: float, debug_name: str):
    """Select a pressure level if level_value is not None."""
    if level_value is None:
        return da
    lev_dim = find_level_dim(da)
    if lev_dim is None:
        print(f"[WARN] {debug_name}: requested level {level_value} but no level dim found. Skip.")
        return None
    try:
        return da.sel({lev_dim: level_value})
    except Exception:
        try:
            return da.sel({lev_dim: level_value}, method="nearest")
        except Exception:
            return None


def plot_grid(
    rows,               # list: {"label": str, "kind": "era"|"model"|"diff", "data": {key: item}, "mae": {key: mae}}
    era_row,            # dict key -> era da (2D)
    col_specs,          # list dicts: key, title, vmin/vmax, dvmin/dvmax, cmap (optional)
    outpath_no_ext,
    cmap_value,
    cmap_diff,
    dpi,
    fmt,
    title_fontsize=18,
    rowlabel_fontsize=14,
    cbar_pad=0.18,
    cbar_size="4.5%",
    show_mae_box=False,
    wspace=0.35,
    hspace=0.35,
    suptitle=None,
    suptitle_fontsize=20,
    rowlabel_x=-0.22,
    title_pad=16,
    model_value_scale="era5",
    transpose=False,    # NEW: Toggle to swap axes
):
    # Determine grid dimensions based on transpose flag
    if not transpose:
        # Standard: Rows = Models, Cols = Vars
        nrows = len(rows)
        ncols = len(col_specs)
    else:
        # Transposed: Rows = Vars, Cols = Models
        nrows = len(col_specs)
        ncols = len(rows)

    fig_w = max(9, 4.7 * ncols)
    fig_h = max(4, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=False)

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    is_vector = fmt.lower() in ["pdf", "svg", "eps"]
    raster_kw = dict(rasterized=is_vector)

    def hide_all(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    def draw_cell(ax, x, y, Z, vmin, vmax, cmap):
        im = ax.pcolormesh(
            x, y, Z,
            shading="nearest",
            vmin=vmin, vmax=vmax, cmap=cmap,
            linewidth=0, edgecolors="none", antialiased=False,
            **raster_kw
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
        fig.colorbar(im, cax=cax)
        return im

    # ---------------------------------------------------------
    # ITERATE LOGICALLY: i = Model Index, j = Variable Index
    # ---------------------------------------------------------
    for i, r in enumerate(rows):
        for j, cs in enumerate(col_specs):
            
            # Map logical index (i,j) to physical grid (row, col)
            if not transpose:
                ax = axes[i, j]
                is_top_row = (i == 0)
                is_first_col = (j == 0)
            else:
                ax = axes[j, i]
                is_top_row = (j == 0)
                is_first_col = (i == 0)

            key = cs["key"]

            # 1. Handle Labels ------------------------------------
            
            # TOP TITLES
            if is_top_row:
                if not transpose:
                    # Standard: Title = Variable Name
                    ax.set_title(cs["title"], fontsize=title_fontsize, pad=title_pad)
                else:
                    # Transposed: Title = Model Name
                    ax.set_title(r["label"], fontsize=title_fontsize, pad=title_pad)

            # LEFT ROW LABELS
            if is_first_col:
                if not transpose:
                    # Standard: Label = Model Name
                    lbl = r["label"]
                else:
                    # Transposed: Label = Variable Name
                    lbl = cs["title"]
                
                ax.text(
                    rowlabel_x, 0.5, lbl,
                    rotation=90, ha="right", va="center",
                    transform=ax.transAxes,
                    fontsize=rowlabel_fontsize, fontweight="bold"
                )

            # 2. Draw Data ---------------------------------------
            value_cmap = cs.get("cmap", cmap_value)
            
            item = r["data"].get(key, None)
            if item is None:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            if isinstance(item, dict):
                da = item["da"]
                model_vmin = item.get("vmin", None)
                model_vmax = item.get("vmax", None)
            else:
                da = item
                model_vmin = model_vmax = None

            da_e = era_row[key]
            x = da_e["longitude"].values
            y = da_e["latitude"].values

            if r["kind"] == "diff":
                vmin, vmax = cs["dvmin"], cs["dvmax"]
                if vmin is None:
                    vmin, vmax = -1.0, 1.0
                draw_cell(ax, x, y, da.values, vmin, vmax, cmap_diff)

            elif r["kind"] == "model" and model_value_scale == "model" and model_vmin is not None:
                draw_cell(ax, x, y, da.values, model_vmin, model_vmax, value_cmap)

            else:
                draw_cell(ax, x, y, da.values, cs["vmin"], cs["vmax"], value_cmap)

            hide_all(ax)

            if show_mae_box and r["kind"] == "model":
                mae = r.get("mae", {}).get(key, float("nan"))
                mae_str = f"{mae:.4g}" if np.isfinite(mae) else "NaN"
                ax.text(
                    0.02, 0.98, f"MAE={mae_str}",
                    ha="left", va="top",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.25", alpha=0.75)
                )

    top_margin = 0.88 if suptitle else 0.92
    fig.subplots_adjust(top=top_margin, wspace=wspace, hspace=hspace)
    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=0.97)

    os.makedirs(os.path.dirname(outpath_no_ext) or ".", exist_ok=True)
    outpath = f"{outpath_no_ext}.{fmt}"
    raster_fmt = fmt.lower() in ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    plt.savefig(outpath, dpi=dpi if raster_fmt else None, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    p = argparse.ArgumentParser(description="Grid compare: ERA5 + models.")
    p.add_argument("--pred_files", type=str, nargs="+", required=True)
    p.add_argument("--pred_labels", type=str, nargs="*", default=None)
    p.add_argument("--era5_upper_file", type=str, required=True)
    p.add_argument("--era5_sfc_file", type=str, required=True)

    p.add_argument("--output", type=str, default="grid_compare")
    p.add_argument("--vars", type=str, nargs="+", required=True,
                   help="Columns. Supports suffix _<level> AND optional title :Title. "
                        "e.g. atmos_t_850:Temperature")

    p.add_argument("--latitude", type=float, nargs=2, metavar=("LAT1", "LAT2"))
    p.add_argument("--longitude", type=float, nargs=2, metavar=("LON1", "LON2"))

    p.add_argument("--transpose", action="store_true", help="Swap axes: Rows=Vars, Cols=Models.")

    p.add_argument("--cmap", type=str, default="cividis", help="Colormap for value maps (fallback).")
    p.add_argument("--diff_cmap", type=str, default="RdBu_r", help="Colormap for diff maps.")
    p.add_argument("--robust", action="store_true", help="Use percentile ranges for scaling (2-98%).")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--fmt", type=str, default="png")

    p.add_argument("--map", action="append", default=[])
    p.add_argument("--map_json", type=str, default=None)
    p.add_argument("--map_mode", type=str, default="aur_to_era", choices=["aur_to_era", "era_to_aur"])

    p.add_argument("--title_fontsize", type=int, default=18)
    p.add_argument("--rowlabel_fontsize", type=int, default=14)
    p.add_argument("--cbar_pad", type=float, default=0.18)
    p.add_argument("--cbar_size", type=str, default="4.5%")
    p.add_argument("--show_mae_box", action="store_true")

    p.add_argument("--wspace", type=float, default=0.35)
    p.add_argument("--hspace", type=float, default=0.35)
    p.add_argument("--suptitle", type=str, default=None)
    p.add_argument("--suptitle_fontsize", type=int, default=20)
    p.add_argument("--rowlabel_x", type=float, default=-0.22)
    p.add_argument("--title_pad", type=float, default=16)

    p.add_argument("--add_diff_rows", action="store_true",
                   help="Insert a diff row after each model row (Model - ERA5).")
    p.add_argument("--model_value_scale", type=str, default="era5", choices=["era5", "model"])
    p.add_argument("--mae_out", type=str, default=None)

    args = p.parse_args()

    mapping = {}
    if args.map_json:
        mapping.update(load_map_json(args.map_json))
    mapping.update(parse_map_pairs(args.map))

    lat_range = tuple(args.latitude) if args.latitude is not None else None
    lon_range = tuple(args.longitude) if args.longitude is not None else None

    era = open_era5_merged(args.era5_upper_file, args.era5_sfc_file)

    labels = args.pred_labels
    if not labels or len(labels) != len(args.pred_files):
        labels = [f"Model{i+1}" for i in range(len(args.pred_files))]

    # Build ERA columns
    era_row = {}
    col_specs = []

    for var_token_input in args.vars:
        # --- MODIFICATION START: Parse "var:Title" syntax ---
        if ":" in var_token_input:
            var_token, custom_title = var_token_input.split(":", 1)
        else:
            var_token, custom_title = var_token_input, None
        # --- MODIFICATION END ---

        base_var, level = parse_var_and_level(var_token)
        era_var = resolve_era_var(base_var, mapping, args.map_mode)

        if era_var not in era.data_vars:
            print(f"[WARN] ERA5 missing '{era_var}' (for token '{var_token}'), skip column.")
            continue

        da_e = maybe_select_first_time(era[era_var])
        da_e = select_latlon_range(da_e, lat_range, lon_range, debug_name=f"ERA5:{era_var}")
        da_e = select_level_if_needed(da_e, level, debug_name=f"ERA5:{era_var}")
        if da_e is None:
            continue

        if set(da_e.dims) != {"latitude", "longitude"}:
            print(f"[WARN] ERA5 '{era_var}' dims {da_e.dims} not 2D lat/lon. Skip.")
            continue

        vmin, vmax = finite_minmax(da_e, robust=args.robust)
        if vmin is None:
            print(f"[WARN] ERA5 '{era_var}' all-NaN. Skip.")
            continue

        # Use var_token as the unique key, but allow custom text for the visible title
        key = var_token
        
        # --- MODIFICATION START: Apply custom title if exists ---
        if custom_title:
            title = custom_title
        else:
            title = base_var if level is None else f"{base_var} ({int(level)} hPa)"
        # --- MODIFICATION END ---
        
        # Determine variable-family colormap
        if base_var.startswith("surf_"):
            col_cmap = "viridis"
        elif base_var.startswith("atmos_"):
            col_cmap = "plasma"
        else:
            col_cmap = args.cmap

        era_row[key] = da_e
        col_specs.append({
            "key": key,
            "title": title,
            "base_var": base_var,
            "level": level,
            "era_var": era_var,
            "vmin": vmin,
            "vmax": vmax,
            "dvmin": None,
            "dvmax": None,
            "cmap": col_cmap,
        })

    if not col_specs:
        raise SystemExit("[ERROR] No valid columns to plot after filtering.")

    # Build model rows + diffs
    models = []
    mae_records = []
    diffs_collect = {cs["key"]: [] for cs in col_specs}

    for path, lab in zip(args.pred_files, labels):
        ds = standardize_latlon(xr.open_dataset(path))
        mdata = {}
        mmae = {}
        mdiff = {}

        for cs in col_specs:
            key = cs["key"]
            base_var = cs["base_var"]
            level = cs["level"]

            if base_var not in ds.data_vars:
                print(f"[WARN] {lab}: missing '{base_var}' (for token '{key}')")
                continue

            da_p = maybe_select_first_time(ds[base_var])
            da_p = select_latlon_range(da_p, lat_range, lon_range, debug_name=f"{lab}:{base_var}")
            da_p = select_level_if_needed(da_p, level, debug_name=f"{lab}:{base_var}")
            if da_p is None:
                continue

            if set(da_p.dims) != {"latitude", "longitude"}:
                continue

            da_e = era_row[key]
            da_p_rg = regrid_pred_to_era5(da_p, da_e)

            mae = mae_l1(da_p_rg, da_e)
            diff = (da_p_rg - da_e)

            if args.model_value_scale == "model":
                mvmin, mvmax = finite_minmax(da_p_rg, robust=args.robust)
            else:
                mvmin = mvmax = None

            mdata[key] = {"da": da_p_rg, "vmin": mvmin, "vmax": mvmax}
            mmae[key] = mae
            mdiff[key] = diff

            diffs_collect[key].append(np.asarray(diff.values).ravel())

            mae_records.append({
                "row_label": lab,
                "token": key,
                "mae": mae,
            })

        models.append({"label": lab, "data": mdata, "mae": mmae, "diff": mdiff})

    # Diff scales
    for cs in col_specs:
        key = cs["key"]
        if diffs_collect.get(key) and len(diffs_collect[key]) > 0:
            allv = np.concatenate(diffs_collect[key], axis=0)
            dvmin, dvmax = symmetric_minmax_from_values(allv, robust=args.robust)
            cs["dvmin"], cs["dvmax"] = dvmin, dvmax
        else:
            cs["dvmin"], cs["dvmax"] = None, None

    rows = []
    rows.append({
        # "label": "ERA5",
        "label": "Ground Truth",
        "kind": "era",
        "data": {cs["key"]: era_row[cs["key"]] for cs in col_specs},
        "mae": {}
    })

    for m in models:
        rows.append({"label": m["label"], "kind": "model", "data": m["data"], "mae": m["mae"]})
        if args.add_diff_rows:
            rows.append({"label": f"Diff: {m['label']}", "kind": "diff", "data": m["diff"], "mae": {}})

    plot_grid(
        rows=rows,
        era_row=era_row,
        col_specs=col_specs,
        outpath_no_ext=args.output,
        cmap_value=args.cmap,
        cmap_diff=args.diff_cmap,
        dpi=args.dpi,
        fmt=args.fmt,
        title_fontsize=args.title_fontsize,
        rowlabel_fontsize=args.rowlabel_fontsize,
        cbar_pad=args.cbar_pad,
        cbar_size=args.cbar_size,
        show_mae_box=args.show_mae_box,
        wspace=args.wspace,
        hspace=args.hspace,
        suptitle=args.suptitle,
        suptitle_fontsize=args.suptitle_fontsize,
        rowlabel_x=args.rowlabel_x,
        title_pad=args.title_pad,
        model_value_scale=args.model_value_scale,
        transpose=args.transpose,
    )

    if args.mae_out:
        os.makedirs(os.path.dirname(args.mae_out) or ".", exist_ok=True)
        if args.mae_out.lower().endswith(".json"):
            with open(args.mae_out, "w", encoding="utf-8") as f:
                json.dump(mae_records, f, indent=2)
        else:
            keys = ["row_label", "token", "mae"]
            with open(args.mae_out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in mae_records:
                    w.writerow(r)
        print(f"Saved MAE table: {args.mae_out}")


if __name__ == "__main__":
    main()