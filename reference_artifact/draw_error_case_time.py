#!/usr/bin/env python3
"""
draw_time_series_compare.py

Generates time-series grid plots comparing MULTIPLE models against Ground Truth.
"""

import os
import json
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Helpers ---
TIME_DIMS = ["valid_time", "time", "history"]
LAT_NAMES = ["latitude", "lat", "Latitude", "LAT"]
LON_NAMES = ["longitude", "lon", "Longitude", "LON"]
LEVEL_DIMS = ["pressure_level", "level", "plev", "isobaricInhPa"]

def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Ensure dimensions are named latitude/longitude and sorted."""
    for n in LAT_NAMES:
        if n in ds.coords or n in ds.dims:
            ds = ds.rename({n: "latitude"})
            break
    for n in LON_NAMES:
        if n in ds.coords or n in ds.dims:
            ds = ds.rename({n: "longitude"})
            break
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    return ds

def open_era5_merged(upper_path, sfc_path):
    """Open upper and surface ERA5 files and merge them."""
    u = standardize_latlon(xr.open_dataset(upper_path))
    s = standardize_latlon(xr.open_dataset(sfc_path))
    return xr.merge([u, s], compat="no_conflicts", join="outer")

def select_latlon_range(da, lat_range, lon_range):
    """Crop DataArray to specific lat/lon box."""
    if not lat_range or not lon_range:
        return da
    lat1, lat2 = lat_range
    lon1, lon2 = lon_range
    lat_min, lat_max = sorted([lat1, lat2])
    lon_min, lon_max = sorted([lon1, lon2])
    return da.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

def parse_var_and_level(varname):
    """Parse 'atmos_z_500' -> ('atmos_z', 500.0)."""
    parts = varname.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), float(parts[-1])
    return varname, None

def select_level(da, lev):
    """Select pressure level if requested."""
    if lev is None:
        return da
    for d in LEVEL_DIMS:
        if d in da.dims:
            try:
                return da.sel({d: lev})
            except:
                try:
                    return da.sel({d: lev}, method="nearest")
                except:
                    pass
    return da

def get_first_time(da):
    """Select the first time step found in the array."""
    for t in TIME_DIMS:
        if t in da.dims:
            return da.isel({t: 0})
    return da

def regrid_to_era(da_p, da_e):
    """Regrid prediction to match ERA5 grid."""
    if "longitude" in da_p.coords:
        if da_e.longitude.min() < 0 and da_p.longitude.min() >= 0:
            da_p = da_p.assign_coords(longitude=(((da_p.longitude + 180) % 360) - 180)).sortby("longitude")
    
    return da_p.interp(
        latitude=da_e.latitude, 
        longitude=da_e.longitude, 
        method="linear", 
        kwargs={"fill_value": np.nan}
    )

def finite_minmax(vals, robust=False):
    v = vals[np.isfinite(vals)]
    if v.size == 0: return 0, 1
    if robust:
        return np.percentile(v, 2), np.percentile(v, 98)
    return v.min(), v.max()

def symmetric_minmax(vals, robust=False):
    v = vals[np.isfinite(vals)]
    if v.size == 0: return -1, 1
    mx = np.percentile(np.abs(v), 98) if robust else np.abs(v).max()
    if mx == 0: mx = 1e-6
    return -mx, mx

# --- Plotting ---

def plot_grid(
    col_data, 
    model_names,     
    var_name,        
    outpath, 
    vmin, vmax, 
    dvmin, dvmax, 
    cmap="viridis", 
    diff_cmap="RdBu_r",
    wspace=0.05, hspace=0.1,
    row_cbar=False   
):
    cols = len(col_data)
    num_models = len(model_names)
    rows = 1 + (2 * num_models)
    
    # Calculate figure size
    fig_w = max(10, 4.0 * cols)
    fig_h = 3.0 * rows 
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    # ADD THIS LINE HERE:
    fig.suptitle(var_name, fontsize=26, fontweight='bold', y=1.02)
    
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # To store the last image of each row for colorbar creation
    row_last_im = [None] * rows

    for j, item in enumerate(col_data):
        ax_col = axes[:, j]
        
        # --- Column Title ---
        ax_col[0].set_title(f"+{item['hour']}h", fontsize=20, pad=16, fontweight='bold')

        # === Row 0: Ground Truth ===
        ax_gt = ax_col[0]
        da_e = item['era']
        
        if da_e is not None:
            im_val = ax_gt.pcolormesh(
                da_e.longitude, da_e.latitude, da_e.values, 
                vmin=vmin, vmax=vmax, cmap=cmap, shading='nearest', rasterized=True  # <--- Add this
            )
            row_last_im[0] = im_val 
        else:
            ax_gt.text(0.5, 0.5, "MISSING ERA", ha='center', va='center')

        ax_gt.set_xticks([])
        ax_gt.set_yticks([])

        # Label Row 0
        if j == 0:
            ax_gt.text(-0.15, 0.5, "Ground Truth", 
                    rotation=90, va='center', ha='center',
                    transform=ax_gt.transAxes, 
                    fontsize=14, fontweight='bold')
        
        # Per-panel colorbar (ONLY if row_cbar is FALSE)
        if not row_cbar and da_e is not None:
             divider = make_axes_locatable(ax_gt)
             cax = divider.append_axes("right", size="5%", pad=0.05)
             fig.colorbar(im_val, cax=cax)

        # === Model Rows ===
        current_row_idx = 1
        
        for m_name in model_names:
            m_data = item['models'].get(m_name, {})
            da_pred = m_data.get('pred')
            da_diff = m_data.get('diff')

            # --- Prediction Row ---
            ax_p = ax_col[current_row_idx]
            if da_pred is not None:
                im_p = ax_p.pcolormesh(
                    da_pred.longitude, da_pred.latitude, da_pred.values, 
                    vmin=vmin, vmax=vmax, cmap=cmap, shading='nearest', rasterized=True  # <--- Add this
                )
                row_last_im[current_row_idx] = im_p
                
                if not row_cbar:
                    divider = make_axes_locatable(ax_p)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im_p, cax=cax)
            else:
                ax_p.text(0.5, 0.5, "MISSING", ha='center', va='center')
            
            ax_p.set_xticks([])
            ax_p.set_yticks([])

            if j == 0:
                ax_p.text(-0.15, 0.5, f"{m_name}", 
                        rotation=90, va='center', ha='center',
                        transform=ax_p.transAxes, fontsize=14, fontweight='bold')

            # --- Difference Row ---
            ax_d = ax_col[current_row_idx + 1]
            if da_diff is not None:
                im_d = ax_d.pcolormesh(
                    da_diff.longitude, da_diff.latitude, da_diff.values, 
                    vmin=dvmin, vmax=dvmax, cmap=diff_cmap, shading='nearest', rasterized=True
                )
                row_last_im[current_row_idx + 1] = im_d

                if not row_cbar:
                    divider = make_axes_locatable(ax_d)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im_d, cax=cax)
            else:
                ax_d.text(0.5, 0.5, "MISSING", ha='center', va='center')

            ax_d.set_xticks([])
            ax_d.set_yticks([])

            if j == 0:
                ax_d.text(-0.15, 0.5, "Diff", 
                        rotation=90, va='center', ha='center',
                        # transform=ax_d.transAxes, fontsize=12, color='gray', style='italic')
                        transform=ax_d.transAxes, fontsize=14, fontweight='bold')

            current_row_idx += 2

    # -- Global Layout & Row Colorbars --
    
    if row_cbar:
        # Reserve more space on the right for the row colorbars
        plt.subplots_adjust(wspace=wspace, hspace=hspace, right=0.88, top=0.95)
        
        for r in range(rows):
            im = row_last_im[r]
            if im is not None:
                # Use fig.add_axes based on the last axis position
                # This prevents the last plot from shrinking
                ax_last = axes[r, -1]
                pos = ax_last.get_position()
                
                cax_x = pos.x1 + 0.015
                cax_y = pos.y0
                cax_w = 0.015
                cax_h = pos.height
                
                cax = fig.add_axes([cax_x, cax_y, cax_w, cax_h])
                
                # # Determine label
                # if r == 0:
                #     label = "Value"
                # elif (r % 2) != 0: 
                #     label = "Value"
                # else: 
                #     label = "Diff"
                
                fig.colorbar(im, cax=cax)
    else:
        # Default tight layout
        plt.subplots_adjust(wspace=wspace, hspace=hspace, right=0.98, top=0.95)

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


# --- Main ---
def main():
    p = argparse.ArgumentParser(description="Plot time-series grid from split files.")
    p.add_argument("--manifest", type=str, required=True, help="JSON file mapping hours to file paths.")
    p.add_argument("--vars", type=str, nargs="+", required=True, help="Variables to plot (e.g. surf_2t atmos_z_500).")
    p.add_argument("--output_prefix", type=str, default="compare_out", help="Output filename prefix.")
    
    # Layout Options
    p.add_argument("--wspace", type=float, default=0.05, help="Horizontal space between subplots.")
    p.add_argument("--hspace", type=float, default=0.1, help="Vertical space between subplots.")
    
    # Colorbar Mode
    p.add_argument("--row_cbar", action="store_true", help="Add ONE colorbar at the end of EACH row (prevents plot resizing).")

    # Data Selection
    p.add_argument("--latitude", type=float, nargs=2, help="Lat range (e.g. 20 50)")
    p.add_argument("--longitude", type=float, nargs=2, help="Lon range (e.g. 100 140)")
    p.add_argument("--map", action="append", default=[], help="Map vars: --map surf_2t=2t")
    p.add_argument("--title_map", action="append", default=[], help="Map var names to titles: --title_map surf_2t='Surface Temp'")
    
    p.add_argument("--robust", action="store_true", help="Use 2nd/98th percentile for scaling.")
    
    args = p.parse_args()

    # 1. Load Manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    manifest.sort(key=lambda x: x.get('hour', 0))

    # Detect models from the first entry
    first_pred = manifest[0].get('pred')
    if isinstance(first_pred, dict):
        model_names = sorted(list(first_pred.keys()))
        print(f"Found models: {model_names}")
    else:
        model_names = ["Prediction"]
        for m in manifest:
            if isinstance(m['pred'], str):
                m['pred'] = {"Prediction": m['pred']}

    # Parse variable mapping
    data_mapping = {}
    for m in args.map:
        if "=" in m:
            k, v = m.split("=", 1)
            data_mapping[k.strip()] = v.strip()

    title_mapping = {}
    for m in args.title_map:
        if "=" in m:
            k, v = m.split("=", 1)
            title_mapping[k.strip()] = v.strip()

    lat_r = args.latitude
    lon_r = args.longitude

    # 2. Process Variables
    for var_token in args.vars:
        print(f"--- Processing {var_token} ---")
        base_var, lev = parse_var_and_level(var_token)
        era_var = data_mapping.get(base_var, base_var)

        col_data = []
        all_vals = []   
        all_diffs = [] 

        for entry in manifest:
            hour = entry.get('hour', '?')
            
            # -- Load ERA5 --
            try:
                ds_e = open_era5_merged(entry['era_upper'], entry['era_sfc'])
                if era_var not in ds_e:
                    print(f"[WARN] '{era_var}' missing in ERA5 for T+{hour}h")
                    da_e = None
                else:
                    da_e = get_first_time(ds_e[era_var])
                    da_e = select_level(select_latlon_range(da_e, lat_r, lon_r), lev)
            except Exception as e:
                print(f"[WARN] Failed to open ERA5 for T+{hour}h: {e}")
                da_e = None

            step_data = {
                'hour': hour,
                'era': da_e,
                'models': {}
            }

            if da_e is not None:
                all_vals.append(da_e.values.ravel())

            # -- Load Models --
            for m_name in model_names:
                pred_path = entry['pred'].get(m_name)
                da_p_reg = None
                diff = None
                
                if pred_path and os.path.exists(pred_path):
                    try:
                        ds_p = standardize_latlon(xr.open_dataset(pred_path))
                        if base_var in ds_p:
                            da_p = get_first_time(ds_p[base_var])
                            da_p = select_level(select_latlon_range(da_p, lat_r, lon_r), lev)
                            
                            # Regrid and Diff
                            if da_e is not None:
                                da_p_reg = regrid_to_era(da_p, da_e)
                                diff = da_p_reg - da_e
                                
                                all_vals.append(da_p_reg.values.ravel())
                                all_diffs.append(diff.values.ravel())
                    except Exception as e:
                        print(f"[WARN] Error loading {m_name} at T+{hour}h: {e}")

                step_data['models'][m_name] = {
                    'pred': da_p_reg,
                    'diff': diff
                }

            col_data.append(step_data)

        # -- Compute Global Min/Max --
        if not all_vals:
            print(f"No valid data found for {var_token}. Skipping.")
            continue
            
        big_v = np.concatenate(all_vals)
        big_d = np.concatenate(all_diffs) if all_diffs else np.array([])
        
        vmin, vmax = finite_minmax(big_v, robust=args.robust)
        if big_d.size > 0:
            dvmin, dvmax = symmetric_minmax(big_d, robust=args.robust)
        else:
            dvmin, dvmax = -1, 1

        if "surf" in base_var:
            cmap = "inferno" 
        elif "atmos" in base_var:
            cmap = "plasma"
        else:
            cmap = "viridis"

        out_filename = f"{args.output_prefix}_{var_token}.pdf"
        display_name = title_mapping.get(var_token, var_token)

        plot_grid(
            col_data, 
            model_names=model_names,
            var_name=display_name, 
            outpath=out_filename, 
            vmin=vmin, vmax=vmax, 
            dvmin=dvmin, dvmax=dvmax, 
            cmap=cmap,
            wspace=args.wspace,
            hspace=args.hspace,
            row_cbar=args.row_cbar
        )

if __name__ == "__main__":
    main()