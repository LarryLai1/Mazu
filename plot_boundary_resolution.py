"""Plot the boundary field as the model actually sees it, across boundary resolutions
and low-res apply modes.

For each (resolution, apply_mode) combination the boundary is loaded through
BoundaryConditionDataset_ERA5, so the field shown is exactly what the rollout injects:
already coarsened and brought back onto the model's 0.25deg grid.

Note: at --boundary_resolution 0.25 the transform is a no-op and the apply mode is
ignored, so both 0.25 panels are identical by construction (kept for layout symmetry).
"""

import argparse
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from datasets.BoundaryConditionDataset import BoundaryConditionDataset_ERA5

# 0.5deg is derived on the fly by pooling the 0.25deg source, so it reuses that root;
# 1.5deg is a genuinely coarser dataset on disk.
DEFAULT_ROOT_025 = "/tmp3/b12902101/era5_tw_forecast_0.25deg"
DEFAULT_ROOT_15 = "/tmp3/b12902101/era5_tw_forecast_1.5deg"

UPPER_ALL = ["u", "v", "t", "q", "z"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var_name', type=str, default='msl', help='Surface variable like t2m, u10, v10, msl (or an upper variable)')
    parser.add_argument('--level', type=int, default=500, help='Pressure level, only used when --var_name is an upper variable')
    parser.add_argument('--init_time', type=str, default='2020-03-01 00:00:00', help='Boundary time to plot (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--output_dir', type=str, default='boundary_resolution_plots')
    parser.add_argument('--plot_mode', type=str, choices=['field', 'diff'], default='field',
                        help="'field' plots the boundary itself; 'diff' plots the difference against "
                             "the 0.25deg baseline (needed to actually see the 0.5deg effect).")
    parser.add_argument('--resolutions', type=float, nargs='+', default=[0.25, 0.5, 1.5], choices=[0.25, 0.5, 1.5])
    parser.add_argument('--apply_modes', type=str, nargs='+', default=['direct', 'interp'], choices=['direct', 'interp'])
    parser.add_argument('--boundary_root_025', type=str, default=DEFAULT_ROOT_025)
    parser.add_argument('--boundary_root_15', type=str, default=DEFAULT_ROOT_15)
    parser.add_argument('--latitude', type=float, nargs=2, default=[39.75, 5])
    parser.add_argument('--longitude', type=float, nargs=2, default=[100, 144.75])
    parser.add_argument('--forecast_cycle_hours', type=int, default=12)
    return parser.parse_args()


def build_dataset(args, resolution, apply_mode, base_time, upper_vars, surface_vars,
                  target_latitude=None, target_longitude=None):
    root = args.boundary_root_15 if resolution == 1.5 else args.boundary_root_025
    return BoundaryConditionDataset_ERA5(
        boundary_root_dir=root,
        start_date_hour=base_time,
        end_date_hour=base_time,
        upper_variables=upper_vars,
        surface_variables=surface_vars,
        levels=[1000, 925, 850, 700, 500, 300, 150, 50],
        latitude=args.latitude,
        longitude=args.longitude,
        boundary_width=0,
        prediction_timedeltas=[0, 12],
        forecast_cycle_hours=args.forecast_cycle_hours,
        time_interp_mode='nearest',
        use_cache=False,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
        boundary_resolution=resolution,
        lowres_apply_mode=apply_mode,
    )


def extract_field(args, ds_bd, base_time, init_time, surface_vars):
    """Return the boundary field at init_time as a 2-D (lat, lon) array."""
    data = ds_bd.get_boundary_at_time(base_time, init_time)
    if data is None:
        return None
    mapped = ds_bd.map_var_name_for_Aurora(args.var_name)
    if surface_vars:
        return data["surf_vars"][mapped].cpu().numpy()
    tensor = data["atmos_vars"][args.var_name].cpu().numpy()
    level_index = [1000, 925, 850, 700, 500, 300, 150, 50].index(args.level)
    return tensor[level_index]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.var_name in UPPER_ALL:
        upper_vars, surface_vars = [args.var_name], []
    else:
        upper_vars, surface_vars = [], [args.var_name]

    init_time = pd.Timestamp(args.init_time)
    base_time = init_time.floor(f"{args.forecast_cycle_hours}h")

    # The 0.25deg boundary already lives on the model's computational grid, so use it as the
    # regrid target for the low-res variants (this is the grid the rollout hands to the model).
    baseline_ds = build_dataset(args, 0.25, 'interp', base_time, upper_vars, surface_vars)
    target_latitude, target_longitude = baseline_ds.get_latitude_longitude()
    lat_arr = np.asarray(target_latitude)
    lon_arr = np.asarray(target_longitude)
    print(f"model grid: {lat_arr.shape[0]}x{lon_arr.shape[0]}  "
          f"lat[{lat_arr.min():.2f}, {lat_arr.max():.2f}] lon[{lon_arr.min():.2f}, {lon_arr.max():.2f}]")

    baseline_field = extract_field(args, baseline_ds, base_time, init_time, surface_vars)
    if baseline_field is None:
        raise RuntimeError(f"No boundary data available at {init_time} (base {base_time}).")

    # Load every panel first so all panels can share one colour scale.
    fields = {}
    for resolution in args.resolutions:
        for apply_mode in args.apply_modes:
            ds_bd = build_dataset(args, resolution, apply_mode, base_time, upper_vars, surface_vars,
                                  target_latitude=target_latitude, target_longitude=target_longitude)
            field = extract_field(args, ds_bd, base_time, init_time, surface_vars)
            if field is None:
                print(f"  res={resolution} {apply_mode}: missing")
                continue
            if args.plot_mode == 'diff':
                field = field - baseline_field
            fields[(resolution, apply_mode)] = field
            print(f"  res={resolution:<4} {apply_mode:<6}: shape={field.shape} "
                  f"range=[{field.min():.2f}, {field.max():.2f}]")

    if not fields:
        raise RuntimeError("No boundary fields could be loaded.")

    # Shared colour scale so panels are directly comparable.
    all_values = np.concatenate([f.ravel() for f in fields.values()])
    if args.plot_mode == 'diff':
        vmax = float(np.abs(all_values).max())
        vmin, cmap = -vmax, 'bwr'
    else:
        vmin, vmax, cmap = float(all_values.min()), float(all_values.max()), 'viridis'

    rows, cols = len(args.resolutions), len(args.apply_modes)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows),
                             subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False)

    for i, resolution in enumerate(args.resolutions):
        for j, apply_mode in enumerate(args.apply_modes):
            ax = axes[i][j]
            field = fields.get((resolution, apply_mode))
            if field is None:
                ax.set_visible(False)
                continue

            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.set_extent([lon_arr.min(), lon_arr.max(), lat_arr.min(), lat_arr.max()],
                          crs=ccrs.PlateCarree())
            im = ax.pcolormesh(lon_arr, lat_arr, field, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading='auto', transform=ccrs.PlateCarree())

            # At 0.25deg the transform is a no-op, so the apply mode is not meaningful.
            label = f"{resolution}°" if resolution == 0.25 else f"{resolution}°, {apply_mode}"
            if resolution == 0.25:
                label += " (apply mode N/A)"
            if args.plot_mode == 'diff':
                label += f"\nmax|diff| = {np.abs(field).max():.2f}"
            ax.set_title(label)
            fig.colorbar(im, ax=ax)

    var_label = args.var_name if surface_vars else f"{args.var_name}_{args.level}"
    mode_label = "difference vs 0.25° baseline" if args.plot_mode == 'diff' else "boundary field"
    plt.suptitle(f"Boundary {var_label} ({mode_label}) @ {init_time.strftime('%Y-%m-%d %H:00')}",
                 fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_path = os.path.join(
        args.output_dir,
        f"boundary_{var_label}_{args.plot_mode}_{init_time.strftime('%Y%m%d_%H%M%S')}.png",
    )
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
