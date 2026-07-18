import argparse
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import sys
import numpy as np

# Add the parent directory to sys.path so we can import datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora

def compute_wavenumber_spectrum(data_2d, lat, lon, crop_width=8):
    # Crop boundaries if requested
    if crop_width > 0:
        data_2d = data_2d[crop_width:-crop_width, crop_width:-crop_width]
        lat = lat[crop_width:-crop_width]
        lon = lon[crop_width:-crop_width]
        
    # Centering (remove spatial mean)
    centered = data_2d - np.nanmean(data_2d)
    
    # 2D FFT
    fft2_vals = np.fft.fft2(centered)
    fft2_mag = np.abs(np.fft.fftshift(fft2_vals))
    
    # Grid spacing (longitude spacing in degrees)
    dlon = float(np.nanmean(np.diff(lon)))
    freq_lon = np.fft.fftshift(np.fft.fftfreq(lon.size, d=dlon))
    
    # Meridional averaging (average along latitude axis = 0)
    zonal_mag = np.nanmean(fft2_mag, axis=0)
    
    # Return positive frequencies only
    pos_mask = freq_lon > 0
    return freq_lon[pos_mask], zonal_mag[pos_mask]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var_name', type=str, default='msl', help='Variable like msl, t2m, u10, v10')
    parser.add_argument('--preds_dirs', nargs='+', required=True, help='List of directories containing predictions')
    parser.add_argument('--labels', nargs='+', required=True, help='Labels corresponding to predictions directories')
    parser.add_argument('--init_times', nargs='+', required=True, help='Initialization times corresponding to directories')
    parser.add_argument('--crop_width', type=int, default=8, help='Number of boundary pixels to ignore')
    parser.add_argument('--data_root_dir', type=str, default='/tmp3/yunye0121/era5_tw', help='Ground truth root directory')
    parser.add_argument('--output_dir', type=str, default='wavenumber_plots', help='Output directory')
    parser.add_argument('--time_interp_mode', type=str, default='nearest', choices=['exact', 'nearest', 'interpolated'])
    parser.add_argument('--boundary_resolutions', type=float, nargs='+', default=None, choices=[0.25, 0.5, 1.5],
                        help='Per-entry ERA5 boundary resolution, parallel to --preds_dirs (defaults to 0.25 '
                             'everywhere). 0.5 pools the 0.25deg source by 2 on the fly; 1.5 needs the entry to '
                             'point at era5_tw_forecast_1.5deg. Ignored for entries holding model predictions.')
    parser.add_argument('--boundary_lowres_apply_modes', type=str, nargs='+', default=None, choices=['direct', 'interp'],
                        help='Per-entry apply mode, parallel to --preds_dirs (defaults to interp everywhere): '
                             'direct=block/nearest footprint; interp=bilinear/linear. Ignored at resolution 0.25.')
    args = parser.parse_args()

    if len(args.preds_dirs) != len(args.labels) or len(args.preds_dirs) != len(args.init_times):
        raise ValueError("Lengths of preds_dirs, labels, and init_times must be equal.")

    num_entries = len(args.preds_dirs)
    boundary_resolutions = args.boundary_resolutions if args.boundary_resolutions is not None else [0.25] * num_entries
    boundary_apply_modes = args.boundary_lowres_apply_modes if args.boundary_lowres_apply_modes is not None else ['interp'] * num_entries
    if len(boundary_resolutions) != num_entries or len(boundary_apply_modes) != num_entries:
        raise ValueError("Lengths of boundary_resolutions and boundary_lowres_apply_modes must match preds_dirs.")

    os.makedirs(args.output_dir, exist_ok=True)

    surface_vars = []
    upper_vars = []
    if args.var_name in ['t2m', 'u10', 'v10', 'msl']:
        surface_vars = [args.var_name]
    else:
        upper_vars = [args.var_name]

    # Convert initial times to DatetimeIndexss
    init_times_parsed = [pd.Timestamp(t) for t in args.init_times]
    lead_times = list(range(0, 241, 12))
    # lead_times = list(range(0, 73, 6))

    # Determine time window range for Ground Truth
    min_init = min(init_times_parsed)
    max_init = max(init_times_parsed)
    start_time = min_init
    end_time = max_init + pd.Timedelta(hours=max(lead_times))

    # Instantiate Ground Truth dataset
    ds_gt = ERA5TWDatasetforAurora(
        data_root_dir=args.data_root_dir,
        start_date_hour=start_time,
        end_date_hour=end_time,
        upper_variables=upper_vars,
        surface_variables=surface_vars,
        static_variables=[],
        levels=[1000, 925, 850, 700, 500, 300, 150, 50],
        latitude=[39.75, 5],
        longitude=[100, 144.75],
        lead_time=1,
        input_time_window=1,
        rollout_step=1,
        sample_stride_hours=1
    )

    # Get coordinate arrays
    first_target_time = init_times_parsed[0] + pd.Timedelta(hours=lead_times[0])
    upper_path, sfc_path = ds_gt._dt_to_path(first_target_time)
    path_to_open = sfc_path if surface_vars else upper_path
    with xr.open_dataset(path_to_open) as ds_coords:
        lat_arr = ds_coords.latitude.sel(latitude=slice(*ds_gt.latitude)).values
        lon_arr = ds_coords.longitude.sel(longitude=slice(*ds_gt.longitude)).values

    # Setup each predictions directory's loader configuration
    loaders_config = []
    for preds_dir, init_time, label, boundary_resolution, apply_mode in zip(
        args.preds_dirs, init_times_parsed, args.labels, boundary_resolutions, boundary_apply_modes
    ):
        # Format Auto-detection
        standard_file_found = False
        for lt in lead_times:
            pred_filename = f"{init_time.strftime('%Y%m%d_%H%M%S')}+{lt}hr.nc"
            pred_path = os.path.join(preds_dir, pred_filename)
            if os.path.exists(pred_path):
                standard_file_found = True
                break
                
        is_era5_forecast = not standard_file_found
        
        ds_bd = None
        bd_source = None
        era5_init_time = None
        
        if is_era5_forecast:
            from datasets.BoundaryConditionDataset import BoundaryConditionDataset_ERA5
            era5_init_time = init_time.floor('12h')
            ds_bd = BoundaryConditionDataset_ERA5(
                boundary_root_dir=preds_dir,
                start_date_hour=era5_init_time,
                end_date_hour=era5_init_time,
                upper_variables=upper_vars,
                surface_variables=surface_vars,
                levels=[1000, 925, 850, 700, 500, 300, 150, 50],
                latitude=[39.75, 5],
                longitude=[100, 144.75],
                boundary_width=0,
                prediction_timedeltas=lead_times,
                forecast_cycle_hours=12,
                time_interp_mode=args.time_interp_mode,
                use_cache=False,
                # The spectrum is compared against the ground truth on lat_arr/lon_arr, so that is
                # the grid a low-res boundary has to be brought back onto.
                target_latitude=lat_arr,
                target_longitude=lon_arr,
                boundary_resolution=boundary_resolution,
                lowres_apply_mode=apply_mode,
            )
            try:
                bd_source = ds_bd.get_boundary_source(era5_init_time)
            except Exception as e:
                print(f"Failed to load boundary source for {label}: {e}")
                
        loaders_config.append({
            'preds_dir': preds_dir,
            'init_time': init_time,
            'label': label,
            'is_era5_forecast': is_era5_forecast,
            'ds_bd': ds_bd,
            'bd_source': bd_source,
            'era5_init_time': era5_init_time
        })

    # Prepare plotting subplots
    num_plots = len(lead_times)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4.5 * rows))
    axes = axes.flatten()

    fig_ratio, axes_ratio = plt.subplots(rows, cols, figsize=(20, 4.5 * rows))
    axes_ratio = axes_ratio.flatten()

    for k in range(num_plots, len(axes)):
        axes[k].set_visible(False)
        axes_ratio[k].set_visible(False)

    for j, lt in enumerate(lead_times):
        ax = axes[j]
        ax.set_title(f"+{lt}hr", fontsize=14)
        ax.set_xlabel("Zonal Wavenumber ($k_x$, cycles/deg)", fontsize=10)
        ax.set_ylabel("Power Spectrum", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

        ax_ratio = axes_ratio[j]
        ax_ratio.set_title(f"+{lt}hr", fontsize=14)
        ax_ratio.set_xlabel("Zonal Wavenumber ($k_x$, cycles/deg)", fontsize=10)
        ax_ratio.set_ylabel("Prediction / Ground Truth", fontsize=10)
        ax_ratio.grid(True, which="both", alpha=0.3)
        ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
        ax_ratio.set_xscale("log")
        ax_ratio.set_ylim(0, 5)

        # Plot Ground Truth (loaded once per target time across all comparisons)
        gt_loaded = False
        gt_val = None
        for config in loaders_config:
            init_time = config['init_time']
            target_time = init_time + pd.Timedelta(hours=lt)
            
            # Load Ground Truth
            upper_path, sfc_path = ds_gt._dt_to_path(target_time)
            upper_nc_gt = None
            sfc_nc_gt = None
            try:
                if upper_vars:
                    upper_nc_gt = xr.open_dataset(upper_path)
                if surface_vars:
                    sfc_nc_gt = xr.open_dataset(sfc_path)
                gt_dict = ds_gt._nc_to_dict(upper_nc_gt, sfc_nc_gt)
                if gt_dict is not None:
                    gt_var_name = ds_gt.map_var_name_for_Aurora(args.var_name)
                    if surface_vars:
                        gt_val = gt_dict["surf_vars"][gt_var_name].numpy()
                    else:
                        gt_val = gt_dict["atmos_vars"][gt_var_name].numpy()
                    gt_loaded = True
                    break
            except Exception as e:
                print(f"Failed to load Ground Truth at target_time {target_time}: {e}")
            finally:
                if upper_nc_gt is not None:
                    upper_nc_gt.close()
                if sfc_nc_gt is not None:
                    sfc_nc_gt.close()
        
        k_gt = None
        power_gt = None
        if gt_loaded and gt_val is not None:
            k_gt, power_gt = compute_wavenumber_spectrum(gt_val, lat_arr, lon_arr, crop_width=args.crop_width)
            ax.loglog(k_gt, power_gt, color='black', linestyle='--', linewidth=2.5, label='Ground Truth')

        # Plot each prediction source
        for config in loaders_config:
            preds_dir = config['preds_dir']
            init_time = config['init_time']
            label = config['label']
            is_era5_forecast = config['is_era5_forecast']
            ds_bd = config['ds_bd']
            bd_source = config['bd_source']
            era5_init_time = config['era5_init_time']
            
            target_time = init_time + pd.Timedelta(hours=lt)
            pred_val = None
            has_pred = False

            if is_era5_forecast and bd_source is not None:
                try:
                    pred_dict = ds_bd.get_boundary_at_time_from_source(bd_source, era5_init_time, target_time)
                    if pred_dict is not None:
                        gt_var_name = ds_gt.map_var_name_for_Aurora(args.var_name)
                        if surface_vars:
                            pred_val = pred_dict["surf_vars"][gt_var_name].cpu().numpy()
                        else:
                            pred_val = pred_dict["atmos_vars"][gt_var_name].cpu().numpy()
                        has_pred = True
                except Exception as e:
                    print(f"Failed to get boundary prediction for {label} at target_time {target_time}: {e}")
            else:
                pred_filename = f"{init_time.strftime('%Y%m%d_%H%M%S')}+{lt}hr.nc"
                pred_path = os.path.join(preds_dir, pred_filename)
                if os.path.exists(pred_path):
                    try:
                        with xr.open_dataset(pred_path) as pred_nc:
                            gt_var_name = ds_gt.map_var_name_for_Aurora(args.var_name)
                            if surface_vars:
                                pred_var_name = f"surf_{gt_var_name}"
                            else:
                                pred_var_name = f"atmos_{gt_var_name}"
                                if pred_var_name not in pred_nc:
                                    for opt in [gt_var_name, f"surf_{gt_var_name}"]:
                                        if opt in pred_nc:
                                            pred_var_name = opt
                                            break
                            pred_val = pred_nc[pred_var_name].values[0]
                            has_pred = True
                    except Exception as e:
                        print(f"Failed to load prediction from {pred_path} for {label}: {e}")

            if has_pred and pred_val is not None:
                k_pred, power_pred = compute_wavenumber_spectrum(pred_val, lat_arr, lon_arr, crop_width=args.crop_width)
                ax.loglog(k_pred, power_pred, alpha=0.8, linewidth=1, label=label)

                if power_gt is not None:
                    # The prediction may sit on a different wavenumber grid, so bring it onto the
                    # ground truth axis before dividing.
                    power_pred_on_gt = np.interp(k_gt, k_pred, power_pred, left=np.nan, right=np.nan)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = power_pred_on_gt / power_gt
                    ax_ratio.plot(k_gt, ratio, alpha=0.8, linewidth=1, label=label)

        ax.legend(fontsize=8, loc='lower left')
        ax_ratio.legend(fontsize=8, loc='lower left')

    fig.suptitle(f"Zonal Wavenumber Spectrum Comparison ({args.var_name}) - Crop Width: {args.crop_width}px", fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(args.output_dir, f"wavenumber_comparison_{args.var_name}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved to {output_path}")

    fig_ratio.suptitle(
        f"Zonal Wavenumber Spectrum Ratio to Ground Truth ({args.var_name}) - Crop Width: {args.crop_width}px",
        fontsize=20,
    )
    fig_ratio.tight_layout(rect=[0, 0.03, 1, 0.95])

    ratio_output_path = os.path.join(args.output_dir, f"wavenumber_ratio_{args.var_name}.png")
    fig_ratio.savefig(ratio_output_path, dpi=150)
    plt.close(fig_ratio)
    print(f"Ratio plot saved to {ratio_output_path}")

if __name__ == '__main__':
    main()
