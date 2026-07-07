import argparse
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Add the parent directory to sys.path so we can import datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var_name', type=str, default='t2m', help='Surface variable like t2m, u10, v10, msl')
    parser.add_argument('--preds_dir', type=str, default='/tmp3/b12902101/LAM_output_preds/era5_boundary4_inject-inside_smooth_gaussian_interp_exact/preds/')
    parser.add_argument('--output_dir', type=str, default='residual_plots')
    parser.add_argument('--plot_mode', type=str, choices=['residual', 'prediction'], default='residual', help='Plot residual or pure prediction')
    parser.add_argument('--init_time', type=str, default='2020-07-01 01:00:00', help='Plotting initialization time (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--time_interp_mode', type=str, default='nearest', choices=['interpolation', 'nearest', 'exact'], help='Time interpolation mode for boundary forecast')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which variables list the target variable belongs to, to load only the required data
    upper_all = ["u", "v", "t", "q", "z"]
    surface_all = ["t2m", "u10", "v10", "msl"]
    
    if args.var_name in upper_all:
        upper_vars = [args.var_name]
        surface_vars = []
    else:
        # Default to surface variable if not in upper
        upper_vars = []
        surface_vars = [args.var_name]

    # Instantiate ground truth dataset
    data_root_dir = "/tmp3/yunye0121/era5_tw"
    
    lead_times = list(range(1, 25)) + [36, 48, 60, 72]
    init_time = pd.Timestamp(args.init_time)
    start_time = init_time
    end_time = init_time + pd.Timedelta(hours=max(lead_times))
    
    ds_gt = ERA5TWDatasetforAurora(
        data_root_dir=data_root_dir,
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

    # Get latitude and longitude arrays once
    # Load coordinates from the first target time file without doing a full dataset load
    first_target_time = init_time + pd.Timedelta(hours=lead_times[0])
    upper_path, sfc_path = ds_gt._dt_to_path(first_target_time)
    path_to_open = sfc_path if surface_vars else upper_path
    with xr.open_dataset(path_to_open) as ds_coords:
        lat_arr = ds_coords.latitude.sel(latitude=slice(*ds_gt.latitude)).values
        lon_arr = ds_coords.longitude.sel(longitude=slice(*ds_gt.longitude)).values

    # Check if preds_dir is in BoundaryConditionDataset_ERA5 format (e.g. era5_tw_forecast_3d)
    is_era5_forecast = False
    standard_file_found = False
    for lt in lead_times:
        pred_filename = f"{init_time.strftime('%Y%m%d_%H%M%S')}+{lt}hr.nc"
        pred_path = os.path.join(args.preds_dir, pred_filename)
        if os.path.exists(pred_path):
            standard_file_found = True
            break
            
    if not standard_file_found:
        date = init_time.normalize()
        name = date.strftime(r"%Y%m%d")
        p1 = os.path.join(args.preds_dir, date.strftime(r"%Y/%Y%m"), f"{name}_upper.nc")
        p2 = os.path.join(args.preds_dir, date.strftime(r"%Y%m%d"), f"{name}_upper.nc")
        p3 = os.path.join(args.preds_dir, f"{name}_upper.nc")
        if os.path.exists(p1) or os.path.exists(p2) or os.path.exists(p3):
            is_era5_forecast = True

    if is_era5_forecast:
        lead_times = list(range(0, 73, 6))
        from datasets.BoundaryConditionDataset import BoundaryConditionDataset_ERA5
        era5_init_time = init_time.floor('12h')
        ds_bd = BoundaryConditionDataset_ERA5(
            boundary_root_dir=args.preds_dir,
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
        )
        try:
            bd_source = ds_bd.get_boundary_source(era5_init_time)
        except Exception as e:
            print(f"Failed to load boundary source for era5_init_time {era5_init_time}: {e}")
            bd_source = None
    else:
        ds_bd = None
        bd_source = None

    num_plots = len(lead_times)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    for k in range(num_plots, len(axes)):
        axes[k].set_visible(False)

    for j, lt in enumerate(lead_times):
        target_time = init_time + pd.Timedelta(hours=lt)
        
        # Load Prediction
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
                print(f"Failed to get boundary prediction at target_time {target_time}: {e}")
        else:
            pred_filename = f"{init_time.strftime('%Y%m%d_%H%M%S')}+{lt}hr.nc"
            pred_path = os.path.join(args.preds_dir, pred_filename)
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
                    print(f"Failed to load prediction from {pred_path}: {e}")
        
        if has_pred:
            # Ground Truth
            upper_path, sfc_path = ds_gt._dt_to_path(target_time)
            
            upper_nc_gt = None
            sfc_nc_gt = None
            try:
                if upper_vars:
                    upper_nc_gt = xr.open_dataset(upper_path)
                if surface_vars:
                    sfc_nc_gt = xr.open_dataset(sfc_path)
                gt_dict = ds_gt._nc_to_dict(upper_nc_gt, sfc_nc_gt)
            finally:
                if upper_nc_gt is not None:
                    upper_nc_gt.close()
                if sfc_nc_gt is not None:
                    sfc_nc_gt.close()
                
            gt_var_name = ds_gt.map_var_name_for_Aurora(args.var_name)
            if surface_vars:
                gt_val = gt_dict["surf_vars"][gt_var_name].numpy()
            else:
                gt_val = gt_dict["atmos_vars"][gt_var_name].numpy()
            
            ax = axes[j]
            
            # Setup map features
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.set_extent([lon_arr.min(), lon_arr.max(), lat_arr.min(), lat_arr.max()], crs=ccrs.PlateCarree())
            
            if args.plot_mode == 'residual':
                residual = pred_val - gt_val
                vmax = max(abs(residual.min()), abs(residual.max()))
                im = ax.pcolormesh(lon_arr, lat_arr, residual, cmap='bwr', vmin=-vmax, vmax=vmax, shading='auto', transform=ccrs.PlateCarree())
            else:
                im = ax.pcolormesh(lon_arr, lat_arr, pred_val, cmap='viridis', shading='auto', transform=ccrs.PlateCarree())
                
            ax.set_title(f"+{lt}hr ({target_time.strftime('%m-%d %H:00')})")
            fig.colorbar(im, ax=ax)
        else:
            axes[j].set_title(f"+{lt}hr (Missing)")
            axes[j].set_visible(False)
            
    plot_title = "Residuals" if args.plot_mode == 'residual' else "Predictions"
    plt.suptitle(f"{plot_title} for Init: {init_time.strftime('%Y-%m-%d %H:00')} ({args.var_name})", fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    prefix = "residual" if args.plot_mode == 'residual' else "prediction"
    output_path = os.path.join(args.output_dir, f"{prefix}_init_{init_time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
