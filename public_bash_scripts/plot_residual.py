import argparse
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import sys

# Add the parent directory to sys.path so we can import datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var_name', type=str, default='t2m', help='Surface variable like t2m, u10, v10, msl')
    parser.add_argument('--preds_dir', type=str, default='/tmp3/b12902101/LAM_output_preds/era5_boundary4_inject-inside_smooth_gaussian_interp_exact/preds/')
    parser.add_argument('--output_dir', type=str, default='residual_plots')
    parser.add_argument('--plot_mode', type=str, choices=['residual', 'prediction'], default='residual', help='Plot residual or pure prediction')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Instantiate ground truth dataset
    data_root_dir = "/tmp3/yunye0121/era5_tw"
    start_time = "2020-08-01 00:00:00"
    end_time = "2020-08-06 23:00:00"
    
    ds_gt = ERA5TWDatasetforAurora(
        data_root_dir=data_root_dir,
        start_date_hour=start_time,
        end_date_hour=end_time,
        upper_variables=["u", "v", "t", "q", "z"],
        surface_variables=["t2m", "u10", "v10", "msl"],
        static_variables=["lsm", "slt", "z"],
        levels=[1000, 925, 850, 700, 500, 300, 150, 50],
        latitude=[39.75, 5],
        longitude=[100, 144.75],
        lead_time=1,
        input_time_window=1,
        rollout_step=1,
        sample_stride_hours=1
    )

    lead_times = list(range(1, 25)) + [36, 48, 60, 72]

    # Get latitude and longitude arrays once
    lat_arr, lon_arr = ds_gt.get_latitude_longitude()
    lat_arr = lat_arr.numpy()
    lon_arr = lon_arr.numpy()

    for i in range(6):
        init_time = pd.Timestamp("2020-08-01 06:00:00") + pd.Timedelta(hours=i)
        
        fig, axes = plt.subplots(7, 4, figsize=(20, 28))
        axes = axes.flatten()

        for j, lt in enumerate(lead_times):
            target_time = init_time + pd.Timedelta(hours=lt)
            
            # Prediction
            pred_filename = f"{init_time.strftime('%Y%m%d_%H%M%S')}+{lt}hr.nc"
            pred_path = os.path.join(args.preds_dir, pred_filename)
            
            if os.path.exists(pred_path):
                # Ground Truth
                upper_path, sfc_path = ds_gt._dt_to_path(target_time)
                with xr.open_dataset(upper_path) as upper_nc_gt, xr.open_dataset(sfc_path) as sfc_nc_gt:
                    gt_dict = ds_gt._nc_to_dict(upper_nc_gt, sfc_nc_gt)
                    
                gt_var_name = ds_gt.map_var_name_for_Aurora(args.var_name)
                gt_val = gt_dict["surf_vars"][gt_var_name].numpy()
                
                with xr.open_dataset(pred_path) as pred_nc:
                    pred_var_name = f"surf_{gt_var_name}"
                    pred_val = pred_nc[pred_var_name].values[0]
                    
                ax = axes[j]
                
                if args.plot_mode == 'residual':
                    residual = pred_val - gt_val
                    vmax = max(abs(residual.min()), abs(residual.max()))
                    im = ax.pcolormesh(lon_arr, lat_arr, residual, cmap='bwr', vmin=-vmax, vmax=vmax, shading='auto')
                else:
                    im = ax.pcolormesh(lon_arr, lat_arr, pred_val, cmap='viridis', shading='auto')
                    
                ax.set_title(f"+{lt}hr ({target_time.strftime('%m-%d %H:00')})")
                fig.colorbar(im, ax=ax)
            else:
                axes[j].set_title(f"+{lt}hr (Missing)")
                axes[j].axis('off')
            
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
