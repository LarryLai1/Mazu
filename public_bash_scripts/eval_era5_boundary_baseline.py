import pandas as pd
import torch
import xarray as xr
import os
import sys
from tqdm import tqdm

# Add the parent directory to sys.path so we can import datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
from datasets.BoundaryConditionDataset import BoundaryConditionDataset_ERA5
from utils.metrics import MSEAggregator

def save_err_agg_to_csv(lead_time_err_agg, out_path, surf_vars, atmos_rows, lead_times):
    rows = []
    lead_time_labels = [f"+{t}h" for t in lead_times]
    row_names = []

    for var in surf_vars:
        row = []
        for t in lead_times:
            agg = lead_time_err_agg[t]["surf_vars"].get(var)
            row.append(agg.mean() if agg is not None else None)
        rows.append(row)
        row_names.append(var)

    for var, lev in atmos_rows:
        row = []
        for t in lead_times:
            agg = lead_time_err_agg[t]["atmos_vars"].get(var, {}).get(lev)
            row.append(agg.mean() if agg is not None else None)
        rows.append(row)
        row_names.append(f"{var}_{lev}")

    df = pd.DataFrame(rows, index=row_names, columns=lead_time_labels)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)
    return df

def main():
    start_date_hour = "2020-07-01 00:00:00"
    end_date_hour = "2020-08-31 23:00:00"
    data_root_dir = "/tmp3/yunye0121/era5_tw"
    boundary_root_dir = "/tmp3/b12902101/era5_tw_forecast_3d"
    
    surface_variables = ["t2m", "u10", "v10", "msl"]
    upper_variables = ["u", "v", "t", "q", "z"]
    static_variables = ["lsm", "slt", "z"]
    levels = [1000, 925, 850, 700, 500, 300, 150, 50]
    latitude = [39.75, 5]
    longitude = [100, 144.75]
    
    lead_times = list(range(1, 25)) + [36, 48, 60, 72] 
    
    out_dir = "/tmp3/b12902101/LAM_output_2mo/era5_boundary_data"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "MSE.csv")

    ds_gt = ERA5TWDatasetforAurora(
        data_root_dir=data_root_dir,
        start_date_hour=start_date_hour,
        end_date_hour=end_date_hour,
        upper_variables=upper_variables,
        surface_variables=surface_variables,
        static_variables=static_variables,
        levels=levels,
        latitude=latitude,
        longitude=longitude,
        lead_time=1,
        input_time_window=1,
        rollout_step=1,
        sample_stride_hours=1
    )

    ds_bd = BoundaryConditionDataset_ERA5(
        boundary_root_dir=boundary_root_dir,
        start_date_hour=start_date_hour,
        end_date_hour=end_date_hour,
        upper_variables=upper_variables,
        surface_variables=surface_variables,
        levels=levels,
        latitude=latitude,
        longitude=longitude,
        boundary_width=0, # Use whole region as per user request
        forecast_cycle_hours=12,
        time_interp_mode="nearest"
    )

    # Initialize aggregators
    agg = {}
    for t in lead_times:
        agg[t] = {'surf_vars': {}, 'atmos_vars': {}}
        for var in surface_variables:
            mapped_var = ds_bd.map_var_name_for_Aurora(var)
            agg[t]['surf_vars'][mapped_var] = MSEAggregator()
        for var in upper_variables:
            mapped_var = ds_bd.map_var_name_for_Aurora(var)
            agg[t]['atmos_vars'][mapped_var] = {}
            for lev in levels:
                agg[t]['atmos_vars'][mapped_var][lev] = MSEAggregator()

    time_axis = pd.date_range(start=start_date_hour, end=end_date_hour, freq="12h")
    loss_fn = torch.nn.MSELoss(reduction="none")
    
    for base_time in tqdm(time_axis, desc="Evaluating Era5 Boundary Baseline"):
        try:
            source = ds_bd.get_boundary_source(base_time)
        except Exception as e:
            print(f"Skipping base_time {base_time} due to error loading boundary source: {e}")
            continue
            
        for lt in lead_times:
            target_time = base_time + pd.Timedelta(hours=lt)
            
            if target_time > pd.Timestamp("2020-09-04 00:00:00"):
                continue
                
            # Ground truth
            try:
                upper_path, sfc_path = ds_gt._dt_to_path(target_time)
                with xr.open_dataset(upper_path) as upper_nc_gt, xr.open_dataset(sfc_path) as sfc_nc_gt:
                    gt_dict = ds_gt._nc_to_dict(upper_nc_gt, sfc_nc_gt)
            except Exception as e:
                # Ground truth file might be missing, skip
                continue

            # Boundary prediction (baseline)
            try:
                pred_dict = ds_bd.get_boundary_at_time_from_source(source, base_time, target_time)
            except Exception as e:
                print(f"Error getting boundary at {target_time}: {e}")
                continue
                
            if pred_dict is None:
                continue

            # Calculate MSE
            for var in surface_variables:
                mapped_var = ds_bd.map_var_name_for_Aurora(var)
                pred_tensor = pred_dict["surf_vars"][mapped_var]
                gt_tensor = gt_dict["surf_vars"][mapped_var]
                
                # Check shapes
                if pred_tensor.shape != gt_tensor.shape:
                    continue
                    
                error_value_tensor = loss_fn(pred_tensor.float(), gt_tensor.float())
                # Mean over spatial dims to match AuroraLoss which reduces 'b t h w -> b'
                loss_mean = error_value_tensor.mean().unsqueeze(0)
                agg[lt]["surf_vars"][mapped_var].update(loss_mean)

            for var in upper_variables:
                mapped_var = ds_bd.map_var_name_for_Aurora(var)
                for i, lev in enumerate(levels):
                    pred_tensor = pred_dict["atmos_vars"][mapped_var][i]
                    gt_tensor = gt_dict["atmos_vars"][mapped_var][i]
                    
                    if pred_tensor.shape != gt_tensor.shape:
                        continue
                        
                    error_value_tensor = loss_fn(pred_tensor.float(), gt_tensor.float())
                    loss_mean = error_value_tensor.mean().unsqueeze(0)
                    agg[lt]["atmos_vars"][mapped_var][lev].update(loss_mean)

    # Save to CSV
    atmos_rows = []
    for v in upper_variables:
        mv = ds_bd.map_var_name_for_Aurora(v)
        for l in levels:
            atmos_rows.append((mv, l))
            
    mapped_surf = [ds_bd.map_var_name_for_Aurora(v) for v in surface_variables]
    
    save_err_agg_to_csv(agg, out_csv, mapped_surf, atmos_rows, lead_times)
    print(f"Evaluation complete. Results saved to {out_csv}")

if __name__ == '__main__':
    main()
