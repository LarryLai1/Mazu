#!/usr/bin/env python
# coding=utf-8

import argparse
import contextlib
import json
import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm.auto import tqdm
import random
import numpy as np
import sys
import os
from pathlib import Path
import xarray as xr

# Add current directory to path just in case
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from datasets.BoundaryConditionDataset import BoundaryConditionDataset_ERA5
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
from utils.metrics import MSEAggregator, MAEAggregator

VAR_NAME_MAPPING = {
    "t2m": "2t",
    "u10": "10u",
    "v10": "10v",
    "msl": "msl",
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate MSE between ERA5 boundary dataset forecast and ground truth.")
    parser.add_argument('--boundary_root_dir', type=str, required=True, help="Directory containing ERA5 boundary forecasts.")
    parser.add_argument('--data_root_dir', type=str, required=True, help="Directory containing ground truth (ERA5 TW).")
    parser.add_argument('--start_date_hour', type=str, required=True, help="Start datetime, e.g., '2020-08-01 00:00:00'.")
    parser.add_argument('--end_date_hour', type=str, required=True, help="End datetime, e.g., '2020-09-01 00:00:00'.")
    parser.add_argument('--surface_variables', type=str, nargs='+', default=["t2m", "u10", "v10", "msl"])
    parser.add_argument('--upper_variables', type=str, nargs='+', default=["u", "v", "t", "q", "z"])
    parser.add_argument('--levels', type=int, nargs='+', default=[1000, 925, 850, 700, 500, 300, 150, 50])
    parser.add_argument('--latitude', type=float, nargs=2, default=[39.75, 5])
    parser.add_argument('--longitude', type=float, nargs=2, default=[100, 144.75])
    parser.add_argument('--forecast_cycle_hours', type=int, default=12, help="Forecast cycle in hours (default: 12).")
    parser.add_argument('--csv_output_folder', type=str, required=True, help="Folder to save the output CSV.")
    parser.add_argument('--prediction_timedelta_hours', type=int, nargs='+', default=None,
                        help="List of forecast lead times in hours. If not specified, all available in files will be used.")
    parser.add_argument('--time_interp_mode', type=str, default="nearest", choices=["interpolation", "nearest", "exact"],
                        help="Time interpolation mode (exact requires matching ground truth target time exactly).")
    parser.add_argument('--eval_metric', type=str, nargs='+', default=["MSE", "MAE"], choices=["MSE", "MAE"])
    parser.add_argument('--seed', type=int, default=1126)
    parser.add_argument('--mp_world_size', type=int, default=None)
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated list of GPU ids to use, e.g. "0,1,2". Spawns one process per GPU.')

    return parser.parse_args()

def _resolve_mp_world_size(args):
    if getattr(args, 'gpus', None):
        gpus = [x for x in args.gpus.split(",") if x.strip() != ""]
        return max(1, len(gpus))
    if args.mp_world_size is not None:
        return max(1, args.mp_world_size)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        n = len([x for x in visible.split(",") if x.strip() != ""])
        return max(1, n)
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1

def create_datasets(args):
    prediction_timedelta_hours = args.prediction_timedelta_hours
    if prediction_timedelta_hours is None:
        try:
            start_dt = pd.Timestamp(args.start_date_hour)
            date = start_dt.normalize()
            name = date.strftime(r"%Y%m%d")
            candidate_path = Path(args.boundary_root_dir) / date.strftime(r"%Y/%Y%m") / f"{name}_upper.nc"
            if not candidate_path.exists():
                candidate_path = Path(args.boundary_root_dir) / date.strftime(r"%Y%m%d") / f"{name}_upper.nc"
            if not candidate_path.exists():
                candidate_path = Path(args.boundary_root_dir) / f"{name}_upper.nc"
            
            if candidate_path.exists():
                with xr.open_dataset(candidate_path) as ds:
                    if "prediction_timedelta" in ds.coords or "prediction_timedelta" in ds.dims:
                        pt_vals = ds["prediction_timedelta"].values
                        if np.issubdtype(pt_vals.dtype, np.timedelta64):
                            prediction_timedelta_hours = sorted(list(set(int(x / np.timedelta64(1, "h")) for x in pt_vals)))
                        else:
                            prediction_timedelta_hours = sorted(list(set(int(x) for x in pt_vals)))
                        prediction_timedelta_hours = [x for x in prediction_timedelta_hours if x <= 72]
        except Exception as e:
            print(f"Failed to auto-detect prediction_timedelta: {e}")
            
    if prediction_timedelta_hours is None:
        prediction_timedelta_hours = list(range(1, 25)) + [36, 48, 60, 72]
        print(f"Using default prediction_timedelta_hours: {prediction_timedelta_hours}")
    else:
        print(f"Using prediction_timedelta_hours: {prediction_timedelta_hours}")

    ds_bd = BoundaryConditionDataset_ERA5(
        boundary_root_dir=args.boundary_root_dir,
        start_date_hour=args.start_date_hour,
        end_date_hour=args.end_date_hour,
        upper_variables=args.upper_variables,
        surface_variables=args.surface_variables,
        levels=args.levels,
        latitude=args.latitude,
        longitude=args.longitude,
        boundary_width=0,
        prediction_timedeltas=prediction_timedelta_hours,
        forecast_cycle_hours=args.forecast_cycle_hours,
        time_interp_mode=args.time_interp_mode,
        use_cache=False,
    )

    ds_gt = ERA5TWDatasetforAurora(
        data_root_dir=args.data_root_dir,
        start_date_hour=args.start_date_hour,
        end_date_hour=args.end_date_hour,
        upper_variables=args.upper_variables,
        surface_variables=args.surface_variables,
        static_variables=["lsm", "slt", "z"],
        levels=args.levels,
        latitude=args.latitude,
        longitude=args.longitude,
        lead_time=1,
        input_time_window=1,
        rollout_step=1,
        sample_stride_hours=1,
    )
    
    return ds_bd, ds_gt, prediction_timedelta_hours

def _build_metric_lists(args, prediction_timedelta_hours):
    err_agg_list = []
    for metric in args.eval_metric:
        if metric == "MSE":
            aggregator = MSEAggregator
        elif metric == "MAE":
            aggregator = MAEAggregator
        else:
            raise ValueError(f"Unsupported eval metric: {metric}")

        agg = {}
        for lt in prediction_timedelta_hours:
            agg[lt] = {'surf_vars': {}, 'atmos_vars': {}}
            for var in args.surface_variables:
                mapped_var = VAR_NAME_MAPPING.get(var, var)
                agg[lt]['surf_vars'][mapped_var] = aggregator()
            for var in args.upper_variables:
                mapped_var = VAR_NAME_MAPPING.get(var, var)
                agg[lt]['atmos_vars'][mapped_var] = {}
                for lev in args.levels:
                    agg[lt]['atmos_vars'][mapped_var][lev] = aggregator()
        err_agg_list.append(agg)
    return err_agg_list

def evaluate(
    args,
    ds_bd,
    ds_gt,
    time_axis,
    prediction_timedelta_hours,
    err_agg_list,
    device,
):
    for base_time in tqdm(time_axis, desc="Evaluating Era5 Boundary Forecast"):
        try:
            source = ds_bd.get_boundary_source(base_time)
        except Exception:
            continue
            
        for lt in prediction_timedelta_hours:
            target_time = base_time + pd.Timedelta(hours=lt)
            
            # Ground truth
            try:
                upper_path, sfc_path = ds_gt._dt_to_path(target_time)
                with xr.open_dataset(upper_path) as upper_nc_gt, xr.open_dataset(sfc_path) as sfc_nc_gt:
                    gt_dict = ds_gt._nc_to_dict(upper_nc_gt, sfc_nc_gt)
            except Exception:
                continue
                
            if gt_dict is None:
                continue
 
            # Boundary forecast prediction
            try:
                pred_dict = ds_bd.get_boundary_at_time_from_source(source, base_time, target_time)
            except Exception:
                continue
                
            if pred_dict is None:
                continue
 
            # Surface Variables
            for var in args.surface_variables:
                mapped_var = ds_bd.map_var_name_for_Aurora(var)
                pred_tensor = pred_dict["surf_vars"][mapped_var].to(device)
                gt_tensor = gt_dict["surf_vars"][mapped_var].to(device)
                
                if pred_tensor.shape != gt_tensor.shape:
                    continue
                
                for metric, err_agg in zip(args.eval_metric, err_agg_list):
                    if metric == "MSE":
                        error_value_tensor = (pred_tensor.float() - gt_tensor.float()) ** 2
                    elif metric == "MAE":
                        error_value_tensor = torch.abs(pred_tensor.float() - gt_tensor.float())
                    loss_mean = error_value_tensor.mean().unsqueeze(0)
                    err_agg[lt]["surf_vars"][mapped_var].update(loss_mean)
 
            # Upper Variables
            for var in args.upper_variables:
                # Note: upper variables are not mapped in atmos_vars keys
                pred_tensor_all_levels = pred_dict["atmos_vars"][var].to(device)
                gt_tensor_all_levels = gt_dict["atmos_vars"][var].to(device)
                
                for i, lev in enumerate(args.levels):
                    pred_tensor = pred_tensor_all_levels[i]
                    gt_tensor = gt_tensor_all_levels[i]
                    
                    if pred_tensor.shape != gt_tensor.shape:
                        continue
                        
                    for metric, err_agg in zip(args.eval_metric, err_agg_list):
                        if metric == "MSE":
                            error_value_tensor = (pred_tensor.float() - gt_tensor.float()) ** 2
                        elif metric == "MAE":
                            error_value_tensor = torch.abs(pred_tensor.float() - gt_tensor.float())
                        loss_mean = error_value_tensor.mean().unsqueeze(0)
                        err_agg[lt]["atmos_vars"][var][lev].update(loss_mean)

def _err_agg_to_state(err_agg):
    state = {}
    for t, t_dict in err_agg.items():
        state[str(t)] = {"surf_vars": {}, "atmos_vars": {}}
        for var, agg in t_dict["surf_vars"].items():
            state[str(t)]["surf_vars"][var] = {
                "error_sum": float(agg.error_sum),
                "count": int(agg.count),
            }
        for var, lev_dict in t_dict["atmos_vars"].items():
            state[str(t)]["atmos_vars"][var] = {}
            for lev, agg in lev_dict.items():
                state[str(t)]["atmos_vars"][var][str(lev)] = {
                    "error_sum": float(agg.error_sum),
                    "count": int(agg.count),
                }
    return state

def _merge_state_into_err_agg(err_agg, state):
    for t, t_dict in state.items():
        ti = int(t)
        for var, s in t_dict["surf_vars"].items():
            err_agg[ti]["surf_vars"][var].error_sum += float(s["error_sum"])
            err_agg[ti]["surf_vars"][var].count += int(s["count"])
        for var, lev_dict in t_dict["atmos_vars"].items():
            for lev, s in lev_dict.items():
                li = int(lev)
                err_agg[ti]["atmos_vars"][var][li].error_sum += float(s["error_sum"])
                err_agg[ti]["atmos_vars"][var][li].count += int(s["count"])

def save_err_agg_to_csv(args, lead_time_err_agg, out_path, prediction_timedelta_hours):
    lead_times = sorted(prediction_timedelta_hours)
    lead_time_labels = [f"{t}h" for t in lead_times]

    surf_vars = sorted(list(set(VAR_NAME_MAPPING.get(v, v) for v in args.surface_variables)))
    
    atmos_rows = []
    for var in sorted(args.upper_variables):
        mapped_var = VAR_NAME_MAPPING.get(var, var)
        for lev in sorted(args.levels, reverse=True):
            atmos_rows.append((mapped_var, lev))

    rows = []
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
    df.to_csv(out_path)
    return df

def _mp_worker_entry(rank, world_size, args, prediction_timedelta_hours):
    set_seed(args.seed + rank)
    if torch.cuda.is_available():
        gpu_id = int(rank)
        if getattr(args, 'gpu_list', None):
            gpu_id = int(args.gpu_list[rank % len(args.gpu_list)])
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    ds_bd, ds_gt, _ = create_datasets(args)
    
    full_time_axis = pd.date_range(
        start=args.start_date_hour,
        end=args.end_date_hour,
        freq=f"{args.forecast_cycle_hours}h"
    )
    worker_time_axis = full_time_axis[rank::world_size]
    
    err_agg_list = _build_metric_lists(args, prediction_timedelta_hours)

    evaluate(
        args=args,
        ds_bd=ds_bd,
        ds_gt=ds_gt,
        time_axis=worker_time_axis,
        prediction_timedelta_hours=prediction_timedelta_hours,
        err_agg_list=err_agg_list,
        device=device,
    )

    tmp_root = Path(args.csv_output_folder)
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_path = tmp_root / f".mp_rank_{rank}_metrics.json"
    
    payload = {
        "rank": rank,
        "metrics": {
            metric: _err_agg_to_state(err_agg)
            for metric, err_agg in zip(args.eval_metric, err_agg_list)
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

def main():
    args = parse_args()
    
    gpu_list = None
    if getattr(args, 'gpus', None):
        gpu_list = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
        args.gpu_list = gpu_list
    else:
        args.gpu_list = None

    world_size = _resolve_mp_world_size(args)
    
    Path(args.csv_output_folder).mkdir(parents=True, exist_ok=True)
    
    # Pre-detect prediction timedelta using single dummy datasets instantiation
    _, _, prediction_timedelta_hours = create_datasets(args)

    if world_size <= 1:
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds_bd, ds_gt, _ = create_datasets(args)
        
        time_axis = pd.date_range(
            start=args.start_date_hour,
            end=args.end_date_hour,
            freq=f"{args.forecast_cycle_hours}h"
        )
        
        err_agg_list = _build_metric_lists(args, prediction_timedelta_hours)

        evaluate(
            args=args,
            ds_bd=ds_bd,
            ds_gt=ds_gt,
            time_axis=time_axis,
            prediction_timedelta_hours=prediction_timedelta_hours,
            err_agg_list=err_agg_list,
            device=device,
        )

        for metric, err_agg in zip(args.eval_metric, err_agg_list):
            csv_output_path = Path(args.csv_output_folder) / f"{metric}.csv"
            save_err_agg_to_csv(args, err_agg, csv_output_path, prediction_timedelta_hours)
            print(f"Results saved to {csv_output_path}")
        return

    print(f"Running multiprocessing evaluation with world_size={world_size}")
    mp_ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        p = mp_ctx.Process(target=_mp_worker_entry, args=(rank, world_size, args, prediction_timedelta_hours))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}.")

    merged_err_agg_list = _build_metric_lists(args, prediction_timedelta_hours)
    
    tmp_root = Path(args.csv_output_folder)
    for rank in range(world_size):
        p = tmp_root / f".mp_rank_{rank}_metrics.json"
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for metric_idx, metric in enumerate(args.eval_metric):
            _merge_state_into_err_agg(merged_err_agg_list[metric_idx], payload["metrics"][metric])
        p.unlink(missing_ok=True)

    for metric, err_agg in zip(args.eval_metric, merged_err_agg_list):
        csv_output_path = Path(args.csv_output_folder) / f"{metric}.csv"
        save_err_agg_to_csv(args, err_agg, csv_output_path, prediction_timedelta_hours)
        print(f"Results saved to {csv_output_path}")

if __name__ == "__main__":
    main()
