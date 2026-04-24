#!/usr/bin/env python
# coding=utf-8

import argparse
import contextlib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
import numpy as np
from pathlib import Path
import xarray as xr
from safetensors.torch import load_file
import logging

# --- Aurora / Pangu Imports ---
from aurora import Batch, Metadata
from utils.custom_rollout import rollout_with_gpu
# from aurora.model.aurora import AuroraSmall  <-- REMOVED
from pangu_lib.models.pangu import PanguModel   # <-- ADDED
from pangu_lib.adapter import PanguAuroraAdapter # <-- ADDED

from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
from utils.metrics import AuroraMAELoss, AuroraMSELoss
from utils.metrics import prepare_each_lead_time_agg

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description = "Pangu-Aurora Inference Script (Single GPU).")
    parser.add_argument('--data_root_dir', type = str, required = True)
    # parser.add_argument("--use_pretrained_weight", action = "store_true") # Not usually used for custom Pangu training
    parser.add_argument('--checkpoint_path', type = str, default = None)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--start_date_hour', type = str, required = True)
    parser.add_argument('--end_date_hour', type = str, required = True)
    parser.add_argument('--upper_variables', type = str, nargs = '+', required = True)
    parser.add_argument('--surface_variables', type = str, nargs = '+', required = True)
    parser.add_argument('--static_variables', type = str, nargs = '+', required = True)
    parser.add_argument('--levels', type = int, nargs = '+', required = True)
    parser.add_argument('--latitude', type = float, nargs = 2, required = True)
    parser.add_argument('--longitude', type = float, nargs = 2, required = True)
    parser.add_argument('--lead_time', type = int, default = 0)
    parser.add_argument('--input_time_window', type = int, default = 2)
    parser.add_argument('--rollout_step', type = int, default = 1)

    parser.add_argument("--timestep_hours", type = int, default = 6)
    parser.add_argument('--use_lora', action = 'store_true') # Kept for args compat, might not be used by Pangu
    parser.add_argument('--bf16_mode', action = 'store_true')
    parser.add_argument('--stabilise_level_agg', action = 'store_true')

    parser.add_argument("--gen_result_folder", type = str, default = './gen_result',)
    parser.add_argument("--save_rollout_step", type = int, nargs = "+", default = None)
    parser.add_argument("--eval_metric", type = str, nargs = "+", default = ["MSE"], choices = ["MSE", "MAE"])

    parser.add_argument("--csv_output_folder", type = str, default = "./errs")
    parser.add_argument('--mixed_precision', type = str, default = None, choices = ["no", "fp16", "bf16"])

    return parser.parse_args()

def create_model(args, device):
    # --- MIGRATED FROM TRAINING SCRIPT ---
    
    # 1. Initialize the Base Pangu Model
    pangu_base = PanguModel(
        data_spatial_shape = (len(args.levels), int(abs(args.latitude[1] - args.latitude[0]) / 0.25) + 1, int(abs(args.longitude[1] - args.longitude[0]) / 0.25) + 1),
        upper_vars = len(args.upper_variables),
        surface_vars = len(args.surface_variables),
        depths =[2, 6],       # Standard Pangu Config (Hardcoded as per training script)
        heads = [6, 12],      # Standard Pangu Config (Hardcoded as per training script)
        embed_dim = 192,      # Standard Pangu Config
        patch_shape = (2, 4, 4),
        window_size = (2, 6, 12),
        # NOTE: Verify these paths exist on your inference machine!
        constant_mask_paths = ["/work/yunye0121/Mazu/pangu_lib/static_npy_data/lsm.npy", "/work/yunye0121/Mazu/pangu_lib/static_npy_data/slt.npy", "/work/yunye0121/Mazu/pangu_lib/static_npy_data/z.npy"], 
        smoothing_kernel_size = None,
        segmented_smooth = False,
        segmented_smooth_boundary_width = None,
        learned_smooth = False,
        residual = False,
        res_conn_after_smooth = False,
    )

    # 2. Load Surface Stats for Aurora Normalization
    # Matches training script dummy stats
    surf_stats_dummy = {}

    # 3. Wrap it
    model = PanguAuroraAdapter(pangu_base, surf_stats = surf_stats_dummy)

    # 4. Load weights
    if args.checkpoint_path:
        logger.info(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = load_file(args.checkpoint_path)
        # Note: In training script, weights were loaded into 'model' (the Adapter) directly
        model.load_state_dict(state_dict, strict = False)
    else:
        logger.warning("No checkpoint path provided! Model is initialized with random weights.")

    model.to(device)
    model.eval()
    return model

def create_dataset(args):
    ds = ERA5TWDatasetforAurora(
        data_root_dir = args.data_root_dir,
        start_date_hour = args.start_date_hour,
        end_date_hour = args.end_date_hour,
        upper_variables = args.upper_variables,
        surface_variables = args.surface_variables,
        static_variables = args.static_variables,
        levels = args.levels,
        latitude = args.latitude,
        longitude = args.longitude,
        lead_time = args.lead_time,
        input_time_window = args.input_time_window,
        rollout_step = args.rollout_step,
    )
    return ds

def log_weather_variable_error_with_lead_time(loss_dict, t, lead_time_agg):
    for v in loss_dict["surf_vars"]:
        lead_time_agg[t]["surf_vars"][v].update( loss_dict["surf_vars"][v] )
    for v in loss_dict["atmos_vars"]:
        for l in loss_dict["atmos_vars"][v]:
            lead_time_agg[t]["atmos_vars"][v][l].update( loss_dict["atmos_vars"][v][l] )

def slice_timeaxis(labels):
    timeaxis_length = next(iter(next(iter(labels.values())).values())).shape[1]
    n_g = {}
    for i in range(timeaxis_length):
        n_g[i] = {}
        for var_type, var_dict in labels.items():
            n_g[i][var_type] = {}
            for var_name, tensor in var_dict.items():
                n_g[i][var_type][var_name] = tensor[:, i : i + 1]
    return n_g

def AuroraBatch_2_nc_files(
    batch,
    args,
):
    surf_vars = batch.surf_vars.keys()
    atmos_vars = batch.atmos_vars.keys()
    static_vars = batch.static_vars.keys()

    def _np(d):
        return d.detach().cpu().numpy()

    _s = set(
        [batch.surf_vars[var].shape[0] for var in surf_vars] +
        [batch.atmos_vars[var].shape[0] for var in atmos_vars]
    )

    assert len(_s) == 1

    batch_dim = next(iter(_s))

    for i in range(batch_dim):
        data_vars = {}

        for k, v in batch.surf_vars.items():
            arr = _np(v)[i]
            data_vars[f"surf_{k}"] = (("history", "latitude", "longitude"), arr)

        for k, v in batch.atmos_vars.items():
            arr = _np(v)[i]
            data_vars[f"atmos_{k}"] = (("history", "level", "latitude", "longitude"), arr)

        for k, v in batch.static_vars.items():
            arr = _np(v)
            data_vars[f"static_{k}"] = (("latitude", "longitude"), arr)

        coords = {
            "latitude": _np(batch.metadata.lat),
            "longitude": _np(batch.metadata.lon),
            "time": [batch.metadata.time[i]],
            "level": list(batch.metadata.atmos_levels),
            "rollout_step": batch.metadata.rollout_step,
        }

        ds = xr.Dataset(data_vars, coords = coords)
        rs = int(batch.metadata.rollout_step)
        output_file_name = f"{(batch.metadata.time[i] - pd.Timedelta(hours = rs * args.lead_time)).strftime('%Y%m%d_%H%M%S')}+{rs * args.lead_time}hr.nc"
        
        gen_result_folder = Path(args.gen_result_folder)
        output_path = gen_result_folder / output_file_name

        ds.to_netcdf( output_path )

def evaluate(args, model, dataloader, criterion_list, err_agg_list, device):
    model.eval()
    latitudes, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    # Optimization: Use inference_mode to reduce memory for gradients
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels, dates = batch
            
            # --- Data moving to device ---
            for _k_var_type in inputs:
                for _k_var in inputs[_k_var_type]:
                    inputs[_k_var_type][_k_var] = inputs[_k_var_type][_k_var].to(device)
            for _k_var_type in labels:
                for _k_var in labels[_k_var_type]:
                    labels[_k_var_type][_k_var] = labels[_k_var_type][_k_var].to(device)
            if isinstance(static_data["static_vars"], torch.Tensor):
                static_data["static_vars"] = static_data["static_vars"].to(device)

            # Pre-slice labels
            _label_list = slice_timeaxis(labels)

            _input = Batch(
                surf_vars=inputs["surf_vars"],
                atmos_vars=inputs["atmos_vars"],
                static_vars=static_data["static_vars"],
                metadata=Metadata(
                    lat=latitudes,
                    lon=longitude,
                    time=tuple(map(lambda d: pd.Timestamp(d), dates)),
                    atmos_levels=levels,
                ),
            )

            # --- Setup Mixed Precision ---
            use_amp = (args.mixed_precision in ("fp16", "bf16")) and (device.type == "cuda")
            dtype = torch.float32  # Default
            if use_amp:
                if args.mixed_precision == "fp16":
                    dtype = torch.float16
                elif args.mixed_precision == "bf16":
                    dtype = torch.bfloat16

            context_manager = torch.cuda.amp.autocast(dtype = dtype) if use_amp else contextlib.nullcontext()
            
            with context_manager:
                # Use the same rollout generator. 
                # PanguAuroraAdapter ensures the model accepts the Batch object.
                generator = rollout_with_gpu(model, _input, steps=args.rollout_step)
                
                for step_index, _pred in enumerate(generator):
                    t = step_index + 1
                    
                    # 1. Get the corresponding label
                    _label_data = _label_list[step_index]
                    
                    _label = Batch(
                        surf_vars=_label_data["surf_vars"],
                        atmos_vars=_label_data["atmos_vars"],
                        static_vars=static_data["static_vars"],
                        metadata=Metadata(
                            lat=latitudes,
                            lon=longitude,
                            time=tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours = t * args.lead_time) , dates)),
                            atmos_levels=levels,
                        ),
                    )

                    # 2. Calculate Loss
                    for (criterion, err_agg) in zip(criterion_list, err_agg_list):
                        # Note: In inference, we typically evaluate on Physical values (Raw).
                        # The training script normalized before loss, but that is common for training stability.
                        # For reporting RMSE/MAE, we usually want real units (Degrees, m/s).
                        # Assuming the Adapter output and Dataset labels are compatible in physical space here.
                        loss_dict = criterion(_pred, _label)
                        log_weather_variable_error_with_lead_time(
                            loss_dict,
                            t * args.lead_time,
                            err_agg,
                        )

                    # 3. Save to disk if needed
                    if args.save_rollout_step and t in args.save_rollout_step:
                        AuroraBatch_2_nc_files(
                            batch=_pred,
                            args=args,
                        )

def export_agg_to_csv(
        args,
        lead_time_err_agg,
        out_path,
    ):

    lead_times = sorted(lead_time_err_agg.keys())
    lead_time_labels = [f"{t}h" for t in lead_times]

    surf_vars = set()
    atmos_vars_levels = dict()
    for t in lead_time_err_agg:
        for var in lead_time_err_agg[t]["surf_vars"]:
            surf_vars.add(var)
        for var in lead_time_err_agg[t]["atmos_vars"]:
            if var not in atmos_vars_levels:
                atmos_vars_levels[var] = set()
            for lev in lead_time_err_agg[t]["atmos_vars"][var]:
                atmos_vars_levels[var].add(lev)
    surf_vars = sorted(list(surf_vars))

    atmos_rows = []
    for var in sorted(atmos_vars_levels.keys()):
        levels = sorted(list(atmos_vars_levels[var]), reverse = True)
        for lev in levels:
            atmos_rows.append((var, lev))

    rows = []
    row_names = []

    for var in surf_vars:
        row = []
        for t in lead_times:
            agg = lead_time_err_agg[t]["surf_vars"].get(var)
            row.append( agg.mean() if agg is not None else None)
        rows.append(row)
        row_names.append(var)

    for var, lev in atmos_rows:
        row = []
        for t in lead_times:
            agg = lead_time_err_agg[t]["atmos_vars"].get(var, {}).get(lev)
            row.append( agg.mean() if agg is not None else None)
        rows.append(row)
        row_names.append(f"{var}_{lev}")

    df = pd.DataFrame(rows, index = row_names, columns = lead_time_labels)
    df.to_csv(out_path)
    return df

def main():
    args = parse_args()
    set_seed(args.seed)
    logger.info("Running single-GPU Pangu evaluation.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = create_model(args, device)
    dataset = create_dataset(args)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = True)

    criterion_list = []
    err_agg_list = []
    for metric in args.eval_metric:
        if metric == "MSE":
            criterion_list.append(AuroraMSELoss)
        elif metric == "MAE":
            criterion_list.append(AuroraMAELoss)
        else:
            raise Exception(f"Unsupported eval metric: {metric}")

        err_agg_list.append(
            prepare_each_lead_time_agg(
                rollout_step = args.rollout_step,
                lead_time = args.lead_time,
                surface_variables = args.surface_variables,
                upper_variables = args.upper_variables,
                levels = args.levels,
                err_type = metric,
            )
        )

    if args.save_rollout_step is not None:
        gen_result_folder = Path(args.gen_result_folder)
        gen_result_folder.mkdir(parents = True, exist_ok = True)
        logger.info(f"Saving lead time outputs to {args.gen_result_folder}")

    evaluate(
        args,
        model,
        dataloader,
        criterion_list,
        err_agg_list,
        device,
    )

    for metric, err_agg in zip(args.eval_metric, err_agg_list):
        if args.csv_output_folder is not None:
            csv_folder = Path(args.csv_output_folder)
            csv_folder.mkdir(parents = True, exist_ok = True)
            csv_output_path = csv_folder / f"{metric}.csv"
            logger.info(f"Exporting results to CSV: {csv_output_path}")
            export_agg_to_csv(args, err_agg, out_path = csv_output_path)

if __name__ == "__main__":
    main()