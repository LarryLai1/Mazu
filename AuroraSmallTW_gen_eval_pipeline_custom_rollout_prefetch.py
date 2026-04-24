#!/usr/bin/env python
# coding=utf-8
"""
Prefetch version of AuroraSmallTW_gen_eval_pipeline_custom_rollout.py

Key difference from the lazy version: targets are loaded via an IterableDataset
wrapped in a DataLoader with workers, so disk I/O for the next target overlaps
with GPU compute for the current step.

Memory usage stays low (only ~2 targets in RAM at a time: current + prefetched),
while speed approaches the eager version.
"""

import argparse
import contextlib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
import numpy as np

from aurora import Batch, Metadata
from utils.custom_rollout import rollout_with_gpu
from aurora.model.aurora import AuroraSmall
from datasets.ERA5TWDatasetforAurora_lazy import ERA5TWDatasetforAuroraLazy
from datasets.ERA5TWTargetIterableDataset import ERA5TWTargetIterableDataset
from utils.metrics import AuroraMAELoss, AuroraMSELoss
from utils.metrics import prepare_each_lead_time_agg

from pathlib import Path

import xarray as xr
from safetensors.torch import load_file

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description = "Aurora Evaluation Script (Single GPU) - Prefetch Target Loading.")
    parser.add_argument('--data_root_dir', type = str, required = True)
    parser.add_argument("--use_pretrained_weight", action = "store_true")
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
    parser.add_argument('--use_lora', action = 'store_true')
    parser.add_argument('--bf16_mode', action = 'store_true')
    parser.add_argument('--stabilise_level_agg', action = 'store_true')

    parser.add_argument("--gen_result_folder", type = str, default = './gen_result',)
    parser.add_argument("--save_rollout_step", type = int, nargs = "+", default = None)
    parser.add_argument("--eval_metric", type = str, nargs = "+", default = ["MSE"], choices = ["MSE", "MAE"])

    parser.add_argument("--csv_output_folder", type = str, default = "./errs")
    parser.add_argument('--mixed_precision', type = str, default = None, choices = ["no", "fp16", "bf16"])

    parser.add_argument('--target_loading_workers', type = int, default = 4,
                        help = "Number of workers for target prefetch DataLoader.")
    # parser.add_argument('--prefetch_factor', type = int, default = 2,
    #                     help = "Number of targets to prefetch per worker.")

    return parser.parse_args()

def load_Aurora_weight(
    Aurora_model,
    checkpoint_path,
):
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
        Aurora_model.load_state_dict(state_dict)

def create_model(args, device):
    model = AuroraSmall(
        use_lora = args.use_lora,
        bf16_mode = args.bf16_mode,
        timestep = pd.Timedelta(hours = args.timestep_hours),
        stabilise_level_agg = args.stabilise_level_agg,
    )
    if args.use_pretrained_weight:
        logger.info("Loading pretrained weights provided by Microsoft Aurora...")
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt", strict = False)
    elif args.checkpoint_path:
        logger.info(f"Loading checkpoint: {args.checkpoint_path}")

        load_Aurora_weight(
            model,
            args.checkpoint_path,
        )

    model.to(device)
    model.eval()
    return model

def create_dataset(args):
    ds = ERA5TWDatasetforAuroraLazy(
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

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, dates = batch

            # --- Data moving to device ---
            for _k_var_type in inputs:
                for _k_var in inputs[_k_var_type]:
                    inputs[_k_var_type][_k_var] = inputs[_k_var_type][_k_var].to(device)
            if isinstance(static_data["static_vars"], torch.Tensor):
                static_data["static_vars"] = static_data["static_vars"].to(device)

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

            assert model.training is False

            # --- Setup Mixed Precision ---
            use_amp = (args.mixed_precision in ("fp16", "bf16")) and (device.type == "cuda")
            dtype = torch.float32
            if use_amp:
                if args.mixed_precision == "fp16":
                    dtype = torch.float16
                elif args.mixed_precision == "bf16":
                    dtype = torch.bfloat16

            context_manager = torch.cuda.amp.autocast(dtype = dtype) if use_amp else contextlib.nullcontext()

            # --- Create prefetch target DataLoader for this batch ---
            target_dataset = ERA5TWTargetIterableDataset(
                data_root_dir=args.data_root_dir,
                base_datetime_strs=dates,
                rollout_steps=args.rollout_step,
                lead_time=args.lead_time,
                upper_variables=args.upper_variables,
                surface_variables=args.surface_variables,
                levels=args.levels,
                latitude=args.latitude,
                longitude=args.longitude,
            )
            target_loader = DataLoader(
                target_dataset,
                batch_size=None,    # already batched inside the dataset
                num_workers=1,
                shuffle=False,
                # prefetch_factor=args.prefetch_factor,
                # persistent_workers=False,
            )

            with context_manager:
                generator = rollout_with_gpu(model, _input, steps=args.rollout_step)

                for step_index, (_pred, _label_data) in enumerate(
                    zip(generator, target_loader)
                ):
                    t = step_index + 1

                    # Move target to device
                    for _k_var in _label_data["surf_vars"]:
                        _label_data["surf_vars"][_k_var] = _label_data["surf_vars"][_k_var].to(device)
                    for _k_var in _label_data["atmos_vars"]:
                        _label_data["atmos_vars"][_k_var] = _label_data["atmos_vars"][_k_var].to(device)

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

                    # Calculate Loss
                    for (criterion, err_agg) in zip(criterion_list, err_agg_list):
                        loss_dict = criterion(_pred, _label)
                        log_weather_variable_error_with_lead_time(
                            loss_dict,
                            t * args.lead_time,
                            err_agg,
                        )

                    # Save to disk if needed
                    if args.save_rollout_step and t in args.save_rollout_step:
                        AuroraBatch_2_nc_files(
                            batch=_pred,
                            args=args,
                        )

                    # _pred and _label_data go out of scope -> can be GC'd

    # --- Memory reporting ---
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        logger.info(f"[Memory] Peak GPU memory allocated: {peak_mem_gb:.2f} GB")

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
    logger.info("Running single-GPU evaluation (prefetch target loading).")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Reset peak memory counter for accurate measurement
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

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
