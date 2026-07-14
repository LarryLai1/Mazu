#!/usr/bin/env python
# coding=utf-8
"""
Multi-GPU (DDP) version of AuroraSmallTW_gen_eval_pipeline_custom_rollout_prefetch.py

Launch with:
    torchrun --nproc_per_node=N AuroraSmallTW_gen_eval_pipeline_custom_rollout_prefetch_multigpu.py ...

Design:
- Each rank loads the same model weights independently (no DDP wrapper needed for
  pure eval — avoids gradient bucket overhead and keeps per-rank numerics identical
  to the single-GPU path).
- The input dataset is sharded across ranks via DistributedSampler(shuffle=False).
  Each rank's per-batch prefetch target dataset is keyed on that rank's local dates,
  so the prefetch pipeline works unchanged.
- Error aggregators' (error_sum, count) pairs are all-reduced (SUM) at the end,
  so the global mean is bit-exact w.r.t. the single-GPU run regardless of how
  samples were split (assuming drop_last=False).
- .nc outputs are written by each rank to unique filenames (keyed on base_time +
  rollout_step), so no collisions.
- CSV export is performed only on rank 0.
"""

import argparse
import contextlib
import os
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """Initialize torch.distributed from torchrun env vars.

    Returns (local_rank, device). Falls back to single-process when the env
    vars are absent, so the script still runs as a regular python invocation.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            device = torch.device(f"cuda:{local_rank}")
        else:
            backend = "gloo"
            device = torch.device("cpu")
        dist.init_process_group(backend=backend)
        logger.info(
            f"[DDP] rank={dist.get_rank()} "
            f"world_size={dist.get_world_size()} "
            f"local_rank={local_rank} device={device}"
        )
        return local_rank, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[DDP] Not launched with torchrun; running single-process on {device}.")
    return 0, device


def cleanup_distributed():
    if is_dist_available_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description = "Aurora Evaluation Script (Multi-GPU DDP) - Prefetch Target Loading.")
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
        if is_main_process():
            logger.info("Loading pretrained weights provided by Microsoft Aurora...")
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt", strict = False)
    elif args.checkpoint_path:
        if is_main_process():
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


def all_reduce_err_agg_list(err_agg_list, device):
    """SUM-reduce every aggregator's (error_sum, count) across ranks.

    After this call, each rank's aggregators hold the global sums, so
    `.mean()` returns the same global value on every rank.
    """
    if not is_dist_available_and_initialized():
        return

    flat_sums = []
    flat_counts = []
    references = []

    for err_agg in err_agg_list:
        for t in err_agg:
            for v, agg in err_agg[t]["surf_vars"].items():
                flat_sums.append(float(agg.error_sum))
                flat_counts.append(int(agg.count))
                references.append(agg)
            for v in err_agg[t]["atmos_vars"]:
                for lev, agg in err_agg[t]["atmos_vars"][v].items():
                    flat_sums.append(float(agg.error_sum))
                    flat_counts.append(int(agg.count))
                    references.append(agg)

    if not references:
        return

    sums_tensor = torch.tensor(flat_sums, dtype=torch.float64, device=device)
    counts_tensor = torch.tensor(flat_counts, dtype=torch.int64, device=device)

    dist.all_reduce(sums_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)

    sums_reduced = sums_tensor.cpu().tolist()
    counts_reduced = counts_tensor.cpu().tolist()

    for agg, s, c in zip(references, sums_reduced, counts_reduced):
        agg.error_sum = s
        agg.count = int(c)


def evaluate(args, model, dataloader, criterion_list, err_agg_list, device):
    model.eval()
    latitudes, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    pbar_disabled = not is_main_process()

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating", disable=pbar_disabled):
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
                batch_size=None,
                num_workers=1,
                shuffle=False,
            )

            with context_manager:
                generator = rollout_with_gpu(model, _input, steps=args.rollout_step)

                for step_index, (_pred, _label_data) in enumerate(
                    zip(generator, target_loader)
                ):
                    t = step_index + 1

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

                    for (criterion, err_agg) in zip(criterion_list, err_agg_list):
                        loss_dict = criterion(_pred, _label)
                        log_weather_variable_error_with_lead_time(
                            loss_dict,
                            t * args.lead_time,
                            err_agg,
                        )

                    if args.save_rollout_step and t in args.save_rollout_step:
                        AuroraBatch_2_nc_files(
                            batch=_pred,
                            args=args,
                        )

    # --- Memory reporting (per-rank) ---
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        logger.info(f"[Memory][rank={get_rank()}] Peak GPU memory allocated: {peak_mem_gb:.2f} GB")


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

    local_rank, device = setup_distributed()

    if is_main_process():
        logger.info(f"Running multi-GPU evaluation (prefetch target loading). world_size={get_world_size()}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    model = create_model(args, device)
    dataset = create_dataset(args)

    sampler = None
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
            drop_last=False,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

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
        if is_main_process():
            gen_result_folder.mkdir(parents = True, exist_ok = True)
            logger.info(f"Saving lead time outputs to {args.gen_result_folder}")
        if is_dist_available_and_initialized():
            dist.barrier()

    evaluate(
        args,
        model,
        dataloader,
        criterion_list,
        err_agg_list,
        device,
    )

    # --- Aggregate errors across ranks so every rank holds the global result ---
    all_reduce_err_agg_list(err_agg_list, device)

    if is_main_process():
        for metric, err_agg in zip(args.eval_metric, err_agg_list):
            if args.csv_output_folder is not None:
                csv_folder = Path(args.csv_output_folder)
                csv_folder.mkdir(parents = True, exist_ok = True)
                csv_output_path = csv_folder / f"{metric}.csv"
                logger.info(f"Exporting results to CSV: {csv_output_path}")
                export_agg_to_csv(args, err_agg, out_path = csv_output_path)

    cleanup_distributed()


if __name__ == "__main__":
    main()
