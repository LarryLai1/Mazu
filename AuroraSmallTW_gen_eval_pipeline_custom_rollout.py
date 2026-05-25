#!/usr/bin/env python
# coding=utf-8

import argparse
import contextlib
import dataclasses
import json
import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import random
import numpy as np
import sys
import os

from aurora import Batch, Metadata
# from aurora import rollout
# from utils.custom_rollout import rollout_with_gpu
from aurora.model.aurora import AuroraSmall
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
from datasets.BoundaryConditionDataset import BoundaryConditionDataset
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
    parser = argparse.ArgumentParser(description = "Aurora Evaluation Script (Single GPU).")
    parser.add_argument('--data_root_dir', type = str, required = True)
    parser.add_argument('--boundary_root_dir', type = str, default = None)
    parser.add_argument('--boundary_width', type = int, default = 0)
    parser.add_argument(
        '--boundary_mode',
        type = str,
        default = "inject-inside",
        choices = ["inject-inside", "pad-outside"],
    )
    parser.add_argument(
        "--boundary_prediction_timedeltas",
        type = int,
        nargs = "+",
        default = [0, 6, 12],
    )
    parser.add_argument(
        "--boundary_pooling",
        type = str,
        default = "no",
        choices = ["no", "yes"],
    )
    parser.add_argument(
        "--boundary_use_cache",
        action = "store_true",
        help = "Preload all boundary files into memory and serve boundary data from cache.",
    )
    parser.add_argument(
        "--gpu_cache",
        action = "store_true",
        help = "Enable GPU boundary cache and preload boundary files into memory.",
    )
    parser.add_argument("--use_pretrained_weight", action = "store_true")
    # parser.add_argument('--checkpoint_path', type = str, required = True)
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
    parser.add_argument('--mp_world_size', type = int, default = 1)
    parser.add_argument(
        '--gpus',
        type = str,
        default = None,
        help = 'Comma-separated list of GPU ids to use, e.g. "0,1,2". If provided, spawns one process per GPU and binds each process to the corresponding GPU.',
    )

    return parser.parse_args()

def _resolve_mp_world_size(args):
    # If user explicitly provided GPU ids, use that
    if getattr(args, 'gpus', None):
        gpus = [x for x in args.gpus.split(",") if x.strip() != ""]
        return max(1, len(gpus))
    if args.mp_world_size > 1:
        return args.mp_world_size
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        n = len([x for x in visible.split(",") if x.strip() != ""])
        return max(1, n)
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1

def _manual_split_dataset(dataset, rank: int, world_size: int):
    if world_size <= 1:
        return dataset
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)

def _build_metric_lists(args):
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
    return criterion_list, err_agg_list

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
    logger.info("Creating Aurora dataset...")
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
        sample_stride_hours=args.timestep_hours,  # Align dataset sampling with model rollout timestep
    )
    return ds

def create_boundary_dataset(args):
    if not args.boundary_root_dir:
        return None
    logger.info("Creating Boundary Condition dataset...")
    boundary_ds_width = args.boundary_width if args.boundary_mode == "pad-outside" else 0
    return BoundaryConditionDataset(
        boundary_root_dir = args.boundary_root_dir,
        start_date_hour = args.start_date_hour,
        end_date_hour = args.end_date_hour,
        upper_variables = args.upper_variables,
        surface_variables = args.surface_variables,
        levels = args.levels,
        latitude = args.latitude,
        longitude = args.longitude,
        boundary_width = boundary_ds_width,
        prediction_timedeltas = args.boundary_prediction_timedeltas,
        enable_pooling = (args.boundary_pooling == "yes"),
        use_cache = args.boundary_use_cache,
    )

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

def _build_boundary_batch(
    boundary_dataset,
    base_times,
    target_times,
):
    surf_vars = {}
    atmos_vars = {}

    for base_time, target_time in zip(base_times, target_times):
        data = boundary_dataset.get_boundary_at_time(base_time, target_time)
        for var_name, tensor in data["surf_vars"].items():
            surf_vars.setdefault(var_name, []).append(tensor)
        for var_name, tensor in data["atmos_vars"].items():
            atmos_vars.setdefault(var_name, []).append(tensor)

    surf_vars = {k: torch.stack(v, dim = 0) for k, v in surf_vars.items()}
    atmos_vars = {k: torch.stack(v, dim = 0) for k, v in atmos_vars.items()}
    return {"surf_vars": surf_vars, "atmos_vars": atmos_vars}

def _get_boundary_source_on_device(
    boundary_dataset,
    base_time,
    gpu_cache,
    device,
):
    if base_time in gpu_cache:
        return gpu_cache[base_time]
    source = boundary_dataset.get_boundary_source(base_time)
    gpu_source = {
        "time_values": source["time_values"],
        "surf_vars": {},
        "atmos_vars": {},
    }
    for var_name, tensor in source["surf_vars"].items():
        gpu_source["surf_vars"][var_name] = tensor.to(device)
    for var_name, tensor in source["atmos_vars"].items():
        gpu_source["atmos_vars"][var_name] = tensor.to(device)
    gpu_cache[base_time] = gpu_source
    return gpu_source

def _build_boundary_batch_from_gpu_cache(
    boundary_dataset,
    gpu_cache,
    base_times,
    target_times,
):
    surf_vars = {}
    atmos_vars = {}

    for base_time, target_time in zip(base_times, target_times):
        source = gpu_cache[base_time]
        time_values = source["time_values"]
        for var_name, tensor in source["surf_vars"].items():
            selected = boundary_dataset._select_from_source(time_values, tensor, target_time)
            surf_vars.setdefault(var_name, []).append(selected)
        for var_name, tensor in source["atmos_vars"].items():
            selected = boundary_dataset._select_from_source(time_values, tensor, target_time)
            atmos_vars.setdefault(var_name, []).append(selected)

    surf_vars = {k: torch.stack(v, dim = 0) for k, v in surf_vars.items()}
    atmos_vars = {k: torch.stack(v, dim = 0) for k, v in atmos_vars.items()}
    return {"surf_vars": surf_vars, "atmos_vars": atmos_vars}

def _is_increasing(coord: torch.Tensor) -> bool:
    if coord.numel() < 2:
        return False
    return coord[0].item() < coord[-1].item()

def _align_boundary_batch(boundary_batch, flip_lat: bool, flip_lon: bool):
    if not (flip_lat or flip_lon):
        return boundary_batch
    lat_dim = -2
    lon_dim = -1
    for var_dict in (boundary_batch["surf_vars"], boundary_batch["atmos_vars"]):
        for k, tensor in var_dict.items():
            if flip_lat:
                tensor = torch.flip(tensor, dims = (lat_dim,))
            if flip_lon:
                tensor = torch.flip(tensor, dims = (lon_dim,))
            var_dict[k] = tensor
    return boundary_batch

def _center_crop_boundary(tensor, boundary_width):
    if boundary_width <= 0:
        return tensor
    return tensor[..., boundary_width:-boundary_width, boundary_width:-boundary_width]

def _pad_interior_with_boundary(interior_tensor, boundary_tensor, boundary_width):
    if boundary_width <= 0:
        return interior_tensor
    h_int, w_int = interior_tensor.shape[-2:]
    h_b, w_b = boundary_tensor.shape[-2:]
    if h_b != h_int + 2 * boundary_width or w_b != w_int + 2 * boundary_width:
        raise ValueError("Boundary tensor shape does not match interior tensor + boundary_width.")
    padded = boundary_tensor.clone()
    padded[..., boundary_width:-boundary_width, boundary_width:-boundary_width] = interior_tensor
    return padded

def _pad_static_vars(static_vars, boundary_width):
    if boundary_width <= 0:
        return static_vars
    padded = {}
    for var_name, tensor in static_vars.items():
        if tensor.dim() == 2:
            padded_tensor = F.pad(
                tensor.unsqueeze(0).unsqueeze(0),
                (boundary_width, boundary_width, boundary_width, boundary_width),
                mode = "replicate",
            ).squeeze(0).squeeze(0)
        elif tensor.dim() == 3:
            padded_tensor = F.pad(
                tensor.unsqueeze(0),
                (boundary_width, boundary_width, boundary_width, boundary_width),
                mode = "replicate",
            ).squeeze(0)
        else:
            padded_tensor = F.pad(
                tensor,
                (boundary_width, boundary_width, boundary_width, boundary_width),
                mode = "replicate",
            )
        padded[var_name] = padded_tensor
    return padded

def _replace_boundary_inside(pred_tensor, boundary_tensor, boundary_width):
    if boundary_width <= 0:
        return pred_tensor
    if pred_tensor.dim() == boundary_tensor.dim() + 1:
        boundary_tensor = boundary_tensor.unsqueeze(1)
    if pred_tensor.dim() != boundary_tensor.dim():
        raise ValueError("Boundary tensor rank does not match prediction tensor.")
    updated = pred_tensor.clone()
    bw = boundary_width
    updated[..., :bw, :] = boundary_tensor[..., :bw, :]
    updated[..., -bw:, :] = boundary_tensor[..., -bw:, :]
    updated[..., :, :bw] = boundary_tensor[..., :, :bw]
    updated[..., :, -bw:] = boundary_tensor[..., :, -bw:]
    return updated

def _slice_interior(tensor, boundary_width):
    if boundary_width <= 0:
        return tensor
    return tensor[..., boundary_width:-boundary_width, boundary_width:-boundary_width]

def _prepare_batch_for_rollout(model, batch):
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    return batch.to(p.device)

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
        # output_file_name = f"{(batch.metadata.time[i] - pd.Timedelta(hours = hours + args.lead_time - 1)).strftime('%Y%m%d_%H%M%S')}+{hours + args.lead_time - 1}hr.nc"
        output_file_name = f"{(batch.metadata.time[i] - pd.Timedelta(hours = rs * args.lead_time)).strftime('%Y%m%d_%H%M%S')}+{rs * args.lead_time}hr.nc"
        
        gen_result_folder = Path(args.gen_result_folder)
        output_path = gen_result_folder / output_file_name

        ds.to_netcdf( output_path )

def evaluate(
    args,
    model,
    dataloader,
    criterion_list,
    err_agg_list,
    device,
    boundary_dataset = None,
    rank = 0,
    metadata_dataset = None,
):
    model.eval()
    ds_ref = metadata_dataset if metadata_dataset is not None else dataloader.dataset
    latitudes, longitude = ds_ref.get_latitude_longitude()
    levels = ds_ref.get_levels()
    static_data = ds_ref.get_static_vars_ds()

    boundary_enabled = boundary_dataset is not None and args.boundary_width > 0
    gpu_boundary_cache = {}

    if boundary_enabled:
        boundary_latitudes, boundary_longitude = boundary_dataset.get_latitude_longitude()
        flip_lat = _is_increasing(boundary_latitudes) != _is_increasing(latitudes)
        flip_lon = _is_increasing(boundary_longitude) != _is_increasing(longitude)
        if flip_lat:
            boundary_latitudes = torch.flip(boundary_latitudes, dims = (0,))
        if flip_lon:
            boundary_longitude = torch.flip(boundary_longitude, dims = (0,))
    else:
        flip_lat = False
        flip_lon = False

    # Optimization: Use inference_mode to reduce memory for gradients
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc = f"Evaluating(rank={rank})", disable = (rank != 0)):
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

            # Pre-slice labels (this is okay to keep in list if it fits in memory, 
            # usually labels are smaller than the computation graph)
            _label_list = slice_timeaxis(labels)

            batch_times = tuple(map(lambda d: pd.Timestamp(d), dates))
            base_times = None
            if boundary_enabled:
                base_times = tuple(boundary_dataset.get_base_time(t) for t in batch_times)

            # Prefetch boundary data for all autoregressive steps and transfer to device
            prefetched_boundary = None
            if boundary_enabled:
                prefetched_boundary = {}
                if args.gpu_cache:
                    for base_time in set(base_times):
                        _get_boundary_source_on_device(
                            boundary_dataset,
                            base_time,
                            gpu_boundary_cache,
                            device,
                        )
                    # include initial step k=0 (current time) and future steps 1..rollout_step
                    for k in range(0, args.rollout_step + 1):
                        target_times_k = tuple(
                            pd.Timestamp(d) + pd.Timedelta(hours = k * args.lead_time)
                            for d in dates
                        )
                        b_step = _build_boundary_batch_from_gpu_cache(
                            boundary_dataset,
                            gpu_boundary_cache,
                            base_times,
                            target_times_k,
                        )
                        b_step = _align_boundary_batch(b_step, flip_lat, flip_lon)
                        prefetched_boundary[k] = b_step
                else:
                    # include initial step k=0 (current time) and future steps 1..rollout_step
                    for k in range(0, args.rollout_step + 1):
                        target_times_k = tuple(
                            pd.Timestamp(d) + pd.Timedelta(hours = k * args.lead_time)
                            for d in dates
                        )
                        b_step = _build_boundary_batch(boundary_dataset, base_times, target_times_k)
                        b_step = _align_boundary_batch(b_step, flip_lat, flip_lon)
                        # Move to device (keep as float/dtype default) to avoid host->device during rollout
                        for var_name, tensor in b_step["surf_vars"].items():
                            b_step["surf_vars"][var_name] = tensor.to(device)
                        for var_name, tensor in b_step["atmos_vars"].items():
                            b_step["atmos_vars"][var_name] = tensor.to(device)
                        prefetched_boundary[k] = b_step

            if boundary_enabled and args.boundary_mode == "pad-outside":
                # use prefetched boundary for initial pad-outside
                boundary_init = prefetched_boundary[0] if prefetched_boundary is not None else _build_boundary_batch(
                    boundary_dataset,
                    base_times,
                    batch_times,
                )

                padded_inputs = {"surf_vars": {}, "atmos_vars": {}}
                for var_name, tensor in inputs["surf_vars"].items():
                    boundary_tensor = boundary_init["surf_vars"][var_name]
                    boundary_tensor = boundary_tensor.unsqueeze(1).expand(-1, tensor.shape[1], -1, -1)
                    padded_inputs["surf_vars"][var_name] = _pad_interior_with_boundary(
                        tensor,
                        boundary_tensor,
                        args.boundary_width,
                    )
                for var_name, tensor in inputs["atmos_vars"].items():
                    boundary_tensor = boundary_init["atmos_vars"][var_name]
                    boundary_tensor = boundary_tensor.unsqueeze(1).expand(-1, tensor.shape[1], -1, -1, -1)
                    padded_inputs["atmos_vars"][var_name] = _pad_interior_with_boundary(
                        tensor,
                        boundary_tensor,
                        args.boundary_width,
                    )
                inputs = padded_inputs
                static_data["static_vars"] = _pad_static_vars(static_data["static_vars"], args.boundary_width)

            # ================== 新增修改：針對 inject-inside 模式進行初始輸入覆寫 ==================
            elif boundary_enabled and args.boundary_mode == "inject-inside":
                # 取得初始時間點的真實邊界（從 prefetched 取得，已在 device）
                boundary_init = prefetched_boundary[0] if prefetched_boundary is not None else _build_boundary_batch(
                    boundary_dataset,
                    base_times,
                    batch_times,
                )
                boundary_init = _align_boundary_batch(boundary_init, flip_lat, flip_lon)

                # boundary_init 已被移到 device during prefetch; use directly
                boundary_inside = {
                    "surf_vars": {
                        k: v
                        for k, v in boundary_init["surf_vars"].items()
                    },
                    "atmos_vars": {
                        k: v
                        for k, v in boundary_init["atmos_vars"].items()
                    },
                }
                
                # 直接覆寫初始 inputs 裡面所有歷史時間步的四周邊界
                for var_name, tensor in inputs["surf_vars"].items():
                    # 配合 inputs 的維度 (Batch, Time, Lat, Lon)，將邊界張量增加 Time 維度並對齊
                    b_tensor = boundary_inside["surf_vars"][var_name].unsqueeze(1).expand(-1, tensor.shape[1], -1, -1)
                    inputs["surf_vars"][var_name] = _replace_boundary_inside(tensor, b_tensor, args.boundary_width)
                    
                for var_name, tensor in inputs["atmos_vars"].items():
                    # 配合 inputs 的維度 (Batch, Time, Level, Lat, Lon)，將邊界張量增加 Time 維度並對齊
                    b_tensor = boundary_inside["atmos_vars"][var_name].unsqueeze(1).expand(-1, tensor.shape[1], -1, -1, -1)
                    inputs["atmos_vars"][var_name] = _replace_boundary_inside(tensor, b_tensor, args.boundary_width)
            # ===================================================================================

            if boundary_enabled and args.boundary_mode == "pad-outside":
                metadata_lat = boundary_latitudes
                metadata_lon = boundary_longitude
            else:
                metadata_lat = latitudes
                metadata_lon = longitude

            _input = Batch(
                surf_vars = inputs["surf_vars"],
                atmos_vars = inputs["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = metadata_lat,
                    lon = metadata_lon,
                    time = batch_times,
                    atmos_levels = levels,
                ),
            )
            # logger.info("Input shape: %s", _input.atmos_vars["t"].shape)
            # # flush stdout
            # sys.stdout.flush()
            

            assert model.training is False

            # --- Setup Mixed Precision ---
            use_amp = (args.mixed_precision in ("fp16", "bf16")) and (device.type == "cuda")
            dtype = torch.float32  # Default
            if use_amp:
                if args.mixed_precision == "fp16":
                    dtype = torch.float16
                elif args.mixed_precision == "bf16":
                    dtype = torch.bfloat16

            # --- THE OPTIMIZED LOOP ---
            # We create a dummy context manager if AMP is not used
            context_manager = torch.amp.autocast(dtype = dtype) if use_amp else contextlib.nullcontext()
            
            with context_manager:
                rollout_batch = _prepare_batch_for_rollout(model, _input)

                for step_index in range(args.rollout_step):
                    # step_index starts at 0, so lead time t is step_index + 1
                    t = step_index + 1

                    _pred = model(rollout_batch)

                    # 1. Get the corresponding label for this specific step
                    _label_data = _label_list[step_index]

                    _label = Batch(
                        surf_vars = _label_data["surf_vars"],
                        atmos_vars = _label_data["atmos_vars"],
                        static_vars = static_data["static_vars"],
                        metadata = Metadata(
                            lat = latitudes,
                            lon = longitude,
                            time = tuple(
                                map(
                                    lambda d: pd.Timestamp(d) + pd.Timedelta(hours = t * args.lead_time),
                                    dates,
                                )
                            ),
                            atmos_levels = levels,
                        ),
                    )

                    if boundary_enabled:
                        target_times = tuple(
                            pd.Timestamp(d) + pd.Timedelta(hours = t * args.lead_time) for d in dates
                        )
                        # print(dates)
                        # print(target_times)
                        # use prefetched per-step boundary (already on device); cast to pred dtype as needed
                        boundary_step = prefetched_boundary[t] if prefetched_boundary is not None else _build_boundary_batch(
                            boundary_dataset,
                            base_times,
                            target_times,
                        )
                        for var_name, tensor in boundary_step["surf_vars"].items():
                            boundary_step["surf_vars"][var_name] = tensor.to(device, dtype = _pred.surf_vars[var_name].dtype)
                        for var_name, tensor in boundary_step["atmos_vars"].items():
                            boundary_step["atmos_vars"][var_name] = tensor.to(device, dtype = _pred.atmos_vars[var_name].dtype)
                        boundary_step = _align_boundary_batch(boundary_step, flip_lat, flip_lon)
                    else:
                        boundary_step = None

                    # 2. Calculate Loss immediately
                    if boundary_enabled and args.boundary_mode == "pad-outside":
                        pred_interior = Batch(
                            surf_vars = {
                                k: _slice_interior(v, args.boundary_width)
                                for k, v in _pred.surf_vars.items()
                            },
                            atmos_vars = {
                                k: _slice_interior(v, args.boundary_width)
                                for k, v in _pred.atmos_vars.items()
                            },
                            static_vars = static_data["static_vars"],
                            metadata = Metadata(
                                lat = latitudes,
                                lon = longitude,
                                time = _label.metadata.time,
                                atmos_levels = levels,
                            ),
                        )
                        loss_pred = pred_interior
                    else:
                        loss_pred = _pred

                    for (criterion, err_agg) in zip(criterion_list, err_agg_list):
                        loss_dict = criterion(loss_pred, _label)
                        log_weather_variable_error_with_lead_time(
                            loss_dict,
                            t * args.lead_time,
                            err_agg,
                        )

                    # 3. Save to disk if needed (then discard from memory)
                    if args.save_rollout_step and t in args.save_rollout_step:
                        AuroraBatch_2_nc_files(
                            batch = _pred,
                            args = args,
                        )

                    if boundary_enabled and boundary_step is not None:
                        if args.boundary_mode == "inject-inside":
                            boundary_inside = {
                                "surf_vars": {
                                    k: v
                                    for k, v in boundary_step["surf_vars"].items()
                                },
                                "atmos_vars": {
                                    k: v
                                    for k, v in boundary_step["atmos_vars"].items()
                                },
                            }
                            pred_for_next = dataclasses.replace(
                                _pred,
                                surf_vars = {
                                    k: _replace_boundary_inside(v, boundary_inside["surf_vars"][k], args.boundary_width)
                                    for k, v in _pred.surf_vars.items()
                                },
                                atmos_vars = {
                                    k: _replace_boundary_inside(v, boundary_inside["atmos_vars"][k], args.boundary_width)
                                    for k, v in _pred.atmos_vars.items()
                                },
                            )
                        else:
                            pred_for_next = dataclasses.replace(
                                _pred,
                                surf_vars = {
                                    k: _replace_boundary_inside(v, boundary_step["surf_vars"][k], args.boundary_width)
                                    for k, v in _pred.surf_vars.items()
                                },
                                atmos_vars = {
                                    k: _replace_boundary_inside(v, boundary_step["atmos_vars"][k], args.boundary_width)
                                    for k, v in _pred.atmos_vars.items()
                                },
                            )
                    else:
                        pred_for_next = _pred

                    rollout_batch = dataclasses.replace(
                        pred_for_next,
                        surf_vars = {
                            k: torch.cat([rollout_batch.surf_vars[k][:, 1:], v], dim = 1)
                            for k, v in pred_for_next.surf_vars.items()
                        },
                        atmos_vars = {
                            k: torch.cat([rollout_batch.atmos_vars[k][:, 1:], v], dim = 1)
                            for k, v in pred_for_next.atmos_vars.items()
                        },
                    )

def export_agg_to_csv(
        args,
        lead_time_err_agg,
        out_path,
    ):

    lead_times = sorted(lead_time_err_agg.keys())
    # lead_time_labels = [f"{t + args.lead_time - 1}h" for t in lead_times]
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

def _mp_worker_entry(rank, world_size, args):
    set_seed(args.seed + rank)
    if torch.cuda.is_available():
        # If user provided explicit GPU list, use mapping; otherwise fall back to rank->device
        gpu_list = getattr(args, 'gpu_list', None)
        if gpu_list is not None:
            gpu_id = int(gpu_list[rank])
        else:
            gpu_id = int(rank)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    model = create_model(args, device)
    full_dataset = create_dataset(args)
    eval_dataset = _manual_split_dataset(full_dataset, rank, world_size)
    boundary_dataset = create_boundary_dataset(args)
    dataloader = DataLoader(
        eval_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True,
    )
    criterion_list, err_agg_list = _build_metric_lists(args)

    evaluate(
        args,
        model,
        dataloader,
        criterion_list,
        err_agg_list,
        device,
        boundary_dataset = boundary_dataset,
        rank = rank,
        metadata_dataset = full_dataset,
    )

    tmp_root = Path(args.csv_output_folder) if args.csv_output_folder is not None else Path(args.gen_result_folder)
    tmp_root.mkdir(parents = True, exist_ok = True)
    out_path = tmp_root / f".mp_rank_{rank}_metrics.json"
    payload = {
        "rank": rank,
        "metrics": {
            metric: _err_agg_to_state(err_agg)
            for metric, err_agg in zip(args.eval_metric, err_agg_list)
        },
    }
    with out_path.open("w", encoding = "utf-8") as f:
        json.dump(payload, f)

def main():
    args = parse_args()
    # print(args.csv_output_folder)
    # If user passed --gpus, parse into list and attach to args for worker mapping
    gpu_list = None
    if getattr(args, 'gpus', None):
        gpu_list = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
        args.gpu_list = gpu_list
    else:
        args.gpu_list = None

    world_size = _resolve_mp_world_size(args)

    if args.save_rollout_step is not None:
        gen_result_folder = Path(args.gen_result_folder)
        gen_result_folder.mkdir(parents = True, exist_ok = True)
        logger.info(f"Saving lead time outputs to {args.gen_result_folder}")

    if args.csv_output_folder is not None:
        Path(args.csv_output_folder).mkdir(parents = True, exist_ok = True)

    if world_size <= 1:
        set_seed(args.seed)
        logger.info("Running single-process evaluation.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(args, device)
        dataset = create_dataset(args)
        boundary_dataset = create_boundary_dataset(args)
        dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = True)
        criterion_list, err_agg_list = _build_metric_lists(args)

        evaluate(
            args,
            model,
            dataloader,
            criterion_list,
            err_agg_list,
            device,
            boundary_dataset = boundary_dataset,
            rank = 0,
            metadata_dataset = dataset,
        )

        for metric, err_agg in zip(args.eval_metric, err_agg_list):
            if args.csv_output_folder is not None:
                csv_folder = Path(args.csv_output_folder)
                csv_folder.mkdir(parents = True, exist_ok = True)
                csv_output_path = csv_folder / f"{metric}.csv"
                logger.info(f"Exporting results to CSV: {csv_output_path}")
                export_agg_to_csv(args, err_agg, out_path = csv_output_path)
        return

    logger.info("Running multiprocessing evaluation with world_size=%s", world_size)

    mp_ctx = mp.get_context("spawn")

    processes = []
    for rank in range(world_size):
        p = mp_ctx.Process(target = _mp_worker_entry, args = (rank, world_size, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}.")

    _, merged_err_agg_list = _build_metric_lists(args)
    tmp_root = Path(args.csv_output_folder) if args.csv_output_folder is not None else Path(args.gen_result_folder)
    for rank in range(world_size):
        p = tmp_root / f".mp_rank_{rank}_metrics.json"
        with p.open("r", encoding = "utf-8") as f:
            payload = json.load(f)
        for metric_idx, metric in enumerate(args.eval_metric):
            _merge_state_into_err_agg(merged_err_agg_list[metric_idx], payload["metrics"][metric])
        p.unlink(missing_ok = True)

    for metric, err_agg in zip(args.eval_metric, merged_err_agg_list):
        if args.csv_output_folder is not None:
            csv_folder = Path(args.csv_output_folder)
            csv_folder.mkdir(parents = True, exist_ok = True)
            csv_output_path = csv_folder / f"{metric}.csv"
            logger.info(f"Exporting results to CSV: {csv_output_path}")
            export_agg_to_csv(args, err_agg, out_path = csv_output_path)

if __name__ == "__main__":
    main()
