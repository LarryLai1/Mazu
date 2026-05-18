#!/usr/bin/env python
# coding=utf-8

import argparse
import contextlib
import dataclasses
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
import numpy as np

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

def create_boundary_dataset(args):
    if not args.boundary_root_dir:
        return None
    return BoundaryConditionDataset(
        boundary_root_dir = args.boundary_root_dir,
        start_date_hour = args.start_date_hour,
        end_date_hour = args.end_date_hour,
        upper_variables = args.upper_variables,
        surface_variables = args.surface_variables,
        levels = args.levels,
        latitude = args.latitude,
        longitude = args.longitude,
        boundary_width = args.boundary_width,
        prediction_timedeltas = args.boundary_prediction_timedeltas,
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

def _build_boundary_time_index(boundary_dataset):
    return {pd.Timestamp(t): i for i, t in enumerate(boundary_dataset.time_axis)}

def _select_boundary_frame(
    boundary_dataset,
    boundary_time_index,
    target_time,
    prefer_leads = (6, 12, 0),
    force_lead = None,
):
    target_time = pd.Timestamp(target_time)
    if force_lead is not None:
        lead = int(force_lead)
        init_time = target_time - pd.Timedelta(hours = lead)
        index = boundary_time_index.get(init_time)
        if index is None or lead not in boundary_dataset.prediction_timedelta_hours:
            return None
        data = boundary_dataset[index]
        lead_idx = boundary_dataset.prediction_timedelta_hours.index(lead)
        return data, lead_idx

    for lead in prefer_leads:
        init_time = target_time - pd.Timedelta(hours = lead)
        index = boundary_time_index.get(init_time)
        if index is None:
            continue
        if lead not in boundary_dataset.prediction_timedelta_hours:
            continue
        data = boundary_dataset[index]
        lead_idx = boundary_dataset.prediction_timedelta_hours.index(lead)
        return data, lead_idx
    return None

def _build_boundary_batch(
    boundary_dataset,
    boundary_time_index,
    target_times,
    prefer_leads = (6, 12, 0),
    force_lead = None,
):
    surf_vars = {}
    atmos_vars = {}

    for target_time in target_times:
        selection = _select_boundary_frame(
            boundary_dataset,
            boundary_time_index,
            target_time,
            prefer_leads = prefer_leads,
            force_lead = force_lead,
        )
        if selection is None:
            raise ValueError(f"No boundary data found for target time {target_time}.")
        data, lead_idx = selection

        for var_name, tensor in data["surf_vars"].items():
            surf_vars.setdefault(var_name, []).append(tensor[lead_idx])
        for var_name, tensor in data["atmos_vars"].items():
            atmos_vars.setdefault(var_name, []).append(tensor[lead_idx])

    surf_vars = {k: torch.stack(v, dim = 0) for k, v in surf_vars.items()}
    atmos_vars = {k: torch.stack(v, dim = 0) for k, v in atmos_vars.items()}
    return {"surf_vars": surf_vars, "atmos_vars": atmos_vars}

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
        padded[var_name] = F.pad(
            tensor,
            (boundary_width, boundary_width, boundary_width, boundary_width),
            mode = "replicate",
        )
    return padded

def _replace_boundary_inside(pred_tensor, boundary_tensor, boundary_width):
    if boundary_width <= 0:
        return pred_tensor
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
):
    model.eval()
    latitudes, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    boundary_enabled = boundary_dataset is not None and args.boundary_width > 0
    boundary_time_index = _build_boundary_time_index(boundary_dataset) if boundary_enabled else None

    if boundary_enabled and args.boundary_mode == "pad-outside":
        boundary_latitudes, boundary_longitude = boundary_dataset.get_latitude_longitude()
    else:
        boundary_latitudes, boundary_longitude = None, None

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

            # Pre-slice labels (this is okay to keep in list if it fits in memory, 
            # usually labels are smaller than the computation graph)
            _label_list = slice_timeaxis(labels)

            batch_times = tuple(map(lambda d: pd.Timestamp(d), dates))

            if boundary_enabled and args.boundary_mode == "pad-outside":
                boundary_init = _build_boundary_batch(
                    boundary_dataset,
                    boundary_time_index,
                    batch_times,
                )
                for var_name, tensor in boundary_init["surf_vars"].items():
                    boundary_init["surf_vars"][var_name] = tensor.to(device)
                for var_name, tensor in boundary_init["atmos_vars"].items():
                    boundary_init["atmos_vars"][var_name] = tensor.to(device)

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
            context_manager = torch.cuda.amp.autocast(dtype = dtype) if use_amp else contextlib.nullcontext()
            
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
                        print(dates)
                        print(target_times)
                        boundary_step = _build_boundary_batch(
                            boundary_dataset,
                            boundary_time_index,
                            target_times,
                        )
                        for var_name, tensor in boundary_step["surf_vars"].items():
                            boundary_step["surf_vars"][var_name] = tensor.to(device, dtype = _pred.surf_vars[var_name].dtype)
                        for var_name, tensor in boundary_step["atmos_vars"].items():
                            boundary_step["atmos_vars"][var_name] = tensor.to(device, dtype = _pred.atmos_vars[var_name].dtype)
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
                                    k: _center_crop_boundary(v, args.boundary_width)
                                    for k, v in boundary_step["surf_vars"].items()
                                },
                                "atmos_vars": {
                                    k: _center_crop_boundary(v, args.boundary_width)
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

def main():
    args = parse_args()
    print(args)
    set_seed(args.seed)
    logger.info("Running single-GPU evaluation.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = create_model(args, device)
    dataset = create_dataset(args)
    boundary_dataset = create_boundary_dataset(args)
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
                # max_lead_time = args.rollout_step,
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
        boundary_dataset = boundary_dataset,
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
