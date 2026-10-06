#!/usr/bin/env python
# coding=utf-8
"""Decoder-only finetuning of AuroraSmallTW with the HRES boundary injected the same way as in
inference (--replace_boundary_position backbone and/or input).

Same data path as train_AuroraSmallTW_fast.py (ERA5 inputs t-1, t -> 1-step target t+1). The
boundary batch is built exactly like rollout step 0 of AuroraSmallTW_gen_eval_pipeline_custom_rollout.py:
BoundaryConditionDataset_HRES + that script's _build_boundary_batch / _align_boundary_batch /
_stack_boundary_time_window, run inside the DataLoader workers via collate_fn. The encoder and
Swin3D backbone are frozen and run under no_grad; only model.decoder is trained. The loss is the
same MAE in normalised space.

Checkpoints are FULL-model .safetensors (frozen encoder/backbone + finetuned decoder), so they can
be passed straight to the inference pipelines' --checkpoint_path.
"""

from functools import partial
from pathlib import Path
import argparse
import pandas as pd
import logging
import shutil

import torch
from torch.utils.data import DataLoader, default_collate
from torch.optim import AdamW
from utils.muon import Muon

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from safetensors.torch import load_file, save_file

from aurora import Batch, Metadata
from aurora.model.aurora import AuroraSmall

from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
from datasets.BoundaryConditionDataset import BoundaryConditionDataset_HRES
from AuroraSmallTW_gen_eval_pipeline_custom_rollout import (
    _align_boundary_batch,
    _build_boundary_batch,
    _is_increasing,
    _stack_boundary_time_window,
)

from utils.decoder_finetune import REPLACE_BOUNDARY_POSITIONS, backbone_forward_with_boundary, decode_from_backbone
from utils.metrics import AuroraMAELoss
from utils.training_scheduler import get_scheduler_with_warmup

from tqdm.auto import tqdm
import wandb

# force: importing the inference pipeline above already configured the root logger.
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force = True,
)
logger = get_logger(__name__, log_level = "INFO")


def _split_muon_and_adamw_params(params):
    muon_params = []
    adamw_params = []
    for param in params:
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


class OptimizerBundle:
    def __init__(self, optimizers):
        self.optimizers = [opt for opt in optimizers if opt is not None]

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


class SchedulerBundle:
    def __init__(self, schedulers):
        self.schedulers = [sch for sch in schedulers if sch is not None]

    def step(self):
        for sch in self.schedulers:
            sch.step()


def parse_args():
    parser = argparse.ArgumentParser(description = "AuroraSmallTW decoder-only finetuning with HRES boundary")

    parser.add_argument("--data_root_dir", type = str, required = True)
    parser.add_argument("--output_dir", type = str, default = "AuroraTW_decoder_only")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--checkpoint_path", type = str, required = True,
                        help = "Full-model weights to finetune (.safetensors, or official .ckpt).")

    parser.add_argument("--train_start_date_hour", type = str, required = True)
    parser.add_argument("--train_end_date_hour", type = str, required = True)
    parser.add_argument("--val_start_date_hour", type = str, required = True)
    parser.add_argument("--val_end_date_hour", type = str, required = True)

    parser.add_argument("--use_lora", action = "store_true")
    parser.add_argument("--timestep_hours", type = int, default = 1)
    parser.add_argument("--stabilise_level_agg", action = "store_true")

    parser.add_argument("--upper_variables", type = str, nargs = "+", required = True)
    parser.add_argument("--surface_variables", type = str, nargs = "+", required = True)
    parser.add_argument("--static_variables", type = str, nargs = "+", required = True)
    parser.add_argument("--levels", type = int, nargs = "+", required = True)
    parser.add_argument("--latitude", type = float, nargs = 2, required = True)
    parser.add_argument("--longitude", type = float, nargs = 2, required = True)
    parser.add_argument("--lead_time", type = int, default = 1)
    parser.add_argument("--input_time_window", type = int, required = True)

    parser.add_argument("--boundary_root_dir", type = str, required = True,
                        help = "HRES root (00/12Z inits, +0/+6/+12h leads per file).")
    parser.add_argument("--replace_boundary_position", type = str, nargs = "+", default = ["backbone"],
                        choices = REPLACE_BOUNDARY_POSITIONS)
    parser.add_argument("--boundary_width", type = int, default = 8, help = "Ring width in grid cells.")
    parser.add_argument("--boundary_smooth_mode", type = str, default = "no",
                        choices = ["no", "linear", "mean", "gaussian"])
    parser.add_argument(
        "--boundary_time_interp_mode",
        type = str,
        default = "interpolation",
        choices = ["interpolation", "nearest"],
        help = "BoundaryConditionDataset_HRES time_interp_mode across the forecast leads (same as inference).",
    )

    parser.add_argument("--use_muon", action = "store_true")
    parser.add_argument("--muon_gradient_accumulation_steps", type = int, default = 4)

    parser.add_argument("--epochs", type = int, default = 5)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--weight_decay", type = float, default = 1e-3)
    parser.add_argument("--warmup_step_ratio", type = float, default = 0.1)
    parser.add_argument("--max_grad_norm", type = float, default = 1.0)
    parser.add_argument("--train_batch_size", type = int, default = 16)
    parser.add_argument("--val_batch_size", type = int, default = 16)
    parser.add_argument("--num_workers", type = int, default = 4)

    parser.add_argument("--checkpointing_epochs", type = int, default = 5)
    parser.add_argument("--checkpoints_total_limit", type = int, default = None)
    parser.add_argument("--save_top_k", type = int, default = 3)

    parser.add_argument("--logging_dir", type = str, default = "logs")
    parser.add_argument("--report_to", type = str, default = "tensorboard")
    parser.add_argument("--tracker_project_name", type = str, default = "AuroraSmallTW")
    parser.add_argument("--mixed_precision", type = str, default = None, choices = ["no", "fp16", "bf16"])
    parser.add_argument("--wandb_name", type = str, default = None)

    args = parser.parse_args()
    if "backbone" in args.replace_boundary_position and args.boundary_width < 4:
        # Backbone replacement works on 4x4 patches: boundary_width // patch_size == 0 is a no-op.
        parser.error("--boundary_width must be >= the patch size (4) for backbone replacement.")
    return args


def create_model(args):
    model = AuroraSmall(
        use_lora = args.use_lora,
        timestep = pd.Timedelta(hours = args.timestep_hours),
        stabilise_level_agg = args.stabilise_level_agg,
    )

    logger.info(f"Loading checkpoint: {args.checkpoint_path}")
    if args.checkpoint_path.endswith(".ckpt"):
        model.load_checkpoint_local(args.checkpoint_path, strict = True)
    else:
        model.load_state_dict(load_file(args.checkpoint_path), strict = True)

    # Freeze everything but the decoder. The frozen parts stay in eval mode for the whole run.
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(True)
    model.eval()
    return model


def _split_dates(args, split):
    if split == "train":
        return args.train_start_date_hour, args.train_end_date_hour
    if split == "val":
        return args.val_start_date_hour, args.val_end_date_hour
    raise Exception("Do not support this dataset split!")


def create_dataset(args, split):
    start_date_hour, end_date_hour = _split_dates(args, split)
    return ERA5TWDatasetforAurora(
            data_root_dir = args.data_root_dir,
            start_date_hour = start_date_hour,
            end_date_hour = end_date_hour,
            upper_variables = args.upper_variables,
            surface_variables = args.surface_variables,
            static_variables = args.static_variables,
            levels = args.levels,
            latitude = args.latitude,
            longitude = args.longitude,
            lead_time = args.lead_time,
            input_time_window = args.input_time_window,
            rollout_step = 1,
        )


def create_boundary_dataset(args, split):
    # Same construction as create_boundary_dataset() in the inference pipeline (hres source).
    start_date_hour, end_date_hour = _split_dates(args, split)
    return BoundaryConditionDataset_HRES(
        boundary_root_dir = args.boundary_root_dir,
        start_date_hour = start_date_hour,
        end_date_hour = end_date_hour,
        upper_variables = args.upper_variables,
        surface_variables = args.surface_variables,
        levels = args.levels,
        latitude = args.latitude,
        longitude = args.longitude,
        boundary_width = 0,
        prediction_timedeltas = [0, 6, 12],
        time_interp_mode = args.boundary_time_interp_mode,
    )


def collate_with_boundary(samples, boundary_dataset, flip_lat, flip_lon, input_time_window, timestep_hours):
    """default_collate the ERA5 samples, then attach the HRES boundary for their input window the
    way the inference pipeline builds it for rollout step 0 (_produce_boundary_step with k = 0).
    Runs in the DataLoader workers, so the HRES reads overlap with training."""
    _input, _target, dates = default_collate(samples)
    base_times = tuple(boundary_dataset.get_base_time(pd.Timestamp(d)) for d in dates)
    single_steps = []
    for tw in range(input_time_window - 1, -1, -1):  # oldest -> newest
        target_times = tuple(pd.Timestamp(d) - pd.Timedelta(hours = tw * timestep_hours) for d in dates)
        single_steps.append(_align_boundary_batch(
            _build_boundary_batch(boundary_dataset, base_times, target_times), flip_lat, flip_lon,
        ))
    _boundary = _stack_boundary_time_window(single_steps)
    if _boundary is None:
        raise RuntimeError(f"No HRES boundary for base times {list(dates)}.")
    return _input, _boundary, _target, dates


def create_dataloader(args, split, batch_size, shuffle):
    ds = create_dataset(args, split)
    boundary_ds = create_boundary_dataset(args, split)

    latitude, longitude = ds.get_latitude_longitude()
    boundary_latitude, boundary_longitude = boundary_ds.get_latitude_longitude()
    collate_fn = partial(
        collate_with_boundary,
        boundary_dataset = boundary_ds,
        flip_lat = _is_increasing(boundary_latitude) != _is_increasing(latitude),
        flip_lon = _is_increasing(boundary_longitude) != _is_increasing(longitude),
        input_time_window = args.input_time_window,
        timestep_hours = args.timestep_hours,
    )
    return DataLoader(
        ds, batch_size = batch_size, shuffle = shuffle,
        num_workers = args.num_workers, pin_memory = True,
        # Keep workers alive across epochs instead of re-spawning (and re-importing) them each epoch.
        persistent_workers = args.num_workers > 0,
        collate_fn = collate_fn,
    )


def save_full_model(model, save_path):
    save_path = Path(save_path)
    save_path.mkdir(parents = True, exist_ok = True)
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(save_path / "model.safetensors"))


def save_checkpoint_by_epoch(args, model, output_dir, epoch):
    output_dir = Path(output_dir)

    if args.checkpointing_epochs > 0 and epoch % args.checkpointing_epochs == 0:
        checkpoints = sorted(
            [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key = lambda x: int(x.name.split("-")[1]),
        )
        if args.checkpoints_total_limit is not None and len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            for removing_checkpoint in checkpoints[:num_to_remove]:
                shutil.rmtree(removing_checkpoint)
                logger.info(f"Removed old checkpoint: {removing_checkpoint}")

        save_path = output_dir / f"checkpoint-{epoch}"
        save_full_model(model, save_path)
        logger.info(f"Saved checkpoint to {save_path}")


def save_checkpoint_best_by_val_loss(
    args,
    model,
    output_dir: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_ckpts: list,
):
    output_dir = Path(output_dir)

    if len(best_ckpts) < args.save_top_k or val_loss < max(best_ckpts, key = lambda x: x[0])[0]:
        save_path = output_dir / f"{epoch}-train_loss={train_loss:.8f}-val_loss={val_loss:.8f}"
        save_full_model(model, save_path)
        logger.info(f"Saved new best checkpoint: {save_path} (train_loss={train_loss:.8f} val_loss={val_loss:.8f})")

        best_ckpts.append((val_loss, save_path))
        best_ckpts.sort(key = lambda x: x[0])

        while len(best_ckpts) > args.save_top_k:
            worst = best_ckpts.pop()
            shutil.rmtree(worst[1], ignore_errors = True)
            logger.info(f"Removed older/worse checkpoint: {worst[1]}")

    return best_ckpts


def compute_loss(args, model, decoder, batch, latitude, longitude, levels, static_data, criterion):
    """Frozen encoder/backbone (no_grad) -> trainable decoder -> per-sample MAE in normalised space."""
    _input, _boundary, _target, dates = batch

    def make_batch(data, times):
        return Batch(
            surf_vars = data["surf_vars"],
            atmos_vars = data["atmos_vars"],
            static_vars = static_data["static_vars"],
            metadata = Metadata(
                lat = latitude,
                lon = longitude,
                time = times,
                atmos_levels = levels,
            ),
        )

    input_times = tuple(map(lambda d: pd.Timestamp(d), dates))
    target_times = tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours = args.lead_time), dates))

    x, prepped_batch = backbone_forward_with_boundary(
        model,
        make_batch(_input, input_times),
        make_batch(_boundary, input_times),
        args.replace_boundary_position,
        args.boundary_width,
        args.boundary_smooth_mode,
    )
    _pred = decode_from_backbone(model, x, prepped_batch, decoder = decoder)
    _label = make_batch(_target, target_times)

    loss_dict = criterion(
        _pred,
        _label.normalise(surf_stats = model.surf_stats),
    )
    return loss_dict["all_vars"]


def train_epoch(
        args,
        model,
        decoder,
        dataloader,
        optimizer_bundle,
        scheduler_bundle,
        criterion,
        accelerator,
        epoch,
        train_global_step,
        train_micro_step,
    ):

    accumulation_steps = args.muon_gradient_accumulation_steps if args.use_muon else 1

    decoder.train()

    total_train_loss = 0.0
    total_train_samples = 0

    latitude, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    pbar = tqdm(
        dataloader,
        disable = not accelerator.is_local_main_process,
        desc = f"train_epoch: {epoch}",
    )

    optimizer_bundle.zero_grad()

    for batch_idx, batch in enumerate(pbar):

        with accelerator.accumulate(decoder):
            with accelerator.autocast():
                loss = compute_loss(args, model, decoder, batch, latitude, longitude, levels, static_data, criterion)

            scaled_loss = loss.mean() / accumulation_steps
            accelerator.backward(scaled_loss)

        if accelerator.is_local_main_process:
            # One fused reduction and a single .item() instead of one GPU sync per parameter.
            grad_norms = [
                param.grad.detach().norm(2)
                for param in decoder.parameters()
                if param.grad is not None
            ]
            total_grad_norm = torch.stack(grad_norms).norm(2).item() if grad_norms else 0.0
        else:
            total_grad_norm = None

        should_step = accelerator.sync_gradients or (batch_idx + 1 == len(dataloader))
        if should_step:
            accelerator.clip_grad_norm_(
                decoder.parameters(),
                args.max_grad_norm,
            )

            optimizer_bundle.step()
            scheduler_bundle.step()
            optimizer_bundle.zero_grad()
        gather_train_loss = accelerator.gather(loss.detach())

        total_train_loss += gather_train_loss.sum().item()
        total_train_samples += gather_train_loss.shape[0]

        current_lr = optimizer_bundle.param_groups[0]["lr"]
        step_loss = gather_train_loss.mean().item()

        if accelerator.is_main_process and should_step:
            pbar.set_postfix({
                "train_step_loss": f"{step_loss:.8f}",
            })
            accelerator.log(
                {
                    "train_global_step": train_global_step,
                    "lr": current_lr,
                    "grad_norm": total_grad_norm,
                },
            )

        if accelerator.is_main_process:
            accelerator.log(
                {
                    "train_micro_step": train_micro_step,
                    "train/step_loss": step_loss,
                },
            )

        train_micro_step += 1

        if should_step:
            train_global_step += 1

    train_epoch_loss = total_train_loss / total_train_samples

    if accelerator.is_main_process:
        accelerator.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_epoch_loss,
            },
        )

    return train_epoch_loss, train_global_step, train_micro_step


def val_epoch(
        args,
        model,
        decoder,
        dataloader,
        criterion,
        accelerator,
        epoch,
        val_global_step,
    ):
    decoder.eval()

    total_val_loss = 0.0
    total_val_samples = 0

    latitude, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    pbar = tqdm(
        dataloader,
        disable = not accelerator.is_local_main_process,
        desc = f"val_epoch: {epoch}",
    )

    with torch.inference_mode():
        for batch in pbar:
            with accelerator.autocast():
                loss = compute_loss(args, model, decoder, batch, latitude, longitude, levels, static_data, criterion)

            gather_val_loss = accelerator.gather(loss)
            total_val_loss += gather_val_loss.sum().item()
            total_val_samples += gather_val_loss.shape[0]

            step_loss = gather_val_loss.mean().item()

            if accelerator.is_main_process:
                pbar.set_postfix({"val_step_loss": f"{step_loss:.8f}"})
                accelerator.log(
                    {
                        "val_global_step": val_global_step,
                        "val/step_loss": step_loss,
                    },
                )

            val_global_step += 1

    val_epoch_loss = total_val_loss / total_val_samples

    if accelerator.is_main_process:
        accelerator.log(
            {
                "epoch": epoch,
                "val/epoch_loss": val_epoch_loss,
            },
        )

    return val_epoch_loss, val_global_step


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    logging_dir = output_dir / args.logging_dir
    ckpt_dir = output_dir / "ckpts"
    ckpt_dir.mkdir(parents = True, exist_ok = True)

    accelerator_project_config = ProjectConfiguration(
        project_dir = args.output_dir,
        logging_dir = logging_dir,
    )

    accelerator = Accelerator(
        mixed_precision = args.mixed_precision,
        log_with = args.report_to,
        project_config = accelerator_project_config,
        gradient_accumulation_steps = args.muon_gradient_accumulation_steps if args.use_muon else 1,
    )

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            config = tracker_config,
            init_kwargs = {"wandb": {"name": args.wandb_name}},
        )

        if args.report_to == "wandb":
            run = wandb.run
            run.define_metric("train/step_loss", step_metric = "train_micro_step")
            run.define_metric("val/step_loss", step_metric = "val_global_step")
            run.define_metric("train/epoch_loss", step_metric = "epoch")
            run.define_metric("val/epoch_loss", step_metric = "epoch")
            run.define_metric("lr", step_metric = "train_global_step")
            run.define_metric("grad_norm", step_metric = "train_global_step")

    logger.info(accelerator.state)

    # The frozen encoder/backbone run outside DDP, so the whole model goes to the device here and
    # only the decoder is handed to accelerator.prepare (DDP only syncs what it wraps).
    model = create_model(args).to(accelerator.device)
    decoder = model.decoder
    logger.info(f"Trainable decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    train_loader = create_dataloader(args, "train", args.train_batch_size, shuffle = True)
    val_loader = create_dataloader(args, "val", args.val_batch_size, shuffle = False)

    effective_lr = args.lr * args.muon_gradient_accumulation_steps if args.use_muon else args.lr
    muon_params, adamw_params = _split_muon_and_adamw_params(decoder.parameters())

    if args.use_muon:
        logger.info(
            "Using hybrid optimizer with accumulation: base_lr=%.8e, accum_steps=%d, effective_lr=%.8e",
            args.lr,
            args.muon_gradient_accumulation_steps,
            effective_lr,
        )
        logger.info(
            "Hybrid split: muon_2d_params=%d, adamw_other_params=%d",
            sum(p.numel() for p in muon_params),
            sum(p.numel() for p in adamw_params),
        )
        muon_optimizer = Muon(
            muon_params,
            lr = effective_lr,
            weight_decay = args.weight_decay,
        ) if len(muon_params) > 0 else None
        adamw_optimizer = AdamW(
            adamw_params,
            lr = effective_lr,
            weight_decay = args.weight_decay,
        ) if len(adamw_params) > 0 else None
    else:
        logger.info("Using AdamW: lr=%.8e", effective_lr)
        muon_optimizer = None
        adamw_optimizer = AdamW(
            decoder.parameters(),
            lr = effective_lr,
            weight_decay = args.weight_decay,
        )

    criterion = AuroraMAELoss

    updates_per_epoch = len(train_loader)
    if args.use_muon:
        updates_per_epoch = (updates_per_epoch + args.muon_gradient_accumulation_steps - 1) // args.muon_gradient_accumulation_steps

    total_training_steps = args.epochs * updates_per_epoch
    warmup_steps = int(args.warmup_step_ratio * total_training_steps)

    adamw_scheduler = get_scheduler_with_warmup(
        adamw_optimizer,
        warmup_steps = warmup_steps,
        training_steps = total_training_steps,
        schedule_type = "cosine",
    ) if adamw_optimizer is not None else None
    muon_scheduler = get_scheduler_with_warmup(
        muon_optimizer,
        warmup_steps = warmup_steps,
        training_steps = total_training_steps,
        schedule_type = "cosine",
    ) if muon_optimizer is not None else None

    if args.use_muon and muon_optimizer is not None and adamw_optimizer is not None:
        decoder, adamw_optimizer, muon_optimizer, train_loader, val_loader, adamw_scheduler, muon_scheduler = accelerator.prepare(
            decoder, adamw_optimizer, muon_optimizer, train_loader, val_loader, adamw_scheduler, muon_scheduler,
        )
    elif muon_optimizer is not None:
        decoder, muon_optimizer, train_loader, val_loader, muon_scheduler = accelerator.prepare(
            decoder, muon_optimizer, train_loader, val_loader, muon_scheduler,
        )
    else:
        decoder, adamw_optimizer, train_loader, val_loader, adamw_scheduler = accelerator.prepare(
            decoder, adamw_optimizer, train_loader, val_loader, adamw_scheduler,
        )

    optimizer_bundle = OptimizerBundle([adamw_optimizer, muon_optimizer])
    scheduler_bundle = SchedulerBundle([adamw_scheduler, muon_scheduler])

    train_global_step = 0
    train_micro_step = 0
    val_global_step = 0
    best_checkpoints = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_global_step, train_micro_step = train_epoch(
            args,
            model,
            decoder,
            train_loader,
            optimizer_bundle,
            scheduler_bundle,
            criterion,
            accelerator,
            epoch,
            train_global_step,
            train_micro_step,
        )
        val_loss, val_global_step = val_epoch(
            args,
            model,
            decoder,
            val_loader,
            criterion,
            accelerator,
            epoch,
            val_global_step,
        )

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            logger.info(f"epoch {epoch} - train_loss: {train_loss:.8f}")
            logger.info(f"epoch {epoch} - val_loss: {val_loss:.8f}")
            # `model` holds the (unwrapped) decoder being trained, so its state dict is the full
            # finetuned model.
            save_checkpoint_by_epoch(
                args,
                model,
                ckpt_dir,
                epoch,
            )
            best_checkpoints = save_checkpoint_best_by_val_loss(
                args,
                model,
                ckpt_dir,
                epoch,
                train_loss,
                val_loss,
                best_checkpoints,
            )

        accelerator.wait_for_everyone()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    accelerator.end_training()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force = True)
    main()
