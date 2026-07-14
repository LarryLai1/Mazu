#!/usr/bin/env python
# coding=utf-8
# Self-feeding scheduled sampling v1 — epoch-updated teacher.
# A separate teacher model provides the self-feeding input; the teacher is
# refreshed from the student's weights at the start of each epoch.

from pathlib import Path
import argparse
import copy
import pandas as pd
import logging
import shutil
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from safetensors.torch import load_file

from aurora import Batch, Metadata
from aurora.model.aurora import AuroraSmall
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
from utils.metrics import AuroraMAELoss
from utils.training_scheduler import get_scheduler_with_warmup

from tqdm.auto import tqdm
import wandb

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = get_logger(__name__, log_level = "INFO")

def parse_args():
    parser = argparse.ArgumentParser(description = "Aurora Training Script (HF Style) - Self-Feeding Scheduled Sampling (Epoch-Updated Teacher)")

    # --- Standard Arguments ---
    parser.add_argument("--data_root_dir", type = str, required = True)
    parser.add_argument("--output_dir", type = str, default = "AuroraTW")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--use_pretrained_weight", action = "store_true")
    parser.add_argument("--checkpoint_path", type = str, default = None)
    parser.add_argument("--resume_from_checkpoint", type = str, default = None,
                        help = "Path to an accelerator checkpoint directory (saved by save_state) to resume training")
    parser.add_argument("--train_start_date_hour", type = str, required = True)
    parser.add_argument("--train_end_date_hour", type = str, required = True)
    parser.add_argument("--val_start_date_hour", type = str, required = True)
    parser.add_argument("--val_end_date_hour", type = str, required = True)
    parser.add_argument("--use_lora", action = "store_true")
    parser.add_argument("--bf16_mode", action = "store_true")
    parser.add_argument("--timestep_hours", type = int, default = 6)
    parser.add_argument("--stabilise_level_agg", action = "store_true")
    parser.add_argument("--upper_variables", type = str, nargs = "+", required = True)
    parser.add_argument("--surface_variables", type = str, nargs = "+", required = True)
    parser.add_argument("--static_variables", type = str, nargs = "+", required = True)
    parser.add_argument("--levels", type = int, nargs = "+", required = True)
    parser.add_argument("--latitude", type = float, nargs = 2, required = True)
    parser.add_argument("--longitude", type = float, nargs = 2, required = True)
    parser.add_argument("--lead_time", type = int, default = 0)
    parser.add_argument("--input_time_window", type = int, required = True)

    parser.add_argument("--rollout_step", type = int, default=1, help="Set to 1 or 2 depending on dataset need")

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

    # --- On-the-fly Scheduled Sampling Arguments (Self-Feeding, Epoch-Updated Teacher) ---
    parser.add_argument("--use_onthefly", action="store_true", help="Enable on-the-fly scheduled sampling (self-feeding)")
    parser.add_argument("--mix_ratio_max", type=float, default=0.7, help="Max prob of using teacher's output as input")
    parser.add_argument("--ramp_end_epoch_ratio", type=float, default=0.7, help="Epoch (as fraction of total) to reach max mixing")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Epochs before starting mixing")

    return parser.parse_args()

def create_model(args):
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
        state_dict = load_file(args.checkpoint_path)
        model.load_state_dict(state_dict, strict = False)
    return model

def create_dataset(args, split):
    return ERA5TWDatasetforAurora(
        data_root_dir = args.data_root_dir,
        start_date_hour = args.train_start_date_hour if split == "train" else args.val_start_date_hour,
        end_date_hour = args.train_end_date_hour if split == "train" else args.val_end_date_hour,
        upper_variables = args.upper_variables,
        surface_variables = args.surface_variables,
        static_variables = args.static_variables,
        levels = args.levels,
        latitude = args.latitude,
        longitude = args.longitude,
        lead_time = args.lead_time,
        input_time_window = args.input_time_window,
        rollout_step = args.rollout_step if split == "train" else 1,
    )

def save_checkpoint_by_epoch(args, accelerator, output_dir, epoch):
    output_dir = Path(output_dir)
    if args.checkpointing_epochs > 0 and epoch % args.checkpointing_epochs == 0:
        if accelerator.is_main_process:
            checkpoints = sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")], key = lambda x: int(x.name.split("-")[1]))
            if args.checkpoints_total_limit is not None and len(checkpoints) >= args.checkpoints_total_limit:
                for removing_checkpoint in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                    shutil.rmtree(removing_checkpoint)
            save_path = output_dir / f"checkpoint-{epoch}"
            save_path.mkdir(parents = True, exist_ok = True)
            accelerator.save_state(str(save_path))

def save_checkpoint_best_by_val_loss(args, accelerator, output_dir, epoch, train_loss, val_loss, best_ckpts):
    output_dir = Path(output_dir)
    if len(best_ckpts) < args.save_top_k or val_loss < max(best_ckpts, key = lambda x: x[0])[0]:
        save_path = output_dir / f"{epoch}-train_loss={train_loss:.6f}-val_loss={val_loss:.6f}"
        save_path.mkdir(parents = True, exist_ok = True)
        accelerator.save_state(str(save_path))
        best_ckpts.append((val_loss, save_path))
        best_ckpts.sort(key = lambda x: x[0])
        while len(best_ckpts) > args.save_top_k:
            worst = best_ckpts.pop()
            shutil.rmtree(worst[1], ignore_errors = True)
    return best_ckpts

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

def get_mix_ratio(args, epoch):
    if not args.use_onthefly:
        return 0.0

    warmup_epochs = args.warmup_epochs
    if epoch <= warmup_epochs:
        return 0.0

    ramp_end_epoch = int(args.epochs * args.ramp_end_epoch_ratio)
    ramp_length = max(1, ramp_end_epoch - warmup_epochs)

    progress = (epoch - warmup_epochs) / ramp_length
    progress = min(1.0, progress)

    return progress * args.mix_ratio_max

def sync_teacher_from_student(student, teacher, accelerator):
    # Copy current student weights into the teacher; teacher stays frozen & in eval mode.
    student_unwrapped = accelerator.unwrap_model(student)
    teacher.load_state_dict(student_unwrapped.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

def train_epoch(
        args,
        model,
        teacher,
        dataloader,
        optimizer,
        scheduler,
        criterion,
        accelerator,
        epoch,
        train_global_step,
    ):

    _model = accelerator.unwrap_model(model)
    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    latitude, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    current_mix_ratio = get_mix_ratio(args, epoch)

    pbar = tqdm(dataloader, disable = not accelerator.is_local_main_process, desc = f"train_epoch: {epoch} (Mix: {current_mix_ratio:.2f})")

    for batch in pbar:
        train_input, train_label, train_dates = batch

        optimizer.zero_grad()
        with accelerator.autocast():

            _label_slices = slice_timeaxis(train_label)

            input_prev_step = Batch(
                surf_vars = train_input["surf_vars"],
                atmos_vars = train_input["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(lat=latitude, lon=longitude, time=tuple(map(lambda d: pd.Timestamp(d), train_dates)), atmos_levels=levels),
            )

            gt_t1_raw = _label_slices[0]
            input_gt = Batch(
                surf_vars = gt_t1_raw["surf_vars"],
                atmos_vars = gt_t1_raw["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(lat=latitude, lon=longitude, time=tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours=args.lead_time), train_dates)), atmos_levels=levels),
            )

            gt_t2_raw = _label_slices[1]
            target = Batch(
                surf_vars = gt_t2_raw["surf_vars"],
                atmos_vars = gt_t2_raw["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(lat=latitude, lon=longitude, time=tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours=args.lead_time*2), train_dates)), atmos_levels=levels),
            )

            # Decide input source: teacher-generated (self-feeding) vs. ground truth
            final_input = input_gt

            if args.use_onthefly and teacher is not None and (random.random() < current_mix_ratio):
                # with torch.inference_mode():
                with torch.no_grad():
                    self_pred_t1 = teacher(input_prev_step)
                    self_pred_t1.metadata.time = input_gt.metadata.time
                    final_input = self_pred_t1

            prediction = model(final_input)

            loss_dict = criterion(
                prediction.normalise(surf_stats = _model.surf_stats),
                target.normalise(surf_stats = _model.surf_stats),
            )
            loss = loss_dict["all_vars"]

        accelerator.backward(loss.mean())

        if accelerator.is_local_main_process:
            total_grad_norm = 0.0
            for param in accelerator.unwrap_model(model).parameters():
                if param.grad is not None:
                    param_norm = param.grad.detach().data.norm(2).item()
                    total_grad_norm += param_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
        else:
            total_grad_norm = None

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()

        gather_train_loss = accelerator.gather(loss)
        total_train_loss += gather_train_loss.sum().item()
        total_train_samples += gather_train_loss.shape[0]
        step_loss = gather_train_loss.mean().item()

        if accelerator.is_main_process:
            pbar.set_postfix({"train_step_loss": f"{step_loss:.8f}"})
            accelerator.log({
                "train_global_step": train_global_step,
                "train/step_loss": step_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "onthefly/mix_ratio": current_mix_ratio,
            })

        train_global_step += 1

    train_epoch_loss = total_train_loss / total_train_samples
    if accelerator.is_main_process:
        accelerator.log({"epoch": epoch, "train/epoch_loss": train_epoch_loss})

    return train_epoch_loss, train_global_step

def val_epoch(
        args,
        model,
        dataloader,
        criterion,
        accelerator,
        epoch,
        val_global_step,
    ):
    unwrapped_model = accelerator.unwrap_model(model)
    model.eval()

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
            val_input, val_label, val_dates = batch
            with accelerator.autocast():
                _input = Batch(
                    surf_vars = val_input["surf_vars"],
                    atmos_vars = val_input["atmos_vars"],
                    static_vars = static_data["static_vars"],
                    metadata = Metadata(
                        lat = latitude,
                        lon = longitude,
                        time = tuple(map(lambda d: pd.Timestamp(d), val_dates)),
                        atmos_levels = levels,
                    ),
                )
                _label = Batch(
                    surf_vars = val_label['surf_vars'],
                    atmos_vars = val_label['atmos_vars'],
                    static_vars = static_data["static_vars"],
                    metadata = Metadata(
                        lat = latitude,
                        lon = longitude,
                        time = tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours = args.lead_time), val_dates)),
                        atmos_levels = levels,
                    ),
                )
                _pred = model(_input)
                loss_dict = criterion(
                    _pred.normalise(surf_stats = unwrapped_model.surf_stats),
                    _label.normalise(surf_stats = unwrapped_model.surf_stats)
                )
                loss = loss_dict["all_vars"]

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
    accelerator = Accelerator(mixed_precision = args.mixed_precision, log_with = args.report_to, project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir))
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)), init_kwargs={"wandb": {"name": args.wandb_name}})

    model = create_model(args)

    # Build the teacher as an independent copy of the student, kept outside accelerator.prepare.
    # It mirrors the student's weights and is refreshed once per epoch.
    teacher = None
    if args.use_onthefly:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    train_dataset = create_dataset(args, "train")
    val_dataset = create_dataset(args, "val")
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler_with_warmup(optimizer, warmup_steps=int(args.warmup_step_ratio * args.epochs * len(train_loader)), training_steps=args.epochs * len(train_loader), schedule_type="cosine")

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    if teacher is not None:
        teacher.to(accelerator.device)

    # ---- Resume from checkpoint ----
    starting_epoch = 1
    train_global_step = 0
    val_global_step = 0
    best_ckpts = []

    if args.resume_from_checkpoint:
        resume_path = Path(args.resume_from_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        logger.info(f"Resuming from checkpoint: {resume_path}")
        accelerator.load_state(str(resume_path))

        dir_name = resume_path.name
        if dir_name.startswith("checkpoint-"):
            resumed_epoch = int(dir_name.split("-")[1])
        else:
            resumed_epoch = int(dir_name.split("-")[0])

        starting_epoch = resumed_epoch + 1
        steps_per_train_epoch = len(train_loader)
        steps_per_val_epoch = len(val_loader)
        train_global_step = resumed_epoch * steps_per_train_epoch
        val_global_step = resumed_epoch * steps_per_val_epoch

        logger.info(f"Resumed training — will start from epoch {starting_epoch} "
                     f"(train_global_step={train_global_step}, val_global_step={val_global_step})")

        # After resume, realign teacher with the loaded student weights.
        if teacher is not None:
            sync_teacher_from_student(model, teacher, accelerator)

    for epoch in range(starting_epoch, args.epochs + 1):
        # Refresh teacher at the start of each epoch so it reflects the last epoch's student.
        if teacher is not None and epoch > 1:
            sync_teacher_from_student(model, teacher, accelerator)

        tl, train_global_step = train_epoch(args, model, teacher, train_loader, optimizer, scheduler, AuroraMAELoss, accelerator, epoch, train_global_step)
        vl, val_global_step = val_epoch(args, model, val_loader, AuroraMAELoss, accelerator, epoch, val_global_step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch}: Train {tl:.6f} Val {vl:.6f}")
            save_checkpoint_by_epoch(args, accelerator, ckpt_dir, epoch)
            best_ckpts = save_checkpoint_best_by_val_loss(args, accelerator, ckpt_dir, epoch, tl, vl, best_ckpts)

    accelerator.end_training()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force = True)
    main()
