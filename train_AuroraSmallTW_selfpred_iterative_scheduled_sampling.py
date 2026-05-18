#!/usr/bin/env python
# coding=utf-8

from pathlib import Path
import argparse
import pandas as pd
import logging
import shutil
import copy
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
    parser = argparse.ArgumentParser(description = "Aurora Training Script (HF Style)")

    # --- Standard Arguments ---
    parser.add_argument("--data_root_dir", type = str, required = True)
    parser.add_argument("--output_dir", type = str, default = "AuroraTW")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--use_pretrained_weight", action = "store_true")
    parser.add_argument("--checkpoint_path", type = str, default = None)
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
    
    # We remove rollout_step arg for training logic, but dataset might need it to fetch t, t+1, t+2
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

    # --- DAgger / Scheduled Sampling Arguments ---
    parser.add_argument("--use_dagger", action="store_true", help="Enable Iterative DAgger / Scheduled Sampling")
    parser.add_argument("--teacher_update_freq", type=int, default=10, help="Epoch frequency to update teacher weights")
    parser.add_argument("--dagger_mix_ratio_max", type=float, default=0.7, help="Max prob of using teacher output as input")
    parser.add_argument("--ramp_end_epoch_ratio", type=float, default=0.7, help="Epoch (as fraction of total) to reach max mixing")
    parser.add_argument("--dagger_warmup_epochs", type=int, default=10, help="Epochs before starting mixing")

    return parser.parse_args()

# ... (create_model and create_dataset remain same) ...
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
    # Dataset needs to return at least 3 steps: t-1, t, t+1
    # Ensure rollout_step in dataset args is sufficient (e.g., 2) to get enough history/future
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
        # rollout_step = args.rollout_step, # Ensure this allows fetching t-1, t, t+1
        rollout_step = args.rollout_step if split == "train" else 1, # For val, only need 1 step ahead
    )

# ... (save_checkpoint helpers remain same) ...
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

# --- Helper: DAgger Mixing Schedule ---
def get_mix_ratio(args, epoch):
    if not args.use_dagger: 
        return 0.0
    
    # 1. Warmup Phase (Fixed or % of total)
    # Let's say warmup is first 10% of training or minimum 5 epochs
    warmup_epochs = args.dagger_warmup_epochs 
    if epoch <= warmup_epochs:
        return 0.0
    
    # 2. Ramp Phase
    # We want to reach max mixing by roughly 50% of training
    # So ramp_length = (Total_Epochs / 2) - Warmup
    # ramp_end_epoch = int(args.epochs * 0.5) 
    ramp_end_epoch = int(args.epochs * args.ramp_end_epoch_ratio)
    ramp_length = max(1, ramp_end_epoch - warmup_epochs)

    # Calculate progress (0.0 to 1.0)
    progress = (epoch - warmup_epochs) / ramp_length
    
    # Clip progress to max 1.0
    progress = min(1.0, progress)
    
    return progress * args.dagger_mix_ratio_max

def train_epoch(
        args,
        model,
        teacher_model, # Passed in
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
    if teacher_model: teacher_model.eval()
    
    total_train_loss = 0.0
    total_train_samples = 0
    
    latitude, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()
    
    current_mix_ratio = get_mix_ratio(args, epoch)

    pbar = tqdm(dataloader, disable = not accelerator.is_local_main_process, desc = f"train_epoch: {epoch} (Mix: {current_mix_ratio:.2f})")

    for batch in pbar:
        # Note: Dataset must provide t-1, t, t+1. 
        # Assume batch contains: 
        # train_input = x_0 (used to generate Teacher Pred for x_1)
        # train_label slices = x_1 (GT input), x_2 (GT Target)
        # We need to structure this correctly.
        
        train_input, train_label, train_dates = batch
        
        # Structure assumptions based on typical Aurora dataset:
        # train_input is at time T (let's call it T_prev)
        # train_label contains T+1, T+2...
        
        # We want to train: Input -> T_next
        # Candidate 1 (GT): Input = Label[0] (Time T+1)
        # Candidate 2 (Teacher): Input = Teacher(Input[T]) -> Pred[T+1]
        # Target: Label[1] (Time T+2)
        
        # We need "previous step" data to run the teacher.
        # Let's assume train_input is t=0. Label is t=1, t=2.
        
        optimizer.zero_grad()
        with accelerator.autocast():
            
            _label_slices = slice_timeaxis(train_label)
            
            # 1. Prepare Ground Truths
            # ------------------------
            # Input_Prev (t=0) used for Teacher
            input_prev_step = Batch(
                surf_vars = train_input["surf_vars"],
                atmos_vars = train_input["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(lat=latitude, lon=longitude, time=tuple(map(lambda d: pd.Timestamp(d), train_dates)), atmos_levels=levels),
            )
            
            # Input_GT (t=1) - The clean input
            gt_t1_raw = _label_slices[0] 
            input_gt = Batch(
                surf_vars = gt_t1_raw["surf_vars"],
                atmos_vars = gt_t1_raw["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(lat=latitude, lon=longitude, time=tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours=args.lead_time), train_dates)), atmos_levels=levels),
            )
            
            # Target (t=2) - What we want to predict
            gt_t2_raw = _label_slices[1]
            target = Batch(
                surf_vars = gt_t2_raw["surf_vars"],
                atmos_vars = gt_t2_raw["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(lat=latitude, lon=longitude, time=tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours=args.lead_time*2), train_dates)), atmos_levels=levels),
            )

            # 2. DECIDE INPUT SOURCE (The Core Logic)
            # ---------------------------------------
            final_input = input_gt # Default to GT
            
            # Only use Teacher if DAgger is ON and randomness hits
            if args.use_dagger and (random.random() < current_mix_ratio):
                with torch.no_grad():
                    # Teacher takes t=0 -> Predicts t=1 (Noisy)
                    teacher_pred_t1 = teacher_model(input_prev_step)
                    
                    # Update metadata for the prediction to be treated as input t=1
                    teacher_pred_t1.metadata.time = input_gt.metadata.time
                    
                    # This is now our input (Detached automatically by no_grad)
                    final_input = teacher_pred_t1

            # 3. TRAIN (Strict 1-Step)
            # ------------------------
            # Model takes (Mixed Input t=1) -> Predicts (Target t=2)
            prediction = model(final_input)
            
            loss_dict = criterion(
                prediction.normalise(surf_stats = _model.surf_stats),
                target.normalise(surf_stats = _model.surf_stats),
            )
            loss = loss_dict["all_vars"]

        accelerator.backward(loss.mean())
        
        # ... (Gradient Clipping, Opt Step, Logging - Same as before) ...
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
                "dagger/mix_ratio": current_mix_ratio,
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
        # ncols = 120,
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
    
    # --- TEACHER INIT ---
    teacher_model = None
    if args.use_dagger:
        logger.info("DAgger Enabled: Initializing Teacher Model")
        teacher_model = copy.deepcopy(model)
        for p in teacher_model.parameters(): p.requires_grad = False
        teacher_model.eval()

    train_dataset = create_dataset(args, "train")
    val_dataset = create_dataset(args, "val")
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler_with_warmup(optimizer, warmup_steps=int(args.warmup_step_ratio * args.epochs * len(train_loader)), training_steps=args.epochs * len(train_loader), schedule_type="cosine")
    
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    if teacher_model: teacher_model = teacher_model.to(accelerator.device)

    train_global_step = 0
    val_global_step = 0
    best_ckpts = []

    for epoch in range(1, args.epochs + 1):
        # --- TEACHER UPDATE ---
        if args.use_dagger and (epoch % args.teacher_update_freq == 0) and epoch > 0:
            if accelerator.is_main_process: logger.info(f"DAgger: Updating Teacher Weights at Epoch {epoch}")
            teacher_model.load_state_dict(accelerator.unwrap_model(model).state_dict())

        tl, train_global_step = train_epoch(args, model, teacher_model, train_loader, optimizer, scheduler, AuroraMAELoss, accelerator, epoch, train_global_step)
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