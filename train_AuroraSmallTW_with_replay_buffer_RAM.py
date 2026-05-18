#!/usr/bin/env python
# coding=utf-8

from pathlib import Path
import argparse
import pandas as pd
import logging
import shutil
import random
from collections import deque

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from safetensors.torch import load_file

from aurora import Batch, Metadata
from aurora.model.aurora import AuroraSmall

# Import your dataset
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

# ==========================================
# 1. Helper Functions
# ==========================================

def batch_to_device(batch_obj, device):
    """
    Recursively moves an Aurora Batch to the specified device (cpu or cuda).
    Detaches tensors to ensure no gradients are stored in the buffer.
    """
    def move_dict(d):
        if d is None: return None
        # detach() removes from graph, to(device) moves memory
        return {k: v.detach().to(device) for k, v in d.items()}

    return Batch(
        surf_vars = move_dict(batch_obj.surf_vars),
        atmos_vars = move_dict(batch_obj.atmos_vars),
        static_vars = move_dict(batch_obj.static_vars),
        metadata = batch_obj.metadata 
    )

def construct_next_input(current_input_batch, prediction_batch, lead_time_hours):
    """
    Implements the sliding window logic for Aurora.
    Takes the Prediction (Surf/Atmos) + Old Static + New Time Metadata.
    """
    # 1. Update Time Metadata
    new_metadata = Metadata(
        lat = current_input_batch.metadata.lat,
        lon = current_input_batch.metadata.lon,
        time = tuple(t + pd.Timedelta(hours=lead_time_hours) for t in current_input_batch.metadata.time),
        atmos_levels = current_input_batch.metadata.atmos_levels,
    )
    
    # 2. Create Next Batch
    next_input = Batch(
        surf_vars = prediction_batch.surf_vars,
        atmos_vars = prediction_batch.atmos_vars,
        static_vars = current_input_batch.static_vars, 
        metadata = new_metadata
    )
    return next_input

def slice_timeaxis(labels):
    """
    Splits a batch with shape [B, Time, Lat, Lon] into a dictionary of time steps.
    Returns: {0: {vars...}, 1: {vars...}}
    """
    # Get the time dimension length from the first available tensor
    first_var = next(iter(next(iter(labels.values())).values()))
    timeaxis_length = first_var.shape[1]
    
    n_g = {}
    for i in range(timeaxis_length):
        n_g[i] = {}
        for var_type, var_dict in labels.items():
            n_g[i][var_type] = {}
            for var_name, tensor in var_dict.items():
                # Slice: keep dims but restrict time to i
                n_g[i][var_type][var_name] = tensor[:, i : i + 1]
    return n_g

# ==========================================
# 2. Optimized Replay Buffer (RAM Only)
# ==========================================

class ReplayBuffer:
    """
    Pure RAM Replay Buffer.
    Since we load T+2 via rollout_step=2, we don't need disk reading logic here.
    """
    def __init__(self, max_size=200):
        self.buffer = deque(maxlen=max_size)

    def put(self, prediction_input_batch, ground_truth_target_batch):
        """
        Stores a pair of (Input, Target) for future training.
        Input: The model's prediction from step T (which becomes input for T+1).
        Target: The actual ground truth for step T+1.
        """
        # Move everything to CPU to save GPU VRAM
        input_cpu = batch_to_device(prediction_input_batch, "cpu")
        target_cpu = batch_to_device(ground_truth_target_batch, "cpu")

        self.buffer.append((input_cpu, target_cpu))

    def sample(self):
        # Returns (Input_Batch_CPU, Target_Batch_CPU)
        if len(self.buffer) == 0:
            return None
        return random.choice(self.buffer)

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. Main Logic
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description = "Aurora Training Script")
    
    # Standard Aurora Args
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
    
    # CRITICAL: We need rollout_step >= 2 for this replay method
    parser.add_argument("--rollout_step", type = int, required = True, help="Must be >= 2 for replay buffer")

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
    
    # --- Replay Buffer Args ---
    parser.add_argument("--replay_buffer_size", type=int, default=200, help="Size of replay buffer")
    parser.add_argument("--finetune_rate", type=int, default=5, help="Number of replay steps per real step")
    parser.add_argument("--replay_start_step", type=int, default=100, help="Global step to start replay mechanism")

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
    # Standard dataset creation
    # Ensure rollout_step is passed correctly
    if split == "train":
        ds = ERA5TWDatasetforAurora(
                data_root_dir = args.data_root_dir,
                start_date_hour = args.train_start_date_hour,
                end_date_hour = args.train_end_date_hour,
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
    elif split == "val":
        ds = ERA5TWDatasetforAurora(
                data_root_dir = args.data_root_dir,
                start_date_hour = args.val_start_date_hour,
                end_date_hour = args.val_end_date_hour,
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
    else:
        raise Exception("Do not support this dataset split!")
    return ds

def train_epoch(args, model, dataloader, optimizer, scheduler, criterion, accelerator, epoch, train_global_step, replay_buffer):
    _model = accelerator.unwrap_model(model)
    model.train()
    
    total_train_loss = 0.0
    total_train_samples = 0

    dataset = dataloader.dataset
    latitude, longitude = dataset.get_latitude_longitude()
    levels = dataset.get_levels()
    static_data = dataset.get_static_vars_ds()

    pbar = tqdm(dataloader, disable = not accelerator.is_local_main_process, desc = f"train_epoch: {epoch}")

    for batch in pbar:
        # train_label contains [Batch, Time=RolloutStep, Lat, Lon]
        train_input, train_label, train_dates = batch

        # Slice the label into steps: 0 -> T+1, 1 -> T+2
        _label_slices = slice_timeaxis(train_label)

        # ==========================================================
        # 1. Main Training Step (T -> T+1)
        # ==========================================================
        optimizer.zero_grad()
        with accelerator.autocast():
            # Prepare Input (Time T)
            _input = Batch(
                surf_vars = train_input["surf_vars"],
                atmos_vars = train_input["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = latitude, lon = longitude,
                    time = tuple(map(lambda d: pd.Timestamp(d), train_dates)),
                    atmos_levels = levels,
                ),
            )
            
            # Prepare Target (Time T+1) -> Index 0 from slices
            _label_t1 = Batch(
                surf_vars = _label_slices[0]['surf_vars'],
                atmos_vars = _label_slices[0]['atmos_vars'],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = latitude, lon = longitude,
                    time = tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours = args.lead_time), train_dates)),
                    atmos_levels = levels,
                ),
            )

            # Forward
            _pred_t1 = model(_input)
            
            # Loss
            loss_dict = criterion(
                _pred_t1.normalise(surf_stats = _model.surf_stats),
                _label_t1.normalise(surf_stats = _model.surf_stats),
            )
            loss = loss_dict["all_vars"]

        accelerator.backward(loss.mean())
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Logging
        gather_loss = accelerator.gather(loss)
        total_train_loss += gather_loss.sum().item()
        total_train_samples += gather_loss.shape[0]
        step_loss = gather_loss.mean().item()

        # ==========================================================
        # 2. Store in Replay Buffer (Zero-Cost Store)
        # ==========================================================
        # If we have Step 2 data (T+2), use it to populate the buffer
        if 1 in _label_slices:
            # Construct Input for next step: (Model Prediction T+1) + (Time T+1)
            # This is the "Synthetic History" input
            next_input_synthetic = construct_next_input(_input, _pred_t1, args.lead_time)

            # Construct Target for next step: (Ground Truth T+2) -> Index 1 from slices
            # We already loaded this from disk! No extra cost.
            _label_t2 = Batch(
                surf_vars = _label_slices[1]['surf_vars'],
                atmos_vars = _label_slices[1]['atmos_vars'],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = latitude, lon = longitude,
                    time = tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours = args.lead_time * 2), train_dates)),
                    atmos_levels = levels,
                ),
            )
            
            # Put into buffer
            replay_buffer.put(next_input_synthetic, _label_t2)

            if train_global_step % 100 == 0 and accelerator.is_main_process:
                print(f"[DEBUG] Replay Buffer Size: {len(replay_buffer)}")

        # ==========================================================
        # 3. Replay Fine-tuning Step
        # ==========================================================
        if train_global_step > args.replay_start_step:
            for _ in range(args.finetune_rate):
                sample_data = replay_buffer.sample()
                if sample_data is None: break # Buffer empty

                r_input_cpu, r_label_cpu = sample_data
                
                # Move to GPU
                r_input = batch_to_device(r_input_cpu, accelerator.device)
                r_label = batch_to_device(r_label_cpu, accelerator.device)

                optimizer.zero_grad()
                with accelerator.autocast():
                    r_pred = model(r_input)
                    r_loss_dict = criterion(
                        r_pred.normalise(surf_stats = _model.surf_stats),
                        r_label.normalise(surf_stats = _model.surf_stats)
                    )
                    r_loss = r_loss_dict["all_vars"]

                accelerator.backward(r_loss.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        # ==========================================================
        # Logging & End of Step
        # ==========================================================
        if accelerator.is_main_process:
            pbar.set_postfix({"loss": f"{step_loss:.6f}"})
            accelerator.log({
                "train_global_step": train_global_step,
                "train/step_loss": step_loss,
                "lr": optimizer.param_groups[0]["lr"]
            })
        
        train_global_step += 1

    train_epoch_loss = total_train_loss / total_train_samples
    if accelerator.is_main_process:
        accelerator.log({"epoch": epoch, "train/epoch_loss": train_epoch_loss})

    return train_epoch_loss, train_global_step

def val_epoch(args, model, dataloader, criterion, accelerator, epoch, val_global_step):
    # Standard Validation (Unchanged)
    _model = accelerator.unwrap_model(model)
    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0
    
    dataset = dataloader.dataset
    latitude, longitude = dataset.get_latitude_longitude()
    levels = dataset.get_levels()
    static_data = dataset.get_static_vars_ds()

    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"val_epoch: {epoch}")

    with torch.inference_mode():
        for batch in pbar:
            val_input, val_label, val_dates = batch
            
            # Val likely also has rollout labels if using same dataset class
            # We just take the first step for standard validation metric, or loop if you want average
            _label_slices = slice_timeaxis(val_label)
            
            with accelerator.autocast():
                _input = Batch(
                    surf_vars = val_input["surf_vars"],
                    atmos_vars = val_input["atmos_vars"],
                    static_vars = static_data["static_vars"],
                    metadata = Metadata(
                        lat = latitude, lon = longitude,
                        time = tuple(map(lambda d: pd.Timestamp(d), val_dates)),
                        atmos_levels = levels,
                    ),
                )
                
                # Validation against first step (T+1)
                _label = Batch(
                    surf_vars = _label_slices[0]['surf_vars'],
                    atmos_vars = _label_slices[0]['atmos_vars'],
                    static_vars = static_data["static_vars"],
                    metadata = Metadata(
                        lat = latitude, lon = longitude,
                        time = tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours = args.lead_time), val_dates)),
                        atmos_levels = levels,
                    ),
                )
                _pred = model(_input)
                loss_dict = criterion(
                    _pred.normalise(surf_stats = _model.surf_stats),
                    _label.normalise(surf_stats = _model.surf_stats)
                )
                loss = loss_dict["all_vars"]

            gather_val_loss = accelerator.gather(loss)
            total_val_loss += gather_val_loss.sum().item()
            total_val_samples += gather_val_loss.shape[0]
            step_loss = gather_val_loss.mean().item()

            if accelerator.is_main_process:
                pbar.set_postfix({"val_step_loss": f"{step_loss:.8f}"})
                accelerator.log({"val_global_step": val_global_step, "val/step_loss": step_loss})
            val_global_step += 1

    val_epoch_loss = total_val_loss / total_val_samples
    if accelerator.is_main_process:
        accelerator.log({"epoch": epoch, "val/epoch_loss": val_epoch_loss})

    return val_epoch_loss, val_global_step

# ... [Save Checkpoint functions remain identical to your provided code] ...
def save_checkpoint_by_epoch(args, accelerator, output_dir, epoch):
    output_dir = Path(output_dir)
    if args.checkpointing_epochs > 0 and epoch % args.checkpointing_epochs == 0:
        if accelerator.is_main_process:
            checkpoints = sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")], key=lambda x: int(x.name.split("-")[1]))
            if args.checkpoints_total_limit is not None and len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                for removing_checkpoint in checkpoints[:num_to_remove]:
                    shutil.rmtree(removing_checkpoint)
                    logger.info(f"Removed old checkpoint: {removing_checkpoint}")
            save_path = output_dir / f"checkpoint-{epoch}"
            save_path.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(str(save_path))
            logger.info(f"Saved checkpoint to {save_path}")

def save_checkpoint_best_by_val_loss(args, accelerator, output_dir, epoch, train_loss, val_loss, best_ckpts):
    output_dir = Path(output_dir)
    if len(best_ckpts) < args.save_top_k or val_loss < max(best_ckpts, key=lambda x: x[0])[0]:
        save_path = output_dir / f"{epoch}-train_loss={train_loss:.8f}-val_loss={val_loss:.8f}"
        save_path.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(save_path))
        logger.info(f"Saved new best checkpoint: {save_path}")
        best_ckpts.append((val_loss, save_path))
        best_ckpts.sort(key=lambda x: x[0])
        while len(best_ckpts) > args.save_top_k:
            worst = best_ckpts.pop()
            shutil.rmtree(worst[1], ignore_errors=True)
    return best_ckpts

def main():
    args = parse_args()
    
    # Assert Rollout Step is sufficient for replay
        
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    logging_dir = output_dir / args.logging_dir
    ckpt_dir = output_dir / "ckpts"
    ckpt_dir.mkdir(parents = True, exist_ok = True)

    accelerator_project_config = ProjectConfiguration(
        project_dir = args.output_dir,
        logging_dir = logging_dir,
    )
    
    # Initialize Buffer
    replay_buffer = ReplayBuffer(max_size=args.replay_buffer_size)

    accelerator = Accelerator(
        mixed_precision = args.mixed_precision,
        log_with = args.report_to,
        project_config = accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            config = tracker_config,
            init_kwargs = {"wandb": {"name": args.wandb_name}},
        )
        if args.report_to == "wandb":
            wandb.run.define_metric("train/step_loss", step_metric="train_global_step")

    logger.info(accelerator.state)

    if args.rollout_step < 2:
        logger.warning(f"Rollout step is {args.rollout_step}. Replay buffer requires at least 2 steps (T+1 and T+2). Replay will be disabled.")


    model = create_model(args)
    
    # Dataset Creation
    train_dataset = create_dataset(args, "train")
    val_dataset = create_dataset(args, "val")

    # OPTIMIZATION: Use persistent_workers and prefetch_factor
    train_loader = DataLoader(
        train_dataset, batch_size = args.train_batch_size, shuffle = True,
        num_workers = args.num_workers, pin_memory = True,
        persistent_workers = True, 
        prefetch_factor = 2
    )
    val_loader = DataLoader(
        val_dataset, batch_size = args.val_batch_size, shuffle = False,
        num_workers = args.num_workers, pin_memory = True,
        persistent_workers = True,
        prefetch_factor = 2
    )

    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    criterion = AuroraMAELoss

    total_training_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_step_ratio * total_training_steps)

    scheduler = get_scheduler_with_warmup(
        optimizer, warmup_steps = warmup_steps, training_steps = total_training_steps, schedule_type = "cosine",
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler,
    )

    train_global_step = 0
    val_global_step = 0
    best_checkpoints = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_global_step = train_epoch(
            args, model, train_loader, optimizer, scheduler, criterion, accelerator,
            epoch, train_global_step, replay_buffer
        )
        val_loss, val_global_step = val_epoch(
            args, model, val_loader, criterion, accelerator, epoch, val_global_step
        )

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            save_checkpoint_by_epoch(args, accelerator, ckpt_dir, epoch)
            best_checkpoints = save_checkpoint_best_by_val_loss(args, accelerator, ckpt_dir, epoch, train_loss, val_loss, best_checkpoints)
        
        accelerator.wait_for_everyone()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    accelerator.end_training()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force = True)
    main()