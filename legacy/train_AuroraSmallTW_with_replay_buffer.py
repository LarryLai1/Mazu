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
from torch.utils.data import DataLoader, Dataset
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
# 1. New Classes for Replay Mechanism
# ==========================================

class ReplayBuffer:
    """
    Stores (Input_Batch, Target_Indices) tuples.
    Fixed size to keep memory usage constant (FengWu usually keeps this small-ish, e.g. 1000).
    """
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def put(self, item):
        # item = (Predicted_Batch, Target_Dataset_Indices_List)
        self.buffer.append(item)

    def sample(self):
        return random.choice(self.buffer)

    def __len__(self):
        return len(self.buffer)

class IndexedDataset(Dataset):
    """
    Wraps your existing dataset to return the INDEX along with data.
    Essential for identifying 'Time T' so we can find 'Time T+1' even when shuffling.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data = self.dataset[index]
        return data, index

    def __len__(self):
        return len(self.dataset)
    
    # Pass-through methods for Aurora dataset attributes
    def get_latitude_longitude(self):
        return self.dataset.get_latitude_longitude()
    def get_levels(self):
        return self.dataset.get_levels()
    def get_static_vars_ds(self):
        return self.dataset.get_static_vars_ds()

# ==========================================
# 2. Helper Functions
# ==========================================

def detach_aurora_batch(batch_obj):
    """
    Manually detaches tensors inside an Aurora Batch object.
    Required because the custom Batch class likely doesn't implement .detach().
    """
    # 1. Detach Surface Variables
    new_surf = {k: v.detach() for k, v in batch_obj.surf_vars.items()}
    
    # 2. Detach Atmospheric Variables
    new_atmos = {k: v.detach() for k, v in batch_obj.atmos_vars.items()}
    
    # 3. Static vars usually don't need grad, but safe to detach if they exist
    new_static = None
    if batch_obj.static_vars is not None:
         new_static = {k: v.detach() for k, v in batch_obj.static_vars.items()}

    # 4. Reconstruct Batch
    # We reuse metadata as-is
    return Batch(
        surf_vars = new_surf,
        atmos_vars = new_atmos,
        static_vars = new_static,
        metadata = batch_obj.metadata
    )

def construct_next_input(current_input_batch, prediction_batch, lead_time_hours):
    """
    Implements the sliding window logic.
    It takes the PREDICTION and makes it the NEW INPUT for the next step.
    """
    # 1. Update Time Metadata
    new_metadata = Metadata(
        lat = current_input_batch.metadata.lat,
        lon = current_input_batch.metadata.lon,
        time = tuple(t + pd.Timedelta(hours=lead_time_hours) for t in current_input_batch.metadata.time),
        atmos_levels = current_input_batch.metadata.atmos_levels,
    )
    
    # 2. Create Next Batch
    # Uses the prediction as the new atmospheric/surface state
    next_input = Batch(
        surf_vars = prediction_batch.surf_vars,
        atmos_vars = prediction_batch.atmos_vars,
        static_vars = current_input_batch.static_vars, # Static vars persist
        metadata = new_metadata
    )
    return next_input

def collate_replay_batch(dataset_items, dataset_obj, lead_time):
    """
    Manually stacks list of (input, label, date) tuples into an Aurora Batch.
    Used for the Replay Step when we fetch random rows from the dataset.
    """
    if not dataset_items: return None

    # Unzip the list of tuples: [(inputs, labels, dates), ...]
    inputs_list, labels_list, dates_list = zip(*dataset_items)
    
    # Helper to stack dictionary of tensors
    def stack_dict(dict_list):
        if not dict_list: return {}
        keys = dict_list[0].keys()
        return {k: torch.stack([d[k] for d in dict_list]) for k in keys}

    # Stack Labels (We only strictly need labels for the Replay Step as Ground Truth)
    lb_surf = stack_dict([item['surf_vars'] for item in labels_list])
    lb_atmos = stack_dict([item['atmos_vars'] for item in labels_list])
    
    # Process Dates
    flat_dates = []
    for d in dates_list:
        # If dates is a list (window), take the last one which implies the target time base
        flat_dates.append(d[-1] if isinstance(d, (list, tuple)) else d)

    lat, lon = dataset_obj.get_latitude_longitude()
    levels = dataset_obj.get_levels()
    static = dataset_obj.get_static_vars_ds()
    
    # We construct the LABEL batch (the Ground Truth for the Replay Step)
    _label = Batch(
        surf_vars=lb_surf,
        atmos_vars=lb_atmos,
        static_vars=static['static_vars'],
        metadata=Metadata(
            lat=lat, lon=lon, 
            time=tuple(pd.Timestamp(d) + pd.Timedelta(hours=lead_time) for d in flat_dates), 
            atmos_levels=levels
        )
    )
    return _label

# ==========================================
# 3. Main Logic
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description = "Aurora Training Script (HF Style)")
    
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
    parser.add_argument("--rollout_step", type = int, required = True)
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
    
    # --- FengWu Replay Args ---
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
    # Select dataset class based on split/args
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
        # CRITICAL: Wrap train dataset to return indices for Replay Logic
        ds = IndexedDataset(ds)
        
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

    # We need access to the underlying dataset to fetch by index
    # dataloader.dataset is IndexedDataset -> .dataset is ERA5TWDataset
    real_dataset = dataloader.dataset.dataset 
    latitude, longitude = real_dataset.get_latitude_longitude()
    levels = real_dataset.get_levels()
    static_data = real_dataset.get_static_vars_ds()

    pbar = tqdm(dataloader, disable = not accelerator.is_local_main_process, desc = f"train_epoch: {epoch}")

    for batch_data, batch_indices in pbar:
        # batch_data is (train_input, train_label, train_dates)
        train_input, train_label, train_dates = batch_data

        # --- 1. Standard Training Step ---
        optimizer.zero_grad()
        with accelerator.autocast():
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
            _label = Batch(
                surf_vars = train_label['surf_vars'],
                atmos_vars = train_label['atmos_vars'],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = latitude, lon = longitude,
                    time = tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours = args.lead_time), train_dates)),
                    atmos_levels = levels,
                ),
            )

            _pred = model(_input)
            loss_dict = criterion(
                _pred.normalise(surf_stats = _model.surf_stats),
                _label.normalise(surf_stats = _model.surf_stats),
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

        # --- 2. Store in Replay Buffer ---
        # FengWu: Store (Input + Prediction) and Next_Index
        
        # Calculate next indices (Assume sequential dataset: next is idx+1)
        next_indices = [(idx.item() + 1) for idx in batch_indices]
        
        # Check validity (ensure we don't go past dataset length)
        # Note: We detach prediction to save memory (no gradient for storage)
        if all(idx < len(real_dataset) for idx in next_indices):
            # _pred_detached = _pred.detach()
            _pred_detached = detach_aurora_batch(_pred)
            next_input_synthetic = construct_next_input(_input, _pred_detached, args.lead_time)
            replay_buffer.put((next_input_synthetic, next_indices))

            # [PRINT 1] Check storage (Only print occasionally to avoid spam, e.g., every 10 steps)
            if train_global_step % 10 == 0 and accelerator.is_main_process:
                print(f"\n[DEBUG] Step {train_global_step}: Stored item in Replay Buffer. Current Size: {len(replay_buffer)}")

        # --- 3. Replay Fine-tuning ---
        if train_global_step > args.replay_start_step and len(replay_buffer) > 0:

            if accelerator.is_main_process and train_global_step % 10 == 0:
                print(f"[DEBUG] Step {train_global_step}: >>> TRIGGERING REPLAY FINE-TUNING <<<")

            for _ in range(args.finetune_rate):
                # A. Sample a synthetic batch from buffer

                r_input, r_target_indices = replay_buffer.sample()

                # B. Fetch Real Targets for these indices
                valid_items = []
                valid_indices = []
                for idx in r_target_indices:
                    if idx < len(real_dataset):
                        valid_items.append(real_dataset[idx])
                        valid_indices.append(idx)
                
                if not valid_items: continue

                # C. Collate Real Targets into Batch
                r_label = collate_replay_batch(valid_items, real_dataset, args.lead_time)
                
                # D. Move Label to correct device (Model device)
                device = next(model.parameters()).device
                def to_device(d): return {k: v.to(device) for k, v in d.items()}
                
                r_label.surf_vars = to_device(r_label.surf_vars)
                r_label.atmos_vars = to_device(r_label.atmos_vars)
                
                # Ensure input is on device (if not already)
                r_input.surf_vars = to_device(r_input.surf_vars)
                r_input.atmos_vars = to_device(r_input.atmos_vars)

                # E. Replay Train Step
                optimizer.zero_grad()
                with accelerator.autocast():
                    r_pred = model(r_input) # Predict based on synthetic history
                    
                    r_loss_dict = criterion(
                        r_pred.normalise(surf_stats = _model.surf_stats),
                        r_label.normalise(surf_stats = _model.surf_stats)
                    )
                    r_loss = r_loss_dict["all_vars"]

                accelerator.backward(r_loss.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                # F. Recursive Push (The FengWu Magic)
                # We predicted T+X, now push T+X so we can train on T+X+1 later
                r_next_indices = [i + 1 for i in valid_indices]
                if all(i < len(real_dataset) for i in r_next_indices):
                    # r_pred_detached = r_pred.detach()
                    r_pred_detached = detach_aurora_batch(r_pred)
                    r_next_input = construct_next_input(r_input, r_pred_detached, args.lead_time)
                    replay_buffer.put((r_next_input, r_next_indices))
        
        # --- End Replay ---

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
    # Standard Validation Loop (Unchanged)
    _model = accelerator.unwrap_model(model)
    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0
    
    # Access dataset directly since Val is not wrapped in IndexedDataset
    dataset = dataloader.dataset
    latitude, longitude = dataset.get_latitude_longitude()
    levels = dataset.get_levels()
    static_data = dataset.get_static_vars_ds()

    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"val_epoch: {epoch}")

    with torch.inference_mode():
        for batch in pbar:
            val_input, val_label, val_dates = batch
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
                _label = Batch(
                    surf_vars = val_label['surf_vars'],
                    atmos_vars = val_label['atmos_vars'],
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

    model = create_model(args)
    
    # Dataset Creation (Now wrapped in IndexedDataset inside function)
    train_dataset = create_dataset(args, "train")
    val_dataset = create_dataset(args, "val")

    train_loader = DataLoader(
        train_dataset, batch_size = args.train_batch_size, shuffle = True,
        num_workers = args.num_workers, pin_memory = True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size = args.val_batch_size, shuffle = False,
        num_workers = args.num_workers, pin_memory = True,
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