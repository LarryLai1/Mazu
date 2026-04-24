#!/usr/bin/env python
# coding=utf-8

from pathlib import Path
import argparse
import pandas as pd
import logging
import shutil

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from safetensors.torch import load_file

from aurora import Batch, Metadata
from aurora.model.aurora import AuroraSmall
from pangu_lib.models.pangu import PanguModel
from pangu_lib.adapter import PanguAuroraAdapter

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

    return parser.parse_args()

# def create_model(
#         args,
#     ):

#     model = AuroraSmall(
#         use_lora = args.use_lora,
#         bf16_mode = args.bf16_mode,
#         timestep = pd.Timedelta(hours = args.timestep_hours),
#         stabilise_level_agg = args.stabilise_level_agg,
#     )

#     if args.use_pretrained_weight:
#         logger.info("Loading pretrained weights provided by Microsoft Aurora...")
#         model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt", strict = False)
#     elif args.checkpoint_path:
#         logger.info(f"Loading checkpoint: {args.checkpoint_path}")
#         state_dict = load_file(args.checkpoint_path)
#         model.load_state_dict(state_dict, strict = False)
#     return model


def create_model(args):
    # 1. Initialize the Base Pangu Model
    # Note: You need to set these shapes to match your ERA5TWDatasetforAurora args
    pangu_base = PanguModel(
        data_spatial_shape = (len(args.levels), int(abs(args.latitude[1] - args.latitude[0]) / 0.25) + 1, int(abs(args.longitude[1] - args.longitude[0]) / 0.25) + 1),
        upper_vars = len(args.upper_variables),
        surface_vars = len(args.surface_variables),
        depths =[2, 6],       # Standard Pangu Config
        heads = [6, 12],      # Standard Pangu Config
        embed_dim = 192,      # Standard Pangu Config
        patch_shape = (2, 4, 4),
        window_size = (2, 6, 12),
        constant_mask_paths = ["/work/yunye0121/Mazu/pangu_lib/static_npy_data/lsm.npy", "/work/yunye0121/Mazu/pangu_lib/static_npy_data/slt.npy", "/work/yunye0121/Mazu/pangu_lib/static_npy_data/z.npy"], # Or provide paths if you have them
        smoothing_kernel_size = None,
        segmented_smooth = False,
        segmented_smooth_boundary_width = None,
        learned_smooth = False,
        residual = False,
        res_conn_after_smooth = False,
    )

    # 2. Load Surface Stats for Aurora Normalization
    # You likely have a JSON or Dict used for normalization in your dataset.
    # The adapter needs this so `unwrapped_model.surf_stats` works in the training loop.
    # Replace this with however you load your stats in the main script.
    # surf_stats_dummy = {
    #     "2t": (280.0, 10.0), "10u": (0.0, 5.0), 
    #     "10v": (0.0, 5.0), "msl": (101325.0, 1000.0)
    # } 

    surf_stats_dummy = {}

    # 3. Wrap it
    model = PanguAuroraAdapter(pangu_base, surf_stats = surf_stats_dummy)

    # 4. Load weights if needed
    if args.checkpoint_path:
        # Note: Be careful loading weights. If loading standard Pangu .onnx/.pth, 
        # you load into model.model. If loading a checkpoint saved by THIS script,
        # you load into model directly.
        state_dict = load_file(args.checkpoint_path)
        model.load_state_dict(state_dict, strict = False)

    return model


def create_dataset(args, split):
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

def save_checkpoint_by_epoch(args, accelerator, output_dir, epoch):
    output_dir = Path(output_dir)

    if args.checkpointing_epochs > 0 and epoch % args.checkpointing_epochs == 0:
        if accelerator.is_main_process:
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
            save_path.mkdir(parents = True, exist_ok = True)
            accelerator.save_state(str(save_path))
            logger.info(f"Saved checkpoint to {save_path}")

def save_checkpoint_best_by_val_loss(
    args,
    accelerator,
    output_dir: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_ckpts: list,
):
    output_dir = Path(output_dir)

    if len(best_ckpts) < args.save_top_k or val_loss < max(best_ckpts, key = lambda x: x[0])[0]:
        save_path = output_dir / f"{epoch}-train_loss={train_loss:.8f}-val_loss={val_loss:.8f}"
        save_path.mkdir(parents = True, exist_ok = True)
        accelerator.save_state(str(save_path))
        logger.info(f"Saved new best checkpoint: {save_path} (train_loss={train_loss:.8f} val_loss={val_loss:.8f})")

        best_ckpts.append((val_loss, save_path))
        best_ckpts.sort(key = lambda x: x[0])

        while len(best_ckpts) > args.save_top_k:
            worst = best_ckpts.pop()
            shutil.rmtree(worst[1], ignore_errors = True)
            logger.info(f"Removed older/worse checkpoint: {worst[1]}")

    return best_ckpts

def train_epoch(
        args,
        model,
        dataloader,
        optimizer,
        scheduler,
        criterion,
        accelerator,
        epoch,
        train_global_step,
    ):

    unwrapped_model = accelerator.unwrap_model(model)

    model.train()
    
    total_train_loss = 0.0
    total_train_samples = 0

    latitude, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    pbar = tqdm(
        dataloader, 
        disable = not accelerator.is_local_main_process, 
        desc = f"train_epoch: {epoch}",
        # ncols = 120,
    )

    for batch in pbar:

        train_input, train_label, train_dates = batch

        assert next(iter(train_input["surf_vars"].values())).shape[0] == next(iter(train_input["atmos_vars"].values())).shape[0] \
            == next(iter(train_label["surf_vars"].values())).shape[0] == next(iter(train_label["atmos_vars"].values())).shape[0]

        optimizer.zero_grad()
        with accelerator.autocast():
            
            _input = Batch(
                surf_vars = train_input["surf_vars"],
                atmos_vars = train_input["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = latitude,
                    lon = longitude,
                    time = tuple(map(lambda d: pd.Timestamp(d), train_dates)),
                    atmos_levels = levels,
                ),
            )

            _label = Batch(
                surf_vars = train_label['surf_vars'],
                atmos_vars = train_label['atmos_vars'],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = latitude,
                    lon = longitude,
                    time = tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours = args.lead_time), train_dates)),
                    atmos_levels = levels,
                ),
            )

            _pred = model(_input)

            loss_dict = criterion(
                _pred.normalise(surf_stats = unwrapped_model.surf_stats),
                _label.normalise(surf_stats = unwrapped_model.surf_stats),
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
            accelerator.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

        optimizer.step()
        scheduler.step()
        gather_train_loss = accelerator.gather( loss )

        total_train_loss += gather_train_loss.sum().item()
        total_train_samples += gather_train_loss.shape[0]

        current_lr = optimizer.param_groups[0]["lr"]
        step_loss = gather_train_loss.mean().item()

        if accelerator.is_main_process:
            pbar.set_postfix({
                "train_step_loss": f"{step_loss:.8f}",
            })
            accelerator.log(
                {
                    "train_global_step": train_global_step,
                    "train/step_loss": step_loss,
                    "lr": current_lr,
                    "grad_norm": total_grad_norm,
                },
            )
        
        train_global_step += 1

    train_epoch_loss = total_train_loss / total_train_samples
    
    if accelerator.is_main_process:
        accelerator.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_epoch_loss,
            },
        )

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

    accelerator_project_config = ProjectConfiguration(
        project_dir = args.output_dir,
        logging_dir = logging_dir,
    )

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
            run = wandb.run
            run.define_metric("train/step_loss", step_metric = "train_global_step")
            run.define_metric("val/step_loss", step_metric = "val_global_step")
            run.define_metric("train/epoch_loss", step_metric = "epoch")
            run.define_metric("val/epoch_loss", step_metric = "epoch")

    logger.info(accelerator.state)

    model = create_model(args)
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

    optimizer = AdamW(
        model.parameters(),
        lr = args.lr,
        weight_decay = args.weight_decay,
    )

    criterion = AuroraMAELoss

    total_training_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_step_ratio * total_training_steps)

    scheduler = get_scheduler_with_warmup(
        optimizer,
        warmup_steps = warmup_steps,
        training_steps = total_training_steps,
        schedule_type = "cosine",
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler,
    )

    train_global_step = 0
    val_global_step = 0
    best_checkpoints = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_global_step = train_epoch(
            args,
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            accelerator,
            epoch,
            train_global_step
        )
        val_loss, val_global_step = val_epoch(
            args,
            model,
            val_loader,
            criterion,
            accelerator,
            epoch,
            val_global_step
        )

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            logger.info(f"epoch {epoch} - train_loss: {train_loss:.8f}")
            logger.info(f"epoch {epoch} - val_loss: {val_loss:.8f}")
            save_checkpoint_by_epoch(
                args,
                accelerator,
                ckpt_dir,
                epoch,
            )
            best_checkpoints = save_checkpoint_best_by_val_loss(
                args,
                accelerator,
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
