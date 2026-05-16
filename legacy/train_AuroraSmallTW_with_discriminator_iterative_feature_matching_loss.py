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

from discriminator.ResNet_discriminator import ResNetDiscriminator

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

    parser.add_argument(
        "--use_feature_matching_loss",
        action = "store_true",
    )
    parser.add_argument(
        "--fm_lambda",
        type = float,
        default = 0.01,
    )
    parser.add_argument(
        "--discriminator_checkpoint_path",
        type = str,
        default = None,
    )
    parser.add_argument(
        "--target_layers", type = str, nargs = "+", default = [
            "backbone.layer1", "backbone.layer2", "backbone.layer3", "backbone.layer4",
        ],
    )

    parser.add_argument(
        "--discriminator_refresh_epochs",
        type=int,
        default=5,
        help="Train new discriminator every N epochs (0 to disable refresh)",
    )
    parser.add_argument(
        "--discriminator_train_epochs",
        type=int,
        default=3,
        help="Number of epochs to train each new discriminator",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=1e-4,
        help="Learning rate for training new discriminators",
    )
    parser.add_argument(
        "--discriminator_batch_size",
        type=int,
        default=32,
        help="Batch size for training discriminators",
    )
    parser.add_argument(
        "--discriminator_shuffle_samples",
        action="store_true",
        help="Shuffle real/fake sample order during discriminator training to prevent position bias",
    )

    parser.add_argument("--freeze_encoder", action = "store_true")
    parser.add_argument("--freeze_backbone", action = "store_true")
    parser.add_argument("--freeze_decoder", action = "store_true")

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

def create_model(
        args,
    ):

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

    # 1. Freeze the Encoder (The "Sensor")
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    

    # 2. Freeze the Backbone (The "Physics")
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # 3. Ensure Decoder is Trainable (The "Generator")
    if args.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False

    # 4. Optional: Log parameter counts to verify
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({(trainable_params/total_params):.2%})")
    return model

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, target_layers):
        super().__init__()
        self.model = model
        self.target_layers = target_layers
        self._features = {
            layer: torch.empty(0) for layer in target_layers
        }
        
        # Register hooks
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(self._save_outputs_hook(name))

    def _save_outputs_hook(self, layer_name):
        def hook(module, input, output):
            self._features[layer_name] = output
        return hook

    def forward(self, x):
        # self._features = {}
        # We ignore the final output (the binary score)
        self.model(x) 
        # return self._features
        return self._features.copy()

def create_discriminator(args):
    discriminator = ResNetDiscriminator(
        surface_variables = args.surface_variables,
        upper_variables = args.upper_variables,
        levels = args.levels,
        backbone_name = "resnet50",
        pretrained = False,
    )

    # print("Discriminator Model:")
    # for name, module in discriminator.named_modules():
    #     print(name) # Look for "layer2", "layer3", etc.

    if args.discriminator_checkpoint_path is not None:
        logger.info(f"Loading discriminator checkpoint from {args.discriminator_checkpoint_path}")
        state_dict = load_file(args.discriminator_checkpoint_path)
        discriminator.load_state_dict(state_dict, strict = False)

    for p in discriminator.parameters():
        p.requires_grad = False

    discriminator.eval()

    # return discriminator
    target_layers = args.target_layers
    feature_discriminator = FeatureExtractor(discriminator, target_layers)
    
    feature_discriminator.eval()

    return feature_discriminator


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

def concatenate_batches(batch1, batch2):
    """
    Concatenate two Aurora Batch objects along the batch dimension.
    This allows single forward pass through discriminator to avoid BatchNorm issues.
    
    Args:
        batch1: First Aurora Batch (e.g., real data)
        batch2: Second Aurora Batch (e.g., fake data)
    
    Returns:
        Combined Aurora Batch with doubled batch size
    """
    from aurora import Batch, Metadata
    
    # Concatenate surface variables
    combined_surf_vars = {}
    for key in batch1.surf_vars.keys():
        combined_surf_vars[key] = torch.cat([batch1.surf_vars[key], batch2.surf_vars[key]], dim=0)
    
    # Concatenate atmospheric variables
    combined_atmos_vars = {}
    for key in batch1.atmos_vars.keys():
        combined_atmos_vars[key] = torch.cat([batch1.atmos_vars[key], batch2.atmos_vars[key]], dim=0)
    
    # Concatenate static variables
    # combined_static_vars = {}
    # for key in batch1.static_vars.keys():
    #     combined_static_vars[key] = torch.cat([batch1.static_vars[key], batch2.static_vars[key]], dim=0)
    
    combined_static_vars = batch1.static_vars  # Assume static vars are the same for both batches

    # Combine metadata (concatenate time tuples)
    # combined_metadata = Metadata(
    #     lat=batch1.metadata.lat,  # Same for both
    #     lon=batch1.metadata.lon,  # Same for both
    #     time=batch1.metadata.time + batch2.metadata.time,  # Concatenate tuples
    #     atmos_levels=batch1.metadata.atmos_levels,  # Same for both
    # )
    combined_metadata = batch1.metadata  # Keep metadata from the first batch (assumed same for both)

    return Batch(
        surf_vars=combined_surf_vars,
        atmos_vars=combined_atmos_vars,
        static_vars=combined_static_vars,
        metadata=combined_metadata,
    )

def train_new_discriminator(
    model,
    train_loader,
    val_loader,
    args,
    accelerator,
    disc_generation,
):
    """
    Train a fresh discriminator to distinguish real data from current model outputs.
    
    Args:
        model: Current Aurora model (frozen during this phase)
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Training arguments
        accelerator: Accelerator instance
        disc_generation: Current discriminator generation number
    
    Returns:
        Trained and frozen discriminator wrapped in FeatureExtractor
    """
    logger.info(f"=" * 80)
    logger.info(f"Training new discriminator (Generation {disc_generation})")
    logger.info(f"=" * 80)
    
    # Create fresh discriminator
    new_discriminator = ResNetDiscriminator(
        surface_variables = args.surface_variables,
        upper_variables = args.upper_variables,
        levels = args.levels,
        backbone_name = "resnet50",
        pretrained = False,
    )
    
    # Make it trainable
    for p in new_discriminator.parameters():
        p.requires_grad = True
    
    # Setup optimizer
    disc_optimizer = AdamW(
        new_discriminator.parameters(),
        lr = args.discriminator_lr,
        weight_decay = args.weight_decay,
    )
    
    # Prepare with accelerator
    new_discriminator, disc_optimizer = accelerator.prepare(
        new_discriminator, disc_optimizer,
    )
    
    # Freeze the generator model during discriminator training
    unwrapped_model = accelerator.unwrap_model(model)
    model.eval()
    
    # Get dataset info
    latitude, longitude = train_loader.dataset.get_latitude_longitude()
    levels = train_loader.dataset.get_levels()
    static_data = train_loader.dataset.get_static_vars_ds()
    

    # Train discriminator for specified epochs
    for disc_epoch in range(1, args.discriminator_train_epochs + 1):
        new_discriminator.train()
        total_disc_loss = 0.0
        total_samples = 0

        all_probs = []
        all_labels = []
        
        pbar = tqdm(
            train_loader,
            disable = not accelerator.is_local_main_process,
            desc = f"Disc Gen {disc_generation} - Epoch {disc_epoch} / {args.discriminator_train_epochs}",
        )
        
        for batch in pbar:
            train_input, train_label, train_dates = batch
            
            disc_optimizer.zero_grad()
            
            with accelerator.autocast():
                # Prepare batches
                _input = Batch(
                    surf_vars=train_input["surf_vars"],
                    atmos_vars=train_input["atmos_vars"],
                    static_vars=static_data["static_vars"],
                    metadata=Metadata(
                        lat=latitude,
                        lon=longitude,
                        time=tuple(map(lambda d: pd.Timestamp(d), train_dates)),
                        atmos_levels=levels,
                    ),
                )
                
                _label = Batch(
                    surf_vars=train_label['surf_vars'],
                    atmos_vars=train_label['atmos_vars'],
                    static_vars=static_data["static_vars"],
                    metadata=Metadata(
                        lat=latitude,
                        lon=longitude,
                        time=tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours=args.lead_time), train_dates)),
                        atmos_levels=levels,
                    ),
                )
                
                # Generate fake data (detached from model)
                with torch.no_grad():
                    _pred = model(_input)
                
                # print("_label and _pred shapes:")
                # print(f"_label.surf_vars.2t.shape: {_label.surf_vars['2t'].shape}")
                # print(f"_pred.surf_vars.2t.shape: {_pred.surf_vars['2t'].shape}")
                # print(f"_label.atmos_vars.u.shape: {_label.atmos_vars['u'].shape}")
                # print(f"_pred.atmos_vars.u.shape: {_pred.atmos_vars['u'].shape}")

                # Discriminator predictions
                # real_score = new_discriminator(_label).view(-1)
                # fake_score = new_discriminator(_pred).view(-1)
                original_batch_size = next(iter(_label.surf_vars.values())).shape[0]
                combined_batch = concatenate_batches(_label, _pred)
                combined_scores = new_discriminator(combined_batch).view(-1)

                # print(f"{original_batch_size=}")

                # batch_size = combined_scores.shape[0] // 2
                combined_labels = torch.cat([
                    torch.ones(original_batch_size, device=combined_scores.device),
                    torch.zeros(original_batch_size, device=combined_scores.device),
                ])
                
                if args.discriminator_shuffle_samples:
                    perm = torch.randperm(combined_scores.shape[0], device = combined_scores.device)
                    combined_scores = combined_scores[perm]
                    combined_labels = combined_labels[perm]

                # Single BCE loss computation
                disc_criterion = torch.nn.BCEWithLogitsLoss(reduction = "mean")
                disc_loss = disc_criterion(
                    combined_scores, combined_labels,
                )

                # print(f"{real_score.shape=}, {fake_score.shape=}")
                
                # # Binary cross-entropy loss
                # # Real data should be classified as 1, fake as 0
                # real_labels = torch.ones_like(real_score)
                # fake_labels = torch.zeros_like(fake_score)
                
                # real_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                #     real_score, real_labels
                # )
                # fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                #     fake_score, fake_labels
                # )
                
                # disc_loss = (real_loss + fake_loss) / 2
            
            accelerator.backward(disc_loss)
            accelerator.clip_grad_norm_(new_discriminator.parameters(), args.max_grad_norm)
            disc_optimizer.step()
            
            # Logging
            gather_disc_loss = accelerator.gather(disc_loss)
            total_disc_loss += gather_disc_loss.sum().item()
            total_samples += gather_disc_loss.shape[0]
            
            # Calculate accuracies for monitoring
            # Apply sigmoid to convert logits to probabilities
            combined_probs = torch.sigmoid(combined_scores)
            gather_probs = accelerator.gather(combined_probs)
            gather_labels = accelerator.gather(combined_labels)
            all_probs.append(gather_probs.cpu())
            all_labels.append(gather_labels.cpu())
            
            if accelerator.is_main_process:
                
                # gather_batch_size = gather_probs.shape[0] // 2
                
                # with torch.no_grad():
                #     real_probs = gather_probs[: gather_batch_size]
                #     fake_probs = gather_probs[gather_batch_size :]
                    # Real samples should have prob > 0.5 (close to 1)
                    # Fake samples should have prob < 0.5 (close to 0)
                    # real_acc = (real_probs > 0.5).float().mean().item()
                    # fake_acc = (fake_probs < 0.5).float().mean().item()
                
                pbar.set_postfix({
                    "disc_loss": f"{gather_disc_loss.mean().item():.6f}",
                    # "real_acc": f"{real_acc:.3f}",
                    # "fake_acc": f"{fake_acc:.3f}",
                })
            # if accelerator.is_main_process:
            #     # Calculate accuracies for monitoring
            #     with torch.no_grad():
            #         real_scores = combined_scores[:batch_size]
            #         fake_scores = combined_scores[batch_size:]
            #         real_acc = (real_scores > 0).float().mean().item()
            #         fake_acc = (fake_scores < 0).float().mean().item()
                
            #     pbar.set_postfix({
            #         "disc_loss": f"{gather_disc_loss.mean().item():.6f}",
            #         "real_acc": f"{real_acc:.3f}",
            #         "fake_acc": f"{fake_acc:.3f}",
            #     })
            # if accelerator.is_main_process:
            #     pbar.set_postfix({
            #         "disc_loss": f"{gather_disc_loss.mean().item():.6f}",
            #         "real_acc": f"{(real_score > 0).float().mean().item():.3f}",
            #         "fake_acc": f"{(fake_score < 0).float().mean().item():.3f}",
            #     })
        
        epoch_disc_loss = total_disc_loss / total_samples
        
        if accelerator.is_main_process:
            # Concatenate all predictions and labels from the epoch
            all_probs = torch.cat(all_probs)
            all_labels = torch.cat(all_labels)
            
            # Binarize predictions (threshold = 0.5)
            pred_labels = (all_probs >= 0.5).long()
            true_labels = all_labels.long()
            
            # Calculate metrics using sklearn (like your reference code)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            epoch_acc = accuracy_score(true_labels, pred_labels)
            epoch_precision = precision_score(true_labels, pred_labels, zero_division=0)
            epoch_recall = recall_score(true_labels, pred_labels, zero_division=0)
            epoch_f1 = f1_score(true_labels, pred_labels, zero_division=0)
            
            logger.info(
                f"Disc Gen {disc_generation} Epoch {disc_epoch}: "
                f"loss={epoch_disc_loss:.6f}, "
                f"acc={epoch_acc:.4f}, "
                f"precision={epoch_precision:.4f}, "
                f"recall={epoch_recall:.4f}, "
                f"f1={epoch_f1:.4f}"
            )
            
            accelerator.log({
                "discriminator_training/epoch": disc_epoch,
                "discriminator_training/loss": epoch_disc_loss,
                "discriminator_training/accuracy": epoch_acc,
                "discriminator_training/precision": epoch_precision,
                "discriminator_training/recall": epoch_recall,
                "discriminator_training/f1": epoch_f1,
                "discriminator_training/generation": disc_generation,
            })

        # if accelerator.is_main_process:
        #     logger.info(f"Disc Gen {disc_generation} Epoch {disc_epoch}: loss={epoch_disc_loss:.6f}")
        #     accelerator.log({
        #         "discriminator_training/epoch": disc_epoch,
        #         "discriminator_training/loss": epoch_disc_loss,
        #         "discriminator_training/generation": disc_generation,
        #     })
    
    # Freeze discriminator after training
    for p in new_discriminator.parameters():
        p.requires_grad = False
    new_discriminator.eval()
    
    # Wrap in FeatureExtractor
    unwrapped_discriminator = accelerator.unwrap_model(new_discriminator)
    feature_discriminator = FeatureExtractor(unwrapped_discriminator, args.target_layers)
    feature_discriminator.eval()
    
    # Save discriminator checkpoint
    if accelerator.is_main_process:
        disc_save_path = Path(args.output_dir) / "discriminators" / f"disc_gen_{disc_generation}.safetensors"
        disc_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        from safetensors.torch import save_file
        save_file(unwrapped_discriminator.state_dict(), disc_save_path)
        logger.info(f"Saved discriminator generation {disc_generation} to {disc_save_path}")
    
    logger.info(f"Finished training discriminator generation {disc_generation}")
    logger.info(f"=" * 80)
    
    return feature_discriminator


def train_epoch(
        args,
        model,
        discriminator,
        dataloader,
        optimizer,
        scheduler,
        criterion,
        feature_matching_loss_criterion,
        accelerator,
        epoch,
        train_global_step,
    ):

    unwrapped_model = accelerator.unwrap_model(model)

    model.train()
    
    total_train_loss = 0.0
    total_train_samples = 0

    total_train_fm_loss = 0.0
    total_train_var_loss = 0.0

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
            
            # loss = loss_dict["all_vars"]
            var_loss = loss_dict["all_vars"]
            
            if args.use_feature_matching_loss and discriminator is not None:
                feature_matching_loss = 0.0
                # with torch.no_grad():
                # discriminator_pred = discriminator(_pred).view(-1)
                # discriminator_label = torch.ones_like(discriminator_pred)
                # feature_matching_loss = feature_matching_loss_criterion( discriminator_pred, discriminator_label )
                # print("Adversarial Loss Computation:")
                # print(f"{discriminator_pred=}")
                # print(f"{discriminator_label=}")
                # print(f"{feature_matching_loss=}")
                # print(f"{var_loss.shape=}")
                # print(f"{discriminator_pred.shape=}")
                # print(f"{discriminator_label.shape=}")
                # print(f"{feature_matching_loss.shape=}")

                fake_features_dict = discriminator(_pred)
                with torch.no_grad():
                    real_features_dict = discriminator(_label)
                for layer_name in fake_features_dict.keys():
                    feat_fake = fake_features_dict[layer_name]
                    feat_real = real_features_dict[layer_name].detach() # Ensure no grad
                    
                    # L1 Loss is usually better for sharpness than MSE (L2)
                    # feature_matching_loss += torch.nn.functional.l1_loss(feat_fake, feat_real)
                    feature_matching_loss += feature_matching_loss_criterion(
                        feat_fake,
                        feat_real,
                    )
                
                # Scale the loss (optional normalization by number of layers)
                feature_matching_loss = feature_matching_loss / len(fake_features_dict)

                # print(f"Feature Matching Loss: {feature_matching_loss.item()}")

                loss = (1 - args.fm_lambda) * var_loss + args.fm_lambda * feature_matching_loss
                # print(f"{loss.shape=}")
            else:
                loss = var_loss

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

        if args.use_feature_matching_loss and discriminator is not None:
            gather_var_loss = accelerator.gather( var_loss )
            gather_fm_loss = accelerator.gather( feature_matching_loss )
            total_train_var_loss += gather_var_loss.sum().item()
            total_train_fm_loss += gather_fm_loss.sum().item()
            step_var_loss = gather_var_loss.mean().item()
            step_fm_loss = gather_fm_loss.mean().item()

        if accelerator.is_main_process:
            if args.use_feature_matching_loss and discriminator is not None:
                pbar.set_postfix({
                    "train_step_loss": f"{step_loss:.8f}",
                    "var_loss": f"{step_var_loss:.8f}",
                    "fm_loss": f"{step_fm_loss:.8f}",
                })
                accelerator.log(
                    {
                        "train_global_step": train_global_step,
                        "train/step_loss": step_loss,
                        "lr": current_lr,
                        "grad_norm": total_grad_norm,
                        "train/step_var_loss": step_var_loss,
                        "train/step_fm_loss": step_fm_loss,
                    },
                )
            else:
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
        # print(f"{total_train_loss=}, {total_train_samples=}")

    train_epoch_loss = total_train_loss / total_train_samples
    
    if accelerator.is_main_process:
        if args.use_feature_matching_loss and discriminator is not None:
            train_epoch_var_loss = total_train_var_loss / total_train_samples
            train_epoch_fm_loss = total_train_fm_loss / total_train_samples
            accelerator.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_epoch_loss,
                    "train/epoch_var_loss": train_epoch_var_loss,
                    "train/epoch_fm_loss": train_epoch_fm_loss,
                },
            )
        else:
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
        discriminator,
        dataloader,
        criterion,
        feature_matching_loss_criterion,
        accelerator,
        epoch,
        val_global_step,
    ):
    unwrapped_model = accelerator.unwrap_model(model)
    model.eval()

    total_val_loss = 0.0
    total_val_samples = 0

    # for logging var/adv loss when using adversarial loss
    total_val_var_loss = 0.0
    total_val_fm_loss = 0.0

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
                    surf_vars = val_label["surf_vars"],
                    atmos_vars = val_label["atmos_vars"],
                    static_vars = static_data["static_vars"],
                    metadata = Metadata(
                        lat = latitude,
                        lon = longitude,
                        time = tuple(map(lambda d: pd.Timestamp(d) + pd.Timedelta(hours = args.lead_time), val_dates)),
                        atmos_levels = levels,
                    ),
                )

                _pred = model(_input)

                # var loss (same as before)
                loss_dict = criterion(
                    _pred.normalise(surf_stats = unwrapped_model.surf_stats),
                    _label.normalise(surf_stats = unwrapped_model.surf_stats),
                )
                var_loss = loss_dict["all_vars"]

                # adversarial part (mirror train_epoch)
                if (
                    args.use_feature_matching_loss
                    and discriminator is not None
                    and feature_matching_loss_criterion is not None
                ):
                    # discriminator is frozen; we still use it in val under no-grad
                    # with torch.no_grad():
                    # discriminator_pred = discriminator(_pred).view(-1)
                    # discriminator_label = torch.ones_like(discriminator_pred)
                    # feature_matching_loss = feature_matching_loss_criterion(
                    #     discriminator_pred,
                    #     discriminator_label,
                    # )

                    with torch.no_grad():
                        fake_features_dict = discriminator(_pred)
                        real_features_dict = discriminator(_label)
                    for layer_name in fake_features_dict.keys():
                        feat_fake = fake_features_dict[layer_name]
                        feat_real = real_features_dict[layer_name].detach() # Ensure no grad
                        
                        # L1 Loss is usually better for sharpness than MSE (L2)
                        # feature_matching_loss += torch.nn.functional.l1_loss(feat_fake, feat_real)
                        feature_matching_loss = feature_matching_loss_criterion(
                            feat_fake,
                            feat_real,
                        )

                    loss = (1 - args.fm_lambda) * var_loss + args.fm_lambda * feature_matching_loss
                else:
                    feature_matching_loss = None  # just for clarity
                    loss = var_loss

            # gather main loss
            gather_val_loss = accelerator.gather(loss)
            total_val_loss += gather_val_loss.sum().item()
            total_val_samples += gather_val_loss.shape[0]
            step_loss = gather_val_loss.mean().item()

            # gather and log components when using adversarial loss
            if (
                args.use_feature_matching_loss
                and discriminator is not None
                and feature_matching_loss_criterion is not None
            ):
                gather_val_var_loss = accelerator.gather(var_loss)
                gather_val_fm_loss = accelerator.gather(feature_matching_loss)

                total_val_var_loss += gather_val_var_loss.sum().item()
                total_val_fm_loss += gather_val_fm_loss.sum().item()

                step_var_loss = gather_val_var_loss.mean().item()
                step_fm_loss = gather_val_fm_loss.mean().item()

            if accelerator.is_main_process:
                if (
                    args.use_feature_matching_loss
                    and discriminator is not None
                    and feature_matching_loss_criterion is not None
                ):
                    pbar.set_postfix(
                        {
                            "val_step_loss": f"{step_loss:.8f}",
                            "val_var_loss": f"{step_var_loss:.8f}",
                            "val_fm_loss": f"{step_fm_loss:.8f}",
                        },
                    )
                    accelerator.log(
                        {
                            "val_global_step": val_global_step,
                            "val/step_loss": step_loss,
                            "val/step_var_loss": step_var_loss,
                            "val/step_fm_loss": step_fm_loss,
                        },
                    )
                else:
                    pbar.set_postfix(
                        {"val_step_loss": f"{step_loss:.8f}"},
                    )
                    accelerator.log(
                        {
                            "val_global_step": val_global_step,
                            "val/step_loss": step_loss,
                        },
                    )

            val_global_step += 1

    val_epoch_loss = total_val_loss / total_val_samples

    if accelerator.is_main_process:
        if (
            args.use_feature_matching_loss
            and discriminator is not None
            and feature_matching_loss_criterion is not None
        ):
            val_epoch_var_loss = total_val_var_loss / total_val_samples
            val_epoch_fm_loss = total_val_fm_loss / total_val_samples
            accelerator.log(
                {
                    "epoch": epoch,
                    "val/epoch_loss": val_epoch_loss,
                    "val/epoch_var_loss": val_epoch_var_loss,
                    "val/epoch_fm_loss": val_epoch_fm_loss,
                },
            )
        else:
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
            if args.use_feature_matching_loss:
                run.define_metric("train/step_var_loss", step_metric = "train_global_step")
                run.define_metric("train/step_fm_loss", step_metric = "train_global_step")
                run.define_metric("train/epoch_var_loss", step_metric = "epoch")
                run.define_metric("train/epoch_fm_loss", step_metric = "epoch")

                run.define_metric("val/step_var_loss", step_metric = "val_global_step")
                run.define_metric("val/step_fm_loss", step_metric = "val_global_step")
                run.define_metric("val/epoch_var_loss", step_metric = "epoch")
                run.define_metric("val/epoch_fm_loss", step_metric = "epoch")


    logger.info(accelerator.state)

    model = create_model(args)
    train_dataset = create_dataset(args, "train")
    val_dataset = create_dataset(args, "val")

    if args.use_feature_matching_loss:
        if args.discriminator_checkpoint_path:
            # Start with provided checkpoint
            logger.info("Using initial discriminator checkpoint")
            discriminator = create_discriminator(args)
            # current_disc_generation = 0
        else:
            # Train initial discriminator from scratch
            logger.info("Training initial discriminator (Generation 0)")
            discriminator = None  # Will be created in train_new_discriminator
            # current_disc_generation = 0
    else:
        discriminator = None
    
    current_disc_generation = 1

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
    # if discriminator:
    #     # feature_matching_loss_criterion = torch.nn.BCEWithLogitsLoss( reduction = "none" )
    #     feature_matching_loss_criterion = torch.nn.L1Loss( reduction = "mean" )
    # else:
    #     feature_matching_loss_criterion = None

    feature_matching_loss_criterion = torch.nn.L1Loss( reduction = "mean" )

    total_training_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_step_ratio * total_training_steps)

    scheduler = get_scheduler_with_warmup(
        optimizer,
        warmup_steps = warmup_steps,
        training_steps = total_training_steps,
        schedule_type = "cosine",
    )

    if discriminator:
        model, discriminator, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, discriminator, optimizer, train_loader, val_loader, scheduler,
        )
    else:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler,
        )

    # if discriminator:
    #     disc_unwrapped = accelerator.unwrap_model(discriminator)
    #     disc_params_before = {
    #         name: p.detach().cpu().clone()
    #         for name, p in disc_unwrapped.named_parameters()
    #     }
    # else:
    #     disc_params_before = None

    train_global_step = 0
    val_global_step = 0
    best_checkpoints = []

    for epoch in range(1, args.epochs + 1):

        # Check if we need to refresh discriminator
        if (
            args.use_feature_matching_loss
            and args.discriminator_refresh_epochs > 0
            and (epoch - 1) % args.discriminator_refresh_epochs == 0
        ):
            
            if discriminator is None or epoch > 1:  # Skip at epoch 1 if we have initial disc
                logger.info(f"\n{'='*80}")
                logger.info(f"Epoch {epoch}: Time to refresh discriminator!")
                logger.info(f"{'='*80}\n")
                
                # Train new discriminator
                discriminator = train_new_discriminator(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    args=args,
                    accelerator=accelerator,
                    disc_generation=current_disc_generation,
                )
                
                # Prepare new discriminator
                discriminator = accelerator.prepare(discriminator)
                current_disc_generation += 1
                
                logger.info(f"Now using discriminator generation {current_disc_generation - 1}")

        train_loss, train_global_step = train_epoch(
            args,
            model,
            discriminator,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            feature_matching_loss_criterion,
            accelerator,
            epoch,
            train_global_step
        )
        val_loss, val_global_step = val_epoch(
            args,
            model,
            discriminator,
            val_loader,
            criterion,
            feature_matching_loss_criterion,
            accelerator,
            epoch,
            val_global_step,
        )

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            logger.info(f"epoch {epoch} - train_loss: {train_loss:.8f}")
            logger.info(f"epoch {epoch} - val_loss: {val_loss:.8f}")
            logger.info(f"epoch {epoch} - using discriminator generation: {current_disc_generation if discriminator else 'None'}")
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

    # if discriminator and accelerator.is_main_process:
    #     disc_unwrapped = accelerator.unwrap_model(discriminator)
    #     changed_params = []
    #     for name, p in disc_unwrapped.named_parameters():
    #         before = disc_params_before[name]
    #         after = p.detach().cpu()
    #         if not torch.allclose(before, after):
    #             changed_params.append(name)

    #     if not changed_params:
    #         logger.info("Discriminator parameters did NOT change. ✅")
    #     else:
    #         logger.warning("Discriminator parameters CHANGED for these tensors:")
    #         for name in changed_params:
    #             logger.warning(f"  - {name}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    accelerator.end_training()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force = True)
    main()
