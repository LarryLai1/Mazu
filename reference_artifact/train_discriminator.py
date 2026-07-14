import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from tqdm import tqdm

# Import your custom dataset and model classes
# from discriminator_ds import Aurora_DS, ERA5_DS, CombinedDataset
# from discriminator_ds_v1 import AuroraPredictionDataset, ERA5TWDataset, DiscriminatorDataset
# from discriminator import WeatherResNetDiscriminator
# from discriminator_v1 import ResNetDiscriminator

from datasets.DiscriminatorDataset import (
    AuroraPredictionDataset,
    ERA5TWDataset,
    DiscriminatorDataset,
)
from discriminator.ResNet_discriminator import ResNetDiscriminator

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shutil
# import os

import pandas as pd

from aurora.batch import Batch, Metadata

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = get_logger(__name__, log_level = "INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Weather Discriminator (PyTorch)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    # Dataset paths & config
    parser.add_argument('--Aurora_input_dir', type=str, required=True, help='Aurora generated data root dir')
    parser.add_argument('--data_root_dir', type=str, required=True, help='ERA5 data root dir')
    parser.add_argument('--train_start_date_hour', type=str, required=True, help='Start datetime (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--train_end_date_hour', type=str, required=True, help='End datetime (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--val_start_date_hour', type=str, required=True, help='Start datetime (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--val_end_date_hour', type=str, required=True, help='End datetime (YYYY-MM-DD HH:MM:SS)')
    # parser.add_argument('--forecast_hour', type=int, default=6, help='Hour selection interval')
    parser.add_argument('--forecast_hour', nargs='+', type=int, default=[6], help='Hour forecast_hourforecast_hourion interval(s)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    # parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Validation batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num workers')
    # Model & train config
    # parser.add_argument('--in_channels', type=int, default=44, help='Number of input channels')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone for discriminator')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    # Weather variable config
    parser.add_argument('--upper_variables', nargs='*', default=['u', 'v', 't', 'q', 'z'])
    parser.add_argument('--surface_variables', nargs='*', default=['t2m', 'u10', 'v10', 'msl'])
    parser.add_argument('--static_variables', nargs='*', default=['lsm', 'slt', 'z'])
    parser.add_argument('--latitude', nargs=2, type=float, default=[39.75, 5], help="lat_min lat_max")
    parser.add_argument('--longitude', nargs=2, type=float, default=[100, 144.75], help="lon_min lon_max")
    parser.add_argument('--levels', nargs='*', type=int, default=[1000, 925, 850, 700, 500, 300, 150, 50])
    # parser.add_argument('--save_path', type=str, default='best_discriminator.pt', help='Best model save path')
    parser.add_argument('--output_dir', type=str, default='wr_discriminator', help='Directory to save checkpoints')
    # Checkpointing
    parser.add_argument('--checkpointing_epochs', type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument('--checkpoints_total_limit', type=int, default=None, help="Max # checkpoints to keep")
    parser.add_argument('--save_top_k', type=int, default=3, help="Save top-K best checkpoints by validation loss")

    # Logging & tracking
    parser.add_argument('--logging_dir', type=str, default="logs", help="Tensorboard log directory")
    parser.add_argument('--report_to', type=str, default="tensorboard", help="Reporting integration")
    parser.add_argument('--tracker_project_name', type=str, default="AuroraTW", help="Accelerate project name")
    # parser.add_argument('--mixed_precision', type=str, default=None, choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--wandb_name", type=str, default = None)

    return parser.parse_args()

def save_checkpoint_by_epoch(args, accelerator, output_dir, epoch):
    """
    Save a checkpoint directory every N epochs. `output_dir` can be str or Path.
    """
    output_dir = Path(output_dir)

    if args.checkpointing_epochs > 0 and epoch % args.checkpointing_epochs == 0:
        if accelerator.is_main_process:
            # List all checkpoint dirs like checkpoint-##
            checkpoints = [
                p for p in output_dir.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint-")
            ]
            checkpoints = sorted(checkpoints, key=lambda p: int(p.name.split("-")[1]))

            # If too many, remove oldest
            if args.checkpoints_total_limit is not None:
                if len(checkpoints) >= args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[:num_to_remove]
                    for removing_path in removing_checkpoints:
                        shutil.rmtree(removing_path, ignore_errors=True)
                        logger.info(f"Removed old checkpoint: {removing_path}")

            save_path = output_dir / f"checkpoint-{epoch}"
            save_path.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(save_path)  # saves model, optimizer, etc.

            logger.info(f"Saved checkpoint to {save_path}")

def save_checkpoint_best_by_val_loss(
    args,
    accelerator,
    output_dir,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_ckpts: list,                # list of (loss, path)
):
    """
    Saves a checkpoint if it belongs in the top-K best (lowest) val losses.
    Returns the updated best_ckpts list.
    """
    output_dir = Path(output_dir)

    # decide whether to save
    if len(best_ckpts) < args.save_top_k or val_loss < max(best_ckpts, key=lambda x: x[0])[0]:
        save_path = output_dir / f"{epoch}-train_loss={train_loss:.6f}-val_loss={val_loss:.6f}"
        save_path.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(save_path)
        logger.info(f"Saved new best checkpoint: {save_path} (train_loss={train_loss:.6f} val_loss={val_loss:.6f})")

        # record & sort
        best_ckpts.append((val_loss, save_path))
        best_ckpts.sort(key=lambda x: x[0])          # ascending

        # trim excess
        while len(best_ckpts) > args.save_top_k:
            worst = best_ckpts.pop()                 # worst = largest loss
            shutil.rmtree(worst[1], ignore_errors = True)
            logger.info(f"Removed older/worse checkpoint: {worst[1]}")

    return best_ckpts

def train_epoch(model, dataloader, criterion, optimizer, accelerator, epoch, train_global_step):
    model.train()
    total_train_loss = 0.0
    total_train_samples = 0

    pbar = tqdm(
        dataloader,
        desc=f"train_epoch: {epoch}",
        disable=not accelerator.is_local_main_process,
        ncols=120,
    )

    latitude, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    for (inputs, input_dates), labels in pbar:
        # inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # outputs = model(inputs)
        # loss = criterion( outputs.view(-1), labels.float() )

        _input = Batch(
            surf_vars = inputs["surf_vars"],
            atmos_vars = inputs["atmos_vars"],
            # static_vars = inputs["static_vars"],
            static_vars = static_data["static_vars"],
            metadata = Metadata(
                lat = latitude,
                lon = longitude,
                time = tuple(map(lambda d: pd.Timestamp(d), input_dates)),
                atmos_levels = levels,
            ),
        )

        outputs = model(_input)
        loss = criterion( outputs.view(-1), labels.float() )

        # loss.backward()
        accelerator.backward( loss.mean() )
        optimizer.step()
        # total_loss += loss.item()
        gather_train_loss = accelerator.gather(loss)
        total_train_loss += gather_train_loss.sum().item()  # Gather losses across all processes
        total_train_samples += gather_train_loss.shape[0]  # Count total samples processed
    
        step_loss = gather_train_loss.mean().item()

        if accelerator.is_main_process:
            pbar.set_postfix({
                "train_step_loss": f"{step_loss:.8f}",
                # "lr": f"{current_lr:.2e}",
                # "grad_norm": f"{total_grad_norm:.2e}",
            })
            # accelerator.log(
            #     {
            #         "train_global_step": train_global_step,
            #         "train/step_loss": step_loss,
            #         # "lr": current_lr,
            #         # "grad_norm": total_grad_norm,
            #     },
            #     # step = train_global_step,
            # )
        
        train_global_step += 1
    
    train_epoch_loss = total_train_loss / total_train_samples
    return train_epoch_loss, train_global_step

def validate_epoch(model, dataloader, criterion, accelerator, epoch, val_global_step):
    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(
        dataloader,
        desc=f"val_epoch: {epoch}",
        disable=not accelerator.is_local_main_process,
        ncols=120,
    )

    latitude, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    with torch.no_grad():
        for (inputs, input_dates), labels in pbar:

            _input = Batch(
                surf_vars = inputs["surf_vars"],
                atmos_vars = inputs["atmos_vars"],
                # static_vars = inputs["static_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = latitude,
                    lon = longitude,
                    time = tuple(map(lambda d: pd.Timestamp(d), input_dates)),
                    atmos_levels = levels,
                ),
            )

            outputs = model(_input)
            
            loss = criterion(outputs.view(-1), labels.float())
            
            gather_val_loss = accelerator.gather(loss)
            total_val_loss += gather_val_loss.sum().item()
            total_val_samples += gather_val_loss.shape[0]


            # probs = torch.sigmoid(outputs.view(-1))
            # Gather predictions and targets from all processes
            probs = torch.sigmoid(outputs.view(-1))
            # gather_probs = accelerator.gather(outputs.view(-1))
            gather_probs = accelerator.gather(probs)
            gather_labels = accelerator.gather(labels.view(-1))
            all_preds.append(gather_probs.cpu())
            all_targets.append(gather_labels.cpu())

            if accelerator.is_main_process:
                pbar.set_postfix({"val_step_loss": f"{gather_val_loss.mean().item():.8f}"})
                # accelerator.log(
                #     {
                #         "val_global_step": val_global_step,
                #         "val/step_loss": gather_val_loss.mean().item(),
                #     },
                #     # step = val_global_step,  # or another global step logic
                # )

            val_global_step += 1

    val_epoch_loss = total_val_loss / total_val_samples

    # print(f"{all_preds}=")

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Binarize predictions (threshold = 0.5)
    pred_labels = (all_preds >= 0.5).long()
    true_labels = all_targets.long()

    # Compute accuracy

    # print( f"{pred_labels=}" )
    # print( f"{true_labels=}" )

    # accuracy = (pred_labels == true_labels).float().mean().item()
    _acc = accuracy_score(true_labels, pred_labels)
    _precision = precision_score(true_labels, pred_labels, zero_division = 0)
    _recall = recall_score(true_labels, pred_labels, zero_division = 0)
    _f1 = f1_score(true_labels, pred_labels, zero_division = 0)

    return val_epoch_loss, (_acc, _precision, _recall, _f1), val_global_step

def main():

    import wandb

    args = parse_args()
    set_seed(args.seed)

    # ---------- pathlib for logging dir ----------
    logging_dir = Path(args.output_dir) / args.logging_dir
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(logging_dir),  # ensure string for downstream tools
    )

    accelerator = Accelerator(
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        # mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )



    # if accelerator.is_main_process:
    #     logger.info("Starting training with the following configuration:")
    #     for arg, value in vars(args).items():
    #         logger.info(f"{arg}: {value}")
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tracker_config.pop("validation_prompts")
        # print(f"{type(args.tracker_project_name)=}")
        # print(f"{type(tracker_config)=}")

        accelerator.init_trackers(
            args.tracker_project_name,
            config = tracker_config,
            init_kwargs={"wandb": {"name": args.wandb_name}}
        )

        if args.report_to == "wandb":
            run = wandb.run
            run.define_metric("train/step_loss", step_metric="train_global_step")
            run.define_metric("val/step_loss", step_metric="val_global_step")
            run.define_metric("train/epoch_loss", step_metric="epoch")
            run.define_metric("val/epoch_loss", step_metric="epoch")


    # Build datasets
    # dataset_aurora = Aurora_DS(
    #     data_root_dir=args.Aurora_input_dir,
    #     start_date_hour=args.start_date_hour,
    #     end_date_hour=args.end_date_hour,
    #     upper_variables=args.upper_variables,
    #     surface_variables=args.surface_variables,
    #     latitude=tuple(args.latitude),
    #     longitude=tuple(args.longitude),
    #     levels=args.levels,
    #     forecast_hour=args.forecast_hour,
    #     flatten=True,
    # )
    # dataset_era5 = ERA5_DS(
    #     data_root_dir=args.data_root_dir,
    #     start_date_hour=args.start_date_hour,
    #     end_date_hour=args.end_date_hour,
    #     upper_variables=args.upper_variables,
    #     surface_variables=args.surface_variables,
    #     latitude=tuple(args.latitude),
    #     longitude=tuple(args.longitude),
    #     levels=args.levels,
    #     flatten=True,
    # )
    # all_ds = CombinedDataset(dataset_aurora, dataset_era5)
    

    # aurora_datasets = []
    # era5_datasets = []


    # train_ds
    train_Aurora_dataset_list = []
    train_ERA5_dataset_list = []

    for s_h in args.forecast_hour:
        train_Aurora_dataset_list.append(
            AuroraPredictionDataset(
                data_root_dir = args.Aurora_input_dir,
                start_date_hour = args.train_start_date_hour,
                end_date_hour = args.train_end_date_hour,
                upper_variables = args.upper_variables,
                surface_variables = args.surface_variables,
                static_variables = args.static_variables,
                latitude = tuple(args.latitude),
                longitude = tuple(args.longitude),
                levels = args.levels,
                forecast_hour = s_h,
                # flatten = True,
            )
        )
        train_ERA5_dataset_list.append(
            ERA5TWDataset(
                data_root_dir = args.data_root_dir,
                start_date_hour = args.train_start_date_hour,
                end_date_hour = args.train_end_date_hour,
                upper_variables = args.upper_variables,
                surface_variables = args.surface_variables,
                static_variables = args.static_variables,
                latitude = tuple(args.latitude),
                longitude = tuple(args.longitude),
                levels = args.levels,
                # forecast_hour = s_h,
                # flatten = True,
            )
        )

    # val_ds
    val_Aurora_dataset_list = []
    val_ERA5_dataset_list = []

    for s_h in args.forecast_hour:
        val_Aurora_dataset_list.append(
            AuroraPredictionDataset(
                data_root_dir = args.Aurora_input_dir,
                start_date_hour = args.val_start_date_hour,
                end_date_hour = args.val_end_date_hour,
                upper_variables = args.upper_variables,
                surface_variables = args.surface_variables,
                static_variables = args.static_variables,
                latitude = tuple(args.latitude),
                longitude = tuple(args.longitude),
                levels = args.levels,
                forecast_hour = s_h,
                # flatten = True,
            )
        )
        val_ERA5_dataset_list.append(
            ERA5TWDataset(
                data_root_dir = args.data_root_dir,
                start_date_hour = args.val_start_date_hour,
                end_date_hour = args.val_end_date_hour,
                upper_variables = args.upper_variables,
                surface_variables = args.surface_variables,
                static_variables = args.static_variables,
                latitude = tuple(args.latitude),
                longitude = tuple(args.longitude),
                levels = args.levels,
                # forecast_hour = s_h,
                # flatten = True,
            )
        )
    
    train_ds = DiscriminatorDataset(
        AuroraTWDataset = train_Aurora_dataset_list,
        ERA5TWDataset = train_ERA5_dataset_list,
        
    )
    val_ds = DiscriminatorDataset(
        AuroraTWDataset = val_Aurora_dataset_list,
        ERA5TWDataset = val_ERA5_dataset_list,
    )

    # for sh in args.forecast_hour:
    #     aurora = Aurora_DS(
    #         data_root_dir=args.Aurora_input_dir,
    #         start_date_hour=args.start_date_hour,
    #         end_date_hour=args.end_date_hour,
    #         upper_variables=args.upper_variables,
    #         surface_variables=args.surface_variables,
    #         latitude=tuple(args.latitude),
    #         longitude=tuple(args.longitude),
    #         levels=args.levels,
    #         forecast_hour=sh,
    #         flatten=True,
    #     )
    #     era5 = ERA5_DS(
    #         data_root_dir=args.data_root_dir,
    #         start_date_hour=args.start_date_hour,
    #         end_date_hour=args.end_date_hour,
    #         upper_variables=args.upper_variables,
    #         surface_variables=args.surface_variables,
    #         latitude=tuple(args.latitude),
    #         longitude=tuple(args.longitude),
    #         levels=args.levels,
    #         # forecast_hour=sh,
    #         flatten=True,
    #     )
    #     aurora_datasets.append(aurora)
    #     era5_datasets.append(era5)

    # Combine as needed
    # all_ds = CombinedDataset(aurora_datasets, era5_datasets)
    # all_ds = DiscriminatorDataset(Aurora_dataset_list, ERA5_dataset_list)

    # val_size = int(len(all_ds) * args.val_ratio)
    # train_size = len(all_ds) - val_size
    # train_ds, val_ds = torch.utils.data.random_split(all_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size = args.train_batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        pin_memory = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = args.val_batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True,
    )

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = WeatherResNetDiscriminator(in_channels=args.in_channels, backbone_name=args.backbone)
    # model.to(device)
    model = ResNetDiscriminator(
        surface_variables = args.surface_variables,
        upper_variables = args.upper_variables,
        levels = args.levels,
        backbone_name = args.backbone,
        pretrained = False,
    )

    # logger.info(f"Using device: {device}")
    # logger.info(f"Model:\n{model}")

    # criterion = torch.nn.BCELoss( reduction = 'none' )
    criterion = torch.nn.BCEWithLogitsLoss( reduction = 'none' )
    # optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = AdamW(
        model.parameters(),
        lr = args.lr,
        weight_decay = args.weight_decay,
    )

    ckpt_dir_path = Path(args.output_dir) / "ckpts"
    ckpt_dir_path.mkdir(parents = True, exist_ok = True)

    # Prepare model, optimizer, and dataloaders with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader,
    )

    best_val_loss = float('inf')
    best_checkpoints = []
    train_global_step = 0
    val_global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        # train_loss = train_epoch(model, train_loader, criterion, optimizer, accelerator)
        train_loss, train_global_step = train_epoch(
            model, train_loader, criterion, optimizer, accelerator, epoch, train_global_step,
        )
        # val_loss = validate_epoch(model, val_loader, criterion, accelerator)
        # val_loss, (val_acc, val_precision, val_recall, val_f1) = validate_epoch(model, val_loader, criterion, accelerator)
        val_loss, (val_acc, val_precision, val_recall, val_f1), val_global_step = validate_epoch(
            model, val_loader, criterion, accelerator, epoch, val_global_step,
        )

        # Log and save checkpoints

        if accelerator.is_main_process:

            logger.info(
                f"Epoch {epoch:04d}: Train Loss = {train_loss:.8f} | Val Loss = {val_loss:.8f} | Val Acc = {val_acc:.4f}, Precision = {val_precision:.4f}, Recall = {val_recall:.4f}, F1 = {val_f1:.4f}"
            )
            # Save best model
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     ckpt_path = ckpt_dir_path / f"epoch={epoch:04d}-train_loss={train_loss:08f}-val_loss={val_loss:08f}.ckpt"
            #     torch.save(model.state_dict(), ckpt_path)
            #     logger.info(f"Saved new best model to {ckpt_path}")
            # save_checkpoint_by_epoch(args, accelerator, output_dir, epoch):
            
            accelerator.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "val/acc": val_acc,
                    "val/precision": val_precision,
                    "val/recall": val_recall,
                    "val/f1": val_f1,
                }
            )
            
            save_checkpoint_by_epoch(
                # args, accelerator, ckpt_dir, epoch, model, optimizer,
                args,
                accelerator,
                ckpt_dir_path,
                epoch,
            )
            save_checkpoint_best_by_val_loss(
                args,
                accelerator,
                ckpt_dir_path,
                epoch,
                train_loss,
                val_loss,
                best_checkpoints,
            )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    accelerator.end_training()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force = True)
    main()
