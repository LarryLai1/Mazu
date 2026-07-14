import argparse
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW
from tqdm import tqdm

# Import your custom dataset and model classes
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
import pandas as pd

from aurora.batch import Batch, Metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Weather Discriminator (PyTorch)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Dataset paths & config
    parser.add_argument(
        "--Aurora_input_dir",
        type=str,
        required=True,
        help="Aurora generated data root dir",
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        required=True,
        help="ERA5 data root dir",
    )

    # Single global interval; we will split randomly by val_ratio
    parser.add_argument(
        "--start_date_hour",
        type=str,
        required=True,
        help="Global start datetime (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end_date_hour",
        type=str,
        required=True,
        help="Global end datetime (YYYY-MM-DD HH:MM:SS)",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (random split on the full interval)",
    )

    parser.add_argument(
        "--forecast_hour",
        nargs="+",
        type=int,
        default=[6],
        help="Hour forecast interval(s)",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=8,
        help="Validation batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num workers",
    )

    # Model & train config
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone for discriminator",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer",
    )

    # Weather variable config
    parser.add_argument(
        "--upper_variables",
        nargs="*",
        default=["u", "v", "t", "q", "z"],
    )
    parser.add_argument(
        "--surface_variables",
        nargs="*",
        default=["t2m", "u10", "v10", "msl"],
    )
    parser.add_argument(
        "--static_variables",
        nargs="*",
        default=["lsm", "slt", "z"],
    )
    parser.add_argument(
        "--latitude",
        nargs=2,
        type=float,
        default=[39.75, 5],
        help="lat_min lat_max",
    )
    parser.add_argument(
        "--longitude",
        nargs=2,
        type=float,
        default=[100, 144.75],
        help="lon_min lon_max",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=[1000, 925, 850, 700, 500, 300, 150, 50],
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="wr_discriminator",
        help="Directory to save checkpoints",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max # checkpoints to keep",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=3,
        help="Save top-K best checkpoints by validation loss",
    )

    # Logging & tracking
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Tensorboard / tracker log directory",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Reporting integration (e.g. 'tensorboard' or 'wandb')",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="AuroraTW",
        help="Accelerate project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
    )

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
                p
                for p in output_dir.iterdir()
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
    best_ckpts: list,  # list of (loss, path)
):
    """
    Saves a checkpoint if it belongs in the top-K best (lowest) val losses.
    Returns the updated best_ckpts list.
    """
    output_dir = Path(output_dir)

    # decide whether to save
    if len(best_ckpts) < args.save_top_k or val_loss < max(
        best_ckpts, key=lambda x: x[0]
    )[0]:
        save_path = output_dir / f"{epoch}-train_loss={train_loss:.6f}-val_loss={val_loss:.6f}"
        save_path.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(save_path)
        logger.info(
            f"Saved new best checkpoint: {save_path} "
            f"(train_loss={train_loss:.6f} val_loss={val_loss:.6f})"
        )

        # record & sort
        best_ckpts.append((val_loss, save_path))
        best_ckpts.sort(key=lambda x: x[0])  # ascending

        # trim excess
        while len(best_ckpts) > args.save_top_k:
            worst = best_ckpts.pop()  # worst = largest loss
            shutil.rmtree(worst[1], ignore_errors=True)
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
        # ncols=120,
    )

    latitude, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    for (inputs, input_dates), labels in pbar:
        optimizer.zero_grad()

        _input = Batch(
            surf_vars=inputs["surf_vars"],
            atmos_vars=inputs["atmos_vars"],
            static_vars=static_data["static_vars"],
            metadata=Metadata(
                lat=latitude,
                lon=longitude,
                time=tuple(map(lambda d: pd.Timestamp(d), input_dates)),
                atmos_levels=levels,
            ),
        )

        outputs = model(_input)
        loss = criterion(outputs.view(-1), labels.float())

        accelerator.backward(loss.mean())
        optimizer.step()

        gather_train_loss = accelerator.gather(loss)
        total_train_loss += gather_train_loss.sum().item()
        total_train_samples += gather_train_loss.shape[0]

        step_loss = gather_train_loss.mean().item()

        if accelerator.is_main_process:
            pbar.set_postfix(
                {
                    "train_step_loss": f"{step_loss:.8f}",
                }
            )

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
        # ncols=120,
    )

    latitude, longitude = dataloader.dataset.get_latitude_longitude()
    levels = dataloader.dataset.get_levels()
    static_data = dataloader.dataset.get_static_vars_ds()

    with torch.no_grad():
        for (inputs, input_dates), labels in pbar:
            _input = Batch(
                surf_vars=inputs["surf_vars"],
                atmos_vars=inputs["atmos_vars"],
                static_vars=static_data["static_vars"],
                metadata=Metadata(
                    lat=latitude,
                    lon=longitude,
                    time=tuple(map(lambda d: pd.Timestamp(d), input_dates)),
                    atmos_levels=levels,
                ),
            )

            outputs = model(_input)

            loss = criterion(outputs.view(-1), labels.float())

            gather_val_loss = accelerator.gather(loss)
            total_val_loss += gather_val_loss.sum().item()
            total_val_samples += gather_val_loss.shape[0]

            probs = torch.sigmoid(outputs.view(-1))
            gather_probs = accelerator.gather(probs)
            gather_labels = accelerator.gather(labels.view(-1))

            all_preds.append(gather_probs.cpu())
            all_targets.append(gather_labels.cpu())

            if accelerator.is_main_process:
                pbar.set_postfix(
                    {"val_step_loss": f"{gather_val_loss.mean().item():.8f}"}
                )

            val_global_step += 1

    val_epoch_loss = total_val_loss / total_val_samples

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Binarize predictions (threshold = 0.5)
    pred_labels = (all_preds >= 0.5).long()
    true_labels = all_targets.long()

    _acc = accuracy_score(true_labels, pred_labels)
    _precision = precision_score(true_labels, pred_labels, zero_division=0)
    _recall = recall_score(true_labels, pred_labels, zero_division=0)
    _f1 = f1_score(true_labels, pred_labels, zero_division=0)

    return val_epoch_loss, (_acc, _precision, _recall, _f1), val_global_step


def main():
    import wandb

    args = parse_args()
    set_seed(args.seed)

    # ---------- pathlib for logging dir ----------
    logging_dir = Path(args.output_dir) / args.logging_dir

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(logging_dir),
    )

    accelerator = Accelerator(
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.wandb_name}},
        )

        if args.report_to == "wandb":
            run = wandb.run
            run.define_metric("train/step_loss", step_metric="train_global_step")
            run.define_metric("val/step_loss", step_metric="val_global_step")
            run.define_metric("train/epoch_loss", step_metric="epoch")
            run.define_metric("val/epoch_loss", step_metric="epoch")

    # ==========================
    # Build ONE full dataset over [start_date_hour, end_date_hour]
    # ==========================
    aurora_dataset_list = []
    era5_dataset_list = []

    for s_h in args.forecast_hour:
        aurora_dataset_list.append(
            AuroraPredictionDataset(
                data_root_dir=args.Aurora_input_dir,
                start_date_hour=args.start_date_hour,
                end_date_hour=args.end_date_hour,
                upper_variables=args.upper_variables,
                surface_variables=args.surface_variables,
                static_variables=args.static_variables,
                latitude=tuple(args.latitude),
                longitude=tuple(args.longitude),
                levels=args.levels,
                forecast_hour=s_h,
            )
        )
        era5_dataset_list.append(
            ERA5TWDataset(
                data_root_dir=args.data_root_dir,
                start_date_hour=args.start_date_hour,
                end_date_hour=args.end_date_hour,
                upper_variables=args.upper_variables,
                surface_variables=args.surface_variables,
                static_variables=args.static_variables,
                latitude=tuple(args.latitude),
                longitude=tuple(args.longitude),
                levels=args.levels,
            )
        )

    full_ds = DiscriminatorDataset(
        AuroraTWDataset=aurora_dataset_list,
        ERA5TWDataset=era5_dataset_list,
    )

    num_samples = len(full_ds)
    val_size = int(num_samples * args.val_ratio)
    # train_size = num_samples - val_size

    # Random split with a seeded generator for reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)
    perm = torch.randperm(num_samples, generator=g).tolist()

    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        full_ds,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        full_ds,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ==========================
    # Model / criterion / optimizer
    # ==========================
    model = ResNetDiscriminator(
        surface_variables=args.surface_variables,
        upper_variables=args.upper_variables,
        levels=args.levels,
        backbone_name=args.backbone,
        pretrained=False,
    )

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ckpt_dir_path = Path(args.output_dir) / "ckpts"
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)

    # Prepare model, optimizer, and dataloaders with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_checkpoints = []
    train_global_step = 0
    val_global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_global_step = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            accelerator,
            epoch,
            train_global_step,
        )

        val_loss, (val_acc, val_precision, val_recall, val_f1), val_global_step = (
            validate_epoch(
                model,
                val_loader,
                criterion,
                accelerator,
                epoch,
                val_global_step,
            )
        )

        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch:04d}: "
                f"Train Loss = {train_loss:.8f} | "
                f"Val Loss = {val_loss:.8f} | "
                f"Val Acc = {val_acc:.4f}, "
                f"Precision = {val_precision:.4f}, "
                f"Recall = {val_recall:.4f}, "
                f"F1 = {val_f1:.4f}"
            )

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
                args,
                accelerator,
                ckpt_dir_path,
                epoch,
            )
            best_checkpoints = save_checkpoint_best_by_val_loss(
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
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
