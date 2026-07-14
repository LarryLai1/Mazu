#!/usr/bin/env python3
"""
Simple, single-GPU/CPU inference for your discriminator (no Accelerate).
- Builds the eval dataset exactly like your val split
- Loads checkpoint_path from a plain model.state_dict file
- Runs inference, computes metrics, and writes metrics.json + predictions.csv
"""
import argparse
from pathlib import Path
import json
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# ---- Your project imports
from datasets.DiscriminatorDataset import (
    AuroraPredictionDataset,
    ERA5TWDataset,
    DiscriminatorDataset,
)
from discriminator.ResNet_discriminator import ResNetDiscriminator
from aurora.batch import Batch, Metadata

from safetensors.torch import load_file

def parse_args():
    p = argparse.ArgumentParser(description="Inference/Eval for Weather Discriminator (no Accelerate)")

    # Repro
    p.add_argument('--seed', type=int, default=42)

    # Data roots + eval window
    p.add_argument('--Aurora_input_dir', type=str, required=True)
    p.add_argument('--data_root_dir', type=str, required=True)
    p.add_argument('--eval_start_date_hour', type=str, required=True,
                   help="Start datetime (YYYY-MM-DD HH:MM:SS)")
    p.add_argument('--eval_end_date_hour', type=str, required=True,
                   help="End datetime (YYYY-MM-DD HH:MM:SS)")
    p.add_argument('--forecast_hour', nargs='+', type=int, default=[6],
                   help="Forecast hour(s) for Aurora preds (same as training)")

    # Variables / domain
    p.add_argument('--upper_variables', nargs='*', default=['u', 'v', 't', 'q', 'z'])
    p.add_argument('--surface_variables', nargs='*', default=['t2m', 'u10', 'v10', 'msl'])
    p.add_argument('--static_variables', nargs='*', default=['lsm', 'slt', 'z'])
    p.add_argument('--latitude', nargs=2, type=float, default=[39.75, 5],
                   help="lat_min lat_max")
    p.add_argument('--longitude', nargs=2, type=float, default=[100, 144.75],
                   help="lon_min lon_max")
    p.add_argument('--levels', nargs='*', type=int,
                   default=[1000, 925, 850, 700, 500, 300, 150, 50])

    # Model
    p.add_argument('--backbone', type=str, default='resnet50')
    p.add_argument('--pretrained', action='store_true', default=False,
                   help="Backbone pretrained flag (kept for parity)")

    # Weights (plain state_dict file saved via `torch.save(model.state_dict(), path)`).
    p.add_argument('--checkpoint_path', type=str, required=True,
                   help="Path to a model.state_dict file (e.g., best.pt)")

    # Loader / batch
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)

    # Output
    p.add_argument('--output_dir', type=str, default='inference_out')
    p.add_argument('--threshold', type=float, default=0.5,
                   help="Decision threshold for metrics")

    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_eval_dataset(args):
    """Build Aurora/ERA5 lists for each forecast hour (same as your val logic)."""
    era5_list, aurora_list = [], []
    for s_h in args.forecast_hour:
        era5_list.append(
            ERA5TWDataset(
                data_root_dir=args.data_root_dir,
                start_date_hour=args.eval_start_date_hour,
                end_date_hour=args.eval_end_date_hour,
                upper_variables=args.upper_variables,
                surface_variables=args.surface_variables,
                static_variables=args.static_variables,
                latitude=tuple(args.latitude),
                longitude=tuple(args.longitude),
                levels=args.levels,
            )
        )
        aurora_list.append(
            AuroraPredictionDataset(
                data_root_dir=args.Aurora_input_dir,
                start_date_hour=args.eval_start_date_hour,
                end_date_hour=args.eval_end_date_hour,
                upper_variables=args.upper_variables,
                surface_variables=args.surface_variables,
                static_variables=args.static_variables,
                latitude=tuple(args.latitude),
                longitude=tuple(args.longitude),
                levels=args.levels,
                forecast_hour=s_h,
            )
        )
    return DiscriminatorDataset(ERA5TWDataset=era5_list, AuroraTWDataset=aurora_list)


# def move_to_device(x: Any, device: torch.device):
#     """Recursively move tensors in nested structures to device."""
#     if torch.is_tensor(x):
#         return x.to(device, non_blocking=True)
#     if isinstance(x, dict):
#         return {k: move_to_device(v, device) for k, v in x.items()}
#     if isinstance(x, (list, tuple)):
#         t = [move_to_device(v, device) for v in x]
#         return type(x)(t) if not isinstance(x, tuple) else tuple(t)
#     return x

def create_model(args, device):
    
    model = ResNetDiscriminator(
        surface_variables=args.surface_variables,
        upper_variables=args.upper_variables,
        levels=args.levels,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
    )

    if args.checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(args.checkpoint_path)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Dataset & loader
    eval_ds = build_eval_dataset(args)
    pin = torch.cuda.is_available()
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # Model
    model = ResNetDiscriminator(
        surface_variables=args.surface_variables,
        upper_variables=args.upper_variables,
        levels=args.levels,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
    )

    # Load checkpoint_path (plain state_dict)
    # if args.checkpoint_path:
    #     # if not checkpoint_path.is_file():
    #     if checkpoint_path.endswith(".safetensors"):
    #         raise FileNotFoundError(f"Weights file not found: {checkpoint_path}")
    #     # state = torch.load(checkpoint_path, map_location = 'cpu')
    #     # checkpoint_path = Path(args.checkpoint_path)
    #     state_dict = load_file(checkpoint_path)
    #     model.load_state_dict(state_dict)
    model = create_model(args, device)

    # Domain info (lat/lon/levels) once
    latitude, longitude = eval_loader.dataset.get_latitude_longitude()
    levels = eval_loader.dataset.get_levels()
    static_data = eval_loader.dataset.get_static_vars_ds()

    all_probs = []
    all_labels = []
    all_dates = []

    with torch.no_grad():
        pbar = tqdm(
            eval_loader,
            desc="inference",
            # ncols=120
        )
        for (inputs, input_dates), labels in pbar:
            # Move tensors
            labels = labels.to(device)

            batch = Batch(
                surf_vars=inputs["surf_vars"],
                atmos_vars=inputs["atmos_vars"],
                # static_vars=inputs["static_vars"],
                static_vars=static_data["static_vars"],
                metadata=Metadata(
                    lat=latitude,
                    lon=longitude,
                    time=tuple(map(lambda d: pd.Timestamp(d), input_dates)),
                    atmos_levels=levels,
                ),
            )

            logits = model(batch).view(-1)
            probs = sigmoid(logits)

            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.view(-1).detach().cpu())
            for d in input_dates:
                all_dates.append(pd.Timestamp(d))

    probs_t = torch.cat(all_probs)
    labels_t = torch.cat(all_labels).to(torch.long)

    thr = float(args.threshold)
    preds_t = (probs_t >= thr).to(torch.long)

    # Metrics
    try:
        auc = roc_auc_score(labels_t.numpy(), probs_t.numpy())
    except ValueError:
        auc = float('nan')
    acc = accuracy_score(labels_t.numpy(), preds_t.numpy())
    prec = precision_score(labels_t.numpy(), preds_t.numpy(), zero_division=0)
    rec = recall_score(labels_t.numpy(), preds_t.numpy(), zero_division=0)
    f1 = f1_score(labels_t.numpy(), preds_t.numpy(), zero_division=0)
    cm = confusion_matrix(labels_t.numpy(), preds_t.numpy()).tolist()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "threshold": thr,
        "confusion_matrix": cm,
        "num_samples": int(labels_t.numel()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Full predictions CSV (aligned 1:1 with local evaluation)
    df = pd.DataFrame({
        "time": all_dates,
        "prob": probs_t.numpy(),
        "pred": preds_t.numpy(),
        "label": labels_t.numpy(),
    })
    df.to_csv(out_dir / "predictions.csv", index=False)

    # Summary print
    print(
        f"[EVAL] N={labels_t.numel()} | "
        f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f} "
        f"(thr={thr:.2f})\n"
        f"Saved to {out_dir}"
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
