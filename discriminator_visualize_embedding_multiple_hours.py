import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy  # <--- Essential for handling the arguments loop

from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.manifold import TSNE
import umap.umap_ as umap

from datasets.DiscriminatorDataset import (
    AuroraPredictionDataset,
    ERA5TWDataset,
    DiscriminatorDataset
)
from discriminator.ResNet_discriminator import ResNetDiscriminator
from aurora.batch import Batch, Metadata

import pandas as pd
from safetensors.torch import load_file

# --------------------------------------------------------------
# Utility: extract features BEFORE final FC
# --------------------------------------------------------------
def extract_backbone_features(model, batch_obj):
    with torch.no_grad():
        # feats = model.backbone(batch_obj)
        feats = model._forward_features(batch_obj)
        return feats.cpu()


# --------------------------------------------------------------
# Build dataset same as in your training script
# --------------------------------------------------------------
def build_val_loader(args):
    val_Aurora_dataset_list = []
    val_ERA5_dataset_list = []

    # This loop works fine because we will pass a list with 1 item in main()
    for s_h in args.forecast_hour:
        val_Aurora_dataset_list.append(
            AuroraPredictionDataset(
                data_root_dir=args.Aurora_input_dir,
                start_date_hour=args.val_start_date_hour,
                end_date_hour=args.val_end_date_hour,
                upper_variables=args.upper_variables,
                surface_variables=args.surface_variables,
                static_variables=args.static_variables,
                latitude=tuple(args.latitude),
                longitude=tuple(args.longitude),
                levels=args.levels,
                forecast_hour=s_h,
            )
        )
        val_ERA5_dataset_list.append(
            ERA5TWDataset(
                data_root_dir=args.data_root_dir,
                start_date_hour=args.val_start_date_hour,
                end_date_hour=args.val_end_date_hour,
                upper_variables=args.upper_variables,
                surface_variables=args.surface_variables,
                static_variables=args.static_variables,
                latitude=tuple(args.latitude),
                longitude=tuple(args.longitude),
                levels=args.levels,
            )
        )

    val_ds = DiscriminatorDataset(
        AuroraTWDataset=val_Aurora_dataset_list,
        ERA5TWDataset=val_ERA5_dataset_list,
    )

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return loader


# --------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # SAME ARGS AS TRAINING
    parser.add_argument('--Aurora_input_dir', type=str, required=True)
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--val_start_date_hour', type=str, required=True)
    parser.add_argument('--val_end_date_hour', type=str, required=True)
    # You can now pass multiple hours here, e.g., --forecast_hour 6 24 48
    parser.add_argument('--forecast_hour', nargs='+', type=int, default=[6])
    parser.add_argument('--upper_variables', nargs='*', default=['u','v','t','q','z'])
    parser.add_argument('--surface_variables', nargs='*', default=['t2m','u10','v10','msl'])
    parser.add_argument('--static_variables', nargs='*', default=['lsm','slt','z'])
    parser.add_argument('--latitude', nargs=2, type=float, default=[39.75,5])
    parser.add_argument('--longitude', nargs=2, type=float, default=[100,144.75])
    parser.add_argument('--levels', nargs='*', type=int,
                        default=[1000,925,850,700,500,300,150,50])

    parser.add_argument("--method", type=str, default="tsne",
                        choices=["tsne", "umap"])
    parser.add_argument("--output_path", type=str, default="embedding_plot.png")

    return parser.parse_args()


# --------------------------------------------------------------
# Main Visualization
# --------------------------------------------------------------
def main():
    args = parse_args()

    # ----------------------
    # Build model
    # ----------------------
    model = ResNetDiscriminator(
        surface_variables=args.surface_variables,
        upper_variables=args.upper_variables,
        levels=args.levels,
        backbone_name="resnet50",
        pretrained=False,
    )

    if args.ckpt.endswith(".safetensors"):
        print(f"Loading safetensors checkpoint: {args.ckpt}")
        state_dict = load_file(args.ckpt, device="cpu")
        model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    # Lists to store combined data from all hours
    all_feats = []
    all_labels = []
    all_hours = [] 

    # --------------------------------------------------------------
    # Loop over each forecast hour separately
    # --------------------------------------------------------------
    # We save the original list of hours passed in command line
    original_hours = args.forecast_hour
    print(f"Processing forecast hours: {original_hours}")

    for hour in original_hours:
        print(f"\n--- Extracting features for Forecast Hour: {hour} ---")
        
        # 1. Create a temporary args object for just this hour
        #    This tricks the build_val_loader into creating a dataset for just this hour
        current_args = copy.deepcopy(args)
        current_args.forecast_hour = [hour] 

        # 2. Build loader for this specific hour
        loader = build_val_loader(current_args)
        
        # Get metadata needed for batch construction
        latitude, longitude = loader.dataset.get_latitude_longitude()
        levels = loader.dataset.get_levels()
        static_data = loader.dataset.get_static_vars_ds()

        # 3. Extract features
        hour_feats = []
        hour_labels = []

        # We need a new pbar for each hour
        pbar = tqdm(loader, desc=f"Hour {hour}")

        for (inputs, input_dates), labels in pbar:
            batch_obj = Batch(
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

            batch_obj = batch_obj.to(next(model.parameters()).device)
            feats = extract_backbone_features(model, batch_obj)

            hour_feats.append(feats)
            hour_labels.append(labels)

        # 4. Concatenate and store for this hour
        if len(hour_feats) > 0:
            hour_feats = torch.cat(hour_feats).numpy()
            hour_labels = torch.cat(hour_labels).numpy()
            
            all_feats.append(hour_feats)
            all_labels.append(hour_labels)
            
            # Track which hour these features belong to
            all_hours.append(np.full(hour_feats.shape[0], hour))

    # Combine everything
    if len(all_feats) == 0:
        print("No data found!")
        return

    all_feats = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    all_hours = np.concatenate(all_hours)

    print(f"\nTotal Feature shape = {all_feats.shape}")

    # --------------------------------------------------------------
    # Dimensionality reduction
    # --------------------------------------------------------------
    print(f"Reducing with {args.method.upper()} ...")

    if args.method == "tsne":
        reducer = TSNE(n_components=2, perplexity=40, learning_rate=200, init='pca')
        emb2d = reducer.fit_transform(all_feats)

    elif args.method == "umap":
        reducer = umap.UMAP(n_components=2, min_dist=0.1, metric="cosine")
        emb2d = reducer.fit_transform(all_feats)


    # --------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    
    unique_hours = sorted(np.unique(all_hours))
    cmap = plt.cm.viridis
    
    # Iterate over hours to assign colors
    for i, h in enumerate(unique_hours):
        # Determine color for this hour
        if len(unique_hours) > 1:
            color = cmap(i / (len(unique_hours) - 1))
        else:
            color = cmap(0.5)
        
        # 1. Plot ERA5 (Real) -> Circle 'o'
        mask_era5 = (all_labels == 0) & (all_hours == h)
        if np.any(mask_era5):
            plt.scatter(
                emb2d[mask_era5, 0],
                emb2d[mask_era5, 1],
                color=color,
                marker='o',
                s=20,
                alpha=0.6,
                edgecolors='none', 
                label=f"ERA5 (Real) - {h}h"
            )

        # 2. Plot Aurora (Fake) -> Cross 'x'
        mask_aurora = (all_labels == 1) & (all_hours == h)
        if np.any(mask_aurora):
            plt.scatter(
                emb2d[mask_aurora, 0],
                emb2d[mask_aurora, 1],
                color=color,
                marker='x',
                s=40,
                alpha=0.8,
                linewidth=1.5,
                label=f"Aurora (Fake) - {h}h"
            )

    plt.title(f"{args.method.upper()} Embedding\nColor = Forecast Hour | Shape = Source")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # Legend outside to avoid clutter
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    # plt.show() 

    print(f"Saved plot → {args.output_path}")


if __name__ == "__main__":
    main()