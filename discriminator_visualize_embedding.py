import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

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

    # ----------------------
    # Build dataset
    # ----------------------
    loader = build_val_loader(args)

    latitude, longitude = loader.dataset.get_latitude_longitude()
    levels = loader.dataset.get_levels()
    static_data = loader.dataset.get_static_vars_ds()

    all_feats = []
    all_labels = []

    # --------------------------------------------------------------
    # Extract feature vectors for ALL validation samples
    # --------------------------------------------------------------
    pbar = tqdm(
        loader,
        desc="Extracting features"
        # ncols=120,
    )

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

        all_feats.append(feats)
        all_labels.append(labels)

    all_feats = torch.cat(all_feats).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print(f"Feature shape = {all_feats.shape}")


    # --------------------------------------------------------------
    # Dimensionality reduction
    # --------------------------------------------------------------
    print(f"Reducing with {args.method.upper()} ...")

    if args.method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
        emb2d = reducer.fit_transform(all_feats)

    elif args.method == "umap":
        reducer = umap.UMAP(n_components=2, min_dist=0.1, metric="cosine")
        emb2d = reducer.fit_transform(all_feats)


    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------
    plt.figure(figsize=(9, 9))
    plt.scatter(
        emb2d[:, 0],
        emb2d[:, 1],
        c=all_labels,
        cmap="coolwarm",
        s=8,
        alpha=0.75,
    )
    plt.title(f"{args.method.upper()} Embedding of ResNetDiscriminator Features")
    plt.colorbar(label="Label (0=ERA5, 1=Aurora)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    # plt.savefig("embedding_plot.png", dpi=200)
    plt.savefig(args.output_path, dpi=300)
    plt.show()

    print(f"Saved plot → {args.output_path}")


if __name__ == "__main__":
    main()
