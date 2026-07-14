import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import linalg
import matplotlib.pyplot as plt
from safetensors.torch import load_file

# ---- Your project imports
from datasets.DiscriminatorDataset import (
    AuroraPredictionDataset,
    ERA5TWDataset,
    DiscriminatorDataset
)
from discriminator.ResNet_discriminator import ResNetDiscriminator
from aurora.batch import Batch, Metadata

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID-like Distance and Logit Confidence")

    # Model & Data paths
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--Aurora_input_dir', type=str, required=True)
    parser.add_argument('--data_root_dir', type=str, required=True)
    
    # Evaluation Window
    parser.add_argument('--val_start_date_hour', type=str, required=True)
    parser.add_argument('--val_end_date_hour', type=str, required=True)
    
    # Model Config (Must match training)
    parser.add_argument('--forecast_hour', nargs='+', type=int, default=[6])
    parser.add_argument('--upper_variables', nargs='*', default=['u','v','t','q','z'])
    parser.add_argument('--surface_variables', nargs='*', default=['t2m','u10','v10','msl'])
    parser.add_argument('--static_variables', nargs='*', default=['lsm','slt','z'])
    parser.add_argument('--latitude', nargs=2, type=float, default=[39.75,5])
    parser.add_argument('--longitude', nargs=2, type=float, default=[100,144.75])
    parser.add_argument('--levels', nargs='*', type=int, default=[1000,925,850,700,500,300,150,50])
    parser.add_argument('--backbone', type=str, default='resnet50')
    
    # System
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_plot", type=str, default="logit_histogram.png")

    return parser.parse_args()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Fréchet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("FID calculation producing infinite values; adding epsilon offset.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight complex component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def build_val_loader(args):
    """Reuses your existing dataset logic"""
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

    return DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    print(f"Loading model from {args.ckpt}...")
    model = ResNetDiscriminator(
        surface_variables=args.surface_variables,
        upper_variables=args.upper_variables,
        levels=args.levels,
        backbone_name=args.backbone,
        pretrained=False,
    )

    if args.ckpt.endswith(".safetensors"):
        state_dict = load_file(args.ckpt, device="cpu")
        model.load_state_dict(state_dict)
    else:
        # Fallback for standard .pt files if needed
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    
    model.to(device)
    model.eval()

    # 2. Prepare Data
    loader = build_val_loader(args)
    latitude, longitude = loader.dataset.get_latitude_longitude()
    levels = loader.dataset.get_levels()
    static_data = loader.dataset.get_static_vars_ds()

    all_feats = []
    all_logits = []
    all_labels = []

    # 3. Inference Loop (Extract Features AND Logits)
    print("Running inference to collect embeddings...")
    with torch.no_grad():
        for (inputs, input_dates), labels in tqdm(loader):
            # Construct Batch object
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
            batch_obj = batch_obj.to(device)
            labels = labels.to(device)

            # Get Features (before final layer)
            feats = model._forward_features(batch_obj)
            
            # Get Logits (final layer output)
            # Note: We need to manually call the head on feats to avoid re-running backbone
            # Assuming model.head is the final layer. 
            # If your ResNetDiscriminator structure is different, just run model(batch_obj) separately.
            logits = model(batch_obj)

            all_feats.append(feats.cpu().numpy())
            all_logits.append(logits.view(-1).cpu().numpy())
            all_labels.append(labels.view(-1).cpu().numpy())

    # Concatenate
    features = np.concatenate(all_feats, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 4. Split into Real (0) and Fake (1)
    # Adjust based on your dataset: usually 0=ERA5(Real), 1=Aurora(Fake)
    real_idx = (labels == 0)
    fake_idx = (labels == 1)

    real_features = features[real_idx]
    fake_features = features[fake_idx]
    
    fake_logits = logits[fake_idx]

    print(f"\nStats: N_Real={len(real_features)}, N_Fake={len(fake_features)}")

    # 5. Calculate Distance Metrics
    print("Calculating Statistics...")
    
    # Calculate Mean and Covariance
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # Metric A: Euclidean Distance of Means
    # Simple measure of how far the "centers" of the clusters are
    euclidean_dist = np.linalg.norm(mu_real - mu_fake)

    # Metric B: Fréchet Distance (FID-like)
    # Measures both center distance and spread/shape overlap
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    print("-" * 40)
    print(f"RESULTS FOR CHECKPOINT: {args.ckpt}")
    print("-" * 40)
    print(f"1. Euclidean Distance of Means: {euclidean_dist:.4f}")
    print(f"2. Fréchet Feature Distance:    {fid_score:.4f}")
    print("-" * 40)

    # 6. Plot Logit Histogram (The "Confidence" argument)
    print(f"Plotting logit histogram to {args.output_plot}...")
    plt.figure(figsize=(10, 6))
    
    # Plot 'Fake' logits (Aurora)
    plt.hist(fake_logits, bins=50, color='red', alpha=0.7, label='Generated (Aurora) Logits')
    
    # Optional: Plot Real logits to show separation
    real_logits = logits[real_idx]
    plt.hist(real_logits, bins=50, color='blue', alpha=0.5, label='Real (ERA5) Logits')

    plt.axvline(0, color='k', linestyle='--', label='Decision Boundary')
    plt.title(f"Discriminator Logit Distribution\n(FID: {fid_score:.2f})")
    plt.xlabel("Logit Value ( <0 = Predicted Real, >0 = Predicted Fake )")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.output_plot)
    print("Done.")

if __name__ == "__main__":
    main()