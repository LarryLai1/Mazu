import torch
import argparse
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from safetensors.torch import load_file

# Reuse your existing imports
from datasets.DiscriminatorDataset import (
    AuroraPredictionDataset, ERA5TWDataset, DiscriminatorDataset
)
from aurora.batch import Batch, Metadata
from aurora.model.aurora import AuroraSmall

# --------------------------------------------------------------
# 1. FID Math Helper
# --------------------------------------------------------------
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Fréchet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

# --------------------------------------------------------------
# 2. Hook to Capture Latent Features
# --------------------------------------------------------------
def attach_perceiver_encoder_hook(model):
    if not hasattr(model, 'encoder'):
        raise RuntimeError("Model does not have an 'encoder' attribute.")
    encoded_buf = {}
    def encoder_hook(module, inputs, output):
        # Output is (B, L, D). Detach and move to CPU.
        encoded_buf["tokens"] = output.detach().cpu()
    handle = model.encoder.register_forward_hook(encoder_hook)
    return handle, encoded_buf

# --------------------------------------------------------------
# 3. Model & Data Setup
# --------------------------------------------------------------
def create_model(args):
    # Standard creation
    model = AuroraSmall(
        use_lora=args.use_lora,
        bf16_mode=args.bf16_mode,
        timestep=pd.Timedelta(hours=args.timestep_hours),
        stabilise_level_agg=args.stabilise_level_agg,
    )
    if args.checkpoint_path:
        print(f"Loading checkpoint for Encoder: {args.checkpoint_path}")
        if args.checkpoint_path.endswith(".safetensors"):
            model.load_state_dict(load_file(args.checkpoint_path), strict=False)
        else:
            model.load_checkpoint_local(args.checkpoint_path, strict=False)
    return model

def build_val_loader(args):
    # This uses your existing logic where DiscriminatorDataset mixes Aurora and ERA5
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
        shuffle=False, # Shuffle false is fine for FID as long as we get enough samples
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader

# --------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # Add your standard args
    parser.add_argument('--Aurora_input_dir', type=str, required=True)
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--val_start_date_hour', type=str, required=True)
    parser.add_argument('--val_end_date_hour', type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    
    # Model/Data args
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--bf16_mode", action="store_true")
    parser.add_argument("--timestep_hours", type=int, default=1)
    parser.add_argument("--stabilise_level_agg", action="store_true")
    parser.add_argument("--use_pretrained_weight", action="store_true")
    parser.add_argument('--forecast_hour', nargs='+', type=int, default=[6])
    parser.add_argument('--upper_variables', nargs='*', default=['u', 'v', 't', 'q', 'z'])
    parser.add_argument('--surface_variables', nargs='*', default=['t2m', 'u10', 'v10', 'msl'])
    parser.add_argument('--static_variables', nargs='*', default=['lsm', 'slt', 'z'])
    parser.add_argument('--latitude', nargs=2, type=float, default=[39.75, 5])
    parser.add_argument('--longitude', nargs=2, type=float, default=[100, 144.75])
    parser.add_argument('--levels', nargs='*', type=int, default=[1000, 925, 850, 700, 500, 300, 150, 50])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model (Acts as the fixed Feature Extractor / "Ruler")
    model = create_model(args).to(device).eval()
    
    # 2. Attach Hook
    hook_handle, hook_buf = attach_perceiver_encoder_hook(model)

    # 3. Load Data (Mixed Aurora and ERA5)
    loader = build_val_loader(args)

    # Containers
    feats_real = [] # ERA5
    feats_fake = [] # Aurora

    print(f"Using the prediction: {args.Aurora_input_dir}")
    print(f"Extracting features for forecast hours: {args.forecast_hour}")

    with torch.no_grad():
        for (inputs, input_dates), labels in tqdm(loader):
            # Construct Batch
            latitude, longitude = loader.dataset.get_latitude_longitude()
            levels = loader.dataset.get_levels()
            static_data = loader.dataset.get_static_vars_ds()
            
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
            # labels = labels.to(device)

            # Clear buffer
            hook_buf.clear()

            # Forward pass (just to trigger encoder hook)
            _ = model(batch_obj)
            
            if "tokens" in hook_buf:
                # Shape: (B, L, D) -> Mean pool -> (B, D)
                # Note: You can also try max pool or just taking the first token depending on Aurora's architecture
                current_feats = hook_buf["tokens"].mean(dim=1)
                
                # Split based on Labels
                # Label 0 = ERA5 (Real)
                # Label 1 = Aurora (Fake)
                
                real_mask = (labels == 0)
                fake_mask = (labels == 1)
                
                if real_mask.any():
                    feats_real.append(current_feats[real_mask].cpu().numpy())
                
                if fake_mask.any():
                    feats_fake.append(current_feats[fake_mask].cpu().numpy())

    hook_handle.remove()

    # 4. Calculate FID
    if len(feats_real) == 0 or len(feats_fake) == 0:
        print("Error: Not enough data in one of the classes (Real or Fake).")
        print(f"Real Batches: {len(feats_real)}, Fake Batches: {len(feats_fake)}")
        return

    # Concatenate
    real_arr = np.concatenate(feats_real, axis=0)
    fake_arr = np.concatenate(feats_fake, axis=0)

    print(f"\nFeature Shapes -> Real: {real_arr.shape}, Fake: {fake_arr.shape}")

    # Compute Statistics
    print("Computing Statistics...")
    mu_real, sigma_real = np.mean(real_arr, axis=0), np.cov(real_arr, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_arr, axis=0), np.cov(fake_arr, rowvar=False)

    # Compute FID
    print("Calculating Fréchet Distance...")
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    print("="*40)
    print(f"Forecast Hour(s): {args.forecast_hour}")
    print(f"FID Score: {fid:.4f}")
    print("="*40)

    # Optional: Save result to a simple text file for logging
    # with open("fid_results.txt", "a") as f:
    #     f.write(f"Forecast_Hour: {args.forecast_hour} | FID: {fid:.4f}\n")

if __name__ == "__main__":
    main()