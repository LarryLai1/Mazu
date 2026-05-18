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
# 1. FID Math Helper (Unchanged)
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
# 2. Hook to Capture Swin3D Backbone Features
# --------------------------------------------------------------
def attach_swin_output_hook(model: torch.nn.Module):
    """
    Hook the Swin3DTransformerBackbone itself to capture its final output tokens.
    Returns (hook_handle, output_tokens_buf).
    """
    # Locate the Swin3D Backbone module inside Aurora
    swin_modules = [
        m for m in model.modules()
        if m.__class__.__name__ == "Swin3DTransformerBackbone"
    ]
    if not swin_modules:
        raise RuntimeError(
            "Could not find a module named 'Swin3DTransformerBackbone' inside Aurora model."
        )

    swin_backbone = swin_modules[0]
    output_tokens_buf = {}

    def backbone_output_hook(module, inputs, output):
        # Swin3D forward returns tokens x: (B, L, D_out)
        # We detach immediately to save memory/graph connection
        output_tokens_buf["tokens"] = output.detach().cpu()

    handle = swin_backbone.register_forward_hook(backbone_output_hook)
    return handle, output_tokens_buf

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
    
    if args.use_pretrained_weight:
        print("Loading pretrained weights provided by Microsoft Aurora...")
        model.load_checkpoint(
            "microsoft/aurora",
            "aurora-0.25-small-pretrained.ckpt",
            strict=False,
        )
    elif args.checkpoint_path:
        print(f"Loading checkpoint for Backbone FID: {args.checkpoint_path}")
        if args.checkpoint_path.endswith(".safetensors"):
            model.load_state_dict(load_file(args.checkpoint_path), strict=False)
        else:
            model.load_checkpoint_local(args.checkpoint_path, strict=False)
    return model

def build_val_loader(args):
    # Identical to previous script
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
# 4. Main
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # Standard args
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
    
    # 1. Load Model
    model = create_model(args).to(device).eval()
    
    # 2. Attach Swin3D Backbone Hook (Changed from Perceiver)
    hook_handle, hook_buf = attach_swin_output_hook(model)

    # 3. Load Data
    loader = build_val_loader(args)

    # Containers
    feats_real = [] # ERA5
    feats_fake = [] # Aurora

    print(f"Using the prediction: {args.Aurora_input_dir}")
    print(f"Extracting Swin3D Backbone features for forecast hours: {args.forecast_hour}")

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

            # Clear buffer
            hook_buf.clear()

            # Forward pass (triggers Swin backbone hook)
            _ = model(batch_obj)
            
            if "tokens" in hook_buf:
                # Shape: (B, L, D_out) -> Mean pool -> (B, D_out)
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

    print(f"\nBackbone Feature Shapes -> Real: {real_arr.shape}, Fake: {fake_arr.shape}")

    # Compute Statistics
    print("Computing Statistics...")
    mu_real, sigma_real = np.mean(real_arr, axis=0), np.cov(real_arr, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_arr, axis=0), np.cov(fake_arr, rowvar=False)

    # Compute FID
    print("Calculating Fréchet Distance (Swin3D Backbone)...")
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    print("="*40)
    print(f"Forecast Hour(s): {args.forecast_hour}")
    print(f"Backbone FID Score: {fid:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()