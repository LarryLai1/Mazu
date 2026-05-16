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

from safetensors.torch import load_file
import pandas as pd

from aurora.batch import Batch, Metadata
from aurora.model.aurora import AuroraSmall


# --------------------------------------------------------------
# Create AuroraSmall model
# --------------------------------------------------------------
def create_model(args):
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
        print(f"Loading checkpoint: {args.checkpoint_path}")
        if args.checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(args.checkpoint_path)
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_checkpoint_local(args.checkpoint_path, strict=False)

    return model


# --------------------------------------------------------------
# Build dataset
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
# Attach hook to Swin3D last encoder layer
# --------------------------------------------------------------
def attach_swin_encoder_hook(model: torch.nn.Module):
    """
    Hook the last encoder layer of the Swin3D backbone.
    Returns (hook_handle, encoded_tokens_buf).
    """
    swin_modules = [
        m for m in model.modules()
        if m.__class__.__name__ == "Swin3DTransformerBackbone"
    ]
    if not swin_modules:
        raise RuntimeError(
            "Could not find a module named 'Swin3DTransformerBackbone' inside Aurora model."
        )

    swin_backbone = swin_modules[0]
    # Access the last layer of the encoder_layers list
    last_enc_layer = swin_backbone.encoder_layers[-1]

    encoded_tokens_buf = {}

    def last_encoder_hook(module, inputs, output):
        # Basic3DEncoderLayer returns (x, x_unscaled)
        x, x_unscaled = output
        encoded_tokens_buf["tokens"] = x.detach().cpu()  # (B, L, D_enc)

    handle = last_enc_layer.register_forward_hook(last_encoder_hook)
    return handle, encoded_tokens_buf


# --------------------------------------------------------------
# Attach hook to Swin3D backbone *output* (after decoder)
# --------------------------------------------------------------
def attach_swin_output_hook(model: torch.nn.Module):
    """
    Hook the Swin3DTransformerBackbone itself to capture its final output tokens.
    Returns (hook_handle, output_tokens_buf).
    """
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
        x = output
        output_tokens_buf["tokens"] = x.detach().cpu()  # (B, L, D_out)

    handle = swin_backbone.register_forward_hook(backbone_output_hook)
    return handle, output_tokens_buf


# --------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # Aurora model config
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--bf16_mode", action="store_true", default=False)
    parser.add_argument("--timestep_hours", type=int, default=1)
    parser.add_argument("--stabilise_level_agg", action="store_true", default=False)

    parser.add_argument("--use_pretrained_weight", action="store_true", default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # Data config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument('--Aurora_input_dir', type=str, required=True)
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--val_start_date_hour', type=str, required=True)
    parser.add_argument('--val_end_date_hour', type=str, required=True)
    parser.add_argument('--forecast_hour', nargs='+', type=int, default=[6])
    parser.add_argument('--upper_variables', nargs='*', default=['u', 'v', 't', 'q', 'z'])
    parser.add_argument('--surface_variables', nargs='*', default=['t2m', 'u10', 'v10', 'msl'])
    parser.add_argument('--static_variables', nargs='*', default=['lsm', 'slt', 'z'])
    parser.add_argument('--latitude', nargs=2, type=float, default=[39.75, 5])
    parser.add_argument('--longitude', nargs=2, type=float, default=[100, 144.75])
    parser.add_argument('--levels', nargs='*', type=int, default=[1000, 925, 850, 700, 500, 300, 150, 50])

    # Visualization
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    
    # Renamed to match the style of your Perceiver script
    parser.add_argument("--encoder_vis_path", type=str, default="viz_backbone_encoder.png")
    parser.add_argument("--output_vis_path", type=str, default="viz_backbone_output.png")

    # The Toggles
    parser.add_argument("--draw_encoder", action="store_true", default=False)
    parser.add_argument("--draw_output", action="store_true", default=False)

    return parser.parse_args()


# --------------------------------------------------------------
# Main Visualization
# --------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = create_model(args)
    model.to(device)
    model.eval()

    # 2. Attach Swin Hooks (Conditional)
    print("Attaching Swin3D Hooks...")
    if args.draw_encoder:
        print(" - Swin Encoder Hook Attached")
        hook_enc, enc_buf = attach_swin_encoder_hook(model)
    
    if args.draw_output:
        print(" - Swin Output Hook Attached")
        hook_out, out_buf = attach_swin_output_hook(model)

    # 3. Load Data
    loader = build_val_loader(args)
    latitude, longitude = loader.dataset.get_latitude_longitude()
    levels = loader.dataset.get_levels()
    static_data = loader.dataset.get_static_vars_ds()

    all_feats_enc = []
    all_feats_out = []
    all_labels = []

    # 4. Extract Loop
    pbar = tqdm(loader, desc="Extracting Swin3D Features")

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
        batch_obj = batch_obj.to(device)

        # Clear buffers
        if args.draw_encoder:
            enc_buf.clear()
        if args.draw_output:
            out_buf.clear()

        # Run Model
        with torch.no_grad():
            _ = model(batch_obj)

        # Check Hooks & Process
        # 1. Swin Encoder
        if args.draw_encoder:
            if "tokens" in enc_buf:
                tokens_enc = enc_buf["tokens"] # (B, L, D_enc)
                feats_enc = tokens_enc.mean(dim=1) 
                all_feats_enc.append(feats_enc)
        
        # 2. Swin Output
        if args.draw_output:
            if "tokens" in out_buf:
                tokens_out = out_buf["tokens"] # (B, L, D_out)
                feats_out = tokens_out.mean(dim=1)
                all_feats_out.append(feats_out)

        all_labels.append(labels.cpu())

    # Remove hooks
    if args.draw_encoder:
        hook_enc.remove()
    if args.draw_output:
        hook_out.remove()

    # Concatenate Labels
    all_labels = torch.cat(all_labels).numpy()

    # 5. Dimensionality Reduction Helper
    def reduce(method, feats):
        print(f"Running {method.upper()} on shape {feats.shape}...")
        if np.isnan(feats).any():
             print("Warning: NaNs found in features, replacing with 0.")
             feats = np.nan_to_num(feats)
             
        if method == "tsne":
            perp = min(30, feats.shape[0] - 1) if feats.shape[0] > 1 else 1
            reducer = TSNE(n_components=2, perplexity=perp, learning_rate=200, init='pca', n_jobs=-1)
        else:
            reducer = umap.UMAP(n_components=2, min_dist=0.1, metric="cosine")
        return reducer.fit_transform(feats)

    # 6. Process Encoder (if active)
    if args.draw_encoder:
        if len(all_feats_enc) > 0:
            all_feats_enc = torch.cat(all_feats_enc).numpy()
            print(f"\nFinal Swin Encoder Matrix: {all_feats_enc.shape}")
            
            emb2d_enc = reduce(args.method, all_feats_enc)
            
            plt.figure(figsize=(9, 9))
            plt.scatter(
                emb2d_enc[:, 0], emb2d_enc[:, 1],
                c=all_labels, cmap="coolwarm", s=15, alpha=0.75, edgecolors='k', linewidth=0.1
            )
            plt.title(f"{args.method.upper()} – Swin3D Encoder")
            plt.colorbar(label="Label (0=ERA5, 1=Aurora)")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
            plt.tight_layout()
            plt.savefig(args.encoder_vis_path, dpi=300)
            print(f"Saved Swin Encoder plot -> {args.encoder_vis_path}")
        else:
            print("Warning: draw_encoder was True, but no features were captured.")

    # 7. Process Output (if active)
    if args.draw_output:
        if len(all_feats_out) > 0:
            all_feats_out = torch.cat(all_feats_out).numpy()
            print(f"\nFinal Swin Output Matrix: {all_feats_out.shape}")
            
            emb2d_out = reduce(args.method, all_feats_out)
            
            plt.figure(figsize=(9, 9))
            plt.scatter(
                emb2d_out[:, 0], emb2d_out[:, 1],
                c=all_labels, cmap="coolwarm", s=15, alpha=0.75, edgecolors='k', linewidth=0.1
            )
            plt.title(f"{args.method.upper()} – Swin3D Output")
            plt.colorbar(label="Label (0=ERA5, 1=Aurora)")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
            plt.tight_layout()
            plt.savefig(args.output_vis_path, dpi=300)
            print(f"Saved Swin Output plot -> {args.output_vis_path}")
        else:
             print("Warning: draw_output was True, but no features were captured.")

if __name__ == "__main__":
    main()