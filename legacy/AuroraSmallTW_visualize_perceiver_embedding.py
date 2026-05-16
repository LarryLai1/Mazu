import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.manifold import TSNE
import umap.umap_ as umap

# Assuming these exist in your environment
from datasets.DiscriminatorDataset import (
    AuroraPredictionDataset,
    ERA5TWDataset,
    DiscriminatorDataset
)
from aurora.batch import Batch, Metadata
from aurora.model.aurora import AuroraSmall

# Safetensors might be needed if you use it, otherwise standard torch.load works
from safetensors.torch import load_file
import pandas as pd


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
        # Support both .pt and .safetensors
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
# HOOK: Perceiver Encoder (Latent Space)
# --------------------------------------------------------------
def attach_perceiver_encoder_hook(model: torch.nn.Module):
    """
    Hooks model.encoder to capture the Latent Embeddings.
    Output of encoder is (B, L, D).
    """
    if not hasattr(model, 'encoder'):
        raise RuntimeError("Model does not have an 'encoder' attribute.")

    encoded_buf = {}

    def encoder_hook(module, inputs, output):
        # Output is the latent tensor x: (B, L, D)
        encoded_buf["tokens"] = output.detach().cpu()

    handle = model.encoder.register_forward_hook(encoder_hook)
    return handle, encoded_buf


# --------------------------------------------------------------
# HOOK: Perceiver Decoder (Physical/Prediction Space)
# --------------------------------------------------------------
def attach_perceiver_decoder_hook(model: torch.nn.Module):
    """
    Hooks model.decoder to capture the Reconstruction/Prediction.
    Output of decoder is a 'Batch' object.
    """
    if not hasattr(model, 'decoder'):
        raise RuntimeError("Model does not have a 'decoder' attribute.")

    decoded_buf = {}

    def decoder_hook(module, inputs, output):
        # Output is a Batch object containing surf_vars and atmos_vars
        # We cannot detach a Batch object directly, we must process it later.
        # We store the reference, but must detach contents immediately to save memory.
        
        # Helper to detach dict of tensors
        def detach_dict(d):
            return {k: v.detach().cpu() for k, v in d.items()}
        
        surf = detach_dict(output.surf_vars)
        atmos = detach_dict(output.atmos_vars)
        
        decoded_buf["surf"] = surf
        decoded_buf["atmos"] = atmos

    handle = model.decoder.register_forward_hook(decoder_hook)
    return handle, decoded_buf


# --------------------------------------------------------------
# Helper: Flatten Batch to Vector
# --------------------------------------------------------------
def flatten_decoder_output(surf_dict, atmos_dict):
    """
    Flattens all surface and atmos variables into a single vector per sample.
    Returns: (B, Flattened_Dim)
    """
    # 1. Flatten Surface: List of (B, 1, H, W) -> List of (B, H*W)
    surf_vecs = [v.flatten(start_dim=1) for v in surf_dict.values()]
    
    # 2. Flatten Atmos: List of (B, 1, Levels, H, W) -> List of (B, Levels*H*W)
    atmos_vecs = [v.flatten(start_dim=1) for v in atmos_dict.values()]
    
    # 3. Concatenate all features
    all_vecs = surf_vecs + atmos_vecs
    if not all_vecs:
        return None
        
    return torch.cat(all_vecs, dim=1)


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
    # parser.add_argument("--ckpt", type=str, default=None)

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
    parser.add_argument("--encoder_vis_path", type=str, default="viz_perceiver_encoder.png")
    parser.add_argument("--decoder_vis_path", type=str, default="viz_perceiver_decoder.png")

    parser.add_argument("--draw_encoder", action="store_true", default=False)
    parser.add_argument("--draw_decoder", action="store_true", default=False)

    return parser.parse_args()


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    args = parse_args()
    # if args.ckpt is not None and args.checkpoint_path is None:
    #     args.checkpoint_path = args.ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = create_model(args)
    model.to(device)
    model.eval()

    # 2. Attach Perceiver Hooks
    print("Attaching Perceiver Hooks...")
    if args.draw_encoder:
        print(" - Encoder Hook Attached")
        hook_enc, enc_buf = attach_perceiver_encoder_hook(model)
    if args.draw_decoder:
        print(" - Decoder Hook Attached")
        hook_dec, dec_buf = attach_perceiver_decoder_hook(model)

    # 3. Load Data
    loader = build_val_loader(args)
    latitude, longitude = loader.dataset.get_latitude_longitude()
    levels = loader.dataset.get_levels()
    static_data = loader.dataset.get_static_vars_ds()

    all_feats_enc = []
    all_feats_dec = []
    all_labels = []

    # 4. Extract Loop
    pbar = tqdm(loader, desc="Extracting Perceiver Features")

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
        if args.draw_decoder:
            dec_buf.clear()

        # Run Model (Forward pass triggers hooks)
        with torch.no_grad():
            _ = model(batch_obj)

        # Check Hooks
        if args.draw_encoder and "tokens" not in enc_buf:
            continue
        if args.draw_decoder and "surf" not in dec_buf:
            continue
        # if "tokens" not in enc_buf:
        #     continue # Should not happen unless skip
        # if "surf" not in dec_buf:
        #     continue

        # --- Process Encoder Features (Latent Space) ---
        # Shape: (B, L, D). We mean-pool over tokens (L) to get (B, D)
        if args.draw_encoder:
            tokens_enc = enc_buf["tokens"] # CPU tensor
            feats_enc = tokens_enc.mean(dim=1) 
            all_feats_enc.append(feats_enc)

        # --- Process Decoder Features (Physical Space) ---
        # Output is a dict of variables. We flatten everything into one vector.
        # This represents the "Reconstruction State"
        if args.draw_decoder:
            feats_dec = flatten_decoder_output(dec_buf["surf"], dec_buf["atmos"])
            all_feats_dec.append(feats_dec)

        all_labels.append(labels.cpu())

    # Remove hooks
    if args.draw_encoder:
        hook_enc.remove()
    if args.draw_decoder:
        hook_dec.remove()

    # Concatenate all batches
    if len(all_feats_enc) == 0:
        print("No features extracted. Check dataset.")
        return

    if args.draw_encoder:
        all_feats_enc = torch.cat(all_feats_enc).numpy()
    if args.draw_decoder:
        all_feats_dec = torch.cat(all_feats_dec).numpy()


    all_labels = torch.cat(all_labels).numpy()

    if args.draw_encoder:
        print(f"\nFinal Encoder Matrix: {all_feats_enc.shape}")
    if args.draw_decoder:
        print(f"Final Decoder Matrix: {all_feats_dec.shape}")

    # 5. Dimensionality Reduction
    def reduce(method, feats):
        print(f"Running {method.upper()} on shape {feats.shape}...")
        # Check for NaNs just in case
        if np.isnan(feats).any():
            print("Warning: NaNs found in features, replacing with 0.")
            feats = np.nan_to_num(feats)
            
        if method == "tsne":
            # Perplexity must be < n_samples
            perp = min(30, feats.shape[0] - 1) if feats.shape[0] > 1 else 1
            reducer = TSNE(n_components=2, perplexity=perp, learning_rate=200, init='pca', n_jobs=-1)
        else:
            reducer = umap.UMAP(n_components=2, min_dist=0.1, metric="cosine")
        return reducer.fit_transform(feats)

    if args.draw_encoder:
        print("\nReducing Encoder Features...")
        emb2d_enc = reduce(args.method, all_feats_enc)
    if args.draw_decoder:
        print("\nReducing Decoder Features...")
        emb2d_dec = reduce(args.method, all_feats_dec)
    # emb2d_enc = reduce(args.method, all_feats_enc)
    # emb2d_dec = reduce(args.method, all_feats_dec)

    # 6. Plot Encoder
    if args.draw_encoder:
        plt.figure(figsize=(9, 9))
        plt.scatter(
            emb2d_enc[:, 0], emb2d_enc[:, 1],
            c=all_labels, cmap="coolwarm", s=15, alpha=0.75, edgecolors='k', linewidth=0.1
        )
        plt.title(f"{args.method.upper()} – Perceiver Encoder (Latent Space)")
        plt.colorbar(label="Label (0=ERA5, 1=Aurora)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(args.encoder_vis_path, dpi=300)
        print(f"Saved encoder plot -> {args.encoder_vis_path}")

    # 7. Plot Decoder
    if args.draw_decoder:
        plt.figure(figsize=(9, 9))
        plt.scatter(
            emb2d_dec[:, 0], emb2d_dec[:, 1],
            c=all_labels, cmap="coolwarm", s=15, alpha=0.75, edgecolors='k', linewidth=0.1
        )
        plt.title(f"{args.method.upper()} – Perceiver Decoder (Physical Space)")
        plt.colorbar(label="Label (0=ERA5, 1=Aurora)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(args.decoder_vis_path, dpi=300)
        print(f"Saved decoder plot -> {args.decoder_vis_path}")

if __name__ == "__main__":
    main()