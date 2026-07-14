import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from safetensors.torch import load_file

# ==========================================
# 1. Reuse Your Existing Imports
# ==========================================
from datasets.DiscriminatorDataset import (
    AuroraPredictionDataset, ERA5TWDataset, DiscriminatorDataset
)
from aurora.batch import Batch, Metadata
from aurora.model.aurora import AuroraSmall

# ==========================================
# 2. The Linear Probe Model (GPU Classifier)
# ==========================================
class LinearProbe(nn.Module):
    """
    A simple linear classifier.
    Input:  Backbone Feature Vector (dim=D)
    Output: Logit (Scalar) - > Sigmoid gives probability of being 'Fake'
    """
    def __init__(self, input_dim):
        super(LinearProbe, self).__init__()
        # One linear layer = Logistic Regression
        self.linear = nn.Linear(input_dim, 1) 

    def forward(self, x):
        return self.linear(x)

# ==========================================
# 3. Training Logic for the Probe
# ==========================================
def train_and_evaluate_probe(real_feats, fake_feats, device, epochs=50, lr=1e-3, 
                             save_path=None, load_path=None):
    """
    Trains the Linear Probe to distinguish Real from Fake.
    - If load_path exists, loads weights and skips training.
    - If save_path is provided, saves weights after training.
    Returns: Test Accuracy (%)
    """
    print(f"\n--- Linear Probe Execution (GPU) ---")
    print(f"Real Samples: {real_feats.shape[0]} | Fake Samples: {fake_feats.shape[0]}")

    # 1. Prepare Labels (0=Real, 1=Fake)
    y_real = torch.zeros(real_feats.shape[0], 1).to(device)
    y_fake = torch.ones(fake_feats.shape[0], 1).to(device)
    
    X = torch.cat([real_feats, fake_feats], dim=0)
    y = torch.cat([y_real, y_fake], dim=0)

    # 2. Shuffle and Split (80% Train, 20% Test)
    indices = torch.randperm(X.size(0))
    split = int(0.8 * X.size(0))
    train_idx, test_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    test_ds  = TensorDataset(X[test_idx], y[test_idx])
    
    probe_loader_train = DataLoader(train_ds, batch_size=1024, shuffle=True)
    probe_loader_test  = DataLoader(test_ds, batch_size=1024, shuffle=False)

    # 3. Setup Model
    input_dim = real_feats.shape[1]
    probe = LinearProbe(input_dim).to(device)
    
    # --- Checkpoint Loading Logic ---
    is_trained = False
    if load_path and os.path.exists(load_path):
        print(f"Found checkpoint at: {load_path}")
        try:
            probe.load_state_dict(torch.load(load_path, map_location=device))
            print("Checkpoint loaded successfully. Skipping training phase.")
            is_trained = True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Proceeding to retrain...")
    
    optimizer = optim.AdamW(probe.parameters(), lr = lr)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Train Loop (Run only if not loaded)
    if not is_trained:
        print("Starting Training...")
        probe.train()
        for epoch in tqdm(range(epochs), desc="Probe Training Epochs"):
            for bx, by in probe_loader_train:
                optimizer.zero_grad()
                logits = probe(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        
        # --- Save Logic ---
        if save_path:
            print(f"Saving trained probe to: {save_path}")
            torch.save(probe.state_dict(), save_path)

    # 5. Evaluation Loop (Always run)
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in probe_loader_test:
            logits = probe(bx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == by).sum().item()
            total += by.size(0)

    acc = 100.0 * correct / total
    return acc

# ==========================================
# 4. Feature Extraction Hook (Swin3D Backbone)
# ==========================================
def attach_swin_output_hook(model: torch.nn.Module):
    """
    Hook the Swin3DTransformerBackbone itself to capture its final output tokens.
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
        # We detach to save memory but keep on device for the probe loop
        output_tokens_buf["tokens"] = output.detach()

    handle = swin_backbone.register_forward_hook(backbone_output_hook)
    return handle, output_tokens_buf

# ==========================================
# 5. Setup Helpers (Model & Data)
# ==========================================
def create_model(args):
    model = AuroraSmall(
        use_lora=args.use_lora,
        bf16_mode=args.bf16_mode,
        timestep=pd.Timedelta(hours=args.timestep_hours),
        stabilise_level_agg=args.stabilise_level_agg,
    )
    if args.use_pretrained_weight:
         # Standard pretrained loading logic if needed
         print("Loading pretrained weights provided by Microsoft Aurora...")
         model.load_checkpoint(
             "microsoft/aurora",
             "aurora-0.25-small-pretrained.ckpt",
             strict=False,
         )
    elif args.checkpoint_path:
        print(f"Loading Backbone Checkpoint: {args.checkpoint_path}")
        if args.checkpoint_path.endswith(".safetensors"):
            model.load_state_dict(load_file(args.checkpoint_path), strict=False)
        else:
            model.load_checkpoint_local(args.checkpoint_path, strict=False)
    return model

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

# ==========================================
# 6. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # Add all your standard args here
    parser.add_argument('--Aurora_input_dir', type=str, required=True)
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--val_start_date_hour', type=str, required=True)
    parser.add_argument('--val_end_date_hour', type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
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
    
    # PROBE Specific Args
    parser.add_argument("--probe_epochs", type=int, default=50, help="Epochs to train linear probe")
    parser.add_argument("--probe_lr", type=float, default=1e-3, help="Learning rate for linear probe")
    
    # ARGS FOR SAVING/LOADING PROBE
    parser.add_argument("--save_probe_path", type=str, default=None, help="Path to save the trained probe weights")
    parser.add_argument("--load_probe_path", type=str, default=None, help="Path to load existing probe weights from")

    args, _ = parser.parse_known_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model (Backbone Target)
    print("Loading Aurora Model for Backbone Extraction...")
    model = create_model(args).to(device).eval()
    
    # 2. Attach Swin3D Hook (Changed from Perceiver)
    hook_handle, hook_buf = attach_swin_output_hook(model)

    # 3. Load Data
    loader = build_val_loader(args)

    # 4. Extract Features
    print(f"Extracting Swin3D Backbone features for forecast hours: {args.forecast_hour}")
    feats_real_list = []
    feats_fake_list = []

    with torch.no_grad():
        for (inputs, input_dates), labels in tqdm(loader):
            # Prepare Batch
            latitude, longitude = loader.dataset.get_latitude_longitude()
            levels = loader.dataset.get_levels()
            static_data = loader.dataset.get_static_vars_ds()
            
            batch_obj = Batch(
                surf_vars=inputs["surf_vars"],
                atmos_vars=inputs["atmos_vars"],
                static_vars=static_data["static_vars"],
                metadata=Metadata(
                    lat=latitude, lon=longitude,
                    time=tuple(map(lambda d: pd.Timestamp(d), input_dates)),
                    atmos_levels=levels,
                ),
            ).to(device)
            
            labels = labels.to(device)

            # Clear buffer and Forward
            hook_buf.clear()
            _ = model(batch_obj)
            
            if "tokens" in hook_buf:
                # Shape: (B, L, D_out) -> Mean pool -> (B, D_out)
                current_feats = hook_buf["tokens"].mean(dim=1)
                
                real_mask = (labels == 0)
                fake_mask = (labels == 1)
                
                if real_mask.any():
                    feats_real_list.append(current_feats[real_mask])
                if fake_mask.any():
                    feats_fake_list.append(current_feats[fake_mask])

    hook_handle.remove()

    if len(feats_real_list) == 0 or len(feats_fake_list) == 0:
        print("Error: Missing data. Need both Real and Fake samples.")
        return

    tensor_real = torch.cat(feats_real_list, dim=0)
    tensor_fake = torch.cat(feats_fake_list, dim=0)

    # 5. Train Probe
    accuracy = train_and_evaluate_probe(
        tensor_real, 
        tensor_fake, 
        device, 
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        save_path=args.save_probe_path, 
        load_path=args.load_probe_path
    )

    # 6. Final Report
    print("\n" + "="*40)
    print(f" EXPERIMENT RESULTS: {args.forecast_hour} hours")
    print(f" Type: Swin3D Backbone Linear Probe")
    print("-" * 40)
    print(f" Linear Probe Test Accuracy: {accuracy:.2f}%")
    print(" (50% = Indistinguishable / Perfect Distribution)")
    print(" (100% = Completely Distinct / Bad Distribution)")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()