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
# 1. Existing Project Imports
# ==========================================
from datasets.DiscriminatorDataset import (
    AuroraPredictionDataset, ERA5TWDataset, DiscriminatorDataset
)
from aurora.batch import Batch, Metadata
from aurora.model.aurora import AuroraSmall

# ==========================================
# 2. The Residual MLP Probe (ResNet-Style)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_blocks=4):
        super(ResidualMLP, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# ==========================================
# 3. Training & Evaluation Logic
# ==========================================
def train_and_evaluate_probe(real_feats, fake_feats, device, epochs=50, lr=1e-3, 
                             save_path=None, load_path=None, eval_only=False):
    print(f"\n--- Residual MLP Probe Execution ---")
    print(f"Real Samples: {real_feats.shape[0]} | Fake Samples: {fake_feats.shape[0]}")

    # Labels
    y_real = torch.zeros(real_feats.shape[0], 1).to(device)
    y_fake = torch.ones(fake_feats.shape[0], 1).to(device)
    
    X = torch.cat([real_feats, fake_feats], dim=0)
    y = torch.cat([y_real, y_fake], dim=0)

    # Setup Model
    input_dim = real_feats.shape[1]
    probe = ResidualMLP(input_dim, hidden_dim=256, num_blocks=1).to(device)

    # Load Weights (If provided)
    is_trained = False
    if load_path and os.path.exists(load_path):
        print(f"Loading probe checkpoint from: {load_path}")
        try:
            probe.load_state_dict(torch.load(load_path, map_location=device))
            is_trained = True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            if eval_only: return 0.0

    elif eval_only:
        print("ERROR: --eval_only set without valid load_probe_path!")
        return 0.0

    # Dataloader Setup
    if eval_only:
        test_ds = TensorDataset(X, y)
        probe_loader_test = DataLoader(test_ds, batch_size=1024, shuffle=False)
        probe_loader_train = None
    else:
        indices = torch.randperm(X.size(0))
        split = int(0.8 * X.size(0))
        train_idx, test_idx = indices[:split], indices[split:]
        train_ds = TensorDataset(X[train_idx], y[train_idx])
        test_ds  = TensorDataset(X[test_idx], y[test_idx])
        probe_loader_train = DataLoader(train_ds, batch_size=1024, shuffle=True)
        probe_loader_test  = DataLoader(test_ds, batch_size=1024, shuffle=False)

    # Training
    if not eval_only and not is_trained:
        optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print("Starting Training...")
        probe.train()
        for epoch in tqdm(range(epochs), desc="Probe Training"):
            for bx, by in probe_loader_train:
                optimizer.zero_grad()
                logits = probe(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        if save_path:
            print(f"Saving trained probe to: {save_path}")
            path_dir = os.path.dirname(save_path)
            os.makedirs(path_dir, exist_ok=True)
            torch.save(probe.state_dict(), save_path)

    # Evaluation
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in probe_loader_test:
            logits = probe(bx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == by).sum().item()
            total += by.size(0)

    return 100.0 * correct / total

# ==========================================
# 4. Helpers
# ==========================================
def attach_perceiver_encoder_hook(model):
    if not hasattr(model, 'encoder'):
        raise RuntimeError("Model does not have an 'encoder' attribute.")
    encoded_buf = {}
    def encoder_hook(module, inputs, output):
        encoded_buf["tokens"] = output.detach() 
    handle = model.encoder.register_forward_hook(encoder_hook)
    return handle, encoded_buf

def create_model(args):
    model = AuroraSmall(
        use_lora=args.use_lora,
        bf16_mode=args.bf16_mode,
        timestep=pd.Timedelta(hours=args.timestep_hours),
        stabilise_level_agg=args.stabilise_level_agg,
    )
    if args.use_pretrained_weight: pass
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
        val_Aurora_dataset_list.append(AuroraPredictionDataset(
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
        ))
        val_ERA5_dataset_list.append(ERA5TWDataset(
            data_root_dir=args.data_root_dir,
            start_date_hour=args.val_start_date_hour,
            end_date_hour=args.val_end_date_hour,
            upper_variables=args.upper_variables,
            surface_variables=args.surface_variables,
            static_variables=args.static_variables,
            latitude=tuple(args.latitude),
            longitude=tuple(args.longitude),
            levels=args.levels,
        ))
    val_ds = DiscriminatorDataset(val_Aurora_dataset_list, val_ERA5_dataset_list)
    return DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # Data & Model Args
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
    
    # PROBE Args
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--save_probe_path", type=str, default=None)
    parser.add_argument("--load_probe_path", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    
    # EMBEDDING Args (New!)
    parser.add_argument("--save_embeddings_path", type=str, default=None, help="Save extracted feats to .pt file")
    parser.add_argument("--load_embeddings_path", type=str, default=None, help="Load feats from .pt file (skips model)")

    args, _ = parser.parse_known_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor_real = None
    tensor_fake = None

    # ---------------------------------------------------------
    # PART A: Check if we can skip extraction
    # ---------------------------------------------------------
    if args.load_embeddings_path and os.path.exists(args.load_embeddings_path):
        print(f"LOADING EMBEDDINGS FROM: {args.load_embeddings_path}")
        data_dict = torch.load(args.load_embeddings_path, map_location=device)
        tensor_real = data_dict['real']
        tensor_fake = data_dict['fake']
        print("Embeddings loaded. Skipping backbone inference.")
    
    # ---------------------------------------------------------
    # PART B: Run heavy extraction (only if needed)
    # ---------------------------------------------------------
    else:
        print("Loading Backbone Model & Data...")
        model = create_model(args).to(device).eval()
        hook_handle, hook_buf = attach_perceiver_encoder_hook(model)
        loader = build_val_loader(args)

        print("Extracting features (this may take time)...")
        feats_real_list = []
        feats_fake_list = []

        with torch.no_grad():
            for (inputs, input_dates), labels in tqdm(loader):
                # Prepare Batch
                lat, lon = loader.dataset.get_latitude_longitude()
                levels = loader.dataset.get_levels()
                static = loader.dataset.get_static_vars_ds()
                
                batch_obj = Batch(
                    surf_vars=inputs["surf_vars"],
                    atmos_vars=inputs["atmos_vars"],
                    static_vars=static["static_vars"],
                    metadata=Metadata(
                        lat=lat, lon=lon,
                        time=tuple(map(lambda d: pd.Timestamp(d), input_dates)),
                        atmos_levels=levels,
                    ),
                ).to(device)
                
                labels = labels.to(device)
                hook_buf.clear()
                _ = model(batch_obj)
                
                if "tokens" in hook_buf:
                    # STRATEGY: Simple Mean Pooling
                    current_feats = hook_buf["tokens"].mean(dim=1).detach()
                    
                    real_mask = (labels == 0)
                    fake_mask = (labels == 1)
                    
                    if real_mask.any(): feats_real_list.append(current_feats[real_mask])
                    if fake_mask.any(): feats_fake_list.append(current_feats[fake_mask])

        hook_handle.remove()
        
        if len(feats_real_list) > 0 and len(feats_fake_list) > 0:
            tensor_real = torch.cat(feats_real_list, dim=0)
            tensor_fake = torch.cat(feats_fake_list, dim=0)
            
            # Save if requested
            if args.save_embeddings_path:
                print(f"SAVING EMBEDDINGS TO: {args.save_embeddings_path}")
                torch.save({'real': tensor_real, 'fake': tensor_fake}, args.save_embeddings_path)
        else:
            print("Error: Extraction failed (missing data).")
            return

    # ---------------------------------------------------------
    # PART C: Train/Eval Probe
    # ---------------------------------------------------------
    accuracy = train_and_evaluate_probe(
        tensor_real, tensor_fake, device, 
        epochs=args.probe_epochs, lr=args.probe_lr,
        save_path=args.save_probe_path,
        load_path=args.load_probe_path,
        eval_only=args.eval_only
    )

    print("\n" + "="*40)
    print(f" Probe Accuracy: {accuracy:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()