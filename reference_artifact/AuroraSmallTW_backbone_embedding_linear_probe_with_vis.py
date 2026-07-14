import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
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
# 2. The Residual MLP Probe (Same as Perceiver Version)
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

    def forward(self, x, return_embedding=False):
        # 1. Project Input
        x = self.input_proj(x)
        
        # 2. Apply ResBlocks
        for block in self.blocks:
            x = block(x)
        
        # 3. Return high-dim space for Viz
        if return_embedding:
            return x  
            
        # 4. Return classification score
        return self.head(x)

# ==========================================
# 3. Visualization Helpers
# ==========================================
def visualize_manifold(embeddings, labels, acc, save_path="backbone_separation.png", method='tsne', max_points=int(1e6)):
    print(f"\n--- Generating {method.upper()} Visualization ---")
    
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy().flatten()

    # Subsample if massive
    if len(embeddings) > max_points:
        print(f"Subsampling from {len(embeddings)} to {max_points} points...")
        indices = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    print(f"Running {method.upper()}...")
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, min_dist=0.1, metric="cosine")
            proj = reducer.fit_transform(embeddings)
        except ImportError:
            print("WARNING: 'umap-learn' not installed. Falling back to t-SNE.")
            method = 'tsne'
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        perp = min(30, len(embeddings) - 1)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
        proj = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    
    # Plot Real (Blue)
    # plt.scatter(proj[labels==0, 0], proj[labels==0, 1], 
    #             c='dodgerblue', label='Real (ERA5)', alpha=0.6, s=20, edgecolors='k', linewidth=0.1)
    plt.scatter(proj[labels==0, 0], proj[labels==0, 1], 
                c='dodgerblue', label='Ground Truth', alpha=0.6, s=20, edgecolors='k', linewidth=0.1)
    
    # Plot Fake (Red)
    # plt.scatter(proj[labels==1, 0], proj[labels==1, 1], 
    #             c='crimson', label='Fake (Aurora Backbone)', alpha=0.6, s=20, edgecolors='k', linewidth=0.1)
    plt.scatter(proj[labels==1, 0], proj[labels==1, 1], 
                c='crimson', label='Prediction', alpha=0.6, s=20, edgecolors='k', linewidth=0.1)

    # plt.title(f"Backbone Separation ({method.upper()}) - Accuracy: {acc:.2f}%", fontsize=16)

    plt.title(f"Embedding Visualization ({method.upper()})", fontsize = 18)

    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.legend()

    plt.legend(fontsize = 14, markerscale=2.0)

    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.close()

# ==========================================
# 4. Training & Viz Logic
# ==========================================
def train_eval_viz(real_feats, fake_feats, device, args):
    print(f"\n--- Residual MLP Probe Execution (Backbone) ---")
    
    y_real = torch.zeros(real_feats.shape[0], 1).to(device)
    y_fake = torch.ones(fake_feats.shape[0], 1).to(device)
    
    X = torch.cat([real_feats, fake_feats], dim=0)
    y = torch.cat([y_real, y_fake], dim=0)

    input_dim = real_feats.shape[1]
    probe = ResidualMLP(input_dim, hidden_dim=256, num_blocks=2).to(device)

    total_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"\n[Model Stats] ResProbe Size: {total_params / 1e6:.2f}M parameters")
    print(f"[Model Stats] Configuration: Hidden={256}, Blocks={2}, Input Dim={input_dim}\n")

    dataset_full = TensorDataset(X, y)
    indices = torch.randperm(X.size(0))
    split = int(0.8 * X.size(0))
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    test_ds  = TensorDataset(X[test_idx], y[test_idx])
    
    probe_loader_train = DataLoader(train_ds, batch_size=1024, shuffle=True)
    probe_loader_test  = DataLoader(test_ds, batch_size=1024, shuffle=False)

    optimizer = optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.probe_epochs)

    print("Starting Training...")
    probe.train()
    for epoch in tqdm(range(args.probe_epochs), desc="Probe Training"):
        for bx, by in probe_loader_train:
            optimizer.zero_grad()
            logits = probe(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

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
    print(f"Probe Accuracy: {acc:.2f}%")

    if args.visualize:
        print("Extracting learned features for visualization...")
        full_loader = DataLoader(dataset_full, batch_size=1024, shuffle=False)
        all_embs = []
        all_lbls = []
        
        with torch.no_grad():
            for bx, by in full_loader:
                emb = probe(bx, return_embedding=True)
                all_embs.append(emb.cpu())
                all_lbls.append(by.cpu())
        
        final_embs = torch.cat(all_embs, dim=0)
        final_lbls = torch.cat(all_lbls, dim=0)
        
        visualize_manifold(final_embs, final_lbls, acc, save_path=args.viz_save_path, method=args.viz_method)

# ==========================================
# 5. Helpers (Updated Hook for Backbone)
# ==========================================
def attach_swin_backbone_hook(model):
    """
    Finds the Swin3DTransformerBackbone module and attaches a hook 
    to capture its output tokens.
    """
    # 1. Find the Backbone Module
    swin_modules = [
        m for m in model.modules()
        if m.__class__.__name__ == "Swin3DTransformerBackbone"
    ]
    
    if not swin_modules:
        # Fallback: Sometimes it might be wrapped differently, check typical attribute names
        if hasattr(model, 'backbone'):
            swin_backbone = model.backbone
        else:
            raise RuntimeError("Could not find 'Swin3DTransformerBackbone' or 'model.backbone' in Aurora.")
    else:
        swin_backbone = swin_modules[0]

    # 2. Attach Hook
    encoded_buf = {}
    def backbone_hook(module, inputs, output):
        # Swin3D output shape is usually (Batch, Tokens, Dim)
        encoded_buf["tokens"] = output.detach() 
        
    handle = swin_backbone.register_forward_hook(backbone_hook)
    print(f"Attached hook to: {swin_backbone.__class__.__name__}")
    return handle, encoded_buf

def create_model(args):
    model = AuroraSmall(
        use_lora=args.use_lora, bf16_mode=args.bf16_mode,
        timestep=pd.Timedelta(hours=args.timestep_hours), stabilise_level_agg=args.stabilise_level_agg,
    )
    if args.checkpoint_path:
        print(f"Loading Backbone: {args.checkpoint_path}")
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
            data_root_dir=args.Aurora_input_dir, start_date_hour=args.val_start_date_hour,
            end_date_hour=args.val_end_date_hour, upper_variables=args.upper_variables,
            surface_variables=args.surface_variables, static_variables=args.static_variables,
            latitude=tuple(args.latitude), longitude=tuple(args.longitude), levels=args.levels, forecast_hour=s_h,
        ))
        val_ERA5_dataset_list.append(ERA5TWDataset(
            data_root_dir=args.data_root_dir, start_date_hour=args.val_start_date_hour,
            end_date_hour=args.val_end_date_hour, upper_variables=args.upper_variables,
            surface_variables=args.surface_variables, static_variables=args.static_variables,
            latitude=tuple(args.latitude), longitude=tuple(args.longitude), levels=args.levels,
        ))
    val_ds = DiscriminatorDataset(val_Aurora_dataset_list, val_ERA5_dataset_list)
    return DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# ==========================================
# 6. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # Basic Args
    parser.add_argument('--Aurora_input_dir', type=str, required=True)
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--val_start_date_hour', type=str, required=True)
    parser.add_argument('--val_end_date_hour', type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--bf16_mode", action="store_true")
    parser.add_argument("--timestep_hours", type=int, default=1)
    parser.add_argument("--stabilise_level_agg", action="store_true")
    parser.add_argument('--forecast_hour', nargs='+', type=int, default=[6])
    parser.add_argument('--upper_variables', nargs='*', default=['u', 'v', 't', 'q', 'z'])
    parser.add_argument('--surface_variables', nargs='*', default=['t2m', 'u10', 'v10', 'msl'])
    parser.add_argument('--static_variables', nargs='*', default=['lsm', 'slt', 'z'])
    parser.add_argument('--latitude', nargs=2, type=float, default=[39.75, 5])
    parser.add_argument('--longitude', nargs=2, type=float, default=[100, 144.75])
    parser.add_argument('--levels', nargs='*', type=int, default=[1000, 925, 850, 700, 500, 300, 150, 50])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Probe & Viz Args
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--save_embeddings_path", type=str, default=None)
    parser.add_argument("--load_embeddings_path", type=str, default=None)
    
    # Visualization Flags
    parser.add_argument("--visualize", action="store_true", help="Enable separation visualization")
    parser.add_argument("--viz_method", type=str, default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--viz_save_path", type=str, default="backbone_separation_viz.png")

    args, _ = parser.parse_known_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Data ---
    if args.load_embeddings_path and os.path.exists(args.load_embeddings_path):
        print(f"Loading cached embeddings: {args.load_embeddings_path}")
        data = torch.load(args.load_embeddings_path, map_location=device)
        tensor_real, tensor_fake = data['real'], data['fake']
    else:
        print("Running Backbone Extraction...")
        model = create_model(args).to(device).eval()
        
        # CHANGED: Attach hook to Swin3D Backbone instead of Perceiver
        handle, buf = attach_swin_backbone_hook(model)
        
        loader = build_val_loader(args)
        
        real_l, fake_l = [], []
        with torch.no_grad():
            for (inp, dates), lbl in tqdm(loader):
                lat, lon = loader.dataset.get_latitude_longitude()
                batch = Batch(
                    surf_vars=inp["surf_vars"], atmos_vars=inp["atmos_vars"],
                    static_vars=loader.dataset.get_static_vars_ds()["static_vars"],
                    metadata=Metadata(lat=lat, lon=lon, time=tuple(map(pd.Timestamp, dates)), atmos_levels=loader.dataset.get_levels())
                ).to(device)
                
                buf.clear()
                _ = model(batch)
                
                if "tokens" in buf:
                    # Global Average Pooling on Backbone Tokens
                    # Shape: [Batch, Tokens, Dim] -> [Batch, Dim]
                    feat = buf["tokens"].mean(1).detach()
                    
                    lbl = lbl.to(device)
                    if (lbl==0).any(): real_l.append(feat[lbl==0])
                    if (lbl==1).any(): fake_l.append(feat[lbl==1])
        
        handle.remove()
        if not real_l or not fake_l: 
            print("Error: Could not collect enough samples for both classes.")
            return
            
        tensor_real = torch.cat(real_l)
        tensor_fake = torch.cat(fake_l)
        
        if args.save_embeddings_path:
            torch.save({'real': tensor_real, 'fake': tensor_fake}, args.save_embeddings_path)

    # --- Run Training & Viz ---
    train_eval_viz(tensor_real, tensor_fake, device, args)

if __name__ == "__main__":
    main()