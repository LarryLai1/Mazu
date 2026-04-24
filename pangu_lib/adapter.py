import torch
import torch.nn as nn
from aurora import Batch
import numpy as np
import pandas as pd

class PanguAuroraAdapter(nn.Module):
    def __init__(self, pangu_model, surf_stats = None):
        super().__init__()
        self.model = pangu_model
        
        # We need these stats to perform normalization inside the forward pass
        self.surf_stats = surf_stats if surf_stats is not None else {}

        # Define variable ordering for Pangu
        # self.surf_order = ["msl", "10u", "10v", "2t"]
        self.surf_order = ["2t", "10u", "10v", "msl"]
        # self.upper_order = ["z", "q", "t", "u", "v"]
        self.upper_order = ["u", "v", "t", "q", "z"]

        self.patch_size = 4
        self.lead_time = 1

    def batch_transform_hook(self, batch: Batch) -> Batch:
        """Transform the batch right after receiving it and before normalisation.

        This function should be idempotent.
        """
        return batch

    def forward(self, batch: Batch) -> Batch:
        device = next(self.model.parameters()).device
        
        # ==========================================================
        # 1. NORMALIZE INPUT
        # ==========================================================
        # The dataloader gives us physical values (e.g., 100000 Pa).
        # Pangu needs normalized values (e.g., 0.1).
        # We use the Batch object's built-in method to handle this.
        norm_batch = batch.normalise(self.surf_stats)

        # ==========================================================
        # 2. DICT -> TENSOR (Using NORMALIZED data)
        # ==========================================================
        
        # --- Surface ---
        surf_tensors = []
        for var_name in self.surf_order:
            # We pull from norm_batch, NOT batch
            data = norm_batch.surf_vars[var_name] 
            if data.shape[1] == 1:
                data = data.squeeze(1)
            else:
                data = data[:, -1, :, :]
            surf_tensors.append(data)
        input_surface = torch.stack(surf_tensors, dim=-1).to(device)

        # --- Upper Air ---
        upper_tensors = []
        for var_name in self.upper_order:
            data = norm_batch.atmos_vars[var_name]
            if data.shape[1] == 1:
                data = data.squeeze(1)
            else:
                data = data[:, -1, :, :, :]
            upper_tensors.append(data)
        input_upper = torch.stack(upper_tensors, dim=-1).to(device)

        # ==========================================================
        # 3. RUN MODEL
        # ==========================================================
        # Pangu runs in "Normalized Space"
        pred_upper, pred_surface = self.model(input_upper, input_surface)

        # ==========================================================
        # 4. TENSOR -> DICT (Still in NORMALIZED Space)
        # ==========================================================
        
        pred_surf_vars = {}
        for i, var_name in enumerate(self.surf_order):
            pred_surf_vars[var_name] = pred_surface[..., i].unsqueeze(1)

        pred_atmos_vars = {}
        for i, var_name in enumerate(self.upper_order):
            pred_atmos_vars[var_name] = pred_upper[..., i].unsqueeze(1)


        batch.metadata.rollout_step += 1  # Increment rollout step for the metadata of the next prediction
        # print(f"Rollout Step: {batch.metadata.rollout_step}")
        # batch.metadata.time=tuple(t + self.lead_time for t in batch.metadata.time),
        batch.metadata.time = tuple(t + pd.Timedelta(hours=self.lead_time) for t in batch.metadata.time)

        # Create a temporary batch containing the normalized predictions
        pred_batch_norm = Batch(
            surf_vars=pred_surf_vars,
            atmos_vars=pred_atmos_vars,
            static_vars=batch.static_vars,
            metadata=batch.metadata
        )

        # ==========================================================
        # 5. UN-NORMALIZE OUTPUT
        # ==========================================================
        # The training loop expects physical values so it can calculate loss properly.
        # We convert Normalized Predictions -> Physical Predictions.
        pred_batch_physical = pred_batch_norm.unnormalise(self.surf_stats)

        return pred_batch_physical