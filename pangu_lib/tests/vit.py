from datetime import datetime
from os.path import join
from typing import Optional

import torch

from ..models.vit import VisionTransformer3D
from ..utils.logging import log_message


def test_vit_model(device: torch.device, data_dir: Optional[str] = None) -> None:
    pl, lat, lon = 8, 141, 181
    upper_vars, surface_vars = 5, 4
    batch_size = 2
    mask_paths = None
    if data_dir is not None:
        mask_paths = [
            join(data_dir, "constant_masks", "land_mask.npy"),
            join(data_dir, "constant_masks", "soil_type.npy"),
            join(data_dir, "constant_masks", "topography.npy"),
        ]

    st = datetime.now()

    model = VisionTransformer3D(
        data_spatial_shape=(pl, lat, lon),
        upper_vars=upper_vars,
        surface_vars=surface_vars,
        depth=12,
        heads=3,
        embed_dim=192,
        patch_shape=(2, 4, 4),
        constant_mask_paths=mask_paths,
        smoothing_kernel_size=None,
    )
    x_upper = torch.randn(batch_size, pl, lat, lon, upper_vars)
    x_surface = torch.randn(batch_size, lat, lon, surface_vars)

    model = model.to(device)
    x_upper = x_upper.to(device)
    x_surface = x_surface.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Total parameters: {total_params:.1e}, Trainable parameters: {trainable_params:.1e}")
    log_message(f"Finish init in {(datetime.now()-st).total_seconds()} seconds")

    y_upper, y_surface = model(x_upper, x_surface)

    log_message(f"Finish forward pass in {(datetime.now()-st).total_seconds()} seconds")

    (y_upper.sum()+y_surface.sum()).backward()
    log_message(f"Finish backward pass in {(datetime.now()-st).total_seconds()} seconds")
