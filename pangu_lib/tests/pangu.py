from datetime import datetime
from os.path import join
from typing import Optional

import torch

from ..models import pangu
from ..models.lightning_modules import PanguLightningModule
from ..utils.constants import local_data_properties
from ..utils.datasets import ERA5TWDataset
from ..utils.logging import log_message

__all__ = ["test_pangu_model", "test_pangu_output"]

upper_vars = local_data_properties.upper_vars
surface_vars = local_data_properties.surface_vars
pressure_levels = local_data_properties.pressure_levels


def create_model_pl_module(ckpt_path: str) -> PanguLightningModule:
    model_pl_module = PanguLightningModule.load_from_checkpoint(
        ckpt_path,
    )
    model_pl_module.freeze()
    return model_pl_module


def test_pangu_model(device: torch.device, data_dir: Optional[str] = None) -> None:
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

    model = pangu.PanguModel(
        data_spatial_shape=(pl, lat, lon),
        upper_vars=upper_vars,
        surface_vars=surface_vars,
        depths=[2, 6],
        heads=[6, 12],
        embed_dim=192,
        patch_shape=(2, 4, 4),
        window_size=(2, 6, 12),
        constant_mask_paths=mask_paths,
        smoothing_kernel_size=3,
        segmented_smooth=False,
        segmented_smooth_boundary_width=8,
        learned_smooth=False,
        residual=True,
        res_conn_after_smooth=False,
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

    log_message(f"{y_upper.shape=} {y_surface.shape=}")
    log_message(f"Finish forward pass in {(datetime.now()-st).total_seconds()} seconds")

    (y_upper.sum()+y_surface.sum()).backward()
    log_message(f"Finish backward pass in {(datetime.now()-st).total_seconds()} seconds")


def test_pangu_output(device: torch.device, data_dir: str, ckpt: str) -> None:
    dataset = ERA5TWDataset(
        root_dir=data_dir,
        start_date_hour=datetime(2017, 1, 8, 0),
        end_date_hour=datetime(2017, 1, 10, 23),
        upper_variables=upper_vars,
        surface_variables=surface_vars,
        standardize=True,
    )
    (input_upper, input_surface), _ = dataset[0]  # type: ignore
    input_upper = input_upper.unsqueeze(0).to(device)
    input_surface = input_surface.unsqueeze(0).to(device)
    pl_ckpt = create_model_pl_module(ckpt)
    pl_ckpt = pl_ckpt.to(device)
    pl_ckpt.eval()
    output_upper, output_surface = pl_ckpt.model(input_upper, input_surface)
    print(f"{output_upper.shape=} {output_surface.shape=}")
    print(f"{output_upper.sum()} {output_surface.sum()}")
