from datetime import datetime, timedelta
from typing import Any

import numpy as np
import torch
from einops import rearrange

from ..models.lightning_modules import PanguLightningModule, ViTLightningModule
from .constants import global_data_properties, local_data_properties
from .datasets import ERA5GlobalDataset, ERA5TWDataset

BOUND_W, BOUND_E = 100, 145
BOUND_S, BOUND_N = 5, 40
BOUNDARY_WIDTH = 8


def get_time_range(start_time: str, end_time: str, total_parts: int, part_idx: int) -> tuple[datetime, datetime]:
    start_dt = datetime.strptime(start_time, r"%Y-%m-%dT%H")
    end_dt = datetime.strptime(end_time, r"%Y-%m-%dT%H")
    assert start_dt < end_dt
    total_hours = (end_dt - start_dt).total_seconds() // 3600
    assert total_hours % total_parts == 0
    hours_per_part = total_hours // total_parts
    start_dt += timedelta(hours=hours_per_part * part_idx)
    end_dt = start_dt + timedelta(hours=hours_per_part)
    return start_dt, end_dt


def prepare_regional_slice(dataset: ERA5GlobalDataset) -> tuple[slice, slice]:
    lon, lat, _ = dataset.get_lon_lat_lev()
    lat_idx = np.where((lat >= -BOUND_N) & (lat <= -BOUND_S))[0]
    lon_idx = np.where((lon >= BOUND_W) & (lon <= BOUND_E))[0]
    lat_slice = slice(lat_idx[0], lat_idx[-1]+1)
    lon_slice = slice(lon_idx[0], lon_idx[-1]+1)
    return lat_slice, lon_slice


def prepare_lev_indices() -> list[int]:
    lev_indices = []
    for lev in local_data_properties.pressure_levels:
        lev_i = global_data_properties.pressure_levels.index(lev)
        lev_indices.append(lev_i)
    return lev_indices


def prepare_boundary_mask(dataset: ERA5TWDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
        boundary_mask_upper: torch.Tensor, shape (h, w, 1)
        boundary_mask_surface: torch.Tensor, shape (1, h, w, 1)
    """
    lon, lat, _ = dataset.get_lon_lat_lev()
    mask_2d = torch.ones((len(lat), len(lon)), dtype=torch.bool, device=device)
    mask_2d[BOUNDARY_WIDTH:-BOUNDARY_WIDTH, BOUNDARY_WIDTH:-BOUNDARY_WIDTH] = False
    return rearrange(mask_2d, "h w -> h w 1"), rearrange(mask_2d, "h w -> 1 h w 1")


def destandardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x*std + mean


def load_pgw_pl_module(ckpt_path: str, device: Any = "cuda:0") -> PanguLightningModule:
    model_pl_module = PanguLightningModule.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False
    )
    model_pl_module.freeze()
    return model_pl_module


def load_vit_pl_module(ckpt_path: str, device: Any = "cuda:0") -> ViTLightningModule:
    model_pl_module = ViTLightningModule.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False
    )
    model_pl_module.freeze()
    return model_pl_module
