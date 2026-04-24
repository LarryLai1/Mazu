from collections.abc import Sequence
from math import ceil
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.profiler import record_function

from ..utils.types import Shape_T, WeatherData
from .pangu import Identity, PatchEmbedding, PatchRecovery, SmoothingBlock

__all__ = ["VisionTransformer3D"]


class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer baseline
    """

    def __init__(
        self,
        data_spatial_shape: Shape_T,
        upper_vars: int,
        surface_vars: int,
        depth: int,
        heads: int,
        embed_dim: int,
        patch_shape: Shape_T,
        constant_mask_paths: Optional[Sequence[str]],
        smoothing_kernel_size: Optional[int],
        # segmented_smooth: bool = False,
        # segmented_smooth_boundary_width: Optional[int] = None,
        # residual: bool = False,
        # res_conn_after_smooth: bool = True,
    ) -> None:
        """
        Args:
            data_spatial_shape (tuple[int, int, int]): Shape of the input tensor (xZ, xH, xW).
            upper_vars (int): Number of upper-air variables.
            surface_vars (int): Number of surface variables.
            depth (int): Number of transformer blocks.
            heads (int): Number of heads.
            embed_dim (int): Dimension of the patch embedding.
            patch_shape (tuple[int, int, int]): Shape of the patch (pZ, pH, pW).
            constant_mask_path (Optional[Sequence[str]]): Paths to the constant mask.
            smoothing_kernel_size (Optional[int]): Kernel size for smoothing.
        """
        if constant_mask_paths is not None:
            print(f"Using constant masks: {constant_mask_paths}")
            assert len(constant_mask_paths) == 3

        super().__init__()
        xZ, xH, xW = data_spatial_shape
        pZ, pH, pW = patch_shape
        # extra Z for the surface data
        Z, H, W = ceil(xZ/pZ)+1, ceil(xH/pH), ceil(xW/pW)

        self.patch_embed = PatchEmbedding(
            in_shape=data_spatial_shape,
            dim=embed_dim,
            upper_vars=upper_vars,
            surface_vars=surface_vars,
            patch_shape=patch_shape,
            constant_mask_paths=constant_mask_paths,
        )

        self.encoder = Encoder(
            seq_len=Z*H*W,
            dim=embed_dim,
            dim_mlp=4*embed_dim,
            depth=depth,
            heads=heads,
            dropout=0.,
        )

        self.patch_recover = PatchRecovery(
            dim=embed_dim,
            in_shape=(Z, H, W),
            out_shape=data_spatial_shape,
            upper_vars=upper_vars,
            surface_vars=surface_vars,
            patch_shape=patch_shape,
        )

        if smoothing_kernel_size is not None:
            assert smoothing_kernel_size % 2 == 1
            smoothing_func = nn.AvgPool3d(
                kernel_size=(1, smoothing_kernel_size, smoothing_kernel_size),
                stride=(1, 1, 1),
                padding=(0, smoothing_kernel_size//2, smoothing_kernel_size//2),
                count_include_pad=False
            )

            self.smoothing_layer = SmoothingBlock(smoothing_func=smoothing_func)
        else:
            self.smoothing_layer = Identity()

    def forward(self, input_upper: Tensor, input_surface: Tensor) -> WeatherData:
        """
        Args:
            input_upper (torch.Tensor): Tensor of shape (B, xZ, xH, xW, C_upper).
            input_surface (torch.Tensor): Tensor of shape (B, xH, xW, C_surface).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of upper-air data and surface data.
        """
        x: Tensor = self.patch_embed(input_upper, input_surface)
        x = self.encoder(x)
        output_upper, output_surface = self.patch_recover(x)

        output_upper = output_upper + input_upper
        output_surface = output_surface + input_surface

        output_upper, output_surface = self.smoothing_layer(output_upper, output_surface)
        return output_upper, output_surface


class Encoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        dim: int,
        dim_mlp: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, dim).normal_(std=0.02))
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim_mlp,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ) for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, Z*H*W, dim)
        Returns:
            torch.Tensor: Tensor of shape (B, Z*H*W, dim)
        """
        y = x + self.pos_embedding
        for encoder_layer in self.encoder_layers:
            y = encoder_layer(y)
        return y
