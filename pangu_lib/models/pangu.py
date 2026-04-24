from collections.abc import Sequence
from itertools import product
from math import ceil, floor
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import einsum, pack, rearrange, repeat, unpack
from timm.layers import DropPath
from torch import Tensor
from torch.profiler import record_function

from ..utils.constant_mask import LoadConstantMask
from ..utils.types import Shape_T, WeatherData
from .smoothing import LearnedSmoothing, SegmentedSmoothingV2

__all__ = ["PanguModel"]


class PanguModel(nn.Module):
    """
    Implementing https://github.com/198808xc/Pangu-Weather
    """

    def __init__(
        self,
        data_spatial_shape: Shape_T,
        upper_vars: int,
        surface_vars: int,
        depths: Sequence[int],
        heads: Sequence[int],
        embed_dim: int,
        patch_shape: Shape_T,
        window_size: Shape_T,
        constant_mask_paths: Optional[Sequence[str]],
        smoothing_kernel_size: Optional[int],
        segmented_smooth: bool,
        segmented_smooth_boundary_width: Optional[int],
        learned_smooth: bool,
        residual: bool,
        res_conn_after_smooth: bool,
    ) -> None:
        """
        Args:
            data_spatial_shape (tuple[int, int, int]): Shape of the input tensor (xZ, xH, xW).
            upper_vars (int): Number of upper-air variables.
            surface_vars (int): Number of surface variables.
            depths (Sequence[int]): Number of blocks in each level, should contain 2 integers.
            heads (Sequence[int]): Number of heads in each level, should contain 2 integers.
            embed_dim (int): Dimension of the patch embedding.
            patch_shape (tuple[int, int, int]): Shape of the patch (pZ, pH, pW).
            window_size (tuple[int, int, int]): window size
            constant_mask_path (Optional[Sequence[str]]): Paths to the constant mask.
            smoothing_kernel_size (Optional[int]): Kernel size for smoothing.
            segmented_smooth (bool): Use segmented smoothing instead of Avgpool.
            segmented_smooth_boundary_width: Parameter for segmented smoothing.
            learned_smooth (bool): Use learned filter.
            residual (bool): Whether to add residual connection.
            res_conn_after_smooth (bool): Change the position of residual connection.
        """
        if constant_mask_paths is not None:
            print(f"Using constant masks: {constant_mask_paths}")
            assert len(constant_mask_paths) == 3

        assert len(depths) == 2
        assert len(heads) == 2
        assert embed_dim % heads[0] == 0
        assert not (segmented_smooth and learned_smooth)

        super().__init__()
        # (xZ * xH * xW) pixels are partitioned into (Z * H * W) patches of size (pZ, pH,pW)
        xZ, xH, xW = data_spatial_shape
        pZ, pH, pW = patch_shape
        # extra Z for the surface data
        Z, H, W = ceil(xZ/pZ)+1, ceil(xH/pH), ceil(xW/pW)
        drop_path_list = np.linspace(0, 0.2, depths[0]+depths[1])

        if smoothing_kernel_size is not None:
            assert smoothing_kernel_size % 2 == 1
            if segmented_smooth:
                assert segmented_smooth_boundary_width is not None
                smoothing_func = SegmentedSmoothingV2(
                    kernel_size=smoothing_kernel_size,
                    boundary_width=segmented_smooth_boundary_width,
                )
            elif learned_smooth:
                smoothing_func = LearnedSmoothing(
                    kernel_size=smoothing_kernel_size
                )
            else:
                smoothing_func = nn.AvgPool3d(
                    kernel_size=(1, smoothing_kernel_size, smoothing_kernel_size),
                    stride=(1, 1, 1),
                    padding=(0, smoothing_kernel_size//2, smoothing_kernel_size//2),
                    count_include_pad=False
                )
            self.smoothing_layer = SmoothingBlock(smoothing_func=smoothing_func)
        else:
            self.smoothing_layer = Identity()

        self.residual = 1 if residual else 0
        self.res_conn_after_smooth = res_conn_after_smooth

        self.patch_embed = PatchEmbedding(
            in_shape=data_spatial_shape,
            dim=embed_dim,
            upper_vars=upper_vars,
            surface_vars=surface_vars,
            patch_shape=patch_shape,
            constant_mask_paths=constant_mask_paths,
        )

        self.layer1 = EarthSpecificLayer(
            in_shape=(Z, H, W),
            dim=embed_dim,
            depth=depths[0],
            heads=heads[0],
            drop_path_ratio=drop_path_list[:depths[0]],  # type: ignore
            window_size=window_size,
        )

        self.layer2 = EarthSpecificLayer(
            in_shape=(Z, ceil(H/2), ceil(W/2)),
            dim=2*embed_dim,
            depth=depths[1],
            heads=heads[1],
            drop_path_ratio=drop_path_list[-depths[1]:],  # type: ignore
            window_size=window_size,
        )

        self.layer3 = EarthSpecificLayer(
            in_shape=(Z, ceil(H/2), ceil(W/2)),
            dim=2*embed_dim,
            depth=depths[1],
            heads=heads[1],
            drop_path_ratio=drop_path_list[-depths[1]:],  # type: ignore
            window_size=window_size,
        )

        self.layer4 = EarthSpecificLayer(
            in_shape=(Z, H, W),
            dim=embed_dim,
            depth=depths[0],
            heads=heads[0],
            drop_path_ratio=drop_path_list[:depths[0]],  # type: ignore
            window_size=window_size,
        )

        self.downsample = DownSample(
            in_shape=(Z, H, W),
            in_channels=embed_dim,
            out_channels=2*embed_dim,
        )

        self.upsample = UpSample(
            in_channels=2*embed_dim,
            out_shape=(Z, H, W),
            out_channels=embed_dim,
        )

        self.patch_recover = PatchRecovery(
            dim=2*embed_dim,
            in_shape=(Z, H, W),
            out_shape=data_spatial_shape,
            upper_vars=upper_vars,
            surface_vars=surface_vars,
            patch_shape=patch_shape,
        )

    def forward(self, input_upper: Tensor, input_surface: Tensor) -> WeatherData:
        """
        Args:
            input_upper (Tensor): Tensor of shape (B, xZ, xH, xW, C_upper).
            input_surface (Tensor): Tensor of shape (B, xH, xW, C_surface).
        Returns:
            WeatherData: Tuple of upper-air data and surface data.
        """

        res_conn_upper = self.residual * input_upper
        res_conn_surface = self.residual * input_surface
        x: Tensor = self.patch_embed(input_upper, input_surface)
        # with torch.cuda.amp.autocast(enabled=False):
        with torch.amp.autocast("cuda", enabled=False):
            x = self.layer1(x.float())
            skip = x
            x = self.downsample(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.upsample(x)
            x = self.layer4(x)
            x, _ = pack([skip, x], "B N *")
        output_upper, output_surface = self.patch_recover(x)
        if not self.res_conn_after_smooth:
            output_surface += res_conn_surface
            output_upper += res_conn_upper
        output_upper, output_surface = self.smoothing_layer(
            output_upper, output_surface)
        if self.res_conn_after_smooth:
            output_upper += res_conn_upper
            output_surface += res_conn_surface
        return output_upper, output_surface


class SmoothingBlock(nn.Module):
    def __init__(
        self,
        smoothing_func: nn.Module,
    ) -> None:
        super().__init__()
        self.smoothing_func = smoothing_func

    def forward(self, x_upper: Tensor, x_surface: Tensor) -> WeatherData:
        x_upper = rearrange(x_upper, "B Z H W C_upper -> B C_upper Z H W")
        x_surface = rearrange(x_surface, "B H W C_surface -> B C_surface () H W")
        x_upper = self.smoothing_func(x_upper)
        x_surface = self.smoothing_func(x_surface)
        x_upper = rearrange(x_upper, "B C_upper Z H W -> B Z H W C_upper")
        x_surface = rearrange(x_surface, "B C_surface () H W -> B H W C_surface")
        return x_upper, x_surface


class Identity(nn.Module):
    """
    Identity layer. Replace smoothing layer when smoothing_kernel_size is None.
    """

    def forward(self, x_upper: Tensor, x_surface: Tensor) -> WeatherData:
        return x_upper, x_surface


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        dim: int,
        upper_vars: int,
        surface_vars: int,
        patch_shape: Shape_T,
        constant_mask_paths: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Convert input fields to patches and linearly embed them.

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (xZ, xH, xW).
            dim (int): Dimension of the output embedding.
            upper_vars (int): Number of upper-air variables.
            surface_vars (int): Number of surface variables.
            patch_shape (tuple[int, int, int]): Size of the patch (pZ, pH, pW).
            constant_mask_paths (Optional[list[str]]): Paths to the constant mask.
        """
        super().__init__()
        if constant_mask_paths is not None:
            assert len(constant_mask_paths) == 3

            land_mask = rearrange(LoadConstantMask(constant_mask_paths[0]), "C H W -> () C H W")
            soil_mask = rearrange(LoadConstantMask(constant_mask_paths[1]), "C H W -> () C H W")
            topography_mask = rearrange(LoadConstantMask(constant_mask_paths[2]), "C H W -> () C H W")
            # Normalize the topography mask to [0, 1]
            topography_mask = (topography_mask - topography_mask.min()) / \
                (topography_mask.max() - topography_mask.min())
            additional_channels = land_mask.shape[1] + soil_mask.shape[1] + topography_mask.shape[1]
        else:
            land_mask, soil_mask, topography_mask = None, None, None
            additional_channels = 0
        self.register_buffer("land_mask", land_mask)
        self.register_buffer("soil_mask", soil_mask)
        self.register_buffer("topography_mask", topography_mask)
        self.patch_shape = patch_shape
        # Use convolution to partition data into cubes
        self.conv_upper = nn.Conv3d(in_channels=upper_vars,
                                    out_channels=dim,
                                    kernel_size=patch_shape,
                                    stride=patch_shape)
        self.conv_surface = nn.Conv2d(in_channels=surface_vars+additional_channels,
                                      out_channels=dim,
                                      kernel_size=patch_shape[1:],
                                      stride=patch_shape[1:])
        self.upper_pad = GetPad3D(in_shape, patch_shape)
        self.surface_pad = GetPad2D(in_shape[1:], patch_shape[1:])

    def forward(self, input_upper: Tensor, input_surface: Tensor) -> Tensor:
        """
        Args:
            input_upper (Tensor): Tensor of shape (B, xZ, xH, xW, C_upper).
            input_surface (Tensor): Tensor of shape (B, xH, xW, C_surface).
        Returns:
            Tensor: Tensor of shape (batch_size, Z*H*W, dim).
        """
        input_upper = rearrange(
            input_upper, "B Z H W C_upper -> B C_upper Z H W")
        input_surface = rearrange(
            input_surface, "B H W C_surface -> B C_surface H W")
        if (self.land_mask is not None
                and self.soil_mask is not None
                and self.topography_mask is not None):
            B = input_surface.shape[0]
            input_surface, _ = pack([
                input_surface,
                repeat(self.land_mask, "B C H W -> (repeat B) C H W", repeat=B),
                repeat(self.soil_mask, "B C H W -> (repeat B) C H W", repeat=B),
                repeat(self.topography_mask, "B C H W -> (repeat B) C H W", repeat=B),
            ], "B * H W")
        # Pad the input to make it divisible by patch_shape
        input_upper = self.upper_pad(input_upper)
        input_surface = self.surface_pad(input_surface)
        x, _ = pack([self.conv_upper(input_upper),
                    self.conv_surface(input_surface)], "B dim * H W")
        x = rearrange(x, "B dim Z H W -> B (Z H W) dim")
        return x


class PatchRecovery(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        dim: int,
        out_shape: Shape_T,
        upper_vars: int,
        surface_vars: int,
        patch_shape: Shape_T,
    ) -> None:
        """
        Recover the output fields from patches.

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            dim (int): Dimension of the input embedding.
            out_shape (tuple[int, int, int]): (xZ, xH, xW).
            patch_shape (tuple[int, int, int]): Size of the patch (pZ, pH, pW).
        """
        super().__init__()
        self.conv_upper = nn.ConvTranspose3d(
            in_channels=dim,
            out_channels=upper_vars,
            kernel_size=patch_shape,
            stride=patch_shape
        )
        self.conv_surface = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=surface_vars,
            kernel_size=patch_shape[1:],
            stride=patch_shape[1:],
        )
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.upper_crop_idx = GetCrop3DIndex(out_shape, patch_shape)
        self.surface_crop_idx = GetCrop2DIndex(out_shape[1:], patch_shape[1:])

    def forward(self, x: Tensor) -> WeatherData:
        """
        Args:
            x (Tensor): Tensor of shape (B, Z*H*W, dim).
        Returns:
            WeatherData: Tuple of upper-air data (B, xZ, xH, xW, C_upper) and surface data (B, xH, xW, C_surface
        """
        x = rearrange(x, "B (Z H W) dim -> B dim Z H W",
                      Z=self.in_shape[0], H=self.in_shape[1], W=self.in_shape[2])
        x_upper, x_surface = unpack(
            x, [[self.in_shape[0]-1], []], "B dim * H W")
        output_upper = self.conv_upper(x_upper)  # (B, C_upper, xZ, xH, xW)
        output_surface = self.conv_surface(x_surface)  # (B, C_surface, xH, xW)
        output_upper = output_upper[(..., *self.upper_crop_idx)]
        output_surface = output_surface[(..., *self.surface_crop_idx)]
        output_upper = rearrange(output_upper, "B C Z H W -> B Z H W C")
        output_surface = rearrange(output_surface, "B C H W -> B H W C")
        return output_upper, output_surface


class DownSample(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        in_channels: int,
        out_channels: int
    ) -> None:
        """
        Reduces the lateral resolution by a factor of 2.

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.linear = nn.Linear(4*in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(4*in_channels)
        self.Z, self.H, self.W = in_shape
        # Pad to make H and W divisible by 2
        self.pad = GetPad2D((self.H, self.W), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, Z*H*W, C_in).
        Returns:
            Tensor: Tensor of shape (B, Z*(H/2)*(W/2) C_out).
        """
        x = rearrange(x, "B (Z H W) C_in -> B C_in Z H W",
                      Z=self.Z, H=self.H, W=self.W)
        x = self.pad(x)
        x = rearrange(
            x, "B C Z (h scale_h) (w scale_w) -> B (Z h w) (scale_h scale_w C)", scale_h=2, scale_w=2)
        x = self.norm(x)  # (B, Z*(H/2)*(W/2), 4*C_in)
        x = self.linear(x)  # (B, Z*(H/2)*(W/2), C_out)
        return x


class UpSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_shape: Shape_T,
        out_channels: int
    ) -> None:
        """
        Increases the lateral resolution by a factor of 2.

        Args:
            in_channels (int): Number of input channels.
            out_shape (tuple[int, int, int]): Shape of the output tensor (Z_out, H_out, W_out).
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, 4*out_channels, bias=False)
        # to mix normalized tensors
        self.linear2 = nn.Linear(out_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)
        self.out_shape = out_shape
        self.crop_idx = GetCrop2DIndex(out_shape[1:], (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, Z*H*W, C_in).
        Returns:
            Tensor: Tensor of shape (B, Z_out*H_out*W_out, C_out). (Cropped to output_shape)
        """
        Z_in, H_in, W_in = self.out_shape[0], ceil(
            self.out_shape[1]/2), ceil(self.out_shape[2]/2)
        x = self.linear(x)  # (B, Z*H*W, 4*C_out)
        x = rearrange(x, "B (Z H W) (scale_h scale_w C) -> B C Z (H scale_h) (W scale_w)",
                      Z=Z_in, H=H_in, W=W_in, scale_h=2, scale_w=2)
        x = x[(..., *self.crop_idx)]  # Crop to output shape
        x = rearrange(x, "B C Z H W -> B (Z H W) C")
        x = self.norm(x)
        x = self.linear2(x)
        return x


class EarthSpecificLayer(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        dim: int,
        depth: int,
        heads: int,
        drop_path_ratio: Sequence[float],
        window_size: Shape_T,
    ) -> None:
        """
        Basic layer of the network, contains either 2 or 6 blocks

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            dim (int): Dimension of the input token.
            depth (int): Number of blocks in the layer.
            heads (int): Number of heads in the attention layer.
            drop_path_ratio (Sequence[float]): Drop path ratio for each block.
            window_size (tuple[int, int, int]): window size.
        """
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList((EarthSpecificBlock(
            dim=dim,
            in_shape=in_shape,
            drop_path_ratio=drop_path_ratio[i],
            heads=heads,
            roll=(i % 2 == 1),
            window_size=window_size,
        ) for i in range(depth)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, Z*H*W, dim).
        Returns:
            Tensor: Tensor of shape (batch_size, Z*H*W, dim).
        """
        for i in range(self.depth):
            x = self.blocks[i](x)
        return x


class EarthSpecificBlock(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        dim: int,
        heads: int,
        drop_path_ratio: float,
        window_size: Shape_T,
        roll: bool,
    ) -> None:
        """
        Earth-specific variant of Swin-Transformer, with 3d window attention and earth-specific bias

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            dim (int): Dimension of the input token.
            heads (int): Number of heads in the attention layer.
            drop_path_ratio (float): DropPath drop probability.
            window_size (tuple[int, int, int]): window size
            roll (bool): Whether to roll the tensor for half a window size.
        """
        super().__init__()
        self.window_size = window_size
        self.wZ, self.wH, self.wW = window_size
        self.shift_size = tuple(i//2 for i in self.window_size)
        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = MLP(dim, 0)
        self.attention = EarthAttention3D(
            dim=dim,
            input_shape=in_shape,
            heads=heads,
            dropout_rate=0,
            window_size=self.window_size,
        )
        self.roll = roll
        self.Z, self.H, self.W = in_shape
        self.pad = GetPad3D(in_shape, window_size)
        self.crop_idx = GetCrop3DIndex(in_shape, window_size)
        # Mask the padding during attention
        pad_pos = rearrange(self.pad(torch.ones(size=(1, 1, self.Z, self.H, self.W))),
                            "B C Z H W -> B Z H W C")  # B=1, C=1
        pad_pos = 1 - pad_pos  # 1 for padding, 0 for real data
        if roll:
            pad_pos = pad_pos.roll(
                shifts=[-i for i in self.shift_size], dims=(1, 2, 3))
        pad_pos = rearrange(pad_pos, "B (mZ wZ) (mH wH) (mW wW) C -> (B mZ mH mW C) (wZ wH wW)",
                            wZ=self.wZ, wH=self.wH, wW=self.wW)
        # If either of the two vectors doing attention is padding, the attention is masked
        pad_mask = rearrange(pad_pos, "wN wP -> wN 1 wP") + \
            rearrange(pad_pos, "wN wP -> wN wP 1")  # 'or' operation
        pad_mask = pad_mask.masked_fill(pad_mask != 0, -100.0)
        pad_mask = pad_mask.masked_fill(pad_mask == 0, 0.0)
        attention_mask = pad_mask  # (wN, wP, wP)
        # Following Swin-Transformer's implementation to get attention mask for SW-MSA
        if roll:
            img_mask = rearrange(self.pad(torch.ones(size=(1, 1, self.Z, self.H, self.W))),
                                 "B C Z H W -> B Z H W C")
            z_slices = (slice(0, -self.wZ),
                        slice(-self.wZ, -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.wH),
                        slice(-self.wH, -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.wW),
                        slice(-self.wW, -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            cnt = 0
            for z, h, w in product(z_slices, h_slices, w_slices):
                img_mask[:, z, h, w, :] = cnt
                cnt += 1
            mask_windows = rearrange(img_mask, "B (mZ wZ) (mH wH) (mW wW) C -> (B mZ mH mW C) (wZ wH wW)",
                                     wZ=self.wZ, wH=self.wH, wW=self.wW)
            nonadjacent_mask = rearrange(
                mask_windows, "wN wP -> wN 1 wP") - rearrange(mask_windows, "wN wP -> wN wP 1")
            nonadjacent_mask = nonadjacent_mask.masked_fill(
                nonadjacent_mask != 0, -100.0)
            nonadjacent_mask = nonadjacent_mask.masked_fill(
                nonadjacent_mask == 0, 0.0)
            attention_mask += nonadjacent_mask
            attention_mask = attention_mask.masked_fill(
                attention_mask != 0, -100.0)
        self.register_buffer("attention_mask", attention_mask)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, Z*H*W, dim).
            Z (int): Number of pressure levels.
            H (int): Number of latitude levels.
            W (int): Number of longitude levels.
            roll (bool): Whether to roll the tensor for half a window size.
        Returns:
            Tensor: Tensor of shape (B, Z*H*W, dim).
        """
        shortcut = x
        x = rearrange(x, "B (Z H W) dim -> B dim Z H W",
                      Z=self.Z, H=self.H, W=self.W)
        x = rearrange(self.pad(x), "B dim Z H W -> B Z H W dim")
        if self.roll:
            x = x.roll(shifts=[-i//2 for i in self.window_size], dims=(1, 2, 3))
        x = rearrange(x, "B (mZ wZ) (mH wH) (mW wW) dim -> (B mZ mH mW) (wZ wH wW) dim",
                      wZ=self.wZ, wH=self.wH, wW=self.wW)
        x = self.attention(x, self.attention_mask)
        x = rearrange(x, "(B mZ mH mW) (wZ wH wW) dim -> B (mZ wZ) (mH wH) (mW wW) dim",
                      wZ=self.wZ, wH=self.wH, wW=self.wW,
                      mZ=ceil(self.Z/self.wZ), mH=ceil(self.H/self.wH), mW=ceil(self.W/self.wW))
        if self.roll:
            x = x.roll(shifts=[i//2 for i in self.window_size], dims=(1, 2, 3))
        x = x[(..., *self.crop_idx, slice(None))]  # Crop to original size
        x = rearrange(x, "B Z H W dim -> B (Z H W) dim")
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.linear(x)))
        return x


class EarthAttention3D(nn.Module):
    def __init__(
        self,
        input_shape: Shape_T,
        dim: int,
        heads: int,
        dropout_rate: float,
        window_size: Shape_T
    ) -> None:
        """
        3D window attention layer.

        Args:
            input_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            dim (int): Dimension of the input token.
            heads (int): Number of heads in the attention layer.
            dropout_rate (float): Dropout rate.
            window_size (tuple[int, int, int]): window size
        """
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.heads = heads
        self.dim = dim
        self.scale = (dim//heads) ** -0.5
        self.window_size = window_size
        wZ, wH, wW = window_size
        self.mZ, self.mH, self.mW = map(lambda x, y: ceil(x/y), input_shape, window_size)
        self.window_types = self.mZ * self.mH  # Bias is longitude-independent
        self.earth_specific_bias = torch.zeros((wZ**2 * wH**2 * (2*wW-1), self.window_types, heads))
        self.earth_specific_bias = nn.Parameter(self.earth_specific_bias, requires_grad=True)
        self.earth_specific_bias = nn.init.trunc_normal_(self.earth_specific_bias, std=0.02)
        self.position_index = self._construct_index()

    def _construct_index(self) -> Tensor:
        """
        Construct the index for reusing symmetrical positional bias.
        """
        wZ, wH, wW = self.window_size
        coords_zi = torch.arange(wZ)
        coords_zj = -torch.arange(wZ)*wZ
        coords_hi = torch.arange(wH)
        coords_hj = -torch.arange(wH)*wH
        coords_w = torch.arange(wW)
        coords_1, _ = pack(torch.meshgrid(coords_zi, coords_hi, coords_w, indexing="ij"), "* wZ wH wW")
        coords_2, _ = pack(torch.meshgrid(coords_zj, coords_hj, coords_w, indexing="ij"), "* wZ wH wW")
        coords_1_flatten = rearrange(coords_1, "n wZ wH wW -> n (wZ wH wW) 1")
        coords_2_flatten = rearrange(coords_2, "n wZ wH wW -> n 1 (wZ wH wW)")
        coords = rearrange(coords_1_flatten - coords_2_flatten, "n wP1 wP2 -> wP1 wP2 n")  # wP=wZ*wH*wW
        coords[:, :, 2] += wW-1
        coords[:, :, 1] *= 2 * wW - 1
        coords[:, :, 0] *= (2 * wW - 1)*wH*wH
        position_index = rearrange(einsum(coords, "i j n -> i j"), "wP1 wP2 -> (wP1 wP2)")
        return position_index

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B*wN, wZ*wH*wW, dim).
            mask (Optional[Tensor]): Attention mask of shape (wN, wZ*wH*wW, wZ*wH*wW).
        Returns:
            Tensor: Tensor of shape (B*wN, wZ*wH*wW, dim).
        """
        qkv = rearrange(self.qkv(
            x), "BN wP (split h dim_h) -> split BN h wP dim_h", split=3, h=self.heads)
        query, key, value = self.scale*(qkv[0]), qkv[1], qkv[2]
        attention = einsum(query, key, "b h i d, b h j d -> b h i j")
        attention = rearrange(attention, "(B mZ mH mW) heads wP1 wP2 -> (B mW) (mZ mH) heads wP1 wP2",
                              mZ=self.mZ, mH=self.mH, mW=self.mW)
        bias = rearrange(self.earth_specific_bias[self.position_index], "(wP1 wP2) wT heads -> wT heads wP1 wP2",
                         wP1=self.window_size[0]*self.window_size[1]*self.window_size[2])
        attention += bias
        attention = rearrange(attention, "(B mW) (mZ mH) heads wP1 wP2 -> (B mZ mH mW) heads wP1 wP2",
                              mZ=self.mZ, mH=self.mH, mW=self.mW)
        if mask is not None:
            attention = rearrange(attention, "(B wN) heads wP1 wP2 -> B wN heads wP1 wP2", wN=self.mZ*self.mH*self.mW)
            attention += rearrange(mask, "wN wP1 wP2 -> 1 wN 1 wP1 wP2")
            attention = rearrange(attention, "B wN heads wP1 wP2 -> (B wN) heads wP1 wP2")
        attention = self.dropout(self.softmax(attention))
        x = einsum(attention, value, "b h i j, b h j d -> b h i d")
        x = rearrange(x, "BN heads wP dim_h -> BN wP (heads dim_h)")
        x = self.dropout(self.proj(x))
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim*4)
        self.linear2 = nn.Linear(dim*4, dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


def GetPad3D(img_shape: tuple[int, int, int], sub_shape: tuple[int, int, int]) -> nn.ZeroPad3d:
    """
    Get nn.ZeroPad3d for padding the input to be divisible by sub_shape.

    Args:
        img_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
        sub_shape (tuple[int, int, int]): Shape of the sub tensor (z, h, w).
    Returns:
        nn.ZeroPad3d: Padding layer.
    """
    Z, H, W = img_shape
    z, h, w = sub_shape
    pad = nn.ZeroPad3d((
        floor((-W % w)/2), ceil((-W % w)/2),
        floor((-H % h)/2), ceil((-H % h)/2),
        floor((-Z % z)/2), ceil((-Z % z)/2),
    ))
    return pad


def GetPad2D(img_shape: tuple[int, int], sub_shape: tuple[int, int]) -> nn.ZeroPad2d:
    """
    Get nn.ZeroPad2d for padding the input to be divisible by sub_shape.

    Args:
        img_shape (tuple[int, int]): Shape of the input tensor (H, W).
        sub_shape (tuple[int, int]): Shape of the sub tensor (h, w).
    Returns:
        nn.ZeroPad2d: Padding layer.
    """
    H, W = img_shape
    h, w = sub_shape
    pad = nn.ZeroPad2d((
        floor((-W % w)/2), ceil((-W % w)/2),
        floor((-H % h)/2), ceil((-H % h)/2),
    ))
    return pad


def GetCrop3DIndex(img_shape: tuple[int, int, int], sub_shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    """
    Get the index for reversing padding via GetPad3D.

    Args:
        img_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
        sub_shape (tuple[int, int, int]): Shape of the sub tensor (z, h, w).
    Returns:
        tuple[slice, slice, slice]: Crop index.
    """
    Z, H, W = img_shape
    z, h, w = sub_shape
    return (
        slice(floor((-Z % z)/2), -ceil((-Z % z)/2)) if Z % z != 0 else slice(None),
        slice(floor((-H % h)/2), -ceil((-H % h)/2)) if H % h != 0 else slice(None),
        slice(floor((-W % w)/2), -ceil((-W % w)/2)) if W % w != 0 else slice(None),
    )


def GetCrop2DIndex(img_shape: tuple[int, int], sub_shape: tuple[int, int]) -> tuple[slice, slice]:
    """
    Get the index for reversing padding via GetPad2D.

    Args:
        img_shape (tuple[int, int]): Shape of the input tensor (H, W).
        sub_shape (tuple[int, int]): Shape of the sub tensor (h, w).
    Returns:
        tuple[slice, slice]: Crop index.
    """
    H, W = img_shape
    h, w = sub_shape
    return (
        slice(floor((-H % h)/2), -ceil((-H % h)/2)) if H % h != 0 else slice(None),
        slice(floor((-W % w)/2), -ceil((-W % w)/2)) if W % w != 0 else slice(None),
    )
