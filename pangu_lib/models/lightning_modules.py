from itertools import product
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from einops import reduce
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities.grads import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from ..utils.lr_scheduler import get_scheduler_with_warmup
from ..utils.plotting import visualize_weather_var
from ..utils.types import Shape_T, Stat_T, WeatherData
from .loss import WeightedL1Loss
from .pangu import PanguModel
from .vit import VisionTransformer3D


def destandardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x*std + mean


class WeatherModel(L.LightningModule):
    def _log_pred_image(self, out_upper: Tensor, out_surface: Tensor, tgt_upper: Tensor, tgt_surface: Tensor) -> None:
        assert isinstance(self.logger, TensorBoardLogger)
        tensorboard_logger: SummaryWriter = self.logger.experiment
        log_vars = [
            ("u", 500), ("v", 500), ("t", 850), ("q", 700), ("z", 500),
            ("msl", None), ("t2m", None), ("u10", None), ("v10", None),
        ]
        for var, pl in log_vars:
            if pl is not None:
                v_idx = self.upper_vars.index(var)
                p_idx = self.pressure_levels.index(pl)
                out_map = out_upper[p_idx, ..., v_idx].cpu()
                tgt_map = tgt_upper[p_idx, ..., v_idx].cpu()
                v_range = float(np.percentile(tgt_map, q=1)), float(np.percentile(tgt_map, q=100-1))

                fig_pred = visualize_weather_var(out_map, v_range)
                tensorboard_logger.add_figure(f"forecast/{var}_{pl}", fig_pred, global_step=self.global_step)
                if self.current_epoch == 0:
                    fig_tgt = visualize_weather_var(tgt_map, v_range)
                    tensorboard_logger.add_figure(f"target/{var}_{pl}", fig_tgt)
            else:
                v_idx = self.surface_vars.index(var)
                out_map = out_surface[..., v_idx].cpu()
                tgt_map = tgt_surface[..., v_idx].cpu()
                v_range = float(np.percentile(tgt_map, q=1)), float(np.percentile(tgt_map, q=100-1))

                fig_pred = visualize_weather_var(out_map, v_range)
                tensorboard_logger.add_figure(f"forecast/{var}", fig_pred, global_step=self.global_step)
                if self.current_epoch == 0:
                    fig_tgt = visualize_weather_var(tgt_map, v_range)
                    tensorboard_logger.add_figure(f"target/{var}", fig_tgt)

    def _calc_criterion_loss(self, out_upper: Tensor, out_surface: Tensor, tgt_upper: Tensor, tgt_surface: Tensor) -> Tensor:
        if self.weighted_loss:
            loss_upper = self.criterion(out_upper, tgt_upper, self.upper_var_weights)
            loss_surface = self.criterion(out_surface, tgt_surface, self.surface_var_weights)
        else:
            loss_upper = self.criterion(out_upper, tgt_upper)
            loss_surface = self.criterion(out_surface, tgt_surface)
        loss = loss_upper + loss_surface * self.loss_params["surface_weight"]
        return loss

    def _log_normalized_L1Loss(self, prefix: str, out_upper: Tensor, out_surface: Tensor, tgt_upper: Tensor, tgt_surface: Tensor) -> None:
        l1_norm_upper = reduce(torch.abs(out_upper - tgt_upper), "b p h w v -> p v", "mean")
        l1_norm_surface = reduce(torch.abs(out_surface - tgt_surface), "b h w v -> v", "mean")
        for (i, pl), (j, var) in product(enumerate(self.pressure_levels), enumerate(self.upper_vars)):
            self.log(name=f"{prefix}_norm_L1/{var}_{pl}", value=l1_norm_upper[i, j],
                     on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        for i, var in enumerate(self.surface_vars):
            self.log(name=f"{prefix}_norm_L1/{var}", value=l1_norm_surface[i],
                     on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def _general_eval_step(self, batch: tuple[WeatherData, WeatherData, Stat_T], batch_idx: int, prefix: str) -> STEP_OUTPUT:
        """
        Shared evaluation step for validation and test.
        """
        (in_upper, in_surface), (tgt_upper, tgt_surface), stat = batch
        out_upper, out_surface = self.model(in_upper, in_surface)
        loss = self._calc_criterion_loss(out_upper, out_surface, tgt_upper, tgt_surface)
        self.log(name=f"{prefix}_loss", value=loss,
                 on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self._log_normalized_L1Loss(prefix, out_upper, out_surface, tgt_upper, tgt_surface)

        # De-standardize
        mean_upper, std_upper, mean_surface, std_surface = stat
        out_upper = destandardize(out_upper, mean_upper, std_upper)
        out_surface = destandardize(out_surface, mean_surface, std_surface)
        tgt_upper = destandardize(tgt_upper, mean_upper, std_upper)
        tgt_surface = destandardize(tgt_surface, mean_surface, std_surface)

        # Calculate and log RMSE of each variable and each pressure level
        rmse_upper = reduce((out_upper - tgt_upper) ** 2, "b p h w v -> p v", "mean").sqrt()
        rmse_surface = reduce((out_surface - tgt_surface) ** 2, "b h w v -> v", "mean").sqrt()
        for (i, pl), (j, var) in product(enumerate(self.pressure_levels), enumerate(self.upper_vars)):
            self.log(name=f"{prefix}_RMSE/{var}_{pl}", value=rmse_upper[i, j],
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
        for i, var in enumerate(self.surface_vars):
            self.log(name=f"{prefix}_RMSE/{var}", value=rmse_surface[i],
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # Visualize forecasts on tensorboard
        if batch_idx == 0:
            self._log_pred_image(out_upper[0], out_surface[0], tgt_upper[0], tgt_surface[0])

        return loss

    def training_step(self, batch: tuple[WeatherData, WeatherData], batch_idx: int) -> STEP_OUTPUT:
        """
        Args:
            batch (tuple[WeatherData, WeatherData]): Tuple of input and target weather data.
                                                     One WeatherData is composed of upper-air data and surface data.
                                                     Upper-air data is of shape (batch, pressure level, latitude, longitude, variable).
                                                     Surface data is of shape (batch, latitude, longitude, variable).
                                                     Data is standardized.
            batch_idx (int): Index of the batch.
        """
        (in_upper, in_surface), (tgt_upper, tgt_surface) = batch
        out_upper, out_surface = self.model(in_upper, in_surface)
        loss = self._calc_criterion_loss(out_upper, out_surface, tgt_upper, tgt_surface)

        self.log(name="train_loss", value=loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        self._log_normalized_L1Loss("train", out_upper, out_surface, tgt_upper, tgt_surface)
        return loss

    def validation_step(self, batch: tuple[WeatherData, WeatherData, Stat_T], batch_idx: int) -> STEP_OUTPUT:
        """
        Args:
            batch (tuple[WeatherData, WeatherData, Stat_T]): Tuple of input, target weather data and (mean_upper, std_upper, mean_surface, std_surface).
                                                             Upper-air data is of shape (batch, pressure level, latitude, longitude, variable).
                                                             Surface data is of shape (batch, latitude, longitude, variable).
            batch_idx (int): Index of the batch.
        """
        return self._general_eval_step(batch, batch_idx, "val")

    def test_step(self, batch: tuple[WeatherData, WeatherData, Stat_T], batch_idx: int) -> STEP_OUTPUT:
        """
        Args:
            batch (tuple[WeatherData, WeatherData, Stat_T]): Tuple of input, target weather data and (mean_upper, std_upper, mean_surface, std_surface).
                                                             Upper-air data is of shape (batch, pressure level, latitude, longitude, variable).
                                                             Surface data is of shape (batch, latitude, longitude, variable).
            batch_idx (int): Index of the batch.
        """
        return self._general_eval_step(batch, batch_idx, "test")

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log(name="gradient_2norm", value=norms["grad_2.0_norm_total"],
                 on_step=True, on_epoch=False)
        norms.pop("grad_2.0_norm_total")
        self.log_dict(norms,
                      on_step=True, on_epoch=False)

    def configure_optimizers(self) -> Any:
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), **self.optimizer_args)
        lr_scheduler = get_scheduler_with_warmup(
            optimizer,
            training_steps=int(self.trainer.estimated_stepping_batches),
            schedule_type=self.lr_scheduler,
            **self.lr_scheduler_args,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def forward(self, input_upper: Tensor, input_surface: Tensor) -> WeatherData:
        return self.model(input_upper, input_surface)


class PanguLightningModule(WeatherModel):
    def __init__(
        self,
        data_spatial_shape: Shape_T = (8, 141, 181),
        pressure_levels: list[int] = [50,  150,  300,  500,  700,  850,  925, 1000],
        upper_vars: list[str] = ["u", "v", "t", "q", "z", "w"],
        surface_vars: list[str] = ["u10", "v10", "t2m", "msl", "sp", "tcwv", "tp", "d2m"],
        depths: list[int] = [2, 6],
        heads: list[int] = [6, 12],
        embed_dim: int = 192,
        patch_shape: Shape_T = (2, 2, 2),
        window_shape: Shape_T = (2, 4, 4),
        surface_weight: float = 0.25,
        optimizer_name: str = "Adam",
        optimizer_args: dict[str, Any] = {},
        lr_scheduler_name: str = "CosineAnnealingLR",
        lr_scheduler_args: dict[str, Any] = {},
        constant_mask_paths: Optional[list[str]] = None,
        upper_var_weights: Optional[dict[str, float]] = None,
        surface_var_weights: Optional[dict[str, float]] = None,
        smoothing_kernel_size: Optional[int] = None,
        segmented_smooth: bool = False,
        segmented_smooth_boundary_width: Optional[int] = None,
        learned_smooth: bool = False,
        residual: bool = False,
        res_conn_after_smooth: bool = True,
    ):
        super().__init__()
        assert len(data_spatial_shape) == 3
        assert len(patch_shape) == 3
        assert len(window_shape) == 3
        assert len(pressure_levels) == data_spatial_shape[0]

        self.save_hyperparameters()

        self.model = PanguModel(
            data_spatial_shape=data_spatial_shape,
            upper_vars=len(upper_vars),
            surface_vars=len(surface_vars),
            depths=depths,
            heads=heads,
            embed_dim=embed_dim,
            patch_shape=patch_shape,
            window_size=window_shape,
            constant_mask_paths=constant_mask_paths,
            smoothing_kernel_size=smoothing_kernel_size,
            segmented_smooth=segmented_smooth,
            segmented_smooth_boundary_width=segmented_smooth_boundary_width,
            learned_smooth=learned_smooth,
            residual=residual,
            res_conn_after_smooth=res_conn_after_smooth,
        )

        if upper_var_weights is None and surface_var_weights is None:
            self.weighted_loss = False
            self.criterion = nn.L1Loss()
            upper_var_weights_tensor = None
            surface_var_weights_tensor = None
        else:
            self.weighted_loss = True
            self.criterion = WeightedL1Loss()
            upper_var_weights_tensor = torch.ones(len(upper_vars))
            surface_var_weights_tensor = torch.ones(len(surface_vars))
            if upper_var_weights is not None:
                for i, var in enumerate(upper_vars):
                    upper_var_weights_tensor[i] = upper_var_weights.get(var, 1.0)
            if surface_var_weights is not None:
                for i, var in enumerate(surface_vars):
                    surface_var_weights_tensor[i] = surface_var_weights.get(var, 1.0)

        self.register_buffer("upper_var_weights", upper_var_weights_tensor)
        self.register_buffer("surface_var_weights", surface_var_weights_tensor)
        self.loss_params = {
            "surface_weight": surface_weight,
        }
        self.pressure_levels = pressure_levels
        self.upper_vars = upper_vars
        self.surface_vars = surface_vars
        self.optimizer = optimizer_name
        self.optimizer_args = optimizer_args
        self.lr_scheduler = lr_scheduler_name
        self.lr_scheduler_args = lr_scheduler_args


class ViTLightningModule(WeatherModel):
    def __init__(
        self,
        data_spatial_shape: Shape_T = (8, 141, 181),
        pressure_levels: list[int] = [50,  150,  300,  500,  700,  850,  925, 1000],
        upper_vars: list[str] = ["u", "v", "t", "q", "z"],
        surface_vars: list[str] = ["u10", "v10", "t2m", "msl",],
        depth: int = 12,
        heads: int = 6,
        embed_dim: int = 192,
        patch_shape: Shape_T = (2, 4, 4),
        surface_weight: float = 0.25,
        optimizer_name: str = "Adam",
        optimizer_args: dict[str, Any] = {},
        lr_scheduler_name: str = "CosineAnnealingLR",
        lr_scheduler_args: dict[str, Any] = {},
        constant_mask_paths: Optional[list[str]] = None,
        upper_var_weights: Optional[dict[str, float]] = None,
        surface_var_weights: Optional[dict[str, float]] = None,
        smoothing_kernel_size: Optional[int] = None,
        # segmented_smooth: bool = False,
        # segmented_smooth_boundary_width: Optional[int] = None,
        # residual: bool = False,
        # res_conn_after_smooth: bool = True,
    ):
        super().__init__()
        assert len(data_spatial_shape) == 3
        assert len(patch_shape) == 3
        assert len(pressure_levels) == data_spatial_shape[0]

        self.save_hyperparameters()

        self.model = VisionTransformer3D(
            data_spatial_shape=data_spatial_shape,
            upper_vars=len(upper_vars),
            surface_vars=len(surface_vars),
            depth=depth,
            heads=heads,
            embed_dim=embed_dim,
            patch_shape=patch_shape,
            constant_mask_paths=constant_mask_paths,
            smoothing_kernel_size=smoothing_kernel_size,
        )

        if upper_var_weights is None and surface_var_weights is None:
            self.weighted_loss = False
            self.criterion = nn.L1Loss()
            upper_var_weights_tensor = None
            surface_var_weights_tensor = None
        else:
            self.weighted_loss = True
            self.criterion = WeightedL1Loss()
            upper_var_weights_tensor = torch.ones(len(upper_vars))
            surface_var_weights_tensor = torch.ones(len(surface_vars))
            if upper_var_weights is not None:
                for i, var in enumerate(upper_vars):
                    upper_var_weights_tensor[i] = upper_var_weights.get(var, 1.0)
            if surface_var_weights is not None:
                for i, var in enumerate(surface_vars):
                    surface_var_weights_tensor[i] = surface_var_weights.get(var, 1.0)

        self.register_buffer("upper_var_weights", upper_var_weights_tensor)
        self.register_buffer("surface_var_weights", surface_var_weights_tensor)
        self.loss_params = {
            "surface_weight": surface_weight,
        }
        self.pressure_levels = pressure_levels
        self.upper_vars = upper_vars
        self.surface_vars = surface_vars
        self.optimizer = optimizer_name
        self.optimizer_args = optimizer_args
        self.lr_scheduler = lr_scheduler_name
        self.lr_scheduler_args = lr_scheduler_args
