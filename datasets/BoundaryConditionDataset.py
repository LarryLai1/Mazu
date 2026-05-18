from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr


class BoundaryConditionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        boundary_root_dir: str,
        start_date_hour: pd.Timestamp,
        end_date_hour: pd.Timestamp,
        upper_variables: list[str],
        surface_variables: list[str],
        levels: list[int],
        latitude: tuple[float, float],
        longitude: tuple[float, float],
        boundary_width: int = 0,
        prediction_timedeltas: list[int] | tuple[int, ...] = (0, 6, 12),
        forecast_cycle_hours: int = 12,
        get_datetime: bool = True,
    ) -> None:
        super().__init__()
        self.boundary_root_dir = boundary_root_dir
        self.start_date_hour = pd.Timestamp(start_date_hour)
        self.end_date_hour = pd.Timestamp(end_date_hour)
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.levels = levels
        self.latitude = latitude
        self.longitude = longitude
        self.boundary_width = boundary_width
        self.prediction_timedelta_hours = tuple(int(x) for x in prediction_timedeltas)
        self.prediction_timedeltas = tuple(pd.Timedelta(hours = x) for x in self.prediction_timedelta_hours)
        self.forecast_cycle_hours = forecast_cycle_hours
        self.get_datetime = get_datetime
        self.time_axis = pd.date_range(
            start = self.start_date_hour,
            end = self.end_date_hour,
            freq = f"{self.forecast_cycle_hours}h",
        )

    def map_var_name_for_Aurora(self, var_name: str) -> str:
        var_name_mapping = {
            "t2m": "2t",
            "u10": "10u",
            "v10": "10v",
            "msl": "msl",
        }
        if var_name in var_name_mapping:
            return var_name_mapping[var_name]
        return var_name

    def _dt_to_path(self, date_hour: pd.Timestamp) -> tuple[str, str]:
        dir_path = Path(self.boundary_root_dir) / date_hour.strftime(r"%Y/%Y%m")
        name = date_hour.strftime(r"%Y%m%d")
        return str(dir_path / f"{name}_upper.nc"), str(dir_path / f"{name}_sfc.nc")

    def _spatial_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        # boundary_width is counted in 0.25 degree grid cells
        boundary_delta = self.boundary_width * 0.25
        latitude_bounds = (self.latitude[0] + boundary_delta, self.latitude[1] - boundary_delta)
        longitude_bounds = (self.longitude[0] - boundary_delta, self.longitude[1] + boundary_delta)
        return latitude_bounds, longitude_bounds

    @staticmethod
    def _build_coord_slice(coord_values: np.ndarray, bounds: tuple[float, float]) -> slice:
        lower = min(bounds)
        upper = max(bounds)
        if coord_values[0] > coord_values[-1]:
            return slice(upper, lower)
        return slice(lower, upper)

    @staticmethod
    def _mean_pool_then_restore_spatial(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            return tensor

        spatial_shape = tensor.shape[-2:]
        if spatial_shape[0] < 2 or spatial_shape[1] < 2:
            return tensor

        leading_shape = tensor.shape[:-2]
        pooled = F.avg_pool2d(tensor.reshape(-1, 1, spatial_shape[0], spatial_shape[1]), kernel_size = 2, stride = 2)
        restored = F.interpolate(pooled, size = spatial_shape, mode = "bilinear", align_corners = False)
        return restored.reshape(*leading_shape, spatial_shape[0], spatial_shape[1])

    def _select_data_array(
        self,
        ds: xr.Dataset,
        var_name: str,
        date_hour: pd.Timestamp,
        prediction_timedelta: pd.Timedelta,
    ) -> torch.Tensor:
        latitude_bounds, longitude_bounds = self._spatial_bounds()
        latitude_slice = self._build_coord_slice(ds.latitude.values, latitude_bounds)
        longitude_slice = self._build_coord_slice(ds.longitude.values, longitude_bounds)
        level_dim = "level" if "level" in ds.dims else "pressure_level"

        data_array = ds[var_name].sel(
            time = date_hour,
            prediction_timedelta = prediction_timedelta,
            latitude = latitude_slice,
            longitude = longitude_slice,
        )

        if level_dim in data_array.dims:
            data_array = data_array.sel({level_dim: self.levels})

        tensor = torch.as_tensor(data_array.values)
        return self._mean_pool_then_restore_spatial(tensor)

    def __len__(self) -> int:
        return len(self.time_axis)

    def get_latitude_longitude(self):
        upper_path, _ = self._dt_to_path(self.time_axis[0])
        latitude_bounds, longitude_bounds = self._spatial_bounds()
        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc:
            upper_nc.load()
            latitude_slice = self._build_coord_slice(upper_nc.latitude.values, latitude_bounds)
            longitude_slice = self._build_coord_slice(upper_nc.longitude.values, longitude_bounds)
            latitude = upper_nc.latitude.sel(latitude = latitude_slice).values
            longitude = upper_nc.longitude.sel(longitude = longitude_slice).values
        return torch.tensor(latitude), torch.tensor(longitude)

    def get_levels(self):
        upper_path, _ = self._dt_to_path(self.time_axis[0])
        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc:
            upper_nc.load()
            level_dim = "level" if "level" in upper_nc.dims else "pressure_level"
            levels = upper_nc[level_dim].values
        return tuple(levels)

    def __getitem__(self, index: int) -> dict:
        date_hour = self.time_axis[index]
        upper_path, surface_path = self._dt_to_path(date_hour)

        result = {
            "prediction_timedelta": torch.tensor(self.prediction_timedelta_hours, dtype = torch.int64),
            "surf_vars": {},
            "atmos_vars": {},
        }

        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc, xr.open_dataset(surface_path, decode_timedelta = True) as surface_nc:
            upper_nc.load()
            surface_nc.load()

            for prediction_timedelta in self.prediction_timedeltas:
                for surface_var in self.surface_variables:
                    mapped_name = self.map_var_name_for_Aurora(surface_var)
                    data = self._select_data_array(surface_nc, surface_var, date_hour, prediction_timedelta)
                    result["surf_vars"].setdefault(mapped_name, []).append(data)

                for upper_var in self.upper_variables:
                    data = self._select_data_array(upper_nc, upper_var, date_hour, prediction_timedelta)
                    result["atmos_vars"].setdefault(upper_var, []).append(data)

        result["surf_vars"] = {
            var_name: torch.stack(tensors, dim = 0)
            for var_name, tensors in result["surf_vars"].items()
        }
        result["atmos_vars"] = {
            var_name: torch.stack(tensors, dim = 0)
            for var_name, tensors in result["atmos_vars"].items()
        }

        if self.get_datetime:
            result["datetime"] = date_hour.strftime("%Y-%m-%d %H:%M:%S")

        return result