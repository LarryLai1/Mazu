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
        forecast_cycle_hours: int = 6,
        get_datetime: bool = True,
        enable_pooling: bool = False,
        interp_mode: str = "forward",
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
        self.enable_pooling = enable_pooling
        self.interp_mode = interp_mode
        if self.interp_mode not in ("forward", "surrounding"):
            raise ValueError(f"Unsupported interp_mode: {self.interp_mode}")
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
        name = date_hour.strftime(r"%Y%m%d_%H")
        # candidate 1: root/YYYY/YYYYMM/YYYYMMDD/{name}_upper.nc
        p1 = Path(self.boundary_root_dir) / date_hour.strftime(r"%Y/%Y%m/%Y%m%d") / f"{name}_upper.nc"
        s1 = Path(self.boundary_root_dir) / date_hour.strftime(r"%Y/%Y%m/%Y%m%d") / f"{name}_sfc.nc"
        if p1.exists() and s1.exists():
            return str(p1), str(s1)

        # candidate 2: root/YYYYMMDD/{name}_upper.nc
        p2 = Path(self.boundary_root_dir) / date_hour.strftime(r"%Y%m%d") / f"{name}_upper.nc"
        s2 = Path(self.boundary_root_dir) / date_hour.strftime(r"%Y%m%d") / f"{name}_sfc.nc"
        if p2.exists() and s2.exists():
            return str(p2), str(s2)

        # candidate 3: root/{name}_upper.nc
        p3 = Path(self.boundary_root_dir) / f"{name}_upper.nc"
        s3 = Path(self.boundary_root_dir) / f"{name}_sfc.nc"
        if p3.exists() and s3.exists():
            return str(p3), str(s3)

        # fallback to default nested path (may raise later when opening)
        dir_path = Path(self.boundary_root_dir) / date_hour.strftime(r"%Y/%Y%m/%Y%m%d")
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

    def _choose_interp_times(
        self,
        time_values: np.ndarray,
        target_time: pd.Timestamp,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        times = pd.DatetimeIndex(pd.to_datetime(time_values))
        if len(times) < 2:
            raise ValueError("Boundary file must contain at least two time steps for interpolation.")

        if self.interp_mode == "surrounding":
            idx = times.searchsorted(target_time, side = "left")
            if idx <= 0:
                return times[0], times[1]
            if idx >= len(times):
                return times[-2], times[-1]
            return times[idx - 1], times[idx]

        idx = times.searchsorted(target_time, side = "left")
        if idx >= len(times):
            return times[-2], times[-1]
        if idx == len(times) - 1:
            return times[idx - 1], times[idx]
        return times[idx], times[idx + 1]

    def _select_data_array_at_time(
        self,
        ds: xr.Dataset,
        var_name: str,
        target_time: pd.Timestamp,
    ) -> torch.Tensor:
        latitude_bounds, longitude_bounds = self._spatial_bounds()
        latitude_slice = self._build_coord_slice(ds.lat.values, latitude_bounds)
        longitude_slice = self._build_coord_slice(ds.lon.values, longitude_bounds)
        level_dim = "level" if "level" in ds.dims else "pressure_level"
        target_time = pd.Timestamp(target_time)
        time_values = pd.to_datetime(ds.time.values)

        if target_time in time_values:
            data_array = ds[var_name].sel(
                time = target_time,
                lat = latitude_slice,
                lon = longitude_slice,
            )
        else:
            t1, t2 = self._choose_interp_times(time_values, target_time)
            data_1 = ds[var_name].sel(
                time = t1,
                lat = latitude_slice,
                lon = longitude_slice,
            )
            data_2 = ds[var_name].sel(
                time = t2,
                lat = latitude_slice,
                lon = longitude_slice,
            )
            weight = (target_time - t1) / (t2 - t1)
            data_array = data_1 + (data_2 - data_1) * float(weight)

        if level_dim in data_array.dims:
            data_array = data_array.sel({level_dim: self.levels})

        tensor = torch.as_tensor(data_array.values)
        if self.enable_pooling:
            return self._mean_pool_then_restore_spatial(tensor)
        return tensor

    def __len__(self) -> int:
        return len(self.time_axis)

    def get_latitude_longitude(self):
        upper_path, _ = self._dt_to_path(self.time_axis[0])
        latitude_bounds, longitude_bounds = self._spatial_bounds()
        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc:
            upper_nc.load()
            latitude_slice = self._build_coord_slice(upper_nc.lat.values, latitude_bounds)
            longitude_slice = self._build_coord_slice(upper_nc.lon.values, longitude_bounds)
            latitude = upper_nc.lat.sel(lat = latitude_slice).values
            longitude = upper_nc.lon.sel(lon = longitude_slice).values
        return torch.tensor(latitude), torch.tensor(longitude)

    def get_levels(self):
        upper_path, _ = self._dt_to_path(self.time_axis[0])
        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc:
            upper_nc.load()
            level_dim = "level" if "level" in upper_nc.dims else "pressure_level"
            levels = upper_nc[level_dim].values
        return tuple(levels)

    def get_base_time(self, target_time: pd.Timestamp) -> pd.Timestamp:
        target_time = pd.Timestamp(target_time)
        return target_time.floor(f"{self.forecast_cycle_hours}h")

    def get_boundary_at_time(
        self,
        base_time: pd.Timestamp,
        target_time: pd.Timestamp,
    ) -> dict:
        base_time = pd.Timestamp(base_time)
        target_time = pd.Timestamp(target_time)
        upper_path, surface_path = self._dt_to_path(base_time)

        result = {"surf_vars": {}, "atmos_vars": {}}

        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc, xr.open_dataset(surface_path, decode_timedelta = True) as surface_nc:
            upper_nc.load()
            surface_nc.load()

            for surface_var in self.surface_variables:
                mapped_name = self.map_var_name_for_Aurora(surface_var)
                data = self._select_data_array_at_time(surface_nc, mapped_name, target_time)
                result["surf_vars"][mapped_name] = data

            for upper_var in self.upper_variables:
                data = self._select_data_array_at_time(upper_nc, upper_var, target_time)
                result["atmos_vars"][upper_var] = data

        return result

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
                target_time = date_hour + prediction_timedelta
                for surface_var in self.surface_variables:
                    mapped_name = self.map_var_name_for_Aurora(surface_var)
                    data = self._select_data_array_at_time(surface_nc, mapped_name, target_time)
                    result["surf_vars"].setdefault(mapped_name, []).append(data)

                for upper_var in self.upper_variables:
                    data = self._select_data_array_at_time(upper_nc, upper_var, target_time)
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