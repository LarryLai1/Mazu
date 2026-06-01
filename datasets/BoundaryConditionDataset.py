from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr


class BoundaryConditionDataset_Aurora(torch.utils.data.Dataset):
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
        use_cache: bool = False,
        time_interp_mode: str = "interpolation",
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
        self.use_cache = use_cache
        self.time_interp_mode = time_interp_mode
        if self.time_interp_mode not in ("interpolation", "nearest", "exact"):
            raise ValueError(f"Unsupported time_interp_mode: {self.time_interp_mode}")
        if self.interp_mode not in ("forward", "surrounding"):
            raise ValueError(f"Unsupported interp_mode: {self.interp_mode}")
        self.time_axis = pd.date_range(
            start = self.start_date_hour,
            end = self.end_date_hour,
            freq = f"{self.forecast_cycle_hours}h",
        )
        self._cache = {}
        self._cache_latitude = None
        self._cache_longitude = None
        self._cache_levels = None
        if self.use_cache:
            self._build_cache()

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

    def _load_boundary_source_from_files(self, date_hour: pd.Timestamp) -> dict:
        upper_path, surface_path = self._dt_to_path(date_hour)
        latitude_bounds, longitude_bounds = self._spatial_bounds()

        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc, xr.open_dataset(surface_path, decode_timedelta = True) as surface_nc:
            upper_nc.load()
            surface_nc.load()

            latitude_slice = self._build_coord_slice(upper_nc.lat.values, latitude_bounds)
            longitude_slice = self._build_coord_slice(upper_nc.lon.values, longitude_bounds)
            level_dim = "level" if "level" in upper_nc.dims else "pressure_level"

            source = {
                "time_values": pd.DatetimeIndex(pd.to_datetime(upper_nc.time.values)),
                "latitude": torch.as_tensor(upper_nc.lat.sel(lat = latitude_slice).values),
                "longitude": torch.as_tensor(upper_nc.lon.sel(lon = longitude_slice).values),
                "levels": tuple(upper_nc[level_dim].sel({level_dim: self.levels}).values),
                "surf_vars": {},
                "atmos_vars": {},
            }

            for surface_var in self.surface_variables:
                mapped_name = self.map_var_name_for_Aurora(surface_var)
                data_array = surface_nc[mapped_name].sel(
                    lat = latitude_slice,
                    lon = longitude_slice,
                )
                source["surf_vars"][mapped_name] = torch.as_tensor(data_array.values)

            for upper_var in self.upper_variables:
                data_array = upper_nc[upper_var].sel(
                    lat = latitude_slice,
                    lon = longitude_slice,
                )
                if level_dim in data_array.dims:
                    data_array = data_array.sel({level_dim: self.levels})
                source["atmos_vars"][upper_var] = torch.as_tensor(data_array.values)

        return source

    def _build_cache(self) -> None:
        for date_hour in self.time_axis:
            self._cache[pd.Timestamp(date_hour)] = self._load_boundary_source_from_files(date_hour)
        first_source = self._cache[pd.Timestamp(self.time_axis[0])]
        self._cache_latitude = first_source["latitude"]
        self._cache_longitude = first_source["longitude"]
        self._cache_levels = first_source["levels"]
        # print(f"Cache built with {len(self._cache)} entries. Latitude shape: {self._cache_latitude.shape}, Longitude shape: {self._cache_longitude.shape}, Levels: {self._cache_levels}")

    def _select_from_source(
        self,
        time_values: pd.DatetimeIndex,
        tensor: torch.Tensor,
        target_time: pd.Timestamp,
    ) -> torch.Tensor:
        target_time = pd.Timestamp(target_time)
        if self.time_interp_mode == "exact":
            if target_time not in time_values:
                return None
            return tensor[time_values.get_loc(target_time)]

        if self.time_interp_mode == "nearest":
            diffs = np.abs(time_values - target_time)
            nearest_idx = np.argmin(diffs)
            return tensor[nearest_idx]

        # Default: interpolation
        if target_time in time_values:
            return tensor[time_values.get_loc(target_time)]

        idx = time_values.searchsorted(target_time, side = "left")
        if idx <= 0:
            return tensor[0]
        if idx >= len(time_values):
            return tensor[-1]

        t1 = time_values[idx - 1]
        t2 = time_values[idx]
        data_1 = tensor[idx - 1]
        data_2 = tensor[idx]
        weight = (target_time - t1) / (t2 - t1)
        return data_1 + (data_2 - data_1) * float(weight)

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
            if self.time_interp_mode == "exact":
                return None
            elif self.time_interp_mode == "nearest":
                diffs = np.abs(time_values - target_time)
                nearest_idx = np.argmin(diffs)
                nearest_time = time_values[nearest_idx]
                data_array = ds[var_name].sel(
                    time = nearest_time,
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
        if self.use_cache:
            return self._cache_latitude, self._cache_longitude
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
        if self.use_cache:
            return self._cache_levels
        upper_path, _ = self._dt_to_path(self.time_axis[0])
        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc:
            upper_nc.load()
            level_dim = "level" if "level" in upper_nc.dims else "pressure_level"
            levels = upper_nc[level_dim].values
        return tuple(levels)

    def get_base_time(self, target_time: pd.Timestamp) -> pd.Timestamp:
        target_time = pd.Timestamp(target_time)
        return target_time.floor(f"{self.forecast_cycle_hours}h")

    def get_boundary_source(self, base_time: pd.Timestamp) -> dict:
        base_time = pd.Timestamp(base_time)
        if self.use_cache:
            return self._cache[base_time]
        return self._load_boundary_source_from_files(base_time)

    def get_boundary_at_time(
        self,
        base_time: pd.Timestamp,
        target_time: pd.Timestamp,
    ) -> dict:
        base_time = pd.Timestamp(base_time)
        target_time = pd.Timestamp(target_time)
        if self.use_cache:
            source = self._cache[base_time]
            result = {"surf_vars": {}, "atmos_vars": {}}
            time_values = source["time_values"]

            for surface_var in self.surface_variables:
                mapped_name = self.map_var_name_for_Aurora(surface_var)
                val = self._select_from_source(
                    time_values,
                    source["surf_vars"][mapped_name],
                    target_time,
                )
                if val is None:
                    return None
                result["surf_vars"][mapped_name] = val

            for upper_var in self.upper_variables:
                val = self._select_from_source(
                    time_values,
                    source["atmos_vars"][upper_var],
                    target_time,
                )
                if val is None:
                    return None
                result["atmos_vars"][upper_var] = val
            return result

        upper_path, surface_path = self._dt_to_path(base_time)

        result = {"surf_vars": {}, "atmos_vars": {}}

        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc, xr.open_dataset(surface_path, decode_timedelta = True) as surface_nc:
            upper_nc.load()
            surface_nc.load()

            for surface_var in self.surface_variables:
                mapped_name = self.map_var_name_for_Aurora(surface_var)
                data = self._select_data_array_at_time(surface_nc, mapped_name, target_time)
                if data is None:
                    return None
                result["surf_vars"][mapped_name] = data

            for upper_var in self.upper_variables:
                data = self._select_data_array_at_time(upper_nc, upper_var, target_time)
                if data is None:
                    return None
                result["atmos_vars"][upper_var] = data

        return result

    def __getitem__(self, index: int) -> dict:
        date_hour = self.time_axis[index]

        result = {
            "prediction_timedelta": torch.tensor(self.prediction_timedelta_hours, dtype = torch.int64),
            "surf_vars": {},
            "atmos_vars": {},
        }

        if self.use_cache:
            source = self._cache[date_hour]
            time_values = source["time_values"]

            for prediction_timedelta in self.prediction_timedeltas:
                target_time = date_hour + prediction_timedelta
                for surface_var in self.surface_variables:
                    mapped_name = self.map_var_name_for_Aurora(surface_var)
                    data = self._select_from_source(
                        time_values,
                        source["surf_vars"][mapped_name],
                        target_time,
                    )
                    result["surf_vars"].setdefault(mapped_name, []).append(data)

                for upper_var in self.upper_variables:
                    data = self._select_from_source(
                        time_values,
                        source["atmos_vars"][upper_var],
                        target_time,
                    )
                    result["atmos_vars"].setdefault(upper_var, []).append(data)
        else:
            upper_path, surface_path = self._dt_to_path(date_hour)

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


class BoundaryConditionDataset_ERA5(torch.utils.data.Dataset):
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
        prediction_timedeltas: list[int] | tuple[int, ...] = (0, 12),
        forecast_cycle_hours: int = 12,
        get_datetime: bool = True,
        enable_pooling: bool = False,
        interp_mode: str = "forward",
        use_cache: bool = False,
        time_interp_mode: str = "interpolation",
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
        self.use_cache = use_cache
        self.time_interp_mode = time_interp_mode
        if self.time_interp_mode not in ("interpolation", "nearest", "exact"):
            raise ValueError(f"Unsupported time_interp_mode: {self.time_interp_mode}")
        if self.interp_mode not in ("forward", "surrounding"):
            raise ValueError(f"Unsupported interp_mode: {self.interp_mode}")
        self.time_axis = pd.date_range(
            start = self.start_date_hour,
            end = self.end_date_hour,
            freq = f"{self.forecast_cycle_hours}h",
        )
        self._cache = {}
        self._cache_latitude = None
        self._cache_longitude = None
        self._cache_levels = None
        if self.use_cache:
            self._build_cache()

    def _dt_to_path(self, date_hour: pd.Timestamp) -> tuple[str, str]:
        date = pd.Timestamp(date_hour).normalize()
        name = date.strftime(r"%Y%m%d")
        # candidate 1: root/YYYY/YYYYMM/{name}_upper.nc
        p1 = Path(self.boundary_root_dir) / date.strftime(r"%Y/%Y%m") / f"{name}_upper.nc"
        s1 = Path(self.boundary_root_dir) / date.strftime(r"%Y/%Y%m") / f"{name}_sfc.nc"
        if p1.exists() and s1.exists():
            return str(p1), str(s1)

        # candidate 2: root/YYYYMMDD/{name}_upper.nc
        p2 = Path(self.boundary_root_dir) / date.strftime(r"%Y%m%d") / f"{name}_upper.nc"
        s2 = Path(self.boundary_root_dir) / date.strftime(r"%Y%m%d") / f"{name}_sfc.nc"
        if p2.exists() and s2.exists():
            return str(p2), str(s2)

        # candidate 3: root/{name}_upper.nc
        p3 = Path(self.boundary_root_dir) / f"{name}_upper.nc"
        s3 = Path(self.boundary_root_dir) / f"{name}_sfc.nc"
        if p3.exists() and s3.exists():
            return str(p3), str(s3)

        # fallback to default nested path (may raise later when opening)
        dir_path = Path(self.boundary_root_dir) / date.strftime(r"%Y/%Y%m")
        return str(dir_path / f"{name}_upper.nc"), str(dir_path / f"{name}_sfc.nc")

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
    def _normalize_prediction_timedelta_hours(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.timedelta64):
            return np.asarray(values / np.timedelta64(1, "h"), dtype = np.float32)
        return np.asarray(values, dtype = np.float32)

    @staticmethod
    def _select_time_coord(ds: xr.Dataset, target_time: pd.Timestamp) -> xr.Dataset:
        if "time" not in ds.coords and "time" not in ds.dims:
            return ds
        time_values = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
        target_time = pd.Timestamp(target_time)
        if target_time in time_values:
            return ds.sel(time = target_time)
        if len(time_values) == 1:
            return ds.sel(time = time_values[0])
        return ds.sel(time = time_values[0])

    @staticmethod
    def _prediction_timedelta_bracketing_indices(
        prediction_timedelta_hours: torch.Tensor,
        target_prediction_timedelta_hours: float,
    ) -> tuple[int, int, torch.Tensor]:
        if prediction_timedelta_hours.numel() == 0:
            raise ValueError("Boundary file must contain at least one prediction_timedelta entry.")

        target = torch.as_tensor(
            float(target_prediction_timedelta_hours),
            device = prediction_timedelta_hours.device,
            dtype = prediction_timedelta_hours.dtype,
        )
        idx = int(torch.searchsorted(prediction_timedelta_hours, target, right = False).item())
        if idx <= 0:
            return 0, 0, torch.zeros((), device = prediction_timedelta_hours.device, dtype = prediction_timedelta_hours.dtype)
        if idx >= prediction_timedelta_hours.numel():
            last = prediction_timedelta_hours.numel() - 1
            return last, last, torch.zeros((), device = prediction_timedelta_hours.device, dtype = prediction_timedelta_hours.dtype)
        if torch.isclose(prediction_timedelta_hours[idx], target):
            return idx, idx, torch.zeros((), device = prediction_timedelta_hours.device, dtype = prediction_timedelta_hours.dtype)

        lower_idx = idx - 1
        upper_idx = idx
        denom = prediction_timedelta_hours[upper_idx] - prediction_timedelta_hours[lower_idx]
        if torch.isclose(denom, torch.zeros_like(denom)):
            return lower_idx, upper_idx, torch.zeros_like(denom)
        weight = (target - prediction_timedelta_hours[lower_idx]) / denom
        return lower_idx, upper_idx, weight

    def _select_from_prediction_timedelta(
        self,
        prediction_timedelta_hours: torch.Tensor,
        tensor: torch.Tensor,
        target_prediction_timedelta_hours: float,
    ) -> torch.Tensor:
        if self.time_interp_mode == "nearest" or self.time_interp_mode == "exact":
            diffs = torch.abs(prediction_timedelta_hours - target_prediction_timedelta_hours)
            nearest_idx = torch.argmin(diffs).item()
            return tensor[nearest_idx]

        lower_idx, upper_idx, weight = self._prediction_timedelta_bracketing_indices(
            prediction_timedelta_hours,
            target_prediction_timedelta_hours,
        )
        if lower_idx == upper_idx:
            return tensor[lower_idx]
        return tensor[lower_idx] + (tensor[upper_idx] - tensor[lower_idx]) * weight

    def _load_era5_source_from_files(self, date_hour: pd.Timestamp) -> dict:
        upper_path, surface_path = self._dt_to_path(date_hour)
        latitude_bounds, longitude_bounds = self._spatial_bounds()

        with xr.open_dataset(upper_path, decode_timedelta = True) as upper_nc, xr.open_dataset(surface_path, decode_timedelta = True) as surface_nc:
            upper_nc.load()
            surface_nc.load()

            upper_nc = self._select_time_coord(upper_nc, date_hour)
            surface_nc = self._select_time_coord(surface_nc, date_hour)

            latitude_slice = self._build_coord_slice(upper_nc.latitude.values, latitude_bounds)
            longitude_slice = self._build_coord_slice(upper_nc.longitude.values, longitude_bounds)
            level_dim = "level" if "level" in upper_nc.dims else "pressure_level"

            if "prediction_timedelta" not in upper_nc.coords and "prediction_timedelta" not in upper_nc.dims:
                raise ValueError("ERA5 boundary file must contain prediction_timedelta coordinate.")

            prediction_timedelta_hours = torch.as_tensor(
                self._normalize_prediction_timedelta_hours(upper_nc["prediction_timedelta"].values),
                dtype = torch.float32,
            )

            time_values = np.atleast_1d(pd.to_datetime(upper_nc.time.values)) if "time" in upper_nc.coords else np.array([pd.Timestamp(date_hour)], dtype = "datetime64[ns]")

            source = {
                "time_values": pd.DatetimeIndex(pd.to_datetime(time_values)),
                "prediction_timedelta_hours": prediction_timedelta_hours,
                "latitude": torch.as_tensor(upper_nc.latitude.sel(latitude = latitude_slice).values),
                "longitude": torch.as_tensor(upper_nc.longitude.sel(longitude = longitude_slice).values),
                "levels": tuple(upper_nc[level_dim].sel({level_dim: self.levels}).values),
                "surf_vars": {},
                "atmos_vars": {},
            }

            for surface_var in self.surface_variables:
                mapped_name = self.map_var_name_for_Aurora(surface_var)
                # Try mapped (Aurora) name first, then the original name as a fallback.
                candidates = []
                if mapped_name not in candidates:
                    candidates.append(mapped_name)
                if surface_var not in candidates:
                    candidates.append(surface_var)

                file_var = None
                for c in candidates:
                    if c in surface_nc.variables:
                        file_var = c
                        break
                if file_var is None:
                    available = list(surface_nc.variables.keys())
                    raise KeyError(f"No variable named '{mapped_name}'. Variables on the dataset include {available}")

                data_array = surface_nc[file_var].sel(
                    latitude = latitude_slice,
                    longitude = longitude_slice,
                )
                if "time" in data_array.dims:
                    data_array = data_array.sel(time = date_hour)
                # store under the Aurora-mapped name for downstream consistency
                source["surf_vars"][mapped_name] = torch.as_tensor(data_array.values)

            for upper_var in self.upper_variables:
                data_array = upper_nc[upper_var].sel(
                    latitude = latitude_slice,
                    longitude = longitude_slice,
                )
                if "time" in data_array.dims:
                    data_array = data_array.sel(time = date_hour)
                if level_dim in data_array.dims:
                    data_array = data_array.sel({level_dim: self.levels})
                source["atmos_vars"][upper_var] = torch.as_tensor(data_array.values)

        return source

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

    def _load_boundary_source_from_files(self, date_hour: pd.Timestamp) -> dict:
        source = self._load_era5_source_from_files(date_hour)
        self.prediction_timedelta_hours = tuple(float(x) for x in source["prediction_timedelta_hours"].detach().cpu().tolist())
        self.prediction_timedeltas = tuple(pd.Timedelta(hours = float(x)) for x in self.prediction_timedelta_hours)
        return source

    def _build_cache(self) -> None:
        for date_hour in self.time_axis:
            self._cache[pd.Timestamp(date_hour)] = self._load_boundary_source_from_files(date_hour)
        first_source = self._cache[pd.Timestamp(self.time_axis[0])]
        self._cache_latitude = first_source["latitude"]
        self._cache_longitude = first_source["longitude"]
        self._cache_levels = first_source["levels"]

    @staticmethod
    def _select_from_source(
        time_values: pd.DatetimeIndex,
        tensor: torch.Tensor,
        target_time: pd.Timestamp,
    ) -> torch.Tensor:
        target_time = pd.Timestamp(target_time)
        if target_time in time_values:
            return tensor[time_values.get_loc(target_time)]

        idx = time_values.searchsorted(target_time, side = "left")
        if idx <= 0:
            return tensor[0]
        if idx >= len(time_values):
            return tensor[-1]

        t1 = time_values[idx - 1]
        t2 = time_values[idx]
        data_1 = tensor[idx - 1]
        data_2 = tensor[idx]
        weight = (target_time - t1) / (t2 - t1)
        return data_1 + (data_2 - data_1) * float(weight)

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
        latitude_slice = self._build_coord_slice(ds.latitude.values, latitude_bounds)
        longitude_slice = self._build_coord_slice(ds.longitude.values, longitude_bounds)
        level_dim = "level" if "level" in ds.dims else "pressure_level"
        target_time = pd.Timestamp(target_time)
        time_values = pd.to_datetime(ds.time.values)

        if target_time in time_values:
            data_array = ds[var_name].sel(
                time = target_time,
                latitude = latitude_slice,
                longitude = longitude_slice,
            )
        else:
            t1, t2 = self._choose_interp_times(time_values, target_time)
            data_1 = ds[var_name].sel(
                time = t1,
                latitude = latitude_slice,
                longitude = longitude_slice,
            )
            data_2 = ds[var_name].sel(
                time = t2,
                latitude = latitude_slice,
                longitude = longitude_slice,
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
        if self.use_cache:
            return self._cache_latitude, self._cache_longitude
        source = self.get_boundary_source(self.time_axis[0])
        return source["latitude"], source["longitude"]

    def get_levels(self):
        if self.use_cache:
            return self._cache_levels
        source = self.get_boundary_source(self.time_axis[0])
        return source["levels"]

    def get_base_time(self, target_time: pd.Timestamp) -> pd.Timestamp:
        target_time = pd.Timestamp(target_time)
        return target_time.floor(f"{self.forecast_cycle_hours}h")

    def get_boundary_source(self, base_time: pd.Timestamp) -> dict:
        base_time = pd.Timestamp(base_time)
        if self.use_cache:
            return self._cache[base_time]
        return self._load_boundary_source_from_files(base_time)

    def _select_boundary_from_source(
        self,
        source: dict,
        base_time: pd.Timestamp,
        target_time: pd.Timestamp,
    ) -> dict:
        base_time = pd.Timestamp(base_time)
        target_time = pd.Timestamp(target_time)
        target_prediction_timedelta_hours = float((target_time - base_time) / pd.Timedelta(hours = 1))
        prediction_timedelta_hours = source["prediction_timedelta_hours"]

        # Exact mode check
        if self.time_interp_mode == "exact":
            diffs = torch.abs(prediction_timedelta_hours - target_prediction_timedelta_hours)
            min_diff = torch.min(diffs).item()
            if min_diff > 1e-4:
                return None

        result = {"surf_vars": {}, "atmos_vars": {}}

        for surface_var in self.surface_variables:
            mapped_name = self.map_var_name_for_Aurora(surface_var)
            result["surf_vars"][mapped_name] = self._select_from_prediction_timedelta(
                prediction_timedelta_hours,
                source["surf_vars"][mapped_name],
                target_prediction_timedelta_hours,
            )

        for upper_var in self.upper_variables:
            result["atmos_vars"][upper_var] = self._select_from_prediction_timedelta(
                prediction_timedelta_hours,
                source["atmos_vars"][upper_var],
                target_prediction_timedelta_hours,
            )

        return result

    def get_boundary_at_time_from_source(
        self,
        source: dict,
        base_time: pd.Timestamp,
        target_time: pd.Timestamp,
    ) -> dict:
        return self._select_boundary_from_source(source, base_time, target_time)

    def get_boundary_at_time(
        self,
        base_time: pd.Timestamp,
        target_time: pd.Timestamp,
    ) -> dict:
        source = self.get_boundary_source(base_time)
        return self.get_boundary_at_time_from_source(source, base_time, target_time)

    def __getitem__(self, index: int) -> dict:
        date_hour = self.time_axis[index]
        source = self.get_boundary_source(date_hour)

        result = {
            "prediction_timedelta": np.array(source["prediction_timedelta_hours"].detach().cpu().numpy(), dtype = "timedelta64[h]"),
            "surf_vars": {var_name: tensor.clone() for var_name, tensor in source["surf_vars"].items()},
            "atmos_vars": {var_name: tensor.clone() for var_name, tensor in source["atmos_vars"].items()},
        }

        if self.get_datetime:
            result["datetime"] = date_hour.strftime("%Y-%m-%d %H:%M:%S")

        return result