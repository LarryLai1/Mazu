from datetime import datetime, timedelta
from math import ceil
from os import path

import numpy as np
import torch
import torch.utils.data as D
import xarray as xr
from einops import pack, rearrange
from netCDF4 import Dataset as NC4Dataset  # type: ignore

from .types import Stat_T, WeatherData, WeatherDataNumpy


class ERA5TWDataset(D.Dataset):
    """
    Dataset for ERA5 reanalysis data.

    Returns:
        tuple[WeatherData, WeatherData]: Tuple of input and target weather data.
                                         One WeatherData is composed of upper-air data and surface data.
                                         Upper-air data is of shape (pressure level, latitude, longitude, variable).
                                         Surface data is of shape (latitude, longitude, variable).
    """

    def __init__(self,
                 root_dir: str,
                 start_date_hour: datetime,
                 end_date_hour: datetime,
                 upper_variables: list[str],
                 surface_variables: list[str],
                 standardize: bool = False,
                 get_stat: bool = False) -> None:
        """
        Args:
            root_dir (str): Root directory of the dataset.
            start_date_hour (datetime): Starting date and hour of the dataset.
            end_date_hour (datetime): Ending date and hour of the dataset, excluded.
            upper_variables (list[str]): List of variables to be included in the upper-air data.
            surface_variables (list[str]): List of variables to be included in the surface data.
            standardize (bool, optional): Whether to standardize the data. Defaults to False.
            get_stat (bool, optional): Whether to return the mean and std of the dataset along with the data. Only works when standardize is True. Defaults to False.
        """
        super().__init__()
        self.root_dir = root_dir
        self.start_date = start_date_hour
        self.end_date = end_date_hour
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.standardize = standardize
        if self.standardize:
            self.upper_mean, self.upper_std, self.surface_mean, self.surface_std = self._load_stat()
        self.get_stat = self.standardize and get_stat

    def _load_stat(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of mean and std of upper-air data and surface data.
        """
        stat = torch.load(path.join(self.root_dir, "stat_dict.pt"))
        upper_mean, _ = pack([stat["mean_upper"][var] for var in self.upper_variables], "z h w *")
        upper_std, _ = pack([stat["std_upper"][var] for var in self.upper_variables], "z h w *")
        surface_mean, _ = pack([stat["mean_surface"][var] for var in self.surface_variables], "h w *")
        surface_std, _ = pack([stat["std_surface"][var] for var in self.surface_variables], "h w *")
        return upper_mean, upper_std, surface_mean, surface_std

    def _stack_nc(self, upper_nc: NC4Dataset, surface_nc: NC4Dataset) -> WeatherData:
        """
        Args:
            upper_nc (netCDF4.Dataset): Dataset of upper-air data.
            surface_nc (netCDF4.Dataset): Dataset of surface data.
        Returns:
            WeatherData: Tuple of upper-air data and surface data.
                                           Upper-air data is of shape(pressure level, latitude, longitude, variable).
                                           Surface data is of shape(latitude, longitude, variable).
        """
        upper_data, _ = pack([rearrange(upper_nc.variables[v][:], "() z h w -> z h w")
                             for v in self.upper_variables], "z h w *")
        upper_data = torch.Tensor(upper_data)
        surface_data, _ = pack([rearrange(surface_nc.variables[v][:], "() h w -> h w")
                               for v in self.surface_variables], "h w *")
        surface_data = torch.Tensor(surface_data)
        if self.standardize:
            upper_data = (upper_data - self.upper_mean) / self.upper_std
            surface_data = (surface_data - self.surface_mean) / self.surface_std
        return upper_data, surface_data

    def _dt_to_path(self, date_hour: datetime) -> tuple[str, str]:
        """
        Args:
            date_hour (datetime): Date and hour to be converted to path.
        Returns:
            tuple[str, str]: Tuple of upper air path and surface path.
        """
        dir_path = path.join(self.root_dir, date_hour.strftime(r'%Y/%Y%m/%Y%m%d'))
        name = date_hour.strftime(r'%Y%m%d%H')
        return path.join(dir_path, f"{name}_upper.nc"), path.join(dir_path, f"{name}_sfc.nc")

    def __len__(self) -> int:
        return round((self.end_date - self.start_date).total_seconds()) // (60*60)

    def __getitem__(self, index: int) -> tuple[WeatherData, WeatherData] | tuple[WeatherData, WeatherData, Stat_T]:
        """
        Args:
            index (int): Index of the hour to be retrieved.
        Returns:
            Return type depends on get_stat.
            tuple[WeatherData, WeatherData]: When get_stat is False, return tuple of input and target weather data.
                                             One WeatherData is composed of upper-air data and surface data.
                                             Upper-air data is of shape (pressure level, latitude, longitude, variable).
                                             Surface data is of shape (latitude, longitude, variable).
            tuple[WeatherData, WeatherData, Stat_T]: When get_stat is True, return tuple of input, target weather data
                                                     and (mean_upper, std_upper, mean_surface, std_surface).
        """
        date_hour_input = self.start_date + timedelta(hours=index)
        upper_path_input, surface_path_input = self._dt_to_path(date_hour_input)
        upper_nc_input = NC4Dataset(upper_path_input, "r")
        surface_input = NC4Dataset(surface_path_input, "r")
        stacked_input = self._stack_nc(upper_nc_input, surface_input)
        upper_nc_input.close()
        surface_input.close()

        date_hour_target = date_hour_input + timedelta(hours=1)
        upper_path_target, surface_path_target = self._dt_to_path(date_hour_target)
        upper_nc_target = NC4Dataset(upper_path_target, "r")
        surface_target = NC4Dataset(surface_path_target, "r")
        stacked_target = self._stack_nc(upper_nc_target, surface_target)
        upper_nc_target.close()
        surface_target.close()

        if self.get_stat:
            return stacked_input, stacked_target, (self.upper_mean, self.upper_std, self.surface_mean, self.surface_std)
        else:
            return stacked_input, stacked_target

    def get_lon_lat_lev(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of longitude, latitude, and pressure level.
        """
        upper_nc_path, _ = self._dt_to_path(self.start_date)
        upper_nc = NC4Dataset(upper_nc_path, "r")
        lon = np.squeeze(upper_nc.variables["longitude"][:])
        lat = np.flip(np.squeeze(upper_nc.variables["latitude"][:]))
        lev = np.squeeze(upper_nc.variables["level"][:])
        return lon, lat, lev


class ERA5GlobalDataset(ERA5TWDataset):
    """
    Dataset for ERA5 reanalysis data. This dataset is for inference using pretrained Pangu-Weather.
    """

    def __init__(self,
                 root_dir: str,
                 start_date_hour: datetime,
                 end_date_hour: datetime,
                 upper_variables: list[str],
                 surface_variables: list[str],
                 return_y: bool = True,
                 step_size: int = 1) -> None:
        """
        Args:
            root_dir (str): Root directory of the dataset.
            start_date_hour (datetime): Starting date and hour of the dataset.
            end_date_hour (datetime): Ending date and hour of the dataset, excluded.
            upper_variables (list[str]): List of variables to be included in the upper-air data.
            surface_variables (list[str]): List of variables to be included in the surface data.
            return_y (bool): Whether to return the target data. Defaults to True.
            step_size (int): Step size of the dataset. Defaults to 1.
        """
        super().__init__(
            root_dir=root_dir,
            start_date_hour=start_date_hour,
            end_date_hour=end_date_hour,
            upper_variables=upper_variables,
            surface_variables=surface_variables,
            standardize=False,
            get_stat=False
        )
        assert step_size > 0
        self.return_y = return_y
        self.step_size = step_size

    def _dt_to_path(self, date_hour: datetime) -> tuple[str, str]:
        """
        Args:
            date_hour (datetime): Date and hour to be converted to path.
        Returns:
            tuple[str, str]: Tuple of upper air path and surface path.
        """
        dir_path = path.join(self.root_dir, date_hour.strftime(r'%Y/%Y%m/%Y%m%d'))
        name = date_hour.strftime(r'%Y%m%d')
        return path.join(dir_path, f"{name}_upper.nc"), path.join(dir_path, f"{name}_sfc.nc")

    def _stack_nc(self, upper_nc: NC4Dataset, surface_nc: NC4Dataset, hour: int) -> WeatherDataNumpy:
        """
        Args:
            upper_nc (netCDF4.Dataset): Dataset of upper-air data.
            surface_nc (netCDF4.Dataset): Dataset of surface data.
            hour (int): Hour of the data. (Because one file contains 24 hours of data.)

        Returns:
            WeatherDataNumpy:
                Tuple of upper-air data and surface data.
                Upper-air data is of shape(pressure level, latitude, longitude, variable).
                Surface data is of shape(latitude, longitude, variable).
        """
        upper_data, _ = pack([upper_nc.variables[v][hour, :, :, :] for v in self.upper_variables], "z h w *")
        surface_data, _ = pack([surface_nc.variables[v][hour, :, :] for v in self.surface_variables], "h w *")
        return upper_data, surface_data

    def __len__(self) -> int:
        return ceil(round((self.end_date - self.start_date).total_seconds()) / (60*60*self.step_size))

    def __getitem__(self, index: int) -> WeatherDataNumpy | tuple[WeatherDataNumpy, WeatherDataNumpy]:
        """
        Args:
            index (int): Index of the hour to be retrieved.
        Returns:
            WeatherDataNumpy:
                When return_y is False, return input weather data.

            tuple[WeatherDataNumpy, WeatherDataNumpy]:
                One WeatherDataNumpy is composed of upper-air data and surface data.
                Upper-air data is of shape (pressure level, latitude, longitude, variable).
                Surface data is of shape (latitude, longitude, variable).
        """
        date_hour_input = self.start_date + timedelta(hours=index*self.step_size)
        upper_path_input, surface_path_input = self._dt_to_path(date_hour_input)
        upper_nc_input = NC4Dataset(upper_path_input, "r")
        surface_input = NC4Dataset(surface_path_input, "r")
        stacked_input = self._stack_nc(upper_nc_input, surface_input, date_hour_input.hour)
        upper_nc_input.close()
        surface_input.close()

        if not self.return_y:
            return stacked_input

        date_hour_target = date_hour_input + timedelta(hours=1)
        upper_path_target, surface_path_target = self._dt_to_path(date_hour_target)
        upper_nc_target = NC4Dataset(upper_path_target, "r")
        surface_target = NC4Dataset(surface_path_target, "r")
        stacked_target = self._stack_nc(upper_nc_target, surface_target, date_hour_target.hour)
        upper_nc_target.close()
        surface_target.close()

        return stacked_input, stacked_target


class PanguBCDataset(D.Dataset):
    def __init__(self,
                 root_dir: str,
                 start_date_hour: datetime,
                 end_date_hour: datetime,
                 upper_variables: list[str],
                 surface_variables: list[str],
                 stat_path: str) -> None:
        """
        Args:
            root_dir (str): Root directory of the dataset.
            start_date_hour (datetime): Starting date and hour of the dataset.
            end_date_hour (datetime): Ending date and hour of the dataset, excluded.
            upper_variables (list[str]): List of variables to be included in the upper-air data.
            surface_variables (list[str]): List of variables to be included in the surface data.
            stat_path (str): Path to the statistics file.
        """
        super().__init__()
        self.root_dir = root_dir
        self.start_date = start_date_hour
        self.end_date = end_date_hour
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.upper_mean, self.upper_std, self.surface_mean, self.surface_std = self._load_stat(stat_path)

    def _load_stat(self, path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of mean and std of upper-air data and surface data.
        """
        stat = torch.load(path)
        upper_mean, _ = pack([stat["mean_upper"][var] for var in self.upper_variables], "z h w *")
        upper_std, _ = pack([stat["std_upper"][var] for var in self.upper_variables], "z h w *")
        surface_mean, _ = pack([stat["mean_surface"][var] for var in self.surface_variables], "h w *")
        surface_std, _ = pack([stat["std_surface"][var] for var in self.surface_variables], "h w *")
        return upper_mean, upper_std, surface_mean, surface_std

    def _dt_to_path(self, date_hour: datetime) -> tuple[str, str]:
        """
        Args:
            date_hour (datetime): Date and hour to be converted to path.
        Returns:
            tuple[str, str]: Tuple of upper air path and surface path.
        """
        dir_path = path.join(self.root_dir, date_hour.strftime(r'%Y/%Y%m/%Y%m%d'))
        name = date_hour.strftime(r'%Y%m%dT%H')
        return path.join(dir_path, f"{name}_upper.nc"), path.join(dir_path, f"{name}_surface.nc")

    def _paths_to_arr(self, path_upper: str, path_surface: str) -> tuple[torch.Tensor, torch.Tensor]:
        ds_upper = xr.load_dataset(path_upper)
        ds_surface = xr.load_dataset(path_surface)
        upper_data, _ = pack([ds_upper[v].values for v in self.upper_variables], "t z h w *")
        surface_data, _ = pack([ds_surface[v].values for v in self.surface_variables], "t h w *")
        return torch.Tensor(upper_data), torch.Tensor(surface_data)

    def __len__(self) -> int:
        return round((self.end_date - self.start_date).total_seconds()) // (60*60)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        date_hour_input = self.start_date + timedelta(hours=index)
        upper_data_path, surface_data_path = self._dt_to_path(date_hour_input)
        upper_data, surface_data = self._paths_to_arr(upper_data_path, surface_data_path)

        upper_data = (upper_data - self.upper_mean) / self.upper_std
        surface_data = (surface_data - self.surface_mean) / self.surface_std

        return upper_data, surface_data
