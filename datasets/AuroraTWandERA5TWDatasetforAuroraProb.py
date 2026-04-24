from logging import info
from pathlib import Path
from typing import Callable, Union
import random

import pandas as pd
import torch
import xarray as xr

class AuroraTWandERA5TWDatasetforAuroraProb(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root_dir: str,
        start_date_hour: pd.Timestamp,
        end_date_hour: pd.Timestamp,
        upper_variables: list[str],
        surface_variables: list[str],
        static_variables: list[str],
        levels: list[int],
        latitude: tuple[int, int],
        longitude: tuple[int, int],
        lead_time: int = 0,
        input_time_window: int = 2,
        rollout_step: int = 1,
        get_datetime: bool = True,
        use_Aurora_input_len: int = 1,
        Aurora_input_dir: str = None,
        aurora_prob: Union[float, Callable[[], float]] = 0.5,
    ) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.start_date_hour = pd.Timestamp(start_date_hour)
        self.end_date_hour = pd.Timestamp(end_date_hour)
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.static_variables = static_variables
        self.levels = levels
        self.lead_time = lead_time
        self.input_time_window = input_time_window
        self.rollout_step = rollout_step
        self.latitude = latitude
        self.longitude = longitude
        self.get_datetime = get_datetime

        self.use_Aurora_input_len = use_Aurora_input_len
        self.Aurora_input_dir = Aurora_input_dir
        self.aurora_prob = aurora_prob

        assert self.use_Aurora_input_len >= 1, \
            "use_Aurora_input_len should be at least 1."
        assert self.Aurora_input_dir, \
            "Aurora_input_dir should be provided when use_Aurora_input_len >= 1."
        assert self.use_Aurora_input_len <= self.input_time_window, \
            "use_Aurora_input_len should be less than or equal to input_time_window."

    def set_aurora_prob(self, prob: Union[float, Callable[[], float]]):
        self.aurora_prob = prob

    def _get_aurora_prob(self) -> float:
        if callable(self.aurora_prob):
            return self.aurora_prob()
        return self.aurora_prob

    def map_var_name_for_Aurora(self, var_name: str) -> str:
        var_name_mapping = {
            "t2m": "2t",
            "u10": "10u",
            "v10": "10v",
            "msl": "msl",
        }
        if var_name in var_name_mapping:
            return var_name_mapping[var_name]
        else:
            return var_name

    def get_latitude_longitude(self):
        upper_path, _ = self._dt_to_path(self.start_date_hour)
        with xr.open_dataset(upper_path) as upper_nc:
            upper_nc.load()
            latitude = upper_nc.latitude.sel(latitude = slice(*self.latitude)).values
            longitude = upper_nc.longitude.sel(longitude = slice(*self.longitude)).values
        return torch.tensor(latitude), torch.tensor(longitude)

    def get_levels(self):
        upper_path, _ = self._dt_to_path(self.start_date_hour)
        with xr.open_dataset(upper_path) as upper_nc:
            upper_nc.load()
            levels = upper_nc.pressure_level.values
        return tuple(levels)

    def get_static_vars_ds(self):
        static_path = Path(self.data_root_dir) / "static" / "static_vars.nc"
        with xr.open_dataset(static_path) as _ds:
            _ds.load()
            _d = {
                "static_vars": {
                    v: torch.tensor(
                        _ds[v].sel(
                            latitude = slice(*self.latitude),
                            longitude = slice(*self.longitude),
                        ).values
                    ).squeeze()
                    for v in self.static_variables
                }
            }
        return _d

    def _dt_to_path(self, date_hour: pd.Timestamp) -> str:
        dir_path = Path(self.data_root_dir) / date_hour.strftime(r"%Y/%Y%m/%Y%m%d")
        name = date_hour.strftime(r"%Y%m%d%H")

        return str(dir_path / f"{name}_upper.nc"), str(dir_path / f"{name}_sfc.nc")

    def _dt_to_path_Aurora(self, date_hour: pd.Timestamp, step: int) -> str:
        dir_path = Path(self.Aurora_input_dir)
        return str(dir_path / f"{date_hour.strftime('%Y%m%d_%H%M%S')}+{step * self.lead_time}hr.nc")

    def __len__(self) -> int:
        duration = self.end_date_hour - self.start_date_hour
        duration_hours = round(duration.total_seconds()) // (60 * 60)
        return duration_hours - (self.input_time_window - 1 + self.rollout_step) * self.lead_time + 1

    def _nc_to_dict(self, upper_nc, sfc_nc) -> dict:
        _d = {
            "surf_vars": {
                self.map_var_name_for_Aurora(v): torch.tensor(
                    sfc_nc[v].sel(
                        latitude = slice(*self.latitude),
                        longitude = slice(*self.longitude),
                    ).values,
                ).squeeze() for v in self.surface_variables
            },
            "atmos_vars": {
                v: torch.tensor(
                    upper_nc[v].sel(
                        pressure_level = self.levels,
                        latitude = slice(*self.latitude),
                        longitude = slice(*self.longitude),
                    ).values
                ).squeeze() for v in self.upper_variables
            },
        }

        return _d

    def _nc_to_dict_AuroraTW(self, ss_ds) -> dict:
        _d = {
            "surf_vars": {
                k.removeprefix("surf_"): \
                    torch.Tensor(v.values).squeeze() for k, v in ss_ds.items() if k.startswith("surf_")
            },
            "atmos_vars": {
                k.removeprefix("atmos_"): \
                    torch.Tensor(v.values).squeeze() for k, v in ss_ds.items() if k.startswith("atmos_")
            },
        }

        return _d

    def concat_ts(self, list_of_ts: list[dict]):
        stacked_dict = {
            'surf_vars': {
                var: torch.stack([d['surf_vars'][var] for d in list_of_ts], dim = 0)
                for var in list_of_ts[0]['surf_vars']
            },
            'atmos_vars': {
                var: torch.stack([d['atmos_vars'][var] for d in list_of_ts], dim = 0)
                for var in list_of_ts[0]['atmos_vars']
            }
        }
        return stacked_dict

    def _load_era5(self, date_hour: pd.Timestamp) -> dict:
        upper_path, sfc_path = self._dt_to_path(date_hour)
        with xr.open_dataset(upper_path) as upper_nc, \
            xr.open_dataset(sfc_path) as sfc_nc:
            upper_nc.load()
            sfc_nc.load()
            return self._nc_to_dict(upper_nc, sfc_nc)

    def _load_aurora(self, aurora_start_dt: pd.Timestamp, step: int) -> dict:
        aurora_path = self._dt_to_path_Aurora(aurora_start_dt, step)
        with xr.open_dataset(aurora_path) as aurora_nc:
            aurora_nc.load()
            return self._nc_to_dict_AuroraTW(aurora_nc)

    def __getitem__(self, index: int) -> tuple:
        date_hour_inputs = [
            self.start_date_hour + pd.Timedelta(hours = index + i * self.lead_time) \
            for i in range(self.input_time_window)
        ]
        date_hour_outputs = [
            date_hour_inputs[-1] + pd.Timedelta(hours = self.lead_time * (i + 1)) \
            for i in range(self.rollout_step)
        ]

        # Always load the first (input_time_window - use_Aurora_input_len) from ERA5
        in_t_list = []
        era5_count = self.input_time_window - self.use_Aurora_input_len
        for in_t in date_hour_inputs[:era5_count]:
            in_t_list.append(self._load_era5(in_t))

        # For the last use_Aurora_input_len slots, probabilistically choose ERA5 or Aurora
        prob = self._get_aurora_prob()
        Aurora_start_dt = date_hour_inputs[-1] - pd.Timedelta(hours = self.use_Aurora_input_len * self.lead_time)
        for i, in_t in enumerate(date_hour_inputs[era5_count:], start=1):
            if random.random() < prob:
                in_t_list.append(self._load_aurora(Aurora_start_dt, step=i))
            else:
                in_t_list.append(self._load_era5(in_t))
        input_data = self.concat_ts(in_t_list)

        out_t_list = []
        for out_t in date_hour_outputs:
            out_t_list.append(self._load_era5(out_t))
        output_data = self.concat_ts(out_t_list)

        result = [input_data, output_data]
        if self.get_datetime:
            result.append(date_hour_inputs[-1].strftime("%Y-%m-%d %H:%M:%S"))
        return tuple(result)

    def check_last_timestep(self):
        last_index = len(self) - 1

        last_input_times = [
            self.start_date_hour + pd.Timedelta(hours=last_index + i * self.lead_time)
            for i in range(self.input_time_window)
        ]

        last_output_times = [
            last_input_times[-1] + pd.Timedelta(hours=self.lead_time * (i + 1))
            for i in range(self.rollout_step)
        ]

        info = {
            "last_index": last_index,
            "last_input_times": last_input_times,
            "last_output_times": last_output_times,
            "dataset_end_date_hour": self.end_date_hour,
            "within_bounds": last_output_times[-1] <= self.end_date_hour,
        }

        return info
