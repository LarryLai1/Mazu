from pathlib import Path
import pandas as pd
import torch
import xarray as xr


class ERA5TWDatasetforAuroraLazy(torch.utils.data.Dataset):
    """
    Lazy-loading version of ERA5TWDatasetforAurora.
    __getitem__ returns ONLY the input window (no rollout targets).
    Use load_single_target() to lazily load one target at a time during the rollout loop.
    """
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

    def __getitem__(self, index: int) -> tuple:
        """Only returns (input_data, date_string). No targets loaded."""
        date_hour_inputs = [
            self.start_date_hour + pd.Timedelta(hours = index + i * self.lead_time)
            for i in range(self.input_time_window)
        ]

        in_t_list = []
        for in_t in date_hour_inputs:
            upper_path_in, sfc_path_in = self._dt_to_path(in_t)
            with xr.open_dataset(upper_path_in) as upper_nc_in, \
                xr.open_dataset(sfc_path_in) as sfc_nc_in:
                upper_nc_in.load()
                sfc_nc_in.load()
                in_t_list.append(self._nc_to_dict(upper_nc_in, sfc_nc_in))
        input_data = self.concat_ts(in_t_list)

        result = [input_data]
        if self.get_datetime:
            result.append(date_hour_inputs[-1].strftime("%Y-%m-%d %H:%M:%S"))
        return tuple(result)

    def load_single_target(self, base_datetime_strs, rollout_step_index: int) -> dict:
        """
        Lazily load ONE target timestep for a given rollout step, for each sample in the batch.

        Args:
            base_datetime_strs: A single datetime string or a list/tuple of datetime strings
                                (one per sample in the batch), representing the last input timestep.
            rollout_step_index: 1-based rollout step (1, 2, 3, ...).

        Returns:
            dict with 'surf_vars' and 'atmos_vars', each value has shape [B, 1, H, W]
            for surface vars or [B, 1, L, H, W] for atmos vars.
        """
        if isinstance(base_datetime_strs, str):
            base_datetime_strs = [base_datetime_strs]

        batch_targets = []
        for dt_str in base_datetime_strs:
            base_time = pd.Timestamp(dt_str)
            target_time = base_time + pd.Timedelta(hours = self.lead_time * rollout_step_index)

            upper_path, sfc_path = self._dt_to_path(target_time)
            with xr.open_dataset(upper_path) as upper_nc, \
                xr.open_dataset(sfc_path) as sfc_nc:
                upper_nc.load()
                sfc_nc.load()
                batch_targets.append(self._nc_to_dict(upper_nc, sfc_nc))

        # Stack along batch dim and add time dim (size 1): [B, 1, ...]
        target_data = {
            "surf_vars": {
                var: torch.stack([b["surf_vars"][var] for b in batch_targets], dim=0).unsqueeze(1)
                for var in batch_targets[0]["surf_vars"]
            },
            "atmos_vars": {
                var: torch.stack([b["atmos_vars"][var] for b in batch_targets], dim=0).unsqueeze(1)
                for var in batch_targets[0]["atmos_vars"]
            },
        }

        return target_data

    def check_last_timestep(self):
        """
        Check the last valid index and the final datetime accessed
        by __getitem__ and load_single_target (both input and output).
        """
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


if __name__ == "__main__":
    # For quick testing

    data_root_dir = "/home/yunye0121/era5_tw"
    train_start_date_hour = "2013-01-01 00:00:00"
    train_end_date_hour = "2013-01-31 23:00:00"
    val_start_date_hour = "2022-01-01 00:00:00"
    val_end_date_hour = "2022-12-31 23:00:00"
    surface_variables = ["t2m", "u10", "v10", "msl"]
    upper_variables = ["u", "v", "t", "q", "z"]
    static_variables = ["lsm", "slt", "z"]
    levels = [1000, 925, 850, 700, 500, 300, 150, 50]
    latitude = [39.75, 5]
    longitude = [100, 144.75]
    lead_time = 6
    input_time_window = 2
    rollout_step = 16

    ds = ERA5TWDatasetforAuroraLazy(
        data_root_dir = data_root_dir,
        start_date_hour = train_start_date_hour,
        end_date_hour = train_end_date_hour,
        upper_variables = upper_variables,
        surface_variables = surface_variables,
        static_variables = static_variables,
        levels = levels,
        latitude = latitude,
        longitude = longitude,
        lead_time = lead_time,
        input_time_window = input_time_window,
        rollout_step = rollout_step,
    )

    info = ds.check_last_timestep()
    for k, v in info.items():
        print(f"{k}: {v}")

