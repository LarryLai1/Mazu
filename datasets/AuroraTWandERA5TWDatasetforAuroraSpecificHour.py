from pathlib import Path
import pandas as pd
import torch
import xarray as xr

class AuroraTWandERA5TWDatasetforAuroraSpecificHour(torch.utils.data.Dataset):
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
        use_Aurora_input_len: list[int] = 1,
        Aurora_input_dir: str = None,
        specific_hour: int = 1,
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
        self.specific_hour = specific_hour

        assert self.use_Aurora_input_len >= 1, \
            "use_Aurora_input_len should be at least 1."
        assert self.Aurora_input_dir, \
            "Aurora_input_dir should be provided when use_Aurora_input_len >= 1."
        assert self.use_Aurora_input_len <= self.input_time_window, \
            "use_Aurora_input_len should be less than or equal to input_time_window."

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

    # def _dt_to_path_Aurora(self, date_hour: pd.Timestamp) -> str:
    #     dir_path = Path(self.Aurora_input_dir)
    #     _ = [
    #         str(dir_path / f"{date_hour.strftime('%Y%m%d_%H%M%S')}+{i}hr.nc") \
    #             for i in range(1, self.use_Aurora_input_len + 1)
    #     ]
    #     return _
    
    def _dt_to_path_Aurora(self, date_hour: pd.Timestamp) -> str:
        dir_path = Path(self.Aurora_input_dir)
        _ = [
            # str(dir_path / f"{date_hour.strftime('%Y%m%d_%H%M%S')}+{i * self.lead_time}hr.nc") \
            str(dir_path / f"{(date_hour + pd.Timedelta(hours = i * self.lead_time) - pd.Timedelta(hours = self.specific_hour)).strftime('%Y%m%d_%H%M%S')}+{self.specific_hour}hr.nc") \
                for i in range(1, self.use_Aurora_input_len + 1)
        ]
        # print("What we get in Aurora input path:", _)
        return _

    # def __len__(self) -> int:
    #     duration = self.end_date_hour - self.start_date_hour \
    #         - pd.Timedelta(hours = self.lead_time + self.rollout_step + self.input_time_window) \
    #         + pd.Timedelta(hours = 3)
    #     return round(duration.total_seconds()) // (60 * 60)
    
    def __len__(self) -> int:
        duration = self.end_date_hour - self.start_date_hour
        duration_interval = round(duration.total_seconds()) // (60 * 60)
        total_interval = duration_interval + 1 - self.input_time_window + 1 - self.lead_time * self.rollout_step
        return total_interval

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

    def __getitem__(self, index: int) -> tuple:
        date_hour_inputs = [
            self.start_date_hour + pd.Timedelta(hours = index + i) \
            for i in range(self.input_time_window)
        ]
        date_hour_outputs = [
            # date_hour_inputs[-1] + pd.Timedelta(hours = self.lead_time + i) \
            date_hour_inputs[-1] + pd.Timedelta(hours = self.lead_time * (i + 1)) \
            for i in range(self.rollout_step)
        ]

        in_t_list = []
        for in_t in date_hour_inputs[: -self.use_Aurora_input_len]:
            upper_path_in, sfc_path_in = self._dt_to_path(in_t)
            with xr.open_dataset(upper_path_in) as upper_nc_in, \
                xr.open_dataset(sfc_path_in) as sfc_nc_in:
                upper_nc_in.load()
                sfc_nc_in.load()
                in_t_list.append(self._nc_to_dict(upper_nc_in, sfc_nc_in))

        Aurora_in_t_list = []
        Aurora_start_dt = date_hour_inputs[-1] - pd.Timedelta(hours = self.use_Aurora_input_len)
        # print(f"{date_hour_inputs=}")
        Aurora_dt_list = self._dt_to_path_Aurora( Aurora_start_dt )
        for Aurora_dt_path in Aurora_dt_list:
            with xr.open_dataset(Aurora_dt_path) as Aurora_nc:
                Aurora_nc.load()
                Aurora_in_t_list.append( self._nc_to_dict_AuroraTW( Aurora_nc ) )
        input_data = self.concat_ts( in_t_list + Aurora_in_t_list )

        out_t_list = []
        for out_t in date_hour_outputs:
            upper_path_out, sfc_path_out = self._dt_to_path(out_t)
            with xr.open_dataset(upper_path_out) as upper_nc_out, \
                xr.open_dataset(sfc_path_out) as sfc_nc_out:
                upper_nc_out.load()
                sfc_nc_out.load()
                out_t_list.append(self._nc_to_dict(upper_nc_out, sfc_nc_out))
        output_data = self.concat_ts(out_t_list)

        result = [input_data, output_data]
        if self.get_datetime:
            result.append(date_hour_inputs[-1].strftime("%Y-%m-%d %H:%M:%S"))
        return tuple(result)
