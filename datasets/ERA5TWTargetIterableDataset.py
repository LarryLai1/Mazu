from pathlib import Path
import pandas as pd
import torch
import xarray as xr


class ERA5TWTargetIterableDataset(torch.utils.data.IterableDataset):
    """
    An IterableDataset that yields one rollout step's targets at a time,
    for a given batch of base datetimes.

    Designed to be used with a DataLoader (num_workers >= 1, prefetch_factor >= 1)
    so that target I/O overlaps with GPU compute.

    Each __next__ call loads N files (one per sample in the batch),
    stacks them, and returns a single-step target dict with shape [B, 1, ...].
    """

    def __init__(
        self,
        data_root_dir: str,
        base_datetime_strs: list[str] | tuple[str, ...],
        rollout_steps: int,
        lead_time: int,
        upper_variables: list[str],
        surface_variables: list[str],
        levels: list[int],
        latitude: tuple[int, int],
        longitude: tuple[int, int],
    ) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.base_datetime_strs = list(base_datetime_strs)
        self.rollout_steps = rollout_steps
        self.lead_time = lead_time
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.levels = levels
        self.latitude = latitude
        self.longitude = longitude

    def map_var_name_for_Aurora(self, var_name: str) -> str:
        var_name_mapping = {
            "t2m": "2t",
            "u10": "10u",
            "v10": "10v",
            "msl": "msl",
        }
        return var_name_mapping.get(var_name, var_name)

    def _dt_to_path(self, date_hour: pd.Timestamp) -> tuple[str, str]:
        dir_path = Path(self.data_root_dir) / date_hour.strftime(r"%Y/%Y%m/%Y%m%d")
        name = date_hour.strftime(r"%Y%m%d%H")
        return str(dir_path / f"{name}_upper.nc"), str(dir_path / f"{name}_sfc.nc")

    def _nc_to_dict(self, upper_nc, sfc_nc) -> dict:
        _d = {
            "surf_vars": {
                self.map_var_name_for_Aurora(v): torch.tensor(
                    sfc_nc[v].sel(
                        latitude=slice(*self.latitude),
                        longitude=slice(*self.longitude),
                    ).values,
                ).squeeze()
                for v in self.surface_variables
            },
            "atmos_vars": {
                v: torch.tensor(
                    upper_nc[v].sel(
                        pressure_level=self.levels,
                        latitude=slice(*self.latitude),
                        longitude=slice(*self.longitude),
                    ).values
                ).squeeze()
                for v in self.upper_variables
            },
        }
        return _d

    def __iter__(self):
        for step_index in range(1, self.rollout_steps + 1):
            batch_targets = []
            for dt_str in self.base_datetime_strs:
                base_time = pd.Timestamp(dt_str)
                target_time = base_time + pd.Timedelta(
                    hours=self.lead_time * step_index
                )
                upper_path, sfc_path = self._dt_to_path(target_time)
                with xr.open_dataset(upper_path) as upper_nc, \
                     xr.open_dataset(sfc_path) as sfc_nc:
                    upper_nc.load()
                    sfc_nc.load()
                    batch_targets.append(self._nc_to_dict(upper_nc, sfc_nc))

            # Stack along batch dim and add time dim (size 1): [B, 1, ...]
            target_data = {
                "surf_vars": {
                    var: torch.stack(
                        [b["surf_vars"][var] for b in batch_targets], dim=0
                    ).unsqueeze(1)
                    for var in batch_targets[0]["surf_vars"]
                },
                "atmos_vars": {
                    var: torch.stack(
                        [b["atmos_vars"][var] for b in batch_targets], dim=0
                    ).unsqueeze(1)
                    for var in batch_targets[0]["atmos_vars"]
                },
            }

            yield target_data
