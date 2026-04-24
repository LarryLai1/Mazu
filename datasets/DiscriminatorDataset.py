from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import torch

class ERA5TWDataset(torch.utils.data.Dataset):
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
        upper_path, sfc_path = self._dt_to_path(self.start_date_hour)
        upper_nc = xr.open_dataset(upper_path).load()
        latitude, longitude = \
            upper_nc.latitude.sel(latitude = slice(*self.latitude)).values, \
            upper_nc.longitude.sel(longitude = slice(*self.longitude)).values
        upper_nc.close()
        return torch.Tensor(latitude), torch.Tensor(longitude)
    
    def get_levels(self):
        upper_path, _ = self._dt_to_path(self.start_date_hour)
        upper_nc = xr.open_dataset(upper_path).load()
        levels = upper_nc.pressure_level.values
        upper_nc.close()
        return tuple(levels)
    
    def get_static_vars_ds(self):
        _ds = xr.open_dataset(self.data_root_dir + "/static/static_vars.nc").load()
        _d = {
            "static_vars": {
                v: torch.Tensor(
                    _ds[v].sel(
                        latitude = slice(*self.latitude), longitude = slice(*self.longitude)
                    ).values
                ).squeeze() for v in self.static_variables
            }
        }
        return _d

    def _dt_to_path(self, date_hour: pd.Timestamp) -> str:
        dir_path = Path(self.data_root_dir) / date_hour.strftime(r"%Y/%Y%m/%Y%m%d")
        name = date_hour.strftime(r"%Y%m%d%H")

        return str(dir_path / f"{name}_upper.nc"), str(dir_path / f"{name}_sfc.nc")

    def __len__(self) -> int:
        duration = self.end_date_hour - self.start_date_hour + pd.Timedelta(hours = 1)
        return round(duration.total_seconds()) // (60 * 60)

    def _nc_to_dict(self, upper_nc, sfc_nc) -> dict:
        _d = {
            "surf_vars": {
                self.map_var_name_for_Aurora(v): torch.Tensor(
                    sfc_nc[v].sel(
                        latitude = slice(*self.latitude),
                        longitude = slice(*self.longitude)
                    ).values,
                ) for v in self.surface_variables
            },
            "atmos_vars": {
                v: torch.Tensor(
                    upper_nc[v].sel(
                        pressure_level = self.levels,
                        latitude = slice(*self.latitude),
                        longitude = slice(*self.longitude),
                    ).values
                ) for v in self.upper_variables
            },
        }
        return _d
        
    def __getitem__(self, index: int) -> tuple:
        cur = self.start_date_hour + pd.Timedelta(hours = index)
        upper_path, sfc_path = self._dt_to_path(cur)
        upper_nc = xr.open_dataset(upper_path).load()
        sfc_nc = xr.open_dataset(sfc_path).load()
        _d = self._nc_to_dict(upper_nc, sfc_nc)
        upper_nc.close()
        sfc_nc.close()

        if self.get_datetime:
            return _d, cur.strftime("%Y-%m-%d %H:%M:%S")
        
        return _d

class AuroraPredictionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root_dir: str,
        start_date_hour: pd.Timestamp,
        end_date_hour: pd.Timestamp,
        upper_variables: list[str],
        surface_variables: list[str],
        static_variables: list[str],
        latitude: tuple[int, int],
        longitude: tuple[int, int],
        levels: list[int],
        forecast_hour: int = 1,
        get_datetime: bool = True,
    ) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.start_date_hour = pd.Timestamp(start_date_hour)
        self.end_date_hour = pd.Timestamp(end_date_hour)
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.static_variables = static_variables
        self.latitude = latitude
        self.longitude = longitude
        self.levels = levels
        self.forecast_hour = forecast_hour
        self.get_datetime = get_datetime

    def __len__(self):
        total_hours = int((self.end_date_hour - self.start_date_hour).total_seconds() // 3600)
        return (total_hours + 1)

    def _nc_to_dict(self, pred_nc) -> dict:

        _d = {
            "surf_vars": {
                k.removeprefix("surf_"): torch.Tensor(
                    v.sel(latitude = slice(*self.latitude), longitude = slice(*self.longitude)).values
                ) for k, v in pred_nc.items() if k.startswith("surf_")
                # k.removeprefix("surf_"): torch.Tensor(v.values) for k, v in pred_nc.items() if k.startswith("surf_")
            },
            "atmos_vars": {
                k.removeprefix("atmos_"): torch.Tensor(
                    v.sel(latitude = slice(*self.latitude), longitude = slice(*self.longitude), level = self.levels).values
                ) for k, v in pred_nc.items() if k.startswith("atmos_")
                # k.removeprefix("atmos_"): torch.Tensor(v.values) for k, v in pred_nc.items() if k.startswith("atmos_")
            },
        }
        return _d

    def get_latitude_longitude(self):
        ds_path = str(Path(self.data_root_dir) / \
            f"{ (self.start_date_hour).strftime('%Y%m%d_%H0000')}+{self.forecast_hour}hr.nc",)
        ds = xr.open_dataset(
            ds_path,
        )
        latitude, longitude = \
            torch.Tensor(ds["latitude"].values), torch.Tensor(ds["longitude"].values)
        return latitude, longitude

    def get_levels(self):
        ds_path = str(Path(self.data_root_dir) / \
            f"{ (self.start_date_hour).strftime('%Y%m%d_%H0000')}+{self.forecast_hour}hr.nc",)
        ds = xr.open_dataset(
            ds_path,
        )
        level = torch.Tensor( ds["level"].values )
        return level

    def get_static_vars_ds(self):
        ds_path = str(Path(self.data_root_dir) / \
            f"{ (self.start_date_hour).strftime('%Y%m%d_%H0000')}+{self.forecast_hour}hr.nc",)
        ds_nc = xr.open_dataset(
            ds_path,
        )
        _d = {
            "static_vars": {
                k.removeprefix("static_"): torch.Tensor(
                    v.sel(latitude = slice(*self.latitude), longitude = slice(*self.longitude)).values
                ) for k, v in ds_nc.items() if k.startswith("static_")
            }
        }
        ds_nc.close()
        return _d

    def __getitem__(self, index: int):
        
        ds_dir = str(Path(self.data_root_dir) / \
            f"{ (self.start_date_hour + pd.Timedelta(hours = index * 1)).strftime('%Y%m%d_%H0000')}+{self.forecast_hour}hr.nc",)
        ds = xr.open_dataset(
            ds_dir,
        )

        _d = self._nc_to_dict( ds )
        dt = ds["time"].values[0]
        ds.close()

        if self.get_datetime:
            return _d, pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")

        return _d

class DiscriminatorDataset(torch.utils.data.Dataset):
    """
    New semantics:
        Aurora predictions  -> label = 0
        ERA5 ground truth   -> label = 1

    Constructor expects:
        DiscriminatorDataset(
            AuroraTWDataset,      # predictions
            ERA5TWDataset,        # ground truth
        )
    """
    def __init__(
        self,
        AuroraTWDataset: AuroraPredictionDataset,
        ERA5TWDataset:   ERA5TWDataset,
    ) -> None:
        super().__init__()

        # Normalize to list form
        if not isinstance(AuroraTWDataset, list):
            AuroraTWDataset = [AuroraTWDataset]
        if not isinstance(ERA5TWDataset, list):
            ERA5TWDataset = [ERA5TWDataset]

        # Store in new intended order:
        #   Aurora first (fake), ERA5 second (real)
        self.Aurora_datasets = AuroraTWDataset
        self.ERA5_datasets   = ERA5TWDataset

        # Lengths
        self.Aurora_lengths = [len(ds) for ds in self.Aurora_datasets]
        self.ERA5_lengths   = [len(ds) for ds in self.ERA5_datasets]

        # Cumulative ranges
        self.Aurora_cum = np.cumsum([0] + self.Aurora_lengths)
        self.ERA5_cum   = np.cumsum([0] + self.ERA5_lengths)

        # Totals
        self.total_Aurora = sum(self.Aurora_lengths)
        self.total_ERA5   = sum(self.ERA5_lengths)
        self.total_len    = self.total_Aurora + self.total_ERA5

    def __len__(self):
        return self.total_len

    def get_latitude_longitude(self):
        # Use ERA5 for reference (ground truth)
        return self.ERA5_datasets[0].get_latitude_longitude()

    def get_levels(self):
        # Use ERA5 as canonical level ordering
        return self.ERA5_datasets[0].get_levels()

    def get_static_vars_ds(self):
        # ERA5 defines the static geography
        return self.ERA5_datasets[0].get_static_vars_ds()

    def __getitem__(self, index):
        if index < 0 or index >= self.total_len:
            raise IndexError(f"Index {index} out of bounds for length {self.total_len}")

        # ------------------------------------------------------------
        # NEW label semantics:
        #
        #   First part   (0 .. total_Aurora-1)   -> Aurora predictions (label = 0)
        #   Second part  (total_Aurora .. end)   -> ERA5 ground truth (label = 1)
        # ------------------------------------------------------------

        if index < self.total_Aurora:
            # Aurora sample → label = 0
            ds_index = np.searchsorted(self.Aurora_cum, index, side="right") - 1
            sample_index = index - self.Aurora_cum[ds_index]
            _x = self.Aurora_datasets[ds_index][sample_index]
            label = 0
        else:
            # ERA5 sample → label = 1
            adjusted = index - self.total_Aurora
            ds_index = np.searchsorted(self.ERA5_cum, adjusted, side="right") - 1
            sample_index = adjusted - self.ERA5_cum[ds_index]
            _x = self.ERA5_datasets[ds_index][sample_index]
            label = 1

        return _x, label

# import numpy as np
# import torch
# from torch.utils.data import Dataset


# import numpy as np
# import torch
# from torch.utils.data import Dataset

class SameDomainDiscriminatorDataset(torch.utils.data.Dataset):
    """
    Generic discriminator dataset for control experiments:

        - Group 0: samples from dataset_group0  -> label = 0
        - Group 1: samples from dataset_group1  -> label = 1

    Both groups are intended to come from the *same* underlying distribution
    (e.g. ERA5 vs ERA5, or Aurora vs Aurora), so a well-behaved discriminator
    should NOT be able to achieve high accuracy.

    The interface is intentionally similar to your original DiscriminatorDataset:
        __len__()
        __getitem__(index) -> ((data_dict, datetime_str), label)

    Plus utility methods:
        get_latitude_longitude()
        get_levels()
        get_static_vars_ds()
    """

    def __init__(
        self,
        dataset_group0,
        dataset_group1,
    ) -> None:
        """
        Args
        ----
        dataset_group0: Dataset or list[Dataset]
            First group of datasets (label = 0).
        dataset_group1: Dataset or list[Dataset]
            Second group of datasets (label = 1).
        """
        super().__init__()

        # Normalize to lists (same as your DiscriminatorDataset)
        if not isinstance(dataset_group0, list):
            dataset_group0 = [dataset_group0]
        if not isinstance(dataset_group1, list):
            dataset_group1 = [dataset_group1]

        self.group0 = dataset_group0
        self.group1 = dataset_group1

        # Lengths
        self.group0_lengths = [len(ds) for ds in self.group0]
        self.group1_lengths = [len(ds) for ds in self.group1]

        # Cumulative ranges
        self.group0_cum = np.cumsum([0] + self.group0_lengths)
        self.group1_cum = np.cumsum([0] + self.group1_lengths)

        # Totals
        self.total_group0 = sum(self.group0_lengths)
        self.total_group1 = sum(self.group1_lengths)
        self.total_len    = self.total_group0 + self.total_group1

    def __len__(self):
        return self.total_len

    # --- Utility methods (mirroring your DiscriminatorDataset) ---

    def get_latitude_longitude(self):
        # Use first dataset in group0 as reference
        ds0 = self.group0[0]
        if hasattr(ds0, "get_latitude_longitude"):
            return ds0.get_latitude_longitude()
        raise AttributeError("Underlying dataset has no get_latitude_longitude().")

    def get_levels(self):
        ds0 = self.group0[0]
        if hasattr(ds0, "get_levels"):
            return ds0.get_levels()
        raise AttributeError("Underlying dataset has no get_levels().")

    def get_static_vars_ds(self):
        ds0 = self.group0[0]
        if hasattr(ds0, "get_static_vars_ds"):
            return ds0.get_static_vars_ds()
        raise AttributeError("Underlying dataset has no get_static_vars_ds().")

    # --- Core indexing logic ---

    def __getitem__(self, index):
        """
        Returns:
            ((data_dict, datetime_str), label)

        Where:
            - data_dict: {"surf_vars": {var: tensor}, "atmos_vars": {var: tensor}}
            - datetime_str: "YYYY-MM-DD HH:MM:SS"
            - label: 0 (from group0) or 1 (from group1)
        """
        if index < 0 or index >= self.total_len:
            raise IndexError(f"Index {index} out of bounds for length {self.total_len}")

        if index < self.total_group0:
            # Group 0 sample → label = 0
            ds_index     = np.searchsorted(self.group0_cum, index, side="right") - 1
            sample_index = index - self.group0_cum[ds_index]
            _x = self.group0[ds_index][sample_index]
            label = 0
        else:
            # Group 1 sample → label = 1
            adjusted     = index - self.total_group0
            ds_index     = np.searchsorted(self.group1_cum, adjusted, side="right") - 1
            sample_index = adjusted - self.group1_cum[ds_index]
            _x = self.group1[ds_index][sample_index]
            label = 1

        return _x, label



class OddEvenSameSourceDiscriminatorDataset(torch.utils.data.Dataset):
    """
    Discriminator dataset that uses a SINGLE underlying dataset (or list of datasets)
    and assigns labels by *index parity*:

        - label = 0  for even sample indices (within each sub-dataset)
        - label = 1  for odd  sample indices

    This is ideal for control experiments like:
        - ERA5 (even indices) vs ERA5 (odd indices)
        - Aurora (even indices) vs Aurora (odd indices)

    The interface mirrors your existing DiscriminatorDataset:

        __len__() -> total samples
        __getitem__(index) -> ((data_dict, datetime_str), label)

    Plus helper methods:
        get_latitude_longitude()
        get_levels()
        get_static_vars_ds()
    """

    def __init__(self, base_datasets):
        """
        Args
        ----
        base_datasets: Dataset or list[Dataset]
            One or more datasets of the same type (e.g. ERA5TWDataset, AuroraPredictionDataset).
            They must return ((data_dict, datetime_str)) from __getitem__.
        """
        super().__init__()

        # Normalize to a list (similar to your DiscriminatorDataset)
        if not isinstance(base_datasets, list):
            base_datasets = [base_datasets]

        self.datasets = base_datasets

        # Lengths and cumulative offsets
        self.lengths = [len(ds) for ds in self.datasets]
        self.cum = np.cumsum([0] + self.lengths)

        self.total_len = sum(self.lengths)

    def __len__(self):
        return self.total_len

    # ----------------- helper methods (same as your DiscriminatorDataset) -----------------

    def get_latitude_longitude(self):
        # Use first underlying dataset as reference
        ds0 = self.datasets[0]
        if hasattr(ds0, "get_latitude_longitude"):
            return ds0.get_latitude_longitude()
        raise AttributeError("Underlying dataset has no get_latitude_longitude().")

    def get_levels(self):
        ds0 = self.datasets[0]
        if hasattr(ds0, "get_levels"):
            return ds0.get_levels()
        raise AttributeError("Underlying dataset has no get_levels().")

    def get_static_vars_ds(self):
        ds0 = self.datasets[0]
        if hasattr(ds0, "get_static_vars_ds"):
            return ds0.get_static_vars_ds()
        raise AttributeError("Underlying dataset has no get_static_vars_ds().")

    # ----------------- core indexing logic -----------------

    def __getitem__(self, index):
        """
        Returns:
            ((data_dict, datetime_str), label)

        label is determined by the *local* sample index within the sub-dataset:
            sample_index % 2 == 0 -> label 0
            sample_index % 2 == 1 -> label 1
        """
        if index < 0 or index >= self.total_len:
            raise IndexError(f"Index {index} out of bounds for length {self.total_len}")

        # Find which sub-dataset and local index
        ds_index = np.searchsorted(self.cum, index, side="right") - 1
        sample_index = index - self.cum[ds_index]

        # Get the actual sample from the underlying dataset
        _x = self.datasets[ds_index][sample_index]   # ((data_dict, dt_str))

        # Label by parity of *local* index
        label = int(sample_index % 2)  # 0 for even, 1 for odd

        return _x, label

if __name__ == "__main__":
    # from ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
    # from AuroraTWandERA5TWDatasetforAurora import AuroraTWandERA5TWDatasetforAurora

    data_root_dir = '/work/yunye0121/era5_tw'
    start_date_hour = '2023-01-01 01:00:00'
    end_date_hour = '2023-01-01 04:00:00'
    upper_variables = ['u', 'v', 't', 'q', 'z']
    surface_variables = ['t2m', "u10", "v10", "msl"]
    static_variables = ['z', 'lsm', 'slt']
    latitude = (39.75, 5)
    longitude = (100, 144.75)
    levels = [1000, 925, 850, 700, 500, 300, 150, 50]
    lead_time = 0
    input_time_window = 1
    rollout_step = 0

    Aurora_input_dir = "/work/yunye0121/AuroraTW_ckpts/392-train_loss=0.022402-val_loss=0.025010/ar_96hrs_202301_result"
    forecast_hour = 1
    # use_Aurora_input_len = 1

    start = pd.Timestamp(start_date_hour)
    end = pd.Timestamp(end_date_hour)
    era5_start = (start + pd.Timedelta(hours=forecast_hour)).strftime('%Y-%m-%d %H:%M:%S')
    era5_end = (end + pd.Timedelta(hours=forecast_hour)).strftime('%Y-%m-%d %H:%M:%S')

    era5_ds= ERA5TWDataset(
        data_root_dir=data_root_dir,
        start_date_hour=era5_start,      # shifted!
        end_date_hour=era5_end,          # shifted!
        upper_variables=upper_variables,
        surface_variables=surface_variables,
        static_variables=static_variables,
        latitude=latitude,
        longitude=longitude,
        levels=levels,
        # get_datetime=False, 
        get_datetime=True,
        # flatten=False,
    )

    aurora_ds = AuroraPredictionDataset(
        data_root_dir=Aurora_input_dir,
        start_date_hour=start,
        end_date_hour=end,
        upper_variables=upper_variables,
        surface_variables=surface_variables,
        static_variables=static_variables,
        latitude=latitude,
        longitude=longitude,
        levels=levels,
        forecast_hour=forecast_hour,
        get_datetime = True,
    )

    discriminator_ds = DiscriminatorDataset(
        AuroraTWDataset = aurora_ds,
        ERA5TWDataset = era5_ds,
    )

    print(f"{len(discriminator_ds)=}")

    print("DiscriminatorDataset length:", len(discriminator_ds))
    print("\nIterating over DiscriminatorDataset samples:")
    for i in range(3):
        (x, input_dates), label = discriminator_ds[i]
        print(f"\nDiscriminator sample[{i}]:")
        print("  Sample:")
        print(f"    surf_vars['2t'] shape: {x['surf_vars']['2t'].shape}")
        print(f"    atmos_vars['u'] shape: {x['atmos_vars']['u'].shape}")
        # print(f"    static_vars['z'] shape: {x['static_vars']['z'].shape}")
        print(f"{input_dates=}")
        print(f"{label=}")
    
    for i in range(len(discriminator_ds)-3, len(discriminator_ds)):
        (x, input_dates), label = discriminator_ds[i]
        print(f"\nDiscriminator sample[{i}]:")
        print("  Sample:")
        print(f"    surf_vars['2t'] shape: {x['surf_vars']['2t'].shape}")
        print(f"    atmos_vars['u'] shape: {x['atmos_vars']['u'].shape}")
        # print(f"    static_vars['z'] shape: {x['static_vars']['z'].shape}")
        print(f"{input_dates=}")
        print(f"{label=}")

    import math
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import numpy as np
    import torch

    def analyze_discriminator_dataset(discriminator_ds, max_samples_per_label=2, do_plots=True):
        """
        Analyze DiscriminatorDataset:

        - Uses up to `max_samples_per_label` samples for each label (0=ERA5, 1=Aurora).
        - Computes mean/std per variable for ERA5 and Aurora separately:
            * Surface vars: aggregated over all dims.
            * Atmospheric vars: aggregated per level (one stat per (var, level)).
        - If do_plots:
            For each used sample (per label):
            * For EVERY surface variable, create a figure:
                - 1 imshow of that surface field.
            * For EVERY atmospheric variable AND EVERY level, create a figure:
                - 1 imshow of that variable at that level.
            => All plots are **per-sample**, not means.
        """

        stats = {
            0: {"surf": defaultdict(lambda: {"sum": 0.0, "sumsq": 0.0, "count": 0}),
                "atmos": defaultdict(lambda: {"sum": 0.0, "sumsq": 0.0, "count": 0})},
            1: {"surf": defaultdict(lambda: {"sum": 0.0, "sumsq": 0.0, "count": 0}),
                "atmos": defaultdict(lambda: {"sum": 0.0, "sumsq": 0.0, "count": 0})},
        }

        used_counts = {0: 0, 1: 0}
        saved_samples = {0: [], 1: []}  # store (orig_index, x, dt_str)

        # Get pressure levels from ERA5 side (this is what we want to align to)
        try:
            levels_from_ds = discriminator_ds.get_levels()  # tuple / list
            # levels_from_ds = np.array(levels_from_ds)
            levels_from_ds = torch.Tensor(levels_from_ds)
            n_levels_ref = levels_from_ds.shape[0]
            print(f"Detected levels from dataset: {levels_from_ds}")
        except Exception as e:
            print(f"Warning: could not get levels from dataset, will use level indices only. Error: {e}")
            levels_from_ds = None
            n_levels_ref = None

        total_n = len(discriminator_ds)
        print(
            f"Targeting first {max_samples_per_label} samples per label (0 and 1). "
            f"Dataset length = {total_n}"
        )

        # ---------- PASS 1: collect stats + remember first N samples per label ----------
        for i in range(total_n):
            # stop when we have enough from both domains
            if used_counts[0] >= max_samples_per_label and used_counts[1] >= max_samples_per_label:
                break

            (x, dt_str), label = discriminator_ds[i]

            if label not in (0, 1):
                raise ValueError(f"Unexpected label {label} at index {i}")

            # If we already have enough for this label, skip it
            if used_counts[label] >= max_samples_per_label:
                continue

            # --- surface vars: stats ---
            for vname, tensor in x["surf_vars"].items():
                t = tensor.float().reshape(-1)
                s = t.sum().item()
                sq = (t * t).sum().item()
                c = t.numel()

                st = stats[label]["surf"][vname]
                st["sum"] += s
                st["sumsq"] += sq
                st["count"] += c

            # --- atmospheric vars: stats per level ---
            for vname, tensor in x["atmos_vars"].items():
                t = tensor.float()

                if t.ndim == 0:
                    t = t.reshape(1, 1)

                # detect which axis is "level"
                if n_levels_ref is not None and n_levels_ref in t.shape:
                    level_axis = t.shape.index(n_levels_ref)
                else:
                    if t.ndim >= 2:
                        level_axis = 1  # assume (time, level, lat, lon) or (batch, level, ...)
                    else:
                        level_axis = 0  # degenerate case

                # move level axis to 0: (L, ...)
                t_lev_first = t.movedim(level_axis, 0)
                L = t_lev_first.shape[0]

                for lev_idx in range(L):
                    if levels_from_ds is not None and L == n_levels_ref:
                        lev_val = float(levels_from_ds[lev_idx])
                        key = f"{vname}_lev{lev_val:g}"
                    else:
                        key = f"{vname}_levIdx{lev_idx}"

                    lev_slice = t_lev_first[lev_idx].reshape(-1)
                    s = lev_slice.sum().item()
                    sq = (lev_slice * lev_slice).sum().item()
                    c = lev_slice.numel()

                    st = stats[label]["atmos"][key]
                    st["sum"] += s
                    st["sumsq"] += sq
                    st["count"] += c

            used_counts[label] += 1
            saved_samples[label].append((i, x, dt_str))

            print(
                f"  Scanned {i + 1}/{total_n} samples | "
                f"used ERA5={used_counts[0]}, Aurora={used_counts[1]}",
                end="\r",
            )

        print(
            f"\n\nFinished: used ERA5={used_counts[0]} samples, "
            f"Aurora={used_counts[1]} samples."
        )

        # ---------- finalize stats ----------
        def finalize_stats(st_dict):
            out = {}
            for vname, st in st_dict.items():
                if st["count"] == 0:
                    continue
                mean = st["sum"] / st["count"]
                var = max(st["sumsq"] / st["count"] - mean * mean, 0.0)
                std = math.sqrt(var)
                out[vname] = {"mean": mean, "std": std, "count": st["count"]}
            return out

        final_stats = {
            label: {
                "surf": finalize_stats(stats[label]["surf"]),
                "atmos": finalize_stats(stats[label]["atmos"]),
            }
            for label in (0, 1)
        }

        label_name = {0: "Aurora (label=0)", 1: "ERA5 (label=1)"}
        for label in (0, 1):
            print(f"\n##### {label_name[label]} #####")

            print("  Surface variables:")
            for vname, st in final_stats[label]["surf"].items():
                print(f"    {vname:14s}  mean={st['mean']:+.4f}, std={st['std']:.4f}, count={st['count']}")

            print("  Atmospheric variables (per level):")
            for vname, st in final_stats[label]["atmos"].items():
                print(f"    {vname:14s}  mean={st['mean']:+.4f}, std={st['std']:.4f}, count={st['count']}")

        # ---------- PASS 2: per-sample, all-variable plots ----------
        if do_plots:
            import os
            def ensure_dir(path):
                if not os.path.exists(path):
                    os.makedirs(path)
        
            # Create debug folder
            out_dir = "debug_plots"
            ensure_dir(out_dir)

            # helper: convert tensor to 2D numpy for imshow
            def to_2d_numpy(tensor):
                field = tensor.float()
                while field.ndim > 2 and field.shape[0] == 1:
                    field = field[0]
                while field.ndim > 2:
                    field = field[0]
                return field.cpu().numpy()

            # helper: extract atmos slice at specific level
            def atmos_level_to_2d_numpy(tensor, lev_idx, levels_from_ds_local):
                t = tensor.float()

                if t.ndim == 0:
                    t = t.reshape(1, 1)

                # detect level axis
                if levels_from_ds_local is not None and levels_from_ds_local.shape[0] in t.shape:
                    level_axis = t.shape.index(levels_from_ds_local.shape[0])
                else:
                    level_axis = 1 if t.ndim >= 2 else 0

                t_lev_first = t.movedim(level_axis, 0)
                field = t_lev_first[lev_idx]
                return to_2d_numpy(field)

            # ------------- loop samples and save plots -------------
            for label in (0, 1):
                samples = saved_samples[label]
                name_prefix = label_name[label].replace(" ", "_")  # ERA5 / Aurora safe name

                for (orig_idx, x, dt_str) in samples:
                    surf_dict = x["surf_vars"]
                    atmos_dict = x["atmos_vars"]

                    # Create subfolder per label
                    label_dir = os.path.join(out_dir, f"{name_prefix}")
                    ensure_dir(label_dir)

                    # Safe timestamp string for filenames
                    dt_safe = dt_str.replace(" ", "_").replace(":", "-")

                    # ------- Surface variable plots -------
                    for surf_name, surf_tensor in surf_dict.items():
                        surf_2d = to_2d_numpy(surf_tensor)

                        plt.figure(figsize=(5, 4))
                        plt.imshow(surf_2d, origin="lower")
                        plt.colorbar(shrink=0.8)
                        plt.title(
                            f"{label_name[label]} | idx={orig_idx}\n"
                            f"{surf_name} @ {dt_str}"
                        )
                        fname = f"{name_prefix}_idx{orig_idx}_{surf_name}_{dt_safe}.png"
                        plt.savefig(os.path.join(label_dir, fname), dpi=150)
                        plt.close()

                    # ------- Atmospheric variables & all levels -------
                    for atmos_name, atmos_tensor in atmos_dict.items():

                        # detect level axis again
                        t = atmos_tensor.float()
                        if n_levels_ref is not None and n_levels_ref in t.shape:
                            level_axis = t.shape.index(n_levels_ref)
                        else:
                            level_axis = 1 if t.ndim >= 2 else 0

                        t_lev_first = t.movedim(level_axis, 0)
                        L = t_lev_first.shape[0]

                        for lev_idx in range(L):
                            field_2d = to_2d_numpy(t_lev_first[lev_idx])

                            if levels_from_ds is not None and L == levels_from_ds.shape[0]:
                                lev_label = f"{levels_from_ds[lev_idx]:g}hPa"
                            else:
                                lev_label = f"lev{lev_idx}"

                            plt.figure(figsize=(5, 4))
                            plt.imshow(field_2d, origin="lower")
                            plt.colorbar(shrink=0.8)
                            plt.title(
                                f"{label_name[label]} | idx={orig_idx}\n"
                                f"{atmos_name} @ {lev_label} | {dt_str}"
                            )
                            fname = (
                                f"{name_prefix}_idx{orig_idx}_{atmos_name}_"
                                f"{lev_label}_{dt_safe}.png"
                            )
                            plt.savefig(os.path.join(label_dir, fname), dpi=150)
                            plt.close()

            print(f"\nSaved all figures to: {out_dir}/")

            return final_stats

    def print_tensor_stats(name, a_tensor, b_tensor):
        """
        a_tensor = Aurora, b_tensor = ERA5
        name = variable name, e.g. "2t" or "u_lev850"
        """
        a = a_tensor.flatten().float()
        b = b_tensor.flatten().float()

        print(f"\n===== {name} =====")

        # First 10 values
        print("Aurora first 10:", a[:10].tolist())
        print("ERA5   first 10:", b[:10].tolist())

        # Last 10 values
        print("Aurora last 10:", a[-10:].tolist())
        print("ERA5   last 10:", b[-10:].tolist())

        # 5×5 center patch
        # Works for any shape >= 5×5
        H, W = a_tensor.shape[-2], a_tensor.shape[-1]
        ch = H // 2
        cw = W // 2

        print("\nAurora center patch:\n", 
            a_tensor[..., ch-2:ch+3, cw-2:cw+3])

        print("\nERA5 center patch:\n", 
            b_tensor[..., ch-2:ch+3, cw-2:cw+3])

        # Correlation
        corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        print(f"\nCorrelation: {corr:.6f}")


    def run_structure_check(discriminator_ds, idx_aurora=0, idx_era5=None):
        """
        idx_aurora = index of Aurora sample in discriminator_ds
        idx_era5 = index of ERA5 sample (default: Aurora index + total_Aurora)
        """
        print("=== STRUCTURE CHECK ===")

        # Find total Aurora length
        total_Aurora = discriminator_ds.total_Aurora

        if idx_era5 is None:
            idx_era5 = idx_aurora + total_Aurora

        # Load samples
        (a_x, a_dt), _ = discriminator_ds[idx_aurora]   # Aurora
        (b_x, b_dt), _ = discriminator_ds[idx_era5]      # ERA5

        print(f"Aurora timestamp: {a_dt}")
        print(f"ERA5   timestamp: {b_dt}")

        # ---- Surface variables ----
        print("\n====== SURFACE VARS ======")
        for v in a_x["surf_vars"]:
            print_tensor_stats(
                name=v,
                a_tensor=a_x["surf_vars"][v][0], 
                b_tensor=b_x["surf_vars"][v][0]
            )

        # ---- Atmospheric variables ----
        print("\n====== ATMOSPHERIC VARS ======")
        for v in a_x["atmos_vars"]:
            a_atm = a_x["atmos_vars"][v]  # shape [1, lev, H, W]
            b_atm = b_x["atmos_vars"][v]

            L = a_atm.shape[1]
            for lev in range(L):
                name = f"{v}_lev{lev}"
                print_tensor_stats(
                    name=name,
                    a_tensor=a_atm[0, lev],
                    b_tensor=b_atm[0, lev]
                )


    stats = analyze_discriminator_dataset(discriminator_ds, max_samples_per_label=3, do_plots=True)
    run_structure_check(discriminator_ds, idx_aurora=1)
    run_structure_check(discriminator_ds, idx_aurora=2)












    # # ============================================================
    # # 2) Same-domain ERA5 vs ERA5 using SameDomainDiscriminatorDataset
    # # ============================================================
    # print("\n==================== ERA5 vs ERA5 (SameDomainDiscriminatorDataset) ====================")
    # same_era5_ds = SameDomainDiscriminatorDataset(
    #     dataset_group0=era5_ds,   # label 0
    #     dataset_group1=era5_ds,   # label 1
    # )

    # print(f"len(same_era5_ds)={len(same_era5_ds)}")
    # print("\nFirst few samples from same_era5_ds:")
    # for i in range(min(6, len(same_era5_ds))):
    #     (x, dt_str), label = same_era5_ds[i]
    #     print(f"\nSameDomain ERA5 sample[{i}]:")
    #     print(f"  datetime: {dt_str}")
    #     print(f"  label:    {label}")
    #     print(f"  surf_vars['2t'] shape: {x['surf_vars']['2t'].shape}")
    #     print(f"  atmos_vars['u'] shape: {x['atmos_vars']['u'].shape}")

    # # Reuse analyzer (label names inside are just text)
    # stats_same_era5 = analyze_discriminator_dataset(
    #     same_era5_ds,
    #     max_samples_per_label=3,
    #     do_plots=False,
    # )

    # # ============================================================
    # # 3) Same-domain Aurora vs Aurora using SameDomainDiscriminatorDataset
    # # ============================================================
    # print("\n==================== Aurora vs Aurora (SameDomainDiscriminatorDataset) ====================")
    # same_aurora_ds = SameDomainDiscriminatorDataset(
    #     dataset_group0=aurora_ds,  # label 0
    #     dataset_group1=aurora_ds,  # label 1
    # )

    # print(f"len(same_aurora_ds)={len(same_aurora_ds)}")
    # print("\nFirst few samples from same_aurora_ds:")
    # for i in range(min(6, len(same_aurora_ds))):
    #     (x, dt_str), label = same_aurora_ds[i]
    #     print(f"\nSameDomain Aurora sample[{i}]:")
    #     print(f"  datetime: {dt_str}")
    #     print(f"  label:    {label}")
    #     print(f"  surf_vars['2t'] shape: {x['surf_vars']['2t'].shape}")
    #     print(f"  atmos_vars['u'] shape: {x['atmos_vars']['u'].shape}")

    # stats_same_aurora = analyze_discriminator_dataset(
    #     same_aurora_ds,
    #     max_samples_per_label=3,
    #     do_plots=False,
    # )

    # # ============================================================
    # # 4) Odd/Even ERA5 using OddEvenSameSourceDiscriminatorDataset
    # # ============================================================
    # print("\n==================== ERA5 odd/even (OddEvenSameSourceDiscriminatorDataset) ====================")
    # odd_even_era5_ds = OddEvenSameSourceDiscriminatorDataset(base_datasets=era5_ds)

    # print(f"len(odd_even_era5_ds)={len(odd_even_era5_ds)}")
    # print("\nFirst few samples from odd_even_era5_ds:")
    # for i in range(min(6, len(odd_even_era5_ds))):
    #     (x, dt_str), label = odd_even_era5_ds[i]
    #     print(f"\nOddEven ERA5 sample[{i}]:")
    #     print(f"  datetime: {dt_str}")
    #     print(f"  label (0=even,1=odd): {label}")
    #     print(f"  surf_vars['2t'] shape: {x['surf_vars']['2t'].shape}")
    #     print(f"  atmos_vars['u'] shape: {x['atmos_vars']['u'].shape}")

    # stats_odd_even_era5 = analyze_discriminator_dataset(
    #     odd_even_era5_ds,
    #     max_samples_per_label=3,
    #     do_plots=False,
    # )

    # # ============================================================
    # # 5) Odd/Even Aurora using OddEvenSameSourceDiscriminatorDataset
    # # ============================================================
    # print("\n==================== Aurora odd/even (OddEvenSameSourceDiscriminatorDataset) ====================")
    # odd_even_aurora_ds = OddEvenSameSourceDiscriminatorDataset(base_datasets=aurora_ds)

    # print(f"len(odd_even_aurora_ds)={len(odd_even_aurora_ds)}")
    # print("\nFirst few samples from odd_even_aurora_ds:")
    # for i in range(min(6, len(odd_even_aurora_ds))):
    #     (x, dt_str), label = odd_even_aurora_ds[i]
    #     print(f"\nOddEven Aurora sample[{i}]:")
    #     print(f"  datetime: {dt_str}")
    #     print(f"  label (0=even,1=odd): {label}")
    #     print(f"  surf_vars['2t'] shape: {x['surf_vars']['2t'].shape}")
    #     print(f"  atmos_vars['u'] shape: {x['atmos_vars']['u'].shape}")

    # stats_odd_even_aurora = analyze_discriminator_dataset(
    #     odd_even_aurora_ds,
    #     max_samples_per_label=3,
    #     do_plots=False,
    # )
    # print("\nTest complete. You are ready to train your model using this paired dataset!")



