import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import xarray as xr

from datasets.BoundaryConditionDataset import BoundaryConditionDataset_HRES
from utils.boundary_replacement import replace_input_boundary


class BoundaryConditionDataset_HRESFixedLead(BoundaryConditionDataset_HRES):
    """HRES boundary source for training: daily files with 00/12Z inits, each holding a few
    forecast leads (+0h/+6h/+12h); a single lead is read per call.

    Reuses BoundaryConditionDataset_HRES for path resolution, variable-name mapping, spatial
    slicing and the latitude-descending normalisation, but (unlike the inference path) fails
    loudly on a missing file / init time / lead time instead of silently falling back to
    another time.
    """

    def __init__(
        self,
        boundary_root_dir: str,
        upper_variables: list[str],
        surface_variables: list[str],
        levels: list[int],
        latitude: tuple[float, float],
        longitude: tuple[float, float],
        target_latitude = None,
        target_longitude = None,
    ) -> None:
        super().__init__(
            boundary_root_dir = boundary_root_dir,
            start_date_hour = pd.Timestamp("2000-01-01 00:00:00"),
            end_date_hour = pd.Timestamp("2000-01-01 00:00:00"),
            upper_variables = upper_variables,
            surface_variables = surface_variables,
            levels = levels,
            latitude = latitude,
            longitude = longitude,
            boundary_width = 0,
            prediction_timedeltas = (0, 6, 12),
            forecast_cycle_hours = 12,
            use_cache = False,
            target_latitude = target_latitude,
            target_longitude = target_longitude,
        )

    @staticmethod
    def _select_time_coord(ds: xr.Dataset, target_time: pd.Timestamp) -> xr.Dataset:
        # Strict version of the parent: the file must really contain this init time.
        if "time" not in ds.coords and "time" not in ds.dims:
            return ds
        time_values = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(ds.time.values)))
        target_time = pd.Timestamp(target_time)
        if target_time not in time_values:
            raise KeyError(f"HRES init time {target_time} not found in file time coordinate {list(time_values)}.")
        if ds.time.ndim == 0:
            return ds
        return ds.sel(time = target_time)

    def load_lead_field(self, init_time: pd.Timestamp, lead_hours: int) -> dict:
        """Return the +lead_hours forecast initialised at `init_time` on the (lat-descending) grid:
        surf_vars {name: [H, W]}, atmos_vars {name: [L, H, W]}, plus latitude / longitude."""
        init_time = pd.Timestamp(init_time)
        lead_hours = int(lead_hours)
        upper_path, surface_path = self._dt_to_path(init_time)
        for path in (upper_path, surface_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing HRES boundary file for init {init_time}: {path}")
        source = self._load_hres_source_from_files(init_time)

        file_lead_hours = np.atleast_1d(source["prediction_timedelta_hours"].numpy())
        matches = np.nonzero(np.isclose(file_lead_hours, lead_hours))[0]
        if matches.size == 0:
            raise ValueError(
                f"HRES file {upper_path} has prediction_timedelta {file_lead_hours.tolist()} h, expected +{lead_hours} h."
            )
        idx = int(matches[0])

        def pick(tensor: torch.Tensor, ndim_without_lead: int) -> torch.Tensor:
            # A length-1 lead axis may have been squeezed away on disk.
            return tensor[idx] if tensor.ndim == ndim_without_lead + 1 else tensor

        return {
            "latitude": source["latitude"],
            "longitude": source["longitude"],
            "surf_vars": {k: pick(v, 2) for k, v in source["surf_vars"].items()},
            "atmos_vars": {k: pick(v, 3) for k, v in source["atmos_vars"].items()},
        }


class ERA5TWDatasetWithHRESInputBoundary(torch.utils.data.Dataset):
    """Wraps ERA5TWDatasetforAurora so that, for every input time step, the outer ring of every
    surf/atmos input variable is replaced by an HRES forecast valid at that time (physical space,
    same helper as inference `--replace_boundary_position input`). Targets and static variables
    are untouched. Runs on CPU inside __getitem__, so DataLoader workers parallelise it.

    HRES runs start at 00/12Z and hold the +0h/+6h/+12h leads, so values exist at the 6-hourly
    marks 00/06/12/18Z. Input times are mapped onto those marks: "nearest" (ties go to the earlier
    mark) or "interpolation" (linear between the two neighbouring marks). Each mark is taken from
    the latest run initialised no later than the sample's base time (the last input frame), so a
    run from the future is never used: with base 00Z, the 23Z input -> mark 00Z from the 00Z run
    at +0h; with base 23Z, the 23Z input -> mark 00Z from the previous 12Z run at +12h.
    """

    MARK_HOURS = 6
    INIT_CYCLE_HOURS = 12

    def __init__(
        self,
        base_dataset,
        boundary_root_dir: str,
        boundary_width: int = 8,
        boundary_smooth_mode: str = "no",
        boundary_time_interp_mode: str = "nearest",
        cache_size: int = 16,
    ) -> None:
        super().__init__()
        if boundary_time_interp_mode not in ("nearest", "interpolation"):
            raise ValueError(f"Unsupported boundary_time_interp_mode: {boundary_time_interp_mode}")
        if boundary_smooth_mode not in ("no", "linear", "mean", "gaussian"):
            raise ValueError(f"Unsupported boundary_smooth_mode: {boundary_smooth_mode}")
        if getattr(base_dataset, "lazy", False) or not getattr(base_dataset, "get_datetime", True):
            raise ValueError("Base dataset must be non-lazy and return the datetime string.")
        self.base = base_dataset
        self.boundary_root_dir = boundary_root_dir
        self.boundary_width = int(boundary_width)
        self.boundary_smooth_mode = boundary_smooth_mode
        self.boundary_time_interp_mode = boundary_time_interp_mode
        self.cache_size = int(cache_size)
        self.hres = BoundaryConditionDataset_HRESFixedLead(
            boundary_root_dir = boundary_root_dir,
            upper_variables = base_dataset.upper_variables,
            surface_variables = base_dataset.surface_variables,
            levels = base_dataset.levels,
            latitude = base_dataset.latitude,
            longitude = base_dataset.longitude,
        )
        # Per-process state (each DataLoader worker builds its own): small LRU of fields keyed
        # by (init time, lead) (a sample needs at most 4, neighbouring samples share them) and the
        # HRES -> ERA5 orientation, resolved on the first file that is loaded.
        self._cache = OrderedDict()
        self._flip = None

    # --- delegation to the wrapped ERA5 dataset --------------------------------------------
    def __len__(self) -> int:
        return len(self.base)

    def get_latitude_longitude(self):
        return self.base.get_latitude_longitude()

    def get_levels(self):
        return self.base.get_levels()

    def get_static_vars_ds(self):
        return self.base.get_static_vars_ds()

    # --- HRES access ---------------------------------------------------------------------
    @staticmethod
    def _is_increasing(coord: torch.Tensor) -> bool:
        return coord.numel() > 1 and coord[0].item() < coord[-1].item()

    def _align_to_era5(self, field: dict) -> dict:
        """Mirror inference's flip handling so the HRES ring has exactly the ERA5 orientation."""
        era5_lat, era5_lon = self.base.get_latitude_longitude()
        if self._flip is None:
            self._flip = (
                self._is_increasing(field["latitude"]) != self._is_increasing(era5_lat),
                self._is_increasing(field["longitude"]) != self._is_increasing(era5_lon),
            )
        flip_lat, flip_lon = self._flip
        lat, lon = field["latitude"], field["longitude"]
        if flip_lat:
            lat = torch.flip(lat, dims = (0,))
        if flip_lon:
            lon = torch.flip(lon, dims = (0,))
        if (
            lat.shape != era5_lat.shape or lon.shape != era5_lon.shape
            or not torch.allclose(lat.double(), era5_lat.double(), atol = 1e-3)
            or not torch.allclose(lon.double(), era5_lon.double(), atol = 1e-3)
        ):
            raise ValueError("HRES boundary grid does not match the ERA5 grid after orientation alignment.")
        for section in ("surf_vars", "atmos_vars"):
            for name, tensor in field[section].items():
                if flip_lat:
                    tensor = torch.flip(tensor, dims = (-2,))
                if flip_lon:
                    tensor = torch.flip(tensor, dims = (-1,))
                field[section][name] = tensor
        return field

    def _forecast_valid_at(self, mark_time: pd.Timestamp, base_time: pd.Timestamp) -> dict:
        """Forecast valid at `mark_time` (a 6h mark) from the latest run initialised no later than
        both `mark_time` and the sample's `base_time` (so a future run is never used)."""
        mark_time = pd.Timestamp(mark_time)
        init_time = min(mark_time, pd.Timestamp(base_time)).floor(f"{self.INIT_CYCLE_HOURS}h")
        lead_hours = int((mark_time - init_time) / pd.Timedelta(hours = 1))
        key = (init_time, lead_hours)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        field = self._align_to_era5(self.hres.load_lead_field(init_time, lead_hours))
        self._cache[key] = field
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last = False)
        return field

    def _boundary_at(self, valid_time: pd.Timestamp, base_time: pd.Timestamp) -> dict:
        valid_time = pd.Timestamp(valid_time)
        cycle = pd.Timedelta(hours = self.MARK_HOURS)
        m0 = valid_time.floor(f"{self.MARK_HOURS}h")
        offset = valid_time - m0
        if offset == pd.Timedelta(0):
            return self._forecast_valid_at(m0, base_time)
        if self.boundary_time_interp_mode == "nearest":
            # Ties (exactly half a cycle away from both marks) go to the earlier mark.
            return self._forecast_valid_at(m0 if offset <= cycle / 2 else m0 + cycle, base_time)
        weight = float(offset / cycle)
        field_0 = self._forecast_valid_at(m0, base_time)
        field_1 = self._forecast_valid_at(m0 + cycle, base_time)
        return {
            section: {
                name: field_0[section][name] + (field_1[section][name] - field_0[section][name]) * weight
                for name in field_0[section]
            }
            for section in ("surf_vars", "atmos_vars")
        }

    # --- sample ----------------------------------------------------------------------------
    def __getitem__(self, index: int) -> tuple:
        sample = self.base[index]
        input_data, date_str = sample[0], sample[-1]
        base_time = pd.Timestamp(date_str)
        window = self.base.input_time_window
        input_times = [
            base_time + pd.Timedelta(hours = (i - (window - 1)) * self.base.lead_time)
            for i in range(window)
        ]
        boundaries = [self._boundary_at(t, base_time) for t in input_times]

        new_input = {"surf_vars": {}, "atmos_vars": {}}
        for section in ("surf_vars", "atmos_vars"):
            for name, main in input_data[section].items():
                bc = torch.stack([b[section][name] for b in boundaries], dim = 0).to(main.dtype)
                new_input[section][name] = replace_input_boundary(
                    main, bc, self.boundary_width, self.boundary_smooth_mode,
                )
        return (new_input, *sample[1:])
