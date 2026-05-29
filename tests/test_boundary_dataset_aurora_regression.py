from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.BoundaryConditionDataset import BoundaryConditionDataset_Aurora


def _write_aurora_boundary_files(root_dir: Path) -> None:
    base_time = pd.Timestamp("2020-07-01 00:00:00")
    time_values = pd.DatetimeIndex([base_time, base_time + pd.Timedelta(hours = 6)])
    latitude = np.array([0.0, 1.0], dtype = np.float32)
    longitude = np.array([100.0, 101.0], dtype = np.float32)
    levels = np.array([850], dtype = np.int32)

    upper_data = np.array(
        [
            [[[10.0, 10.0], [10.0, 10.0]]],
            [[[20.0, 20.0], [20.0, 20.0]]],
        ],
        dtype = np.float32,
    )
    surface_data = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ],
        dtype = np.float32,
    )

    upper_ds = xr.Dataset(
        data_vars = {
            "t": (("time", "level", "lat", "lon"), upper_data),
        },
        coords = {
            "time": time_values,
            "level": levels,
            "lat": latitude,
            "lon": longitude,
        },
    )
    surface_ds = xr.Dataset(
        data_vars = {
            "2t": (("time", "lat", "lon"), surface_data),
        },
        coords = {
            "time": time_values,
            "lat": latitude,
            "lon": longitude,
        },
    )

    upper_path = root_dir / "2020" / "202007" / "20200701" / "20200701_00_upper.nc"
    surface_path = root_dir / "2020" / "202007" / "20200701" / "20200701_00_sfc.nc"
    upper_path.parent.mkdir(parents = True, exist_ok = True)
    upper_ds.to_netcdf(upper_path)
    surface_ds.to_netcdf(surface_path)


def test_aurora_surface_mapping_regression(tmp_path):
    root_dir = tmp_path / "aurora_boundary"
    _write_aurora_boundary_files(root_dir)

    ds = BoundaryConditionDataset_Aurora(
        boundary_root_dir = str(root_dir),
        start_date_hour = "2020-07-01 00:00:00",
        end_date_hour = "2020-07-01 00:00:00",
        upper_variables = ["t"],
        surface_variables = ["t2m"],
        levels = [850],
        latitude = (0.0, 1.0),
        longitude = (100.0, 101.0),
        boundary_width = 0,
        prediction_timedeltas = [0, 6],
        use_cache = False,
        get_datetime = False,
    )

    source = ds.get_boundary_source(pd.Timestamp("2020-07-01 00:00:00"))
    assert "2t" in source["surf_vars"]
    assert "t2m" not in source["surf_vars"]
    assert source["surf_vars"]["2t"].shape[0] == 2


if __name__ == "__main__":
    test_aurora_surface_mapping_regression(Path("/tmp"))
    print("ok")