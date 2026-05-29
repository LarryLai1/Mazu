from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.BoundaryConditionDataset import BoundaryConditionDataset_ERA5


def _write_era5_boundary_files(root_dir: Path) -> None:
    base_time = pd.Timestamp("2020-07-01 00:00:00")
    time_values = pd.DatetimeIndex([base_time])
    prediction_timedelta = pd.to_timedelta([0, 6], unit = "h")
    latitude = np.array([0.0, 1.0], dtype = np.float32)
    longitude = np.array([100.0, 101.0], dtype = np.float32)
    levels = np.array([850], dtype = np.int32)

    surf_data = np.array(
        [[
            [[0.0, 0.0], [0.0, 0.0]],
            [[6.0, 6.0], [6.0, 6.0]],
        ]],
        dtype = np.float32,
    )
    upper_data = np.array(
        [[
            [[[10.0, 10.0], [10.0, 10.0]]],
            [[[16.0, 16.0], [16.0, 16.0]]],
        ]],
        dtype = np.float32,
    )

    upper_ds = xr.Dataset(
        data_vars = {
            "t": (("time", "prediction_timedelta", "level", "latitude", "longitude"), upper_data),
        },
        coords = {
            "time": time_values,
            "prediction_timedelta": prediction_timedelta,
            "level": levels,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    surface_ds = xr.Dataset(
        data_vars = {
            "2t": (("time", "prediction_timedelta", "latitude", "longitude"), surf_data),
        },
        coords = {
            "time": time_values,
            "prediction_timedelta": prediction_timedelta,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    upper_path = root_dir / "2020" / "202007" / "20200701_upper.nc"
    surface_path = root_dir / "2020" / "202007" / "20200701_sfc.nc"
    upper_path.parent.mkdir(parents = True, exist_ok = True)
    upper_ds.to_netcdf(upper_path)
    surface_ds.to_netcdf(surface_path)


def _move_source_to_device(source: dict, device: torch.device) -> dict:
    return {
        "time_values": source["time_values"],
        "prediction_timedelta_hours": source["prediction_timedelta_hours"].to(device),
        "latitude": source["latitude"].to(device),
        "longitude": source["longitude"].to(device),
        "levels": source["levels"],
        "surf_vars": {k: v.to(device) for k, v in source["surf_vars"].items()},
        "atmos_vars": {k: v.to(device) for k, v in source["atmos_vars"].items()},
    }


def test_era5_uses_prediction_timedelta_coordinate(tmp_path):
    root_dir = tmp_path / "era5_boundary"
    _write_era5_boundary_files(root_dir)

    ds = BoundaryConditionDataset_ERA5(
        boundary_root_dir = str(root_dir),
        start_date_hour = "2020-07-01 00:00:00",
        end_date_hour = "2020-07-01 00:00:00",
        upper_variables = ["t"],
        surface_variables = ["t2m"],
        levels = [850],
        latitude = (0.0, 1.0),
        longitude = (100.0, 101.0),
        boundary_width = 0,
        use_cache = False,
        get_datetime = False,
    )

    source = ds.get_boundary_source(pd.Timestamp("2020-07-01 00:00:00"))
    assert tuple(ds.prediction_timedelta_hours) == (0.0, 6.0)
    assert torch.equal(source["prediction_timedelta_hours"], torch.tensor([0.0, 6.0], dtype = torch.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = _move_source_to_device(source, device)

    exact = ds.get_boundary_at_time_from_source(
        source,
        pd.Timestamp("2020-07-01 00:00:00"),
        pd.Timestamp("2020-07-01 06:00:00"),
    )
    interp = ds.get_boundary_at_time_from_source(
        source,
        pd.Timestamp("2020-07-01 00:00:00"),
        pd.Timestamp("2020-07-01 03:00:00"),
    )

    expected_exact_surface = torch.full((2, 2), 6.0, device = device)
    expected_interp_surface = torch.full((2, 2), 3.0, device = device)
    expected_exact_upper = torch.full((1, 2, 2), 16.0, device = device)
    expected_interp_upper = torch.full((1, 2, 2), 13.0, device = device)

    torch.testing.assert_close(exact["surf_vars"]["2t"], expected_exact_surface)
    torch.testing.assert_close(interp["surf_vars"]["2t"], expected_interp_surface)
    torch.testing.assert_close(exact["atmos_vars"]["t"], expected_exact_upper)
    torch.testing.assert_close(interp["atmos_vars"]["t"], expected_interp_upper)


if __name__ == "__main__":
    test_era5_uses_prediction_timedelta_coordinate(Path("/tmp"))
    print("ok")