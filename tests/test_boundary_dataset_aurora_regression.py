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


def _write_aurora_ascending_lat_files(root_dir: Path) -> pd.Timestamp:
    """Write an aurora `_HH` cycle with ASCENDING latitude and a 6-hourly forecast trajectory that
    starts at base+6h (mirroring the real earth2 output offset). Field values encode
    time_index*10 + lat_index so orientation/selection can be checked exactly."""
    base_time = pd.Timestamp("2020-07-01 06:00:00")
    time_values = pd.DatetimeIndex([base_time + pd.Timedelta(hours = h) for h in (6, 12, 18)])
    latitude = np.array([5.0, 6.0, 7.0], dtype = np.float32)   # ASC -> must be flipped to DESC
    longitude = np.array([100.0, 101.0], dtype = np.float32)   # ASC
    levels = np.array([850], dtype = np.int32)

    nt, nlev, nlat, nlon = len(time_values), len(levels), len(latitude), len(longitude)
    upper = np.zeros((nt, nlev, nlat, nlon), dtype = np.float32)
    surface = np.zeros((nt, nlat, nlon), dtype = np.float32)
    for ti in range(nt):
        for la in range(nlat):
            upper[ti, :, la, :] = ti * 10 + la
            surface[ti, la, :] = ti * 10 + la

    upper_ds = xr.Dataset(
        data_vars = {"t": (("time", "level", "lat", "lon"), upper)},
        coords = {"time": time_values, "level": levels, "lat": latitude, "lon": longitude},
    )
    surface_ds = xr.Dataset(
        data_vars = {"2t": (("time", "lat", "lon"), surface)},
        coords = {"time": time_values, "lat": latitude, "lon": longitude},
    )
    upper_path = root_dir / "2020" / "202007" / "20200701" / "20200701_06_upper.nc"
    surface_path = root_dir / "2020" / "202007" / "20200701" / "20200701_06_sfc.nc"
    upper_path.parent.mkdir(parents = True, exist_ok = True)
    upper_ds.to_netcdf(upper_path)
    surface_ds.to_netcdf(surface_path)
    return base_time


def _make_ascending_lat_dataset(root_dir: Path):
    return BoundaryConditionDataset_Aurora(
        boundary_root_dir = str(root_dir),
        start_date_hour = "2020-07-01 06:00:00",
        end_date_hour = "2020-07-01 06:00:00",
        upper_variables = ["t"],
        surface_variables = ["t2m"],
        levels = [850],
        latitude = (7.0, 5.0),
        longitude = (100.0, 101.0),
        boundary_width = 0,
        prediction_timedeltas = [6, 12, 18],
        forecast_cycle_hours = 6,
        use_cache = False,
        get_datetime = False,
    )


def test_aurora_loader_orientation_normalization(tmp_path):
    root_dir = tmp_path / "aurora_boundary_orient"
    base_time = _write_aurora_ascending_lat_files(root_dir)
    ds = _make_ascending_lat_dataset(root_dir)

    source = ds.get_boundary_source(base_time)

    lat = source["latitude"]
    lon = source["longitude"]
    # Loader must force latitude descending / longitude ascending (Aurora Metadata requirement).
    assert bool((lat[1:] - lat[:-1] < 0).all()), f"latitude not strictly decreasing: {lat}"
    assert bool((lon[1:] - lon[:-1] > 0).all()), f"longitude not strictly increasing: {lon}"

    # prediction_timedelta_hours is derived from absolute time minus the base/init time.
    assert list(source["prediction_timedelta_hours"].tolist()) == [6.0, 12.0, 18.0]

    # Field spatial dims are (lat, lon) and the lat axis was flipped in step with the coord flip:
    # after flipping, lat row 0 corresponds to the original highest-index lat (value ti*10 + 2).
    t = source["atmos_vars"]["t"]
    assert t.shape[-2:] == (len(lat), len(lon))
    assert float(t[0, 0, 0, 0]) == 2.0   # time 0, level 0, flipped-lat row 0, lon 0
    two_t = source["surf_vars"]["2t"]
    assert float(two_t[0, 0, 0]) == 2.0


def test_aurora_prediction_timedelta_selection(tmp_path):
    root_dir = tmp_path / "aurora_boundary_select"
    base_time = _write_aurora_ascending_lat_files(root_dir)
    ds = _make_ascending_lat_dataset(root_dir)

    # Exact lead time -> that forecast step (time index 1 == base+12h == pred_td 12h).
    exact = ds.get_boundary_at_time(base_time, base_time + pd.Timedelta(hours = 12))
    assert float(exact["atmos_vars"]["t"][0, 0, 0]) == 10 + 2  # time index 1, flipped lat row 0

    # Between lead times -> linear interpolation on prediction_timedelta (6h and 12h, weight 0.5).
    mid = ds.get_boundary_at_time(base_time, base_time + pd.Timedelta(hours = 9))
    expected = 0.5 * (2.0) + 0.5 * (12.0)  # ti0->2, ti1->12 at flipped lat row 0
    assert abs(float(mid["atmos_vars"]["t"][0, 0, 0]) - expected) < 1e-5


if __name__ == "__main__":
    test_aurora_surface_mapping_regression(Path("/tmp"))
    print("ok")