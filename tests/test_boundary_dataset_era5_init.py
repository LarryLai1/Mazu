from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.BoundaryConditionDataset import BoundaryConditionDataset_ERA5


def test_era5_init_defaults():
    ds = BoundaryConditionDataset_ERA5(
        boundary_root_dir=".",
        start_date_hour="2020-01-01 00:00:00",
        end_date_hour="2020-01-02 00:00:00",
        upper_variables=["t"],
        surface_variables=["u10", "v10", "t2m", "msl"],
        levels=[850],
        latitude=(0.0, 1.0),
        longitude=(0.0, 1.0),
        boundary_width=0,
        use_cache=False,
        get_datetime=False,
    )
    assert ds.forecast_cycle_hours == 12
    assert ds.prediction_timedelta_hours == (0, 12)
    assert len(ds) == 3


if __name__ == "__main__":
    test_era5_init_defaults()
    print("ok")
