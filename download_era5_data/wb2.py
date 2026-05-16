from pathlib import Path
import json
import xarray as xr
import gcsfs
import numpy as np
import pandas as pd
from tqdm import tqdm

mapping_path = Path("/tmp3/b12902101/Mazu/download_era5_data/variable_mapping.json")
with mapping_path.open() as f:
    mapping = json.load(f)

def _collect_vars(section):
    out = {}
    for target, vals in mapping.get(section, {}).items():
        src = vals.get("forecast_var")
        if src:
            out[src] = vals.get("data_root_var") or target
    return out

rename_map = {}
rename_map.update(_collect_vars("surface"))
rename_map.update(_collect_vars("atmospheric"))
needed_vars = sorted(rename_map.keys())

lat = slice(2.5, 41.25)
lon = slice(97.5, 147.25)
# time = slice("2016-01-01T00:00:00.000000000", "2016-12-31T23:00:00.000000000")
time = slice("2017-01-01T00:00:00.000000000", "2017-12-31T23:00:00.000000000")
pred_time = slice(0, 12)
levels = [1000, 925, 850, 700, 500, 300, 150, 50]

fs = gcsfs.GCSFileSystem(token='anon')
mapper = fs.get_mapper("gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr")
ds = xr.open_zarr(mapper, consolidated=True, decode_timedelta=False)
ds = ds.sel(
    latitude=lat,
    longitude=lon,
    time=time,
    level=levels,
    prediction_timedelta=pred_time,
)
existing_vars = [v for v in needed_vars if v in ds.data_vars]
missing_vars = [v for v in needed_vars if v not in ds.data_vars]
if missing_vars:
    print("Missing forecast vars:", missing_vars)
ds = ds[existing_vars]

out_root = Path("/tmp3/b12902101/era5_tw_forecast")
out_root.mkdir(parents=True, exist_ok=True)

def _to_valid_time(init_time, pred_value):
    if np.issubdtype(type(pred_value), np.timedelta64):
        delta = pred_value
    else:
        delta = np.timedelta64(int(pred_value), "h")
    return np.datetime64(init_time) + delta

def split_and_save_by_date(ds, out_root, time_values=None, pred_values=None, limit=None):
    if time_values is None:
        time_values = ds["time"].values
    if pred_values is None:
        pred_values = ds["prediction_timedelta"].values
    count = 0
    buckets = {}
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    for init_time in tqdm(time_values, desc="Init time", unit="init", bar_format=bar_format):
        for pred_value in pred_values:
            ds_step = ds.sel(time=[init_time], prediction_timedelta=[pred_value])
            valid_time = _to_valid_time(init_time, ds_step["prediction_timedelta"].values[0])
            ds_step = ds_step.isel(prediction_timedelta=0, drop=True)
            ds_step = ds_step.assign_coords(time=("time", [valid_time]))
            ds_step = ds_step.rename({
                k: v for k, v in rename_map.items() if k in ds_step.data_vars
            })

            upper_vars = [v for v in ds_step.data_vars if "level" in ds_step[v].dims]
            sfc_vars = [v for v in ds_step.data_vars if "level" not in ds_step[v].dims]
            ds_upper = ds_step[upper_vars].copy() if upper_vars else None
            ds_sfc = None
            if sfc_vars:
                ds_sfc = ds_step[sfc_vars].copy()
                ds_sfc = ds_sfc.drop_vars("level", errors="ignore")

            dt = pd.to_datetime(valid_time)
            y = dt.strftime("%Y")
            ym = dt.strftime("%Y%m")
            ymd = dt.strftime("%Y%m%d")
            month_dir = out_root / y / ym
            month_dir.mkdir(parents=True, exist_ok=True)
            if ymd not in buckets:
                buckets[ymd] = {"dir": month_dir, "sfc": [], "upper": []}
            if ds_sfc is not None:
                buckets[ymd]["sfc"].append(ds_sfc)
            if ds_upper is not None:
                buckets[ymd]["upper"].append(ds_upper)

            count += 1
            if limit is not None and count >= limit:
                return _flush_buckets(buckets)

    _flush_buckets(buckets)

def _flush_buckets(buckets):
    for ymd, payload in buckets.items():
        month_dir = payload["dir"]
        if payload["sfc"]:
            ds_sfc_day = xr.concat(payload["sfc"], dim="time").sortby("time")
            ds_sfc_day.to_netcdf(month_dir / f"{ymd}_sfc.nc")
        if payload["upper"]:
            ds_upper_day = xr.concat(payload["upper"], dim="time").sortby("time")
            ds_upper_day.to_netcdf(month_dir / f"{ymd}_upper.nc")

# Example: limit to 2 steps for a quick sanity check
split_and_save_by_date(ds, out_root, limit=None)