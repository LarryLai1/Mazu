import sys
import argparse
import time
import pandas as pd
from pathlib import Path
import cdsapi

var_sfc = [
    '10m_u_component_of_wind',         # u10
    '10m_v_component_of_wind',         # v10
    '2m_temperature',                  # t2m
    'mean_sea_level_pressure',         # msl
    'surface_pressure',                # sp
    'total_column_water_vapour',       # tcwv
    '2m_dewpoint_temperature',         # d2m
    "sea_surface_temperature",         # sst
    # "mean_wave_direction",
    # "mean_wave_period",
    # "significant_height_of_combined_wind_waves_and_swell",
    # 'total_precipitation',
]
var_upper = [
    'u_component_of_wind',             # u
    'v_component_of_wind',             # v
    'temperature',                     # t
    'specific_humidity',               # q
    'geopotential',                    # z
    'vertical_velocity'                # w
]

# pressure_level = [50, 150, 300, 500, 700, 850, 925, 1000]
pressure_level = [1000, 925, 850, 700, 500, 300, 150, 50]

def retrieve_with_retry(
        client: cdsapi.Client,
        dataset: str,
        request: dict,
        file_path: str,
        max_retries = 10
    ):
    retries = 0
    while retries < max_retries:
        try:
            client.retrieve(dataset, request, file_path)
            print(f'Data successfully retrieved and saved to {file_path}.')
            break
        except Exception as e:
            retries += 1
            print(f'Attempt {retries} failed: {e}.')
            print('Retrying in 10 seconds...')
            time.sleep(10)
    else:
        print(f'Failed to retrieve data after {max_retries} attempts.')

def main(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    save_dir: str,
    region: str
):
    if isinstance(start, str):
        start = pd.Timestamp(start)
    if isinstance(end, str):
        end = pd.Timestamp(end)

    # end += pd.DateOffset(months=1)
    dt = start
    C = cdsapi.Client()
    grid = [0.25, 0.25]

    if region == 'tw':
        area = [40, 100, 5, 145]
    elif region == 'global':
        area = [90, -180, -90, 180]
    elif region == "eu":
        area = [70, -10, 35, 35]

    while dt < end:
        Y = f'{dt.year:0>4d}'
        M = f'{dt.month:0>2d}'
        D = f'{dt.day:0>2d}'
        print(f'***** Downloading [{Y}/{M}/{D}]. *****')
 
        output_dir = Path(save_dir) / region / f'{Y}/{Y}{M}/{Y}{M}{D}'
        if Path(output_dir).exists():
            print(f'Warning: Directory {output_dir} already exists.', file=sys.stderr)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        retrieve_with_retry(
            C,
            'reanalysis-era5-single-levels', {
                'variable': var_sfc,
                'product_type': 'reanalysis',
                'year': Y,
                'month': M,
                'day': D,
                'time': [f'{i:02}:00' for i in range(24)],
                'area': area,
                'grid': grid,
                "download_format": "unarchived",
                'format': 'netcdf',
        }, Path(output_dir) / f'{Y}{M}{D}_sfc.nc')

        retrieve_with_retry(
            C,
            'reanalysis-era5-pressure-levels', {
                'variable': var_upper,
                'pressure_level': pressure_level,
                'product_type': 'reanalysis',
                'year': Y,
                'month': M,
                'day': D,
                'time': [f'{i:02}:00' for i in range(24)],
                'area': area,
                'grid': grid,
                "download_format": "unarchived",
                'format': 'netcdf'
        }, Path(output_dir) / f'{Y}{M}{D}_upper.nc')

        dt += pd.DateOffset(days = 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start', type=str, required=True,
        help='Start date in the format of YYYY-MM.'
    )
    parser.add_argument(
        '--end', type=str, required=True,
        help='End date (exclusive) in the format of YYYY-MM.'
    )
    parser.add_argument(
        '--save_dir', type=str, default='/work/yunye0121/era5_data',
        help='Directory to save the downloaded data.',
    )
    parser.add_argument(
        '--region', type=str,
        default='tw', choices=['global', 'tw', "eu"],
        help='Specify the region to download data.'
    )
    args = parser.parse_args()
    main(**vars(args))