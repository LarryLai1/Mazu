import os
import sys
import argparse
import time
import pandas as pd
from pathlib import Path
import cdsapi

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

def retrieve_static_vars(
        client: cdsapi.Client,
        save_dir: str,
        region: str
):
    if region == 'tw':
        area = [40, 100, 5, 145]
    elif region == 'global':
        area = [90, -180, -90, 180]
    
    grid = [0.25, 0.25]
    output_dir = Path(save_dir) / region / 'static'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download static fields
    retrieve_with_retry(
        client,
        'reanalysis-era5-single-levels', {
            'variable': [
                'geopotential',  # 'z' in the code
                'land_sea_mask', # 'lsm' in the code
                'soil_type',     # 'slt' in the code
            ],
            'product_type': ["reanalysis"],
            'year': '2020',      # Any year will do for invariant data
            'month': '01',       # Any month will do
            'day': '01',         # Any day will do
            'time': '00:00',     # Any time will do
            'area': area,
            'grid': grid,
            'format': 'netcdf'
        },
        Path(output_dir) / 'static_vars.nc'
    )

def main(
    save_dir: str,
    region: str
):
    C = cdsapi.Client()
    retrieve_static_vars(C, save_dir, region)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir', type=str, default='/work/yunye0121/era5_data',
        help='Directory to save the downloaded data.',
    )
    parser.add_argument(
        '--region', type=str,
        default='tw', choices=['global', 'tw'],
        help='Specify the region to download data.'
    )
    args = parser.parse_args()
    main(**vars(args))