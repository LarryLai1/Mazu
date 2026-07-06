#!/usr/bin/env python3
import os
import glob
import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from concurrent.futures import ProcessPoolExecutor, as_completed

def plot_single_nc(nc_path, out_dir, dpi=150, cmap='jet'):
    try:
        # Load dataset
        with xr.open_dataset(nc_path) as ds:
            # Look for mean sea level pressure variable
            if 'surf_msl' in ds:
                da = ds['surf_msl']
            elif 'msl' in ds:
                da = ds['msl']
            else:
                # Search case-insensitively
                msl_vars = [v for v in ds.data_vars if 'msl' in v.lower()]
                if msl_vars:
                    da = ds[msl_vars[0]]
                else:
                    print(f"Skipping {nc_path}: MSL variable not found.")
                    return False
            
            # Select first history slice if present
            if 'history' in da.dims:
                da = da.isel(history=0)
            
            # Lat/lon coordinate names discovery
            lat_name = None
            lon_name = None
            for n in ['latitude', 'lat', 'Latitude', 'LAT']:
                if n in da.coords:
                    lat_name = n
                    break
            for n in ['longitude', 'lon', 'Longitude', 'LON']:
                if n in da.coords:
                    lon_name = n
                    break
            
            if lat_name is None or lon_name is None:
                print(f"Skipping {nc_path}: latitude/longitude coords not found.")
                return False
            
            lat_arr = da[lat_name].values
            lon_arr = da[lon_name].values
            val_arr = da.values
            
            # Convert Pa to hPa if values are in Pa range (> 50000 Pa)
            is_hpa = False
            mean_val = float(da.mean())
            if mean_val > 50000:
                val_arr = val_arr / 100.0
                is_hpa = True
            
            # Setup Plot with Cartopy Projection
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Add geographical features
            ax.coastlines(resolution='50m', linewidth=1.0, color='black')
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1.0)
            
            # Add grid lines
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False
            
            # Set map extent based on coordinate range
            ax.set_extent([lon_arr.min(), lon_arr.max(), lat_arr.min(), lat_arr.max()], crs=ccrs.PlateCarree())
            
            # Plot surface MSL
            unit_str = "hPa" if is_hpa else "Pa"
            im = ax.pcolormesh(lon_arr, lat_arr, val_arr, cmap=cmap, shading='auto', transform=ccrs.PlateCarree())
            
            # Colorbar
            fig.colorbar(im, ax=ax, label=f'Mean Sea Level Pressure ({unit_str})', shrink=0.7)
            
            # Set Title
            base_name = os.path.basename(nc_path)
            plt.title(f"MSL Forecast - {base_name}", fontsize=14)
            plt.tight_layout()
            
            # Save figure
            png_name = base_name.replace('.nc', '.png')
            out_path = os.path.join(out_dir, png_name)
            plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return True
            
    except Exception as e:
        print(f"Error processing {nc_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Plot surf_msl from all .nc files in a directory.")
    parser.add_argument('--preds_dir', type=str, 
                        default='/tmp3/b12902101/LAM_output_preds/era5_boundary8_inject-inside_no_nearest_backbone/preds',
                        help='Directory containing the predictions (.nc files)')
    parser.add_argument('--output_dir', type=str, 
                        default='/tmp3/b12902101/LAM_output_preds/era5_boundary8_inject-inside_no_nearest_backbone/plots_msl',
                        help='Directory to save output figures')
    parser.add_argument('--dpi', type=int, default=150, help='Resolution for output figures')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers for plotting')
    parser.add_argument('--cmap', type=str, default='jet', help='Colormap for MSL plot')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .nc files
    nc_pattern = os.path.join(args.preds_dir, "*.nc")
    nc_files = sorted(glob.glob(nc_pattern))
    total_files = len(nc_files)
    
    if total_files == 0:
        print(f"No .nc files found in {args.preds_dir}")
        return
        
    print(f"Found {total_files} .nc files in {args.preds_dir}")
    print(f"Saving figures to {args.output_dir}")
    print(f"Starting processing with {args.num_workers} parallel workers...")
    
    success_count = 0
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(plot_single_nc, file_path, args.output_dir, args.dpi, args.cmap): file_path 
            for file_path in nc_files
        }
        
        for i, future in enumerate(as_completed(futures), start=1):
            file_path = futures[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
            except Exception as e:
                print(f"Unexpected exception processing {file_path}: {e}")
                
            if i % 50 == 0 or i == total_files:
                print(f"Progress: {i}/{total_files} files processed ({success_count} successful plots)")
                
    print(f"Done! Successfully generated {success_count}/{total_files} figures.")

if __name__ == '__main__':
    main()
