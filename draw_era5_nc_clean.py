import xarray as xr
import matplotlib.pyplot as plt
import os
import argparse

# --- Helper 1: Select Lat/Lon Range ---
def select_latlon_range(da, latitude_range, longitude_range):
    lat_min, lat_max = latitude_range
    lon_min, lon_max = longitude_range
    return da.sel(latitude=slice(lat_min, lat_max),
                  longitude=slice(lon_min, lon_max))

# --- Helper 2: The Clean Plotting Logic ---
def plot_clean_heatmap(da, filename, cmap, figsize):
    """
    Plots a DataArray as a strictly clean image:
    - No axis, no ticks, no border spines.
    - No labels, no title, no colorbar.
    - Zero whitespace padding.
    """
    plt.figure(figsize=figsize)
    
    # Plot with xarray: disable standard labels
    da.plot(cmap=cmap, add_colorbar=False, add_labels=False)
    
    # Force disable axis
    plt.axis('off')
    
    # Double check title is gone
    plt.title("")
    
    # Save tightly
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {filename}")

# --- Surface Variables Processing ---
def plot_surf_vars(ds, outdir, figsize=(8, 6), cmap="cividis",
                   latitude_range=None, longitude_range=None):
    os.makedirs(outdir, exist_ok=True)
    
    for var in ds.data_vars:
        da = ds[var]
        
        # 1. Handle Time Dimension (ERA5 often has 'valid_time' or 'time')
        if "valid_time" in da.dims:
            da = da.isel(valid_time=0)
        elif "time" in da.dims:
            da = da.isel(time=0)
            
        # 2. Check if 2D (lat/lon)
        if set(da.dims) == {"latitude", "longitude"}:
            # Crop
            if latitude_range and longitude_range:
                da = select_latlon_range(da, latitude_range, longitude_range)
            
            # Plot
            # filename = os.path.join(outdir, f"{var}.png")
            filename = os.path.join(outdir, f"{var}.pdf")
            plot_clean_heatmap(da, filename, cmap, figsize)
        else:
            print(f"Skipping {var}: dims {da.dims} (not 2D lat/lon)")

# --- Atmospheric Variables Processing ---
def plot_atmos_vars_all_levels(ds, outdir, figsize=(8, 6), cmap="viridis",
                              latitude_range=None, longitude_range=None):
    os.makedirs(outdir, exist_ok=True)
    
    for var in ds.data_vars:
        da = ds[var]
        
        # 1. Handle Time
        if "valid_time" in da.dims:
            da = da.isel(valid_time=0)
        elif "time" in da.dims:
            da = da.isel(time=0)

        # 2. Handle Pressure Levels
        if "pressure_level" in da.dims:
            for lev in ds["pressure_level"].values:
                da_level = da.sel(pressure_level=lev)
                
                if set(da_level.dims) == {"latitude", "longitude"}:
                    # Crop
                    if latitude_range and longitude_range:
                        da_level = select_latlon_range(da_level, latitude_range, longitude_range)
                    
                    # Plot (filename includes level)
                    # filename = os.path.join(outdir, f"{var}_{int(lev)}.png")
                    filename = os.path.join(outdir, f"{var}_{int(lev)}.pdf")
                    plot_clean_heatmap(da_level, filename, cmap, figsize)
                else:
                    print(f"Skipping {var} at {lev}: dims {da_level.dims}")
        else:
            print(f"Skipping {var}: no 'pressure_level' dimension.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot strictly clean ERA5 images (prediction maps).")
    
    parser.add_argument("--surface_file", type=str, required=True, help="Path to ERA5 surface .nc file")
    parser.add_argument("--atmos_file", type=str, required=True, help="Path to ERA5 atmos .nc file")
    parser.add_argument("--output_dir", type=str, default="plots_clean_out", help="Output directory")
    
    parser.add_argument("--latitude", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Lat range (e.g. 20 25)")
    parser.add_argument("--longitude", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Lon range (e.g. 120 125)")

    args = parser.parse_args()

    # Ranges
    lat_range = tuple(args.latitude) if args.latitude else None
    lon_range = tuple(args.longitude) if args.longitude else None

    # --- Process Surface File ---
    if os.path.exists(args.surface_file):
        ds_surf = xr.open_dataset(args.surface_file)
        date_str = os.path.splitext(os.path.basename(args.surface_file))[0]
        surf_out = os.path.join(args.output_dir, "surface", date_str)
        
        print(f"Processing Surface: {args.surface_file}")
        plot_surf_vars(ds_surf, outdir=surf_out, cmap="cividis",
                       latitude_range=lat_range, longitude_range=lon_range)
    
    # --- Process Atmos File ---
    if os.path.exists(args.atmos_file):
        ds_atm = xr.open_dataset(args.atmos_file)
        date_str = os.path.splitext(os.path.basename(args.atmos_file))[0]
        atm_out = os.path.join(args.output_dir, "atmos", date_str)
        
        print(f"Processing Atmos: {args.atmos_file}")
        plot_atmos_vars_all_levels(ds_atm, outdir=atm_out, cmap="viridis",
                                  latitude_range=lat_range, longitude_range=lon_range)