import xarray as xr
import matplotlib.pyplot as plt
import os
import argparse

# def plot_coast(ax: Axes) -> None:
#     if not hasattr(plot_coast, "coast"):
#         plot_coast.coast = pd.read_csv("coast.csv")
#     ax.plot(
#         (plot_coast.coast.lon_map - 100) / 0.25,
#         (plot_coast.coast.lat_map - 5) / 0.25, linewidth = 0.5,
#         color = "grey",
#     )

# --- Helper function ---
def select_latlon_range(da, latitude_range, longitude_range):
    lat_min, lat_max = latitude_range
    lon_min, lon_max = longitude_range
    return da.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

def plot_surf_vars(ds, figsize=(8, 6), outdir="plots_surf", cmap="cividis",
                   latitude_range=None, longitude_range=None):
    os.makedirs(outdir, exist_ok=True)
    for var in ds.data_vars:
        da = ds[var]
        if "valid_time" in da.dims:
            da = da.isel(valid_time=0)
        if set(da.dims) == {"latitude", "longitude"}:
            # Crop by lat/lon if needed
            if latitude_range is not None and longitude_range is not None:
                da = select_latlon_range(da, latitude_range, longitude_range)
            plt.figure(figsize=figsize)
            im = da.plot(cmap=cmap)
            plt.title(var)
            plt.tight_layout()
            plt.savefig(f"{outdir}/{var}.png", dpi=300)
            plt.close()
            print(f"Saved: {outdir}/{var}.png")
        else:
            print(f"Skipping {var}: dims {da.dims} (not 2D lat/lon)")
    print("Done! All surf vars plotted.")

def plot_atmos_vars_all_levels(ds, figsize=(8, 6), outdir="plots_atmos", cmap="viridis",
                              latitude_range=None, longitude_range=None):
    os.makedirs(outdir, exist_ok=True)
    for var in ds.data_vars:
        da = ds[var]
        if "valid_time" in da.dims:
            da = da.isel(valid_time=0)
        if "pressure_level" in da.dims:
            for lev in ds["pressure_level"].values:
                da_level = da.sel(pressure_level=lev)
                if set(da_level.dims) == {"latitude", "longitude"}:
                    # Crop by lat/lon if needed
                    if latitude_range is not None and longitude_range is not None:
                        da_level = select_latlon_range(da_level, latitude_range, longitude_range)
                    plt.figure(figsize=figsize)
                    da_level.plot(cmap=cmap)
                    plt.title(f"{var} (level={lev})")
                    plt.tight_layout()
                    plt.savefig(f"{outdir}/{var}_{lev}.png", dpi=300)
                    plt.close()
                    print(f"Saved: {outdir}/{var}_{lev}.png")
                else:
                    print(f"Skipping {var} at level {lev}: dims {da_level.dims} (not 2D lat/lon)")
        else:
            print(f"Skipping {var}: no 'pressure_level' dimension found")
    print("Done! All atmos vars at all levels plotted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot surface and atmospheric ERA5 variables from NetCDF files.")
    parser.add_argument("--surface_file", type=str, required=True,
                        help="Path to ERA5 surface NetCDF file.")
    parser.add_argument("--atmos_file", type=str, required=True,
                        help="Path to ERA5 atmospheric NetCDF file.")
    parser.add_argument("--output_dir", type=str, default="plots_era5_gt_out",
                        help="Parent output directory for plots. Default: 'plots_era5_gt_out'")
    parser.add_argument("--latitude", type=float, nargs=2, metavar=("LAT_MIN", "LAT_MAX"),
                        help="Latitude range: min max (e.g., --latitude 20 25)")
    parser.add_argument("--longitude", type=float, nargs=2, metavar=("LON_MIN", "LON_MAX"),
                        help="Longitude range: min max (e.g., --longitude 120 125)")

    args = parser.parse_args()

    latitude_range = tuple(args.latitude) if args.latitude is not None else None
    longitude_range = tuple(args.longitude) if args.longitude is not None else None

    # Load and plot surface variables
    surface_ds = xr.open_dataset(args.surface_file)
    surface_date_str = os.path.splitext(os.path.basename(args.surface_file))[0]
    surf_outdir = os.path.join(args.output_dir, "plots_surf_era5", surface_date_str)
    plot_surf_vars(surface_ds, figsize=(8, 6), outdir=surf_outdir, cmap="cividis",
                   latitude_range=latitude_range, longitude_range=longitude_range)

    # Load and plot atmospheric variables
    atmos_ds = xr.open_dataset(args.atmos_file)
    atmos_date_str = os.path.splitext(os.path.basename(args.atmos_file))[0]
    atmos_outdir = os.path.join(args.output_dir, "plots_atmos_era5", atmos_date_str)
    plot_atmos_vars_all_levels(atmos_ds, figsize=(8, 6), outdir=atmos_outdir, cmap="viridis",
                              latitude_range=latitude_range, longitude_range=longitude_range)
