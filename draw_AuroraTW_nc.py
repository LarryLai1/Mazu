import xarray as xr
import matplotlib.pyplot as plt
import math
import argparse
import os

# --- Helper function to select lat/lon range ---
def select_latlon_range(da, latitude_range, longitude_range):
    lat_min, lat_max = latitude_range
    lon_min, lon_max = longitude_range
    return da.sel(latitude=slice(lat_min, lat_max),
                  longitude=slice(lon_min, lon_max))

def plot_surf_vars(ds, figsize=(8, 6), outdir="plots_surf", cmap="cividis",
                   latitude_range=None, longitude_range=None):
    """
    Plots all 'surf_' variables as 2D heatmaps and saves each to a file.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened dataset.
    outdir : str
        Directory to save output plots.
    cmap : str
        Colormap for the plots.
    latitude_range : tuple or None
        Tuple (min_lat, max_lat) to restrict plot. If None, plot all.
    longitude_range : tuple or None
        Tuple (min_lon, max_lon) to restrict plot. If None, plot all.
    """
    os.makedirs(outdir, exist_ok=True)
    for var in ds.data_vars:
        if var.startswith("surf_"):
            da = ds[var]
            if "history" in da.dims:
                da = da.isel(history=0)
            if set(da.dims) == {"latitude", "longitude"}:
                # --- Select by range if provided ---
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
    """
    Plots all 'atmos_' variables at all available pressure levels as 2D heatmaps and saves each to a file.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened dataset.
    outdir : str
        Directory to save output plots.
    cmap : str
        Colormap for the plots.
    latitude_range : tuple or None
        Tuple (min_lat, max_lat) to restrict plot. If None, plot all.
    longitude_range : tuple or None
        Tuple (min_lon, max_lon) to restrict plot. If None, plot all.
    """
    os.makedirs(outdir, exist_ok=True)
    for var in ds.data_vars:
        if var.startswith("atmos_"):
            da = ds[var]
            if "history" in da.dims:
                da = da.isel(history=0)
            if "level" in da.dims:
                for lev in ds["level"].values:
                    da_level = da.sel(level=lev)
                    if set(da_level.dims) == {"latitude", "longitude"}:
                        # --- Select by range if provided ---
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
                print(f"Skipping {var}: no 'level' dimension found")
    print("Done! All atmos vars at all levels plotted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot surface and atmospheric variables from a NetCDF file."
    )
    parser.add_argument(
        "--file_path", type=str, required=True,
        help="Path to the NetCDF file to plot."
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots_out",
        help="Parent output directory for plots. Default: 'plots_out'"
    )
    parser.add_argument(
        "--latitude", type=float, nargs=2, metavar=("LAT_MIN", "LAT_MAX"),
        help="Latitude range: min max (e.g., --latitude 20 25)"
    )
    parser.add_argument(
        "--longitude", type=float, nargs=2, metavar=("LON_MIN", "LON_MAX"),
        help="Longitude range: min max (e.g., --longitude 120 125)"
    )

    args = parser.parse_args()

    latitude_range = tuple(args.latitude) if args.latitude is not None else None
    longitude_range = tuple(args.longitude) if args.longitude is not None else None

    batch_output_ds = xr.open_dataset(args.file_path)
    print(batch_output_ds)

    # Extract filename (used as folder name)
    date_str = os.path.splitext(os.path.basename(args.file_path))[0]
    # Create subfolders in the output directory
    surf_outdir = os.path.join(args.output_dir, "plots_surf", date_str)
    atmos_outdir = os.path.join(args.output_dir, "plots_atmos", date_str)

    # Run Plotting
    plot_surf_vars(batch_output_ds, outdir=surf_outdir, cmap="viridis",
                   latitude_range=latitude_range, longitude_range=longitude_range)
    plot_atmos_vars_all_levels(batch_output_ds, outdir=atmos_outdir, cmap="plasma",
                              latitude_range=latitude_range, longitude_range=longitude_range)