#!/bin/bash
set -euo pipefail

output_dir="wavenumber_plots_output"

era5_boundary_root="/tmp3/b12902101/era5_tw_forecast_0.25deg"
era5_boundary_root_15="/tmp3/b12902101/era5_tw_forecast_1.5deg"
preds_root="/tmp3/b12902101/LAM_output_preds"

smooth="no"
bd_position="backbone"
interp="nearest"
resols=(0.25 0.5 1.5)

# The ERA5 forecast is initialised on the 12h cycle; the model rollouts start at 01:00.
era5_init_time="2020-03-01 00:00:00"
model_init_time="2020-03-01 01:00:00"

# Every series lands on one set of axes, so stick to one apply mode to keep the plot readable.
bd_apply_mode="direct"

preds_dirs=()
labels=()
init_times=()
boundary_resolutions=()
boundary_apply_modes=()

# HRES: the ERA5 forecast itself, at each resolution.
for resol in "${resols[0]}"; do
    # The 1.5deg boundary lives in its own native low-res dataset.
    era5_dir="${era5_boundary_root}"
    if [ "${resol}" = "1.5" ]; then
        era5_dir="${era5_boundary_root_15}"
    fi

    preds_dirs+=("${era5_dir}")
    labels+=("HRES ${resol} ${bd_apply_mode}")
    init_times+=("${era5_init_time}")
    boundary_resolutions+=("${resol}")
    boundary_apply_modes+=("${bd_apply_mode}")
done

# Baseline model run (boundary_width=0).
preds_dirs+=("${preds_root}/era5_boundary0_${smooth}_${interp}_${bd_position}_res0.25_direct/preds")
labels+=("Baseline (boundary_width=0)")
init_times+=("${model_init_time}")
boundary_resolutions+=(0.25)
boundary_apply_modes+=("${bd_apply_mode}")

# Model rollouts driven by each boundary variant. The resolution is already baked into these
# prediction files, so the per-entry resolution is carried along only for readability.
for resol in "${resols[@]}"; do
    preds_dirs+=("${preds_root}/era5_boundary8_${smooth}_${interp}_${bd_position}_res${resol}_${bd_apply_mode}/preds")
    labels+=("Boundary 8 ${resol} ${bd_apply_mode}")
    init_times+=("${model_init_time}")
    boundary_resolutions+=("${resol}")
    boundary_apply_modes+=("${bd_apply_mode}")
done

# preds_dirs+=("${preds_root}/aurora_boundary8_no_nearest_backbone_res0.25_direct/preds")
# labels+=("Aurora boundary 8")
# init_times+=("${model_init_time}")
# boundary_resolutions+=(0.25)
# boundary_apply_modes+=("direct")

for var_name in "msl" "u10"; do
    echo "Drawing wavenumber spectrum + ratio comparisons for ${var_name}..."
    python plot_wavenumber.py \
        --var_name "${var_name}" \
        --crop_width 8 \
        --output_dir "${output_dir}" \
        --preds_dirs "${preds_dirs[@]}" \
        --labels "${labels[@]}" \
        --init_times "${init_times[@]}" \
        --boundary_resolutions "${boundary_resolutions[@]}" \
        --boundary_lowres_apply_modes "${boundary_apply_modes[@]}"
done

echo "Wavenumber spectrum + ratio comparisons generation complete!"
