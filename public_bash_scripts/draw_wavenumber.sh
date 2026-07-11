#!/bin/bash
set -euo pipefail

output_dir="wavenumber_plots_output"

echo "Drawing wavenumber comparisons for msl..."
python plot_wavenumber.py \
    --var_name msl \
    --crop_width 8 \
    --output_dir "${output_dir}" \
    --preds_dirs \
        "/tmp3/b12902101/era5_tw_forecast_3d" \
        "/tmp3/b12902101/LAM_output_preds/era5_boundary0_inject-inside_no_exact_encoder/preds" \
        "/tmp3/b12902101/LAM_output_preds/era5_boundary8_inject-inside_no_nearest_backbone/preds/" \
    --labels \
        "HRES" \
        "Baseline (boundary_width=0)" \
        "Boundary 8 (nearest, no, backbone)" \
    --init_times \
        "2020-03-01 00:00:00" \
        "2020-03-01 01:00:00" \
        "2020-03-01 01:00:00"

echo "Drawing wavenumber comparisons for u10..."
python plot_wavenumber.py \
    --var_name u10 \
    --crop_width 8 \
    --output_dir "${output_dir}" \
    --preds_dirs \
        "/tmp3/b12902101/era5_tw_forecast_3d" \
        "/tmp3/b12902101/LAM_output_preds/era5_boundary0_inject-inside_no_exact_encoder/preds" \
        "/tmp3/b12902101/LAM_output_preds/era5_boundary8_inject-inside_no_nearest_backbone/preds/" \
    --labels \
        "HRES" \
        "Baseline (boundary_width=0)" \
        "Boundary 8 (nearest, no, backbone)" \
    --init_times \
        "2020-03-01 00:00:00" \
        "2020-03-01 01:00:00" \
        "2020-03-01 01:00:00"

echo "Wavenumber comparisons generation complete!"
