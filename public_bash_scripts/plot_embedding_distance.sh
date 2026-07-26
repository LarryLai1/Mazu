#!/bin/bash
set -euo pipefail

# The ECMWF HRES forecast trajectory, added as a baseline line. It holds no <INIT>+<LEAD>hr.nc
# files, so plot_embedding_distance.py auto-detects it and serves it through
# BoundaryConditionDataset_HRES. Its 6-hourly forecast steps are sampled at the hourly leads
# below via --hres_time_interp_mode interpolation, and each init time is floored to the 12h HRES
# cycle to pick the forecast base time.
hres_forecast_dir="/tmp3/b12902101/hres_tw_forecast_0.25deg"

python plot_embedding_distance.py \
    --preds_dirs /tmp3/b12902101/LAM_output_preds/hres_boundary0_no_nearest_backbone_res0.25_direct/preds \
    /tmp3/b12902101/LAM_output_preds/hres_boundary8_no_nearest_backbone_res0.25_direct/preds "${hres_forecast_dir}" \
    --lead_times $(seq 1 168) \
    --init_times '2020-03-01 03:00:00' \
    --hres_time_interp_mode nearest
