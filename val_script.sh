#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_custom_rollout.sh"
BOUNDARY_MODE="inject-inside"

# BOUNDARY_WIDTHS=(2 6)
# GPU="0,1,2,3"
# GPU="0"
GPU="0,1,2,3,4,5"

pooling="no"
for interp in "nearest"; do
    for smooth in "no" "mean" "gaussian"; do
        for bd_position in "backbone" "encoder"; do
            echo "Starting boundary_width=8, boundary_smoothing=${smooth}, boundary_time_interp_mode=${interp} on GPU ${GPU}..."
            "${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width 8 --boundary_mode "${BOUNDARY_MODE}" \
                --boundary_pooling "${pooling}" --boundary_smooth_mode "${smooth}" \
                --boundary_time_interp_mode "${interp}" --replace_boundary_position "${bd_position}"
        done
    done
done

"${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width 0 --boundary_mode "inject-inside" \
        --boundary_pooling no --boundary_smooth_mode "no" \
        --boundary_time_interp_mode exact

TOTAL_TIME=$((SECONDS))
echo "All jobs completed in ${TOTAL_TIME}s."

python ~/notify_line.py "Aurora Inference" "Run Complete within ${TOTAL_TIME}s"