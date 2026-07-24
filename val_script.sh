#!/bin/bash
set -euo pipefail

# LINE notification failure handler
failure_handler() {
    local exit_code=$?
    local line_num=$1
    local detail_msg="val_script.sh failed at line ${line_num} with exit code ${exit_code}."
    if [[ -n "${smooth:-}" && -n "${interp:-}" && -n "${bd_position:-}" ]]; then
        detail_msg="${detail_msg} Parameters: boundary_smooth_mode=${smooth}, boundary_time_interp_mode=${interp}, replace_boundary_position=${bd_position}"
    fi
    python ~/notify_line.py "Aurora Inference Error" "${detail_msg}"
}
trap 'failure_handler $LINENO' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_custom_rollout.sh"

GPU="0,1,2,3"
# GPU="4"

interp="nearest"
smooth="no"
bd_position="backbone"

for resol in 0.5 1.5; do
    for apply_mode in "direct" "interp"; do
        echo "Starting boundary_width=8, boundary_smoothing=${smooth}, boundary_time_interp_mode=${interp} on GPU ${GPU}..."
        LOG_FILE="./bash_outputs/hres_custom_rollout_8_${smooth}_${interp}_${bd_position}.log"
        "${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width 8 \
            --boundary_smooth_mode "${smooth}" \
            --boundary_time_interp_mode "${interp}" --replace_boundary_position "${bd_position}" \
            --boundary_resolution "${resol}" --boundary_lowres_apply_mode "${apply_mode}" \
            --pred "true"
    done
done

for interp in "nearest"; do
    for smooth in "no"; do
        for bd_position in "backbone"; do
            echo "Starting boundary_width=8, boundary_smoothing=${smooth}, boundary_time_interp_mode=${interp} on GPU ${GPU}..."
            LOG_FILE="./bash_outputs/aurora_custom_rollout_8_${smooth}_${interp}_${bd_position}.log"
            "${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width 8 \
                --boundary_smooth_mode "${smooth}" \
                --boundary_time_interp_mode "${interp}" --replace_boundary_position "${bd_position}" \
                --boundary_resolution 0.25 --boundary_lowres_apply_mode "direct" \
                --pred "true"
        done
    done
done

"${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width 0 \
        --boundary_smooth_mode "no" \
        --boundary_time_interp_mode "nearest" \
        --replace_boundary_position "backbone" \
        --boundary_lowres_apply_mode "direct" \
        --pred "true"

TOTAL_TIME=$((SECONDS))
echo "All jobs completed in ${TOTAL_TIME}s."

python ~/notify_line.py "Aurora Inference" "Run Complete within ${TOTAL_TIME}s"
