#!/bin/bash
set -euo pipefail

# LINE notification failure handler
failure_handler() {
    local exit_code=$?
    local line_num=$1
    local detail_msg="val_script_with_embeddings.sh failed at line ${line_num} with exit code ${exit_code}."
    if [[ -n "${smooth:-}" && -n "${interp:-}" && -n "${bd_position:-}" ]]; then
        detail_msg="${detail_msg} Parameters: boundary_smooth_mode=${smooth}, boundary_time_interp_mode=${interp}, replace_boundary_position=${bd_position}"
    fi
    python ~/notify_line.py "Aurora Inference Error" "${detail_msg}"
}
trap 'failure_handler $LINENO' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_with_embeddings.sh"

# GPU="0,1,2,3"
GPU="4,6,7"

EMBEDDING_SAVE_STEPS="${EMBEDDING_SAVE_STEPS:-1 6 24 72 120 168}"

interp="nearest"
smooth="no"
bd_position="backbone"

for resol in 0.25; do
    for apply_mode in "direct"; do
        echo "Starting boundary_width=8, boundary_smoothing=${smooth}, boundary_time_interp_mode=${interp} on GPU ${GPU}..."
        LOG_FILE="./bash_outputs/hres_custom_rollout_8_${smooth}_${interp}_${bd_position}_embeddings.log"
        "${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width 8 \
            --boundary_smooth_mode "${smooth}" \
            --boundary_time_interp_mode "${interp}" --replace_boundary_position "${bd_position}" \
            --boundary_resolution "${resol}" --boundary_lowres_apply_mode "${apply_mode}" \
            --embedding_save_steps "${EMBEDDING_SAVE_STEPS}"
    done
done

"${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width 0 \
        --boundary_smooth_mode "no" \
        --boundary_time_interp_mode "nearest" \
        --replace_boundary_position "backbone" \
        --boundary_lowres_apply_mode "direct" \
        --embedding_save_steps "${EMBEDDING_SAVE_STEPS}"

TOTAL_TIME=$((SECONDS))
echo "All jobs completed in ${TOTAL_TIME}s."

python ~/notify_line.py "Aurora Inference" "Run Complete within ${TOTAL_TIME}s"
