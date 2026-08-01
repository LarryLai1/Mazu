#!/bin/bash
set -euo pipefail

# Same as val_script.sh, but points at the embedding-extraction wrapper
# (AuroraSmallTW_gen_eval_pipeline_with_embeddings.sh) instead of the plain rollout wrapper,
# so Swin3D bottleneck embeddings (flattened, pred + ground truth) get extracted alongside
# every prediction run below. Embeddings land under
# /tmp3/b12902101/mazu_embedding_output/embeddings/<run-config>/ (outside Mazu/), tagged per
# run config so plot_embedding_tsne_hooked.py can compare them afterwards.

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
GPU="4,5,6,7"

# Which rollout steps to extract embeddings for, passed through to every RUN_SCRIPT call
# below. Kept small: flattened bottleneck vectors are ~3.4 MB/sample/step. Override with
# EMBEDDING_SAVE_STEPS=... in the environment if you want a different set.
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
