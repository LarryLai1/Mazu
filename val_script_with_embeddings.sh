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
GPU="2"

EMBEDDING_SAVE_STEPS="${EMBEDDING_SAVE_STEPS:-1 6 24 72 120 168}"

interp="nearest"
smooth="no"
# bd_position="input"
# bd_position="backbone"
resol=0.25

# Checkpoint alias -> model.safetensors. The alias becomes the second-level output folder
# (LAM_output*/<alias>/<config>, finetuned_embedding_output/*/<alias>/<config>).
declare -A CKPT_PATHS=(
    [baseline]="/tmp2/yuanlim0919/lateral_smooth/model_weights/Aurora/model.safetensors"
    [backbone_decoder]="/tmp3/b12902101/train/50-train_loss=0.02831841-val_loss=0.03047114/model.safetensors"
    [input_full]="/tmp3/b12902101/train/full_finetune_input_200epochs/model.safetensors"
)
# Checkpoints to run; every config below is run once per checkpoint.
CKPTS=("baseline" "backbone_decoder" "input_full")

for ckpt in "${CKPTS[@]}"; do
ckpt_args=(--checkpoint_name "${ckpt}" --checkpoint_path "${CKPT_PATHS[${ckpt}]}")

for bd_position in "input" "backbone"; do
    for apply_mode in "direct"; do
        echo "Starting ckpt=${ckpt}, boundary_width=8, boundary_smoothing=${smooth}, boundary_time_interp_mode=${interp} on GPU ${GPU}..."
        LOG_FILE="./bash_outputs/hres_custom_rollout_8_${smooth}_${interp}_${bd_position}_embeddings.log"
        "${RUN_SCRIPT}" --gpus "${GPU}" "${ckpt_args[@]}" --boundary_width 8 \
            --boundary_smooth_mode "${smooth}" \
            --boundary_time_interp_mode "${interp}" --replace_boundary_position "${bd_position}" \
            --boundary_resolution "${resol}" --boundary_lowres_apply_mode "${apply_mode}" \
            --embedding_save_steps "${EMBEDDING_SAVE_STEPS}" \
            --pred true
    done
done

"${RUN_SCRIPT}" --gpus "${GPU}" "${ckpt_args[@]}" --boundary_width 0 \
        --boundary_smooth_mode "no" \
        --boundary_time_interp_mode "nearest" \
        --replace_boundary_position "backbone" \
        --boundary_lowres_apply_mode "direct" \
        --embedding_save_steps "${EMBEDDING_SAVE_STEPS}" \
        --pred true

done

TOTAL_TIME=$((SECONDS))
echo "All jobs completed in ${TOTAL_TIME}s."

python ~/notify_line.py "Aurora Inference" "Run Complete within ${TOTAL_TIME}s"
