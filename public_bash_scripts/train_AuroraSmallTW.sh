#!/bin/bash

set -euo pipefail

DEFAULT_GPUS="5,6"
USE_MUON=0
USE_SWIGLU_FFN=0
USE_ROPE_EMBEDDING=0
CUDA_VISIBLE_DEVICES="${DEFAULT_GPUS}"

print_usage() {
    cat <<'EOF'
Usage: train_AuroraSmallTW.sh [options]

Options:
    --gpus LIST               CUDA_VISIBLE_DEVICES value, e.g. 0,1 or 5,6
    --use-muon                Enable Muon
    --use-swiglu              Enable SwiGLU FFN
    --use-rope                Enable RoPE embedding
    -h, --help                Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            [[ $# -ge 2 ]] || { echo "Missing value for --gpus" >&2; exit 2; }
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --use-muon)
            USE_MUON=1
            shift
            ;;
        --use-swiglu)
            USE_SWIGLU_FFN=1
            shift
            ;;
        --use-rope)
            USE_ROPE_EMBEDDING=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage >&2
            exit 2
            ;;
    esac
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=1,2,3,4

export WANDB_API_KEY="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"
export WANDB_ENTITY="noiselarry1234-taiwan"
export WANDB_DIR="./wandb_logs"

# Boolean toggles for optional model features (1: enable, 0: disable)

PROJECT="Mazu"
# NAME="${PROJECT}-epochs=400-traindt=20130101--20181231-valdt=2022-intw=1-rs=1-sd=1126-lr=3e-5-bs=8"
NAME="${PROJECT}-MUON:${USE_MUON}_SWIGLU:${USE_SWIGLU_FFN}_ROPE:${USE_ROPE_EMBEDDING}-epochs=50"
OUTPUT_DIR="./${PROJECT}_training_results/${NAME}"

# Derive worker/process count from CUDA_VISIBLE_DEVICES.
if [[ -z "${CUDA_VISIBLE_DEVICES}" || "${CUDA_VISIBLE_DEVICES}" == "all" ]]; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
else
    GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
fi

# Keep a safe minimum of 1 in case detection fails.
if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${GPU_COUNT}" -lt 1 ]]; then
    GPU_COUNT=1
fi


OPTIONAL_ARGS=()
if [[ "$USE_MUON" == "1" ]]; then
    OPTIONAL_ARGS+=("--use_muon")
fi
if [[ "$USE_SWIGLU_FFN" == "1" ]]; then
    OPTIONAL_ARGS+=("--use_swiglu_ffn")
fi
if [[ "$USE_ROPE_EMBEDDING" == "1" ]]; then
    OPTIONAL_ARGS+=("--use_rope_embedding")
fi

time \
accelerate launch --config_file ./public_bash_scripts/accelerate_training_config.yaml \
    --num_processes "${GPU_COUNT}" \
    ./train_AuroraSmallTW_otter_test.py \
    --data_root_dir "/tmp2/yunye0121/era5_tw" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 1126 \
    --train_start_date_hour "2013-01-01 00:00:00" \
    --train_end_date_hour "2018-12-31 23:00:00" \
    --val_start_date_hour "2022-01-01 00:00:00" \
    --val_end_date_hour "2022-12-31 23:00:00" \
    --surface_variables t2m u10 v10 msl \
    --upper_variables u v t q z \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 \
    --longitude 100 144.75 \
    --lead_time 1 \
    --input_time_window 1 \
    --rollout_step 1 \
    --timestep_hours 1 \
    --use_pretrained_weight \
    --epochs 50 \
    --lr 3e-5 \
    --weight_decay 1e-3 \
    --warmup_step_ratio 0.1 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --num_workers "${GPU_COUNT}"  \
    --checkpointing_epochs 25 \
    --report_to wandb \
    --tracker_project_name "${PROJECT}" \
    --wandb_name "${NAME}" \
    "${OPTIONAL_ARGS[@]}" \
    --mixed_precision "no"
