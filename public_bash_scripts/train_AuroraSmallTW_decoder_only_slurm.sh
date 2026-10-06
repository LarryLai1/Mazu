#!/bin/bash
# Decoder-only finetuning (1 node, 8x H200): encoder + backbone frozen, HRES boundary injected
# as in inference (--replace_boundary_position backbone by default), only the decoder is trained.
# Submit from the repo root:
#   sbatch public_bash_scripts/train_AuroraSmallTW_decoder_only_slurm.sh [--checkpoint /path/to/weights] [options]
# Saved checkpoints are full-model model.safetensors, usable directly as inference weights.
#SBATCH --job-name=Mazu_train_decoder
#SBATCH --account=mst115137
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_results/%x_%j.out

set -euo pipefail

USE_MUON=0
EPOCHS=${EPOCHS:-50}
LR=${LR:-3e-5}
# DataLoader workers per GPU process. Loading is bound by opening many small netCDF
# files per sample, so read them in parallel (8 GPUs x (1 + 4) procs fits 48 CPUs).
NUM_WORKERS=${NUM_WORKERS:-4}
# The weights to finetune: the full AuroraSmallTW model inference runs with (.safetensors).
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/work/b12902101/checkpoints/aurora/model.safetensors}
# HRES forecasts, 00/12Z inits with +0/+6/+12h leads per file.
BOUNDARY_ROOT_DIR=${BOUNDARY_ROOT_DIR:-/work/b12902101/hres_0.25_12h}
REPLACE_BOUNDARY_POSITION=${REPLACE_BOUNDARY_POSITION:-backbone}
BOUNDARY_WIDTH=${BOUNDARY_WIDTH:-8}
BOUNDARY_SMOOTH_MODE=${BOUNDARY_SMOOTH_MODE:-no}
BOUNDARY_TIME_INTERP_MODE=${BOUNDARY_TIME_INTERP_MODE:-interpolation}

ENV_PREFIX="/home/b12902101/micromamba/envs/AS"
DATA_ROOT_DIR="/work/b12902101/era5_tw"

print_usage() {
    cat <<'EOF'
Usage: sbatch public_bash_scripts/train_AuroraSmallTW_decoder_only_slurm.sh [options]

Options:
    --checkpoint PATH         Full-model weights to finetune (.safetensors or .ckpt)
                              (default: /work/b12902101/checkpoints/aurora/model.safetensors)
    --boundary-root PATH      HRES root (default: /work/b12902101/hres_0.25_12h)
    --replace-boundary-position "P ..."
                              backbone and/or input (default: backbone)
    --boundary-width N        Ring width in grid cells, >= 4 for backbone (default: 8)
    --boundary-smooth-mode M  no|linear|mean|gaussian (default: no)
    --boundary-time-interp-mode M
                              interpolation|nearest across the HRES forecast leads, as in inference
                              (default: interpolation)
    --use-muon                Enable Muon
    --epochs N                Training epochs (default: 50)
    --lr LR                   Learning rate (default: 3e-5)
    -h, --help                Show this help message

GPUs come from the #SBATCH --gres line; override at submit time with e.g.
    sbatch --gres=gpu:4 public_bash_scripts/train_AuroraSmallTW_decoder_only_slurm.sh ...
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint" >&2; exit 2; }
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --boundary-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --boundary-root" >&2; exit 2; }
            BOUNDARY_ROOT_DIR="$2"
            shift 2
            ;;
        --replace-boundary-position)
            [[ $# -ge 2 ]] || { echo "Missing value for --replace-boundary-position" >&2; exit 2; }
            REPLACE_BOUNDARY_POSITION="$2"
            shift 2
            ;;
        --boundary-width)
            [[ $# -ge 2 ]] || { echo "Missing value for --boundary-width" >&2; exit 2; }
            BOUNDARY_WIDTH="$2"
            shift 2
            ;;
        --boundary-smooth-mode)
            [[ $# -ge 2 ]] || { echo "Missing value for --boundary-smooth-mode" >&2; exit 2; }
            BOUNDARY_SMOOTH_MODE="$2"
            shift 2
            ;;
        --boundary-time-interp-mode)
            [[ $# -ge 2 ]] || { echo "Missing value for --boundary-time-interp-mode" >&2; exit 2; }
            BOUNDARY_TIME_INTERP_MODE="$2"
            shift 2
            ;;
        --use-muon)
            USE_MUON=1
            shift
            ;;
        --epochs)
            [[ $# -ge 2 ]] || { echo "Missing value for --epochs" >&2; exit 2; }
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            [[ $# -ge 2 ]] || { echo "Missing value for --lr" >&2; exit 2; }
            LR="$2"
            shift 2
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

# Slurm runs a copy of this script, so locate the repo via the submit directory.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
TRAIN_SCRIPT="${REPO_ROOT}/train_AuroraSmallTW_decoder_only.py"
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Cannot find ${TRAIN_SCRIPT}; submit this job from the Mazu repo root." >&2
    exit 1
fi
cd "${REPO_ROOT}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi
if [[ ! -d "${BOUNDARY_ROOT_DIR}" ]]; then
    echo "Boundary root not found: ${BOUNDARY_ROOT_DIR}" >&2
    exit 1
fi

# WandB credentials are read from WANDB_API_KEY or ~/.netrc (run `wandb login` once).
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" "${HOME}/.netrc"; then
    echo "No WandB credentials: run 'wandb login' or export WANDB_API_KEY before sbatch." >&2
    exit 1
fi
export WANDB_ENTITY="noiselarry1234-taiwan"
export WANDB_DIR="./wandb_logs"

# Use the micromamba env directly (no shell hook needed in a batch job).
export PATH="${ENV_PREFIX}/bin:${PATH}"
# Repo-root packages (aurora, datasets, utils) must be importable.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT="Boundary"
NAME="${PROJECT}_DecoderOnly_$(echo ${REPLACE_BOUNDARY_POSITION} | tr ' ' '+')_bw${BOUNDARY_WIDTH}"
if [[ "$USE_MUON" == "1" ]]; then
    NAME="${NAME}_Muon"
fi
NAME="${NAME}_epochs=${EPOCHS}"
OUTPUT_DIR="./${PROJECT}_training_results/${NAME}"

# Slurm sets CUDA_VISIBLE_DEVICES to the allocated GPUs.
GPU_COUNT="${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${GPU_COUNT}" -lt 1 ]]; then
    GPU_COUNT=1
fi

OPTIONAL_ARGS=()
if [[ "$USE_MUON" == "1" ]]; then
    OPTIONAL_ARGS+=("--use_muon")
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname): ${GPU_COUNT} GPU(s) [${CUDA_VISIBLE_DEVICES:-unset}], run ${NAME}"

time \
accelerate launch --config_file ./public_bash_scripts/accelerate_training_config.yaml \
    --num_processes "${GPU_COUNT}" \
    "${TRAIN_SCRIPT}" \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 1126 \
    --train_start_date_hour "2016-01-01 00:00:00" \
    --train_end_date_hour "2020-12-31 23:00:00" \
    --val_start_date_hour "2022-01-01 00:00:00" \
    --val_end_date_hour "2022-12-31 23:00:00" \
    --surface_variables t2m u10 v10 msl \
    --upper_variables u v t q z \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 \
    --longitude 100 144.75 \
    --lead_time 1 \
    --input_time_window 2 \
    --timestep_hours 1 \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --boundary_root_dir "${BOUNDARY_ROOT_DIR}" \
    --replace_boundary_position ${REPLACE_BOUNDARY_POSITION} \
    --boundary_width "${BOUNDARY_WIDTH}" \
    --boundary_smooth_mode "${BOUNDARY_SMOOTH_MODE}" \
    --boundary_time_interp_mode "${BOUNDARY_TIME_INTERP_MODE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay 1e-3 \
    --warmup_step_ratio 0.1 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --num_workers "${NUM_WORKERS}" \
    --checkpointing_epochs 25 \
    --report_to wandb \
    --tracker_project_name "${PROJECT}" \
    --wandb_name "${NAME}" \
    "${OPTIONAL_ARGS[@]}" \
    --mixed_precision "no"
