#!/bin/bash
# Slurm version of train_AuroraSmallTW.sh (1 node, 8x H200).
# Submit from the repo root:
#   sbatch public_bash_scripts/train_AuroraSmallTW_slurm.sh [--checkpoint /path/to/weights] [options]
#SBATCH --job-name=Mazu_train
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
USE_SWIGLU_FFN=0
USE_ROPE_EMBEDDING=0
RANDOM_MLP=0
EPOCHS=${EPOCHS:-50}
# DataLoader workers per GPU process. Loading is bound by opening many small netCDF
# files per sample, so read them in parallel (8 GPUs x (1 + 4) procs fits 48 CPUs).
NUM_WORKERS=${NUM_WORKERS:-4}
# Local pretrained weights: .safetensors (safetensors.load_file) or official Aurora .ckpt
# (model.load_checkpoint_local). Defaults to the official AuroraSmall pretrained checkpoint.
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/work/b12902101/checkpoints/aurora/aurora-0.25-small-pretrained.ckpt}
# HRES +6h forecasts (6-hourly inits) used to replace the outer ring of the training/validation
# inputs, matching inference's input-space boundary replacement. Empty value disables it.
BOUNDARY_ROOT_DIR=${BOUNDARY_ROOT_DIR-/work/b12902101/hres_tw_forecast_0.25deg}
BOUNDARY_WIDTH=${BOUNDARY_WIDTH:-8}
BOUNDARY_SMOOTH_MODE=${BOUNDARY_SMOOTH_MODE:-no}
BOUNDARY_TIME_INTERP_MODE=${BOUNDARY_TIME_INTERP_MODE:-nearest}

ENV_PREFIX="/home/b12902101/micromamba/envs/AS"
DATA_ROOT_DIR="/work/b12902101/era5_tw"

print_usage() {
    cat <<'EOF'
Usage: sbatch public_bash_scripts/train_AuroraSmallTW_slurm.sh [options]

Options:
    --checkpoint PATH         Pretrained weights (.safetensors or .ckpt)
                              (default: official aurora-0.25-small-pretrained.ckpt)
    --use-muon                Enable Muon
    --use-swiglu              Enable SwiGLU FFN
    --use-rope                Enable RoPE embedding
    --random-mlp              Randomly init MLP blocks after loading
    --epochs N                Training epochs (default: 50)
    --boundary-root PATH      HRES +6h forecast root; the outer ring of every input step (train and
                              val) is replaced by the HRES forecast valid at that time. An empty
                              value disables replacement
                              (default: /tmp2/b12902101/hres_tw_forecast_0.25deg, or $BOUNDARY_ROOT_DIR)
    --boundary-width N        Ring width in grid cells (default: 8)
    --boundary-smooth-mode M  no|linear|mean|gaussian (default: no)
    --boundary-time-interp-mode M
                              nearest|interpolation: map input times onto the 6-hourly HRES marks
                              (default: nearest; ties go to the earlier mark)
    -h, --help               Show this help message

GPUs come from the #SBATCH --gres line; override at submit time with e.g.
    sbatch --gres=gpu:4 public_bash_scripts/train_AuroraSmallTW_slurm.sh ...
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint" >&2; exit 2; }
            CHECKPOINT_PATH="$2"
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
        --random-mlp)
            RANDOM_MLP=1
            shift
            ;;
        --epochs)
            [[ $# -ge 2 ]] || { echo "Missing value for --epochs" >&2; exit 2; }
            EPOCHS="$2"
            shift 2
            ;;
        --boundary-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --boundary-root" >&2; exit 2; }
            BOUNDARY_ROOT_DIR="$2"
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
TRAIN_SCRIPT="${REPO_ROOT}/reference_artifact/train_AuroraSmallTW_fast.py"
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Cannot find ${TRAIN_SCRIPT}; submit this job from the Mazu repo root." >&2
    exit 1
fi
cd "${REPO_ROOT}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "No pretrained weights given: pass --checkpoint /path/to/weights (.safetensors or .ckpt)." >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi
if [[ -n "${BOUNDARY_ROOT_DIR}" && ! -d "${BOUNDARY_ROOT_DIR}" ]]; then
    echo "Boundary root not found: ${BOUNDARY_ROOT_DIR} (pass --boundary-root '' to disable boundary replacement)." >&2
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
# The training script lives in reference_artifact/ but imports repo-root packages.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT="Boundary"
NAME_SUFFIX=()
if [[ "$USE_MUON" == "1" ]]; then
    NAME_SUFFIX+=("Muon")
fi
if [[ "$USE_SWIGLU_FFN" == "1" ]]; then
    NAME_SUFFIX+=("SwiGLU")
fi
if [[ "$USE_ROPE_EMBEDDING" == "1" ]]; then
    NAME_SUFFIX+=("RoPE")
fi
if [[ "$RANDOM_MLP" == "1" ]]; then
    NAME_SUFFIX+=("RandomMLP")
fi

if [[ ${#NAME_SUFFIX[@]} -gt 0 ]]; then
    NAME="${PROJECT}_$(IFS=+; echo "${NAME_SUFFIX[*]}")_epochs=${EPOCHS}"
else
    NAME="${PROJECT}_Reference_epochs=${EPOCHS}"
fi
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
if [[ "$USE_SWIGLU_FFN" == "1" ]]; then
    OPTIONAL_ARGS+=("--use_swiglu_ffn")
fi
if [[ "$USE_ROPE_EMBEDDING" == "1" ]]; then
    OPTIONAL_ARGS+=("--use_rope_embedding")
fi
if [[ "$RANDOM_MLP" == "1" ]]; then
    OPTIONAL_ARGS+=("--random-mlp")
fi
if [[ -n "${BOUNDARY_ROOT_DIR}" ]]; then
    OPTIONAL_ARGS+=(
        "--boundary_root_dir" "${BOUNDARY_ROOT_DIR}"
        "--boundary_width" "${BOUNDARY_WIDTH}"
        "--boundary_smooth_mode" "${BOUNDARY_SMOOTH_MODE}"
        "--boundary_time_interp_mode" "${BOUNDARY_TIME_INTERP_MODE}"
    )
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
    --rollout_step 1 \
    --timestep_hours 1 \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --epochs "${EPOCHS}" \
    --lr 3e-5 \
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
