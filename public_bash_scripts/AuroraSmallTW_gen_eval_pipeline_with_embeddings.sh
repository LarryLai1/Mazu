#!/bin/bash
# Single/multi-GPU inference script for AuroraTW weather model, WITH Swin3D bottleneck
# embedding extraction (flattened, pred + ground truth) piggybacked onto the same rollout.
#
# This is a copy of AuroraSmallTW_gen_eval_pipeline_custom_rollout.sh pointed at
# AuroraSmallTW_gen_eval_pipeline_with_embeddings.py instead of the original pipeline script,
# with --embedding_output_dir/--embedding_save_steps added. Everything else (data roots,
# boundary config, --lazy_mode, --gpus, ...) is unchanged, so predictions/.nc/.csv output
# stay exactly where the original script would put them; only the new embeddings/ output
# is added, and it is written OUTSIDE Mazu/, under /tmp3/b12902101/mazu_embedding_output/.
set -eo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID

usage() {
    echo "Usage: $0 [--gpus GPU_IDS] [--boundary_width N] [--pred true|false] [--embedding_save_steps \"1 6 24 72\"]" >&2
    echo "  --gpus GPU_IDS           CUDA_VISIBLE_DEVICES value (default: 0,1)" >&2
    echo "  --boundary_width N       Boundary width (default: 2)" >&2
    echo "  --boundary_resolution R  Boundary resolution: 0.25|0.5|1.5 (default: 0.25)" >&2
    echo "  --boundary_lowres_apply_mode M  direct|interp (default: interp)" >&2
    echo "  --pred VALUE             Enable prediction mode (true/false, default: false)" >&2
    echo "  --embedding_save_steps \"S1 S2 ...\"  Rollout steps to extract embeddings for (default: \"1 6 24 72\")" >&2
    echo "  --no_embeddings          Disable embedding extraction entirely (predictions only, like the original script)" >&2
    echo "  --embedding_metrics      Compute embedding distance (cos/L2 vs ERA5) at EVERY rollout step" >&2
    echo "                           on the fly and write a CSV + plots; stores no embeddings and" >&2
    echo "                           no .nc predictions. Implies --no_embeddings." >&2
}

CUDA_VISIBLE_DEVICES_VALUE="0,1"
boundary_width=2
boundary_smooth_mode="no"
boundary_smooth_width_adjustment=0
boundary_time_interp_mode="interpolation"
replace_boundary_position="encoder"
boundary_resolution="0.25"
boundary_lowres_apply_mode="interp"
pred="false"
embedding_save_steps="1 6 24 72"
enable_embeddings="true"
embedding_metrics="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pred)
            pred="$2"
            shift 2
            ;;
        --gpus)
            CUDA_VISIBLE_DEVICES_VALUE="$2"
            shift 2
            ;;
        --boundary_width)
            boundary_width="$2"
            shift 2
            ;;
        --boundary_smooth_mode)
            boundary_smooth_mode="$2"
            shift 2
            ;;
        --boundary_smooth_width_adjustment)
            boundary_smooth_width_adjustment="$2"
            shift 2
            ;;
        --boundary_time_interp_mode)
            boundary_time_interp_mode="$2"
            shift 2
            ;;
        --replace_boundary_position)
            replace_boundary_position="$2"
            shift 2
            ;;
        --boundary_resolution)
            boundary_resolution="$2"
            shift 2
            ;;
        --boundary_lowres_apply_mode)
            boundary_lowres_apply_mode="$2"
            shift 2
            ;;
        --embedding_save_steps)
            embedding_save_steps="$2"
            shift 2
            ;;
        --no_embeddings)
            enable_embeddings="false"
            shift 1
            ;;
        --embedding_metrics)
            embedding_metrics="true"
            # The metric mode exists precisely because the per-sample embeddings do not fit on
            # disk, so never dump them alongside it.
            enable_embeddings="false"
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

MODEL_CKPT_PATH="/tmp2/yuanlim0919/lateral_smooth/model_weights/Aurora/model.safetensors"

batch_size=8
if [[ "${pred}" == "true" ]]; then
    start_time="2020-03-01 00:00:00"
    end_time="2020-03-31 23:00:00"
    extra_args=("--save_rollout_step" 72)
    output_root="/tmp3/b12902101/LAM_output_preds"
else
    start_time="2020-03-10 00:00:00"
    end_time="2020-03-31 23:00:00"
    extra_args=()
    output_root="/tmp3/b12902101/LAM_output"
fi
aurora_boundary_root="/tmp3/b12902101/earth2/outputs"
hres_boundary_root="/tmp3/b12902101/hres_tw_forecast_0.25deg"
hres_boundary_root_15="/tmp3/b12902101/hres_tw_forecast_1.5deg"
ground_truth_root="/tmp3/yunye0121/era5_tw"

# boundary_source="ground_truth"
boundary_source="hres"
# boundary_source="aurora"

OUTPUT_FOLDER_NAME="${output_root}/${boundary_source}_boundary${boundary_width}_${boundary_smooth_mode}_${boundary_time_interp_mode}_${replace_boundary_position}_res${boundary_resolution}_${boundary_lowres_apply_mode}"

# Embedding output is kept OUTSIDE Mazu/, tagged with the same run-config suffix as
# OUTPUT_FOLDER_NAME above so multiple configs (e.g. baseline vs. boundary-replacement) can
# later be compared side by side in plot_embedding_tsne_hooked.py via --embeddings_dirs/--labels.
EMBEDDING_OUTPUT_ROOT="/tmp3/b12902101/mazu_embedding_output"
RUN_CONFIG_SUFFIX="${boundary_source}_boundary${boundary_width}_${replace_boundary_position}_res${boundary_resolution}_${boundary_lowres_apply_mode}"
EMBEDDING_OUTPUT_DIR="${EMBEDDING_OUTPUT_ROOT}/embeddings/${RUN_CONFIG_SUFFIX}"
# On-the-fly cos/L2-vs-ERA5 curves. Same run-config suffix, so several configs can be compared
# by pointing plot_embedding_distance.py-style plots at the sibling directories.
EMBEDDING_METRICS_DIR="${EMBEDDING_OUTPUT_ROOT}/embedding_distance/${RUN_CONFIG_SUFFIX}"

embedding_args=()
if [[ "${enable_embeddings}" == "true" ]]; then
    mkdir -p "${EMBEDDING_OUTPUT_DIR}"
    embedding_args=(
        "--embedding_output_dir" "${EMBEDDING_OUTPUT_DIR}"
        "--embedding_save_steps" ${embedding_save_steps}
    )
fi
if [[ "${embedding_metrics}" == "true" ]]; then
    mkdir -p "${EMBEDDING_METRICS_DIR}"
    embedding_args+=(
        "--embedding_metrics_output_dir" "${EMBEDDING_METRICS_DIR}"
        "--embedding_metrics_label" "${RUN_CONFIG_SUFFIX}"
    )
    # Metrics mode covers every rollout step by default and writes no predictions: drop any
    # --save_rollout_step so nothing lands on disk but the CSV and the plots.
    extra_args=()
fi

EXPERIMENT_ID=$(basename "$(dirname "$(dirname "$MODEL_CKPT_FOLDER")")")
CKPT_NAME=$(basename "$MODEL_CKPT_FOLDER")
LOG_FILE="./bash_outputs/${EXPERIMENT_ID}_${CKPT_NAME}.log"

if [[ "${boundary_source}" == "hres" ]]; then
    boundary_root_dir="${hres_boundary_root}"
    # Point at the native 1.5deg dataset when that resolution is requested.
    if [[ "${boundary_resolution}" == "1.5" ]]; then
        boundary_root_dir="${hres_boundary_root_15}"
    fi
elif [[ "${boundary_source}" == "ground_truth" ]]; then
    boundary_root_dir="${ground_truth_root}"
else
    boundary_root_dir="${aurora_boundary_root}"
fi

touch "${LOG_FILE}"

# Inference-progress checkpoints are always written by the python script. To resume a
# previous run that was killed mid-inference, invoke with RESUME_INFERENCE=1 (or true).
resume_args=()
if [[ "${RESUME_INFERENCE:-}" == "1" || "${RESUME_INFERENCE:-}" == "true" ]]; then
    resume_args=("--resume_inference")
fi

time \
python ./AuroraSmallTW_gen_eval_pipeline_with_embeddings.py \
    --data_root_dir /work/yunye0121/era5_tw \
    --boundary_root_dir "${boundary_root_dir}" \
    --checkpoint_path "${MODEL_CKPT_PATH}" \
    --batch_size ${batch_size} \
    --num_workers 1 \
    --seed 1126 \
    --start_date_hour "${start_time}" \
    --end_date_hour "${end_time}" \
    --surface_variables t2m u10 v10 msl \
    --upper_variables u v t q z \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 \
    --longitude 100 144.75 \
    --lead_time 1 \
    --input_time_window 2 \
    --rollout_step 169 \
    --timestep_hours 1 \
    --boundary_width ${boundary_width} \
    --boundary_source ${boundary_source} \
    --boundary_smooth_mode "${boundary_smooth_mode}" \
    --boundary_time_interp_mode "${boundary_time_interp_mode}" \
    --replace_boundary_position "${replace_boundary_position}" \
    --boundary_resolution "${boundary_resolution}" \
    --boundary_lowres_apply_mode "${boundary_lowres_apply_mode}" \
    --gpu_cache \
    --eval_metric MSE MAE \
    --csv_output_folder "${OUTPUT_FOLDER_NAME}" \
    --gen_result_folder "${OUTPUT_FOLDER_NAME}/preds" \
    --gpus "${CUDA_VISIBLE_DEVICES_VALUE}" \
    --lazy_mode \
    --lazy_prefetch_steps 2 \
    "${resume_args[@]}" \
    "${extra_args[@]}" \
    "${embedding_args[@]}" \
    # 2>&1 | tee "${LOG_FILE}"
