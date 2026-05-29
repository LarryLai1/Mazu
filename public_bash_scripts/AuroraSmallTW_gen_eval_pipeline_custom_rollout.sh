#!/bin/bash
# Singe_GPU inference script for AuroraTW weather model.

export CUDA_DEVICE_ORDER=PCI_BUS_ID

usage() {
    echo "Usage: $0 [--gpus GPU_IDS] [--boundary_width N] [--boundary_mode MODE]" >&2
    echo "  --gpus GPU_IDS           CUDA_VISIBLE_DEVICES value (default: 3,4,5)" >&2
    echo "  --boundary_width N       Boundary width (default: 2)" >&2
    echo "  --boundary_mode MODE     Boundary mode (default: inject-inside)" >&2
    echo "  --boundary_pooling MODE  Boundary pooling then reshape (default: no)" >&2
    echo "  --boundary_use_cache     Preload boundary data into memory (default: off)" >&2
}

CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3,4,5,6,7"
boundary_width=2
boundary_mode="inject-inside"
# boundary_pooling="yes"
boundary_pooling="no"
boundary_use_cache=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            CUDA_VISIBLE_DEVICES_VALUE="$2"
            shift 2
            ;;
        --boundary_width)
            boundary_width="$2"
            shift 2
            ;;
        --boundary_mode)
            boundary_mode="$2"
            shift 2
            ;;
        --boundary_pooling)
            boundary_pooling="$2"
            shift 2
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

MODEL_CKPT_FOLDER="/tmp3/b12902101/Mazu/checkpoint-50"
MODEL_CKPT_PATH="${MODEL_CKPT_FOLDER}/model.safetensors"

start_time="2020-07-01 00:00:00"
# end_time="2020-07-04 23:00:00"
end_time="2020-08-31 23:00:00"
aurora_boundary_root="/tmp3/b12902101/earth2/outputs"
era5_boundary_root="/tmp3/b12902101/era5_tw_forecast_3d"

boundary_source="era5"
# boundary_source="aurora"

OUTPUT_FOLDER_NAME="/home/LarryLai/LAM_output/${boundary_source}_boundary${boundary_width}_${boundary_mode}_pooling${boundary_pooling}"

EXPERIMENT_ID=$(basename "$(dirname "$(dirname "$MODEL_CKPT_FOLDER")")")
CKPT_NAME=$(basename "$MODEL_CKPT_FOLDER")
LOG_FILE="./bash_outputs/${EXPERIMENT_ID}_${CKPT_NAME}.log"

if [[ "${boundary_source}" == "era5" ]]; then
    boundary_root_dir="${era5_boundary_root}"
else
    boundary_root_dir="${aurora_boundary_root}"
fi

touch "${LOG_FILE}"

time \
python ./AuroraSmallTW_gen_eval_pipeline_custom_rollout.py \
    --data_root_dir /tmp3/yunye0121/era5_tw \
    --boundary_root_dir "${boundary_root_dir}" \
    --checkpoint_path "${MODEL_CKPT_PATH}" \
    --batch_size 8 \
    --num_workers 4 \
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
    --input_time_window 1 \
    --rollout_step 72 \
    --timestep_hours 1 \
    --boundary_width ${boundary_width} \
    --boundary_mode ${boundary_mode} \
    --boundary_pooling ${boundary_pooling} \
    --boundary_source ${boundary_source} \
    --gpu_cache \
    --eval_metric MSE MAE \
    --csv_output_folder "${OUTPUT_FOLDER_NAME}" \
    --gen_result_folder "${OUTPUT_FOLDER_NAME}/preds" \
    --gpus "${CUDA_VISIBLE_DEVICES_VALUE}" \
    2>&1 | tee "${LOG_FILE}" \

# --save_rollout_step 1 2 4 8 12 24 36 48 60 72 \