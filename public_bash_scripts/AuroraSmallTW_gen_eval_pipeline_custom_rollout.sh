#!/bin/bash
# Singe_GPU inference script for AuroraTW weather model.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3,4,5

MODEL_CKPT_FOLDER="/tmp3/b12902101/Mazu/checkpoint-50"
MODEL_CKPT_PATH="${MODEL_CKPT_FOLDER}/model.safetensors"

start_time="2016-12-01 00:00:00"
end_time="2016-12-02 23:00:00"
OUTPUT_FOLDER_NAME="ar_rs1_bs1_dt20161201-20161202_lt1_intw1"

EXPERIMENT_ID=$(basename "$(dirname "$(dirname "$MODEL_CKPT_FOLDER")")")
CKPT_NAME=$(basename "$MODEL_CKPT_FOLDER")
LOG_FILE="./bash_outputs/${EXPERIMENT_ID}_${CKPT_NAME}.log"

touch "${LOG_FILE}"

time \
python ./AuroraSmallTW_gen_eval_pipeline_custom_rollout.py \
    --data_root_dir /tmp3/yunye0121/era5_tw \
    --boundary_root_dir /tmp3/b12902101/era5_tw_forecast \
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
    --lead_time 6 \
    --input_time_window 1 \
    --rollout_step 1 \
    --save_rollout_step 1 \
    --timestep_hours 6 \
    --boundary_width 1 \
    --boundary_mode "inject-inside" \
    --mixed_precision 'no' \
    --eval_metric MSE MAE \
    --gen_result_folder "${MODEL_CKPT_FOLDER}/${OUTPUT_FOLDER_NAME}/preds" \
    --csv_output_folder "${MODEL_CKPT_FOLDER}/${OUTPUT_FOLDER_NAME}/errs" \
    2>&1 | tee "${LOG_FILE}" \
