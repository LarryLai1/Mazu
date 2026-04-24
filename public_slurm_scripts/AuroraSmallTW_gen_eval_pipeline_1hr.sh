#!/bin/bash

# your slurm header here

set -euo pipefail


# Loading environment, if needed


MODEL_CKPT_FOLDER="/work/yunye0121/MazuTW_training_results/MazuTW_trainwithselfpred400eps_seed1126_traindt2013010101-2018123123_valdt2022010101-2022123123_lt1_intw1_rs1_th1_epoch100_lr3e-5_trainbs32_valbs32/ckpts/checkpoint-50"
MODEL_CKPT_PATH="${MODEL_CKPT_FOLDER}/model.safetensors"
OUTPUT_FOLDER_NAME="ar_bs1_dt2023010100-2023013123_lt1_intw1_rs168_th1"

time \
python ./AuroraSmallTW_gen_eval_pipeline.py \
    --data_root_dir /home/yunye0121/era5_tw \
    --checkpoint_path "${MODEL_CKPT_PATH}" \
    --batch_size 1 \
    --num_workers 8 \
    --seed 1126 \
    --start_date_hour "2023-01-01 00:00:00" \
    --end_date_hour "2023-01-31 23:00:00" \
    --surface_variables t2m u10 v10 msl \
    --upper_variables u v t q z \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 \
    --longitude 100 144.75 \
    --lead_time 1 \
    --input_time_window 1 \
    --rollout_step 240 \
    --save_rollout_step 24 72 120 168 240 \
    --timestep_hours 1 \
    --mixed_precision 'no' \
    --eval_metric MSE MAE \
    --gen_result_folder "${MODEL_CKPT_FOLDER}/${OUTPUT_FOLDER_NAME}/preds" \
    --csv_output_folder "${MODEL_CKPT_FOLDER}/${OUTPUT_FOLDER_NAME}/errs" \
