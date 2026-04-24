#!/bin/bash

# your slurm header here

set -euo pipefail

# activate your conda environment if needed

MODEL_CKPT_FOLDER="/work/yunye0121/MazuTW_training_results/MazuTW_trainwithreplaybufferRAM_st400eps_seed1126_traindt2013010100-2018123123_valdt2022010100-2022123123_lt1_intw1_rs2_th1_epoch100_lr3e-5_trainbs16_valbs16_rbs200_fs1_rss200/ckpts/100-train_loss=0.02819251-val_loss=0.03063994"
MODEL_CKPT_PATH="${MODEL_CKPT_FOLDER}/model.safetensors"
OUTPUT_FOLDER_NAME="ar_bs1_dt2013010100-2018123123_lt1_intw1_rs1_th1"

time \
python ./AuroraSmallTW_gen_eval_pipeline_custom_rollout.py \
    --data_root_dir /home/yunye0121/era5_tw \
    --batch_size 1 \
    --num_workers 4 \
    --checkpoint_path "${MODEL_CKPT_PATH}" \
    --seed 1126 \
    --start_date_hour "2013-01-01 00:00:00" \
    --end_date_hour "2018-12-31 23:00:00" \
    --surface_variables t2m u10 v10 msl \
    --upper_variables u v t q z \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 \
    --longitude 100 144.75 \
    --lead_time 1 \
    --input_time_window 1 \
    --rollout_step 1 \
    --save_rollout_step 1 \
    --timestep_hours 1 \
    --mixed_precision 'no' \
    --eval_metric MSE MAE \
    --gen_result_folder "${MODEL_CKPT_FOLDER}/${OUTPUT_FOLDER_NAME}/preds" \
    --csv_output_folder "${MODEL_CKPT_FOLDER}/${OUTPUT_FOLDER_NAME}/errs" \
