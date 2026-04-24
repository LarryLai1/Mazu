#!/bin/bash

# your slurm option


export WANDB_API_KEY="your_wandb_api_key"
export WANDB_ENTITY="your_wandb_entity"
export WANDB_DIR="./wandb_logs"

set -euo pipefail

# activate your conda environment if needed

PROJECT="MazuTW"
NAME="${PROJECT}_seed1126_traindt2013010100-2018123123_valdt2022010100-2022123123_lt1_intw1_rs1_th1_epoch400_lr3e-5_trainbs32_valbs32"
OUTPUT_DIR="/work/yunye0121/${PROJECT}_training_results/${NAME}"

time \
accelerate launch --config_file public_slurm_scripts/accelerate_training_config.yaml \
    train_AuroraSmallTW.py \
    --data_root_dir "/home/yunye0121/era5_tw" \
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
    --epochs 400 \
    --lr 3e-5 \
    --weight_decay 1e-3 \
    --warmup_step_ratio 0.1 \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --num_workers 8 \
    --checkpointing_epochs 50 \
    --report_to wandb \
    --tracker_project_name "${PROJECT}" \
    --wandb_name "${NAME}" \
    --mixed_precision "no" \
