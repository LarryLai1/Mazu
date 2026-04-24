#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_API_KEY="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"
export WANDB_ENTITY="noiselarry1234-taiwan"
export WANDB_DIR="./wandb_logs"

PROJECT="Mazu"
NAME="${PROJECT}-epochs=400-traindt=20130101--20181231-valdt=2022-intw=1-rs=1-sd=1126-lr=3e-5-bs=8"
OUTPUT_DIR="./${PROJECT}_training_results/${NAME}"

time \
accelerate launch --config_file ./public_bash_scripts/accelerate_training_config.yaml \
    ./train_AuroraSmallTW.py \
    --data_root_dir "/work/yunye0121/era5_tw" \
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
    --epochs 1 \
    --lr 3e-5 \
    --weight_decay 1e-3 \
    --warmup_step_ratio 0.1 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --num_workers 4 \
    --checkpointing_epochs 50 \
    --report_to wandb \
    --tracker_project_name "${PROJECT}" \
    --wandb_name "${NAME}" \
    --mixed_precision "no" \
