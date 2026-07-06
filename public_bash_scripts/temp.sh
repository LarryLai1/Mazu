export CUDA_VISIBLE_DEVICES=5

MODEL_FOLDER="/tmp2/yuanlim0919/lateral_smooth/model_weights/Aurora/"
OUTPUT_FOLDER="/tmp3/b12902101/LAM_output_test/tmp"

python AuroraTW_gen_eval_pipeline.py \
    --data_root_dir /work/yunye0121/era5_tw \
    --checkpoint_path "${MODEL_FOLDER}/model.safetensors" \
    --batch_size 8 \
    --num_workers 1 \
    --seed 1126 \
    --start_date_hour "2020-07-01 00:00:00" \
    --end_date_hour "2020-07-04 23:00:00" \
    --surface_variables t2m u10 v10 msl \
    --upper_variables u v t q z \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 \
    --longitude 100 144.75 \
    --lead_time 1 \
    --rollout_step 72 \
    --timestep_hours 1 \
    --mixed_precision 'no' \
    --gen_result_folder "${OUTPUT_FOLDER}/preds" \
    --csv_output_path "${OUTPUT_FOLDER}/errs" \