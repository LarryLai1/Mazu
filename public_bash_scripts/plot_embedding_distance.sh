set -eo pipefail

if [[ $# -gt 0 ]]; then
    echo "$0 takes no arguments; edit the settings at the top of the script instead." >&2
    exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# --- Fixed run settings ----------------------------------------------------------------
start_time="2020-03-01 00:00:00"
end_time="2020-03-31 23:00:00"   # results are averaged over every init time in this range
rollout_step=169                 # hourly lead times 1..168
batch_size=8

boundary_source="hres"
boundary_smooth_mode="no"
boundary_smooth_width_adjustment=0
boundary_time_interp_mode="nearest"
replace_boundary_position="backbone"
boundary_resolution="0.25"
# Only consulted below 0.25deg; kept here because it is part of the run-config name.
boundary_lowres_apply_mode="direct"

# The two compared configurations: baseline first, then boundary replacement.
boundary_widths=(0 8)
config_labels=("baseline" "boundary replacement (w8)")

MODEL_CKPT_PATH="/tmp2/yuanlim0919/lateral_smooth/model_weights/Aurora/model.safetensors"
DATA_ROOT_DIR="/work/yunye0121/era5_tw"
BOUNDARY_ROOT_DIR="/tmp3/b12902101/hres_tw_forecast_0.25deg"
# Kept OUTSIDE Mazu/, next to the other embedding artefacts.
EMBEDDING_OUTPUT_ROOT="/tmp3/b12902101/mazu_embedding_output"

# Inference-progress checkpoints are always written by the python script. To resume a previous
# run that was killed mid-inference, invoke with RESUME_INFERENCE=1 (or true).
resume_args=()
if [[ "${RESUME_INFERENCE:-}" == "1" || "${RESUME_INFERENCE:-}" == "true" ]]; then
    resume_args=("--resume_inference")
fi

run_dirs=()

for i in "${!boundary_widths[@]}"; do
    boundary_width="${boundary_widths[$i]}"
    RUN_CONFIG_SUFFIX="${boundary_source}_boundary${boundary_width}_${replace_boundary_position}_res${boundary_resolution}_${boundary_time_interp_mode}"
    EMBEDDING_METRICS_DIR="${EMBEDDING_OUTPUT_ROOT}/embedding_distance/${RUN_CONFIG_SUFFIX}"
    mkdir -p "${EMBEDDING_METRICS_DIR}"
    run_dirs+=("${EMBEDDING_METRICS_DIR}")

    echo "=== embedding distance: ${config_labels[$i]} -- ${RUN_CONFIG_SUFFIX} (${start_time} .. ${end_time}, ${rollout_step} leads) ==="

    # --csv_output_folder / --gen_result_folder only hold the pixel-space MSE csv, the per-rank
    # metric handoff and the resume checkpoint; no predictions are written because
    # --save_rollout_step is never passed.
    time \
    python ./AuroraSmallTW_gen_eval_pipeline_with_embeddings.py \
        --data_root_dir "${DATA_ROOT_DIR}" \
        --boundary_root_dir "${BOUNDARY_ROOT_DIR}" \
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
        --rollout_step ${rollout_step} \
        --timestep_hours 1 \
        --boundary_width ${boundary_width} \
        --boundary_source ${boundary_source} \
        --boundary_smooth_mode "${boundary_smooth_mode}" \
        --boundary_smooth_width_adjustment ${boundary_smooth_width_adjustment} \
        --boundary_time_interp_mode "${boundary_time_interp_mode}" \
        --replace_boundary_position "${replace_boundary_position}" \
        --boundary_resolution "${boundary_resolution}" \
        --boundary_lowres_apply_mode "${boundary_lowres_apply_mode}" \
        --gpu_cache \
        --eval_metric MSE \
        --csv_output_folder "${EMBEDDING_METRICS_DIR}/run" \
        --gen_result_folder "${EMBEDDING_METRICS_DIR}/run" \
        --gpus "${CUDA_VISIBLE_DEVICES}" \
        --lazy_mode \
        --lazy_prefetch_steps 2 \
        --embedding_metrics_output_dir "${EMBEDDING_METRICS_DIR}" \
        --embedding_metrics_label "${config_labels[$i]}" \
        "${resume_args[@]}"

    echo "wrote ${EMBEDDING_METRICS_DIR}/embedding_distance.csv"
done

# --- HRES forecast baseline --------------------------------------------------------------
# The ECMWF HRES trajectory vs. the same ERA5 ground truth. There is no rollout to hook here, so
# it is computed separately: both sides go straight through utils.embedding.encode_batch (the
# pre-encoder + encoder + backbone encoder layers, stopping at the bottleneck), batched over init
# times on the GPU. Two encoder passes per (batch, lead) instead of a full forward, so this is
# far quicker than the rollouts above.
#
# Sampling mode for HRES's 6-hourly steps at these hourly leads. It follows the rollout runs by
# default so all three lines see the same HRES data. Set it to "interpolation" if you would rather
# the two history slots of the HRES window always differ: at "nearest", leads one hour apart
# frequently snap to the same 6-hourly forecast step, so the HRES window becomes [X, X] while the
# ERA5 window it is compared against holds two distinct states.
hres_baseline_time_interp_mode="${boundary_time_interp_mode}"

HRES_METRICS_DIR="${EMBEDDING_OUTPUT_ROOT}/embedding_distance/hres_forecast_res${boundary_resolution}_${hres_baseline_time_interp_mode}"
mkdir -p "${HRES_METRICS_DIR}"
run_dirs+=("${HRES_METRICS_DIR}")
config_labels+=("HRES forecast")

echo "=== embedding distance: HRES forecast -- $(basename "${HRES_METRICS_DIR}") (${start_time} .. ${end_time}, ${rollout_step} leads) ==="

time \
python ./compute_hres_embedding_distance.py \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --boundary_root_dir "${BOUNDARY_ROOT_DIR}" \
    --checkpoint_path "${MODEL_CKPT_PATH}" \
    --start_date_hour "${start_time}" \
    --end_date_hour "${end_time}" \
    --rollout_step ${rollout_step} \
    --lead_time 1 \
    --batch_size ${batch_size} \
    --surface_variables t2m u10 v10 msl \
    --upper_variables u v t q z \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 \
    --longitude 100 144.75 \
    --timestep_hours 1 \
    --boundary_source "${boundary_source}" \
    --boundary_time_interp_mode "${hres_baseline_time_interp_mode}" \
    --boundary_resolution "${boundary_resolution}" \
    --boundary_lowres_apply_mode "${boundary_lowres_apply_mode}" \
    --output_dir "${HRES_METRICS_DIR}" \
    --label "HRES forecast"

echo "wrote ${HRES_METRICS_DIR}/embedding_distance.csv"

# --- Overlay both configurations in one figure set --------------------------------------
# Each run above produced its own per-config CSV + PNGs; this draws them as one line per
# configuration, which is what the old plot_embedding_distance.py did in a single pass.
COMBINED_DIR="${EMBEDDING_OUTPUT_ROOT}/embedding_distance/combined"
echo "=== combining ${#run_dirs[@]} configuration(s) -> ${COMBINED_DIR} ==="
python ./combine_embedding_distance_csv.py \
    --inputs "${run_dirs[@]}" \
    --labels "${config_labels[@]}" \
    --output_dir "${COMBINED_DIR}" \
    --title "${start_time} .. ${end_time}, ${boundary_source} res${boundary_resolution} ${boundary_time_interp_mode}"
