root_dir="../LAM_output_preds"
var_dir="${root_dir}"
suffix="MAE.csv"
colors=("#FF0000" "#006000" "#00EC00" "#C07AB8" "#8600FF")

# Inference results live under ${root_dir}/<checkpoint alias>/<config>/ (aliases defined in val_script*.sh).
# Checkpoints compared below (boundary 8, backbone injection), one color each.
ckpts=("baseline" "backbone_decoder" "input_full")
# Checkpoint whose boundary-0 (no boundary) run is drawn as the reference curve.
baseline_ckpt="baseline"

use_baseline=true
if [ "${use_baseline}" = true ]; then
    csv_paths=("${root_dir}/${baseline_ckpt}/hres_boundary0_no_nearest_backbone_res0.25_direct/${suffix}")
    styles=("linestyle=-,linewidth=2,color=#000000")
    legend_names=("${baseline_ckpt} Boundary 0")
else
    csv_paths=()
    styles=()
    legend_names=()
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

hres_boundary_root="/tmp3/b12902101/hres_tw_forecast_0.25deg"
hres_boundary_root_15="/tmp3/b12902101/hres_tw_forecast_1.5deg"

smooth="no"
bd_position="input"
# bd_position="backbone"
interp="nearest"
resols=(0.25)
# resols=(0.25 0.5 1.5)

# Cap the highest forecast lead time shown in the plots (hours). The CSVs still hold every
# lead time; this only trims the x-axis at draw time. Leave empty to plot all available hours.
max_lead_hours="168"

# HRES boundary forecast itself (no model), scored at each resolution in direct apply mode.
bd_apply_mode="direct"
for index in 0; do
# for index in "${!resols[@]}"; do
    resol="${resols[${index}]}"
    # The 1.5deg boundary lives in its own native low-res dataset.
    boundary_root_dir="${hres_boundary_root}"
    if [ "${resol}" = "1.5" ]; then
        boundary_root_dir="${hres_boundary_root_15}"
    fi
    bd_out_dir="${root_dir}/hres_boundary_data_res${resol}_${bd_apply_mode}"

    if [ ! -f "${bd_out_dir}/MAE.csv" ]; then
        python eval_hres_boundary_forecast.py \
            --boundary_root_dir "${boundary_root_dir}" \
            --data_root_dir "/tmp3/yunye0121/era5_tw" \
            --start_date_hour "2021-03-01 00:00:00" \
            --end_date_hour "2021-03-01 00:00:00" \
            --batch_size 4 \
            --num_workers 1 \
            --csv_output_folder "${bd_out_dir}" \
            --eval_metric "MAE" \
            --boundary_resolution "${resol}" \
            --boundary_lowres_apply_mode "${bd_apply_mode}" \
            --gpus "2"
    fi

    csv_paths+=("${bd_out_dir}/${suffix}")
    legend_names+=("HRES ${resol} ${bd_apply_mode}")
    styles+=("linestyle=-,linewidth=2,color=${colors[${index}]}")
done

# # Model rollouts driven by each boundary variant.
# for index in "${!resols[@]}"; do
#     for apply_mode in "direct" "interp"; do
#         resol="${resols[${index}]}"
#         if [ "${resol}" = 0.25 ] && [ "${apply_mode}" = "interp" ]; then
#             continue
#         fi
#         csv_paths+=("${var_dir}/hres_boundary8_${smooth}_${interp}_${bd_position}_res${resol}_${apply_mode}/${suffix}")
#         legend_names+=("Boundary 8 ${interp} ${smooth} ${bd_position} ${resol} ${apply_mode}")
#         if [ "${apply_mode}" == "interp" ]; then
#             styles+=("linestyle=--,linewidth=2,color=${colors[${index}]}")
#         else
#             styles+=("linestyle=:,linewidth=2,color=${colors[${index}]}")
#         fi
#     done
# done

bd_position="backbone"
resol=0.25
apply_mode="direct"
# Boundary-8 backbone rollouts, one per checkpoint (colors[0] is taken by the HRES curve).
for index in "${!ckpts[@]}"; do
    ckpt="${ckpts[${index}]}"
    csv_paths+=("${var_dir}/${ckpt}/hres_boundary8_${smooth}_${interp}_${bd_position}_res${resol}_${apply_mode}/${suffix}")
    legend_names+=("${ckpt} Boundary 8 ${bd_position}")
    styles+=("linestyle=:,linewidth=2,color=${colors[$((index + 1))]}")
done

echo csv_paths[@]: "${csv_paths[@]}"

max_hours_args=()
if [ -n "${max_lead_hours}" ]; then
    max_hours_args=(--max_hours "${max_lead_hours}")
fi

python draw_error_plots.py \
    --csv_paths "${csv_paths[@]}" \
    --legend_names "${legend_names[@]}" \
    --styles "${styles[@]}" \
    "${max_hours_args[@]}" \
    --output_dir plots --ext png --dpi 150