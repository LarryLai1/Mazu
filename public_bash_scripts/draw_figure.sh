root_dir="../LAM_output"
var_dir="../LAM_output"
suffix="MAE.csv"
colors=("#FF0000" "#006000" "#00EC00" "#C07AB8" "#8600FF")

use_baseline=false
if [ "${use_baseline}" = true ]; then
    csv_paths=("${root_dir}/era5_boundary0_no_nearest_backbone_res0.25_direct/${suffix}")
    styles=("linestyle=-,linewidth=2")
    legend_names=("Baseline")
else
    csv_paths=()
    styles=()
    legend_names=()
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

era5_boundary_root="/tmp3/b12902101/era5_tw_forecast_0.25deg"
era5_boundary_root_15="/tmp3/b12902101/era5_tw_forecast_1.5deg"

smooth="no"
bd_position="backbone"
interp="nearest"
resols=(0.25 0.5 1.5)

# Cap the highest forecast lead time shown in the plots (hours). The CSVs still hold every
# lead time; this only trims the x-axis at draw time. Leave empty to plot all available hours.
max_lead_hours="72"

# ERA5 boundary forecast itself (no model), scored at each resolution in direct apply mode.
bd_apply_mode="direct"
for index in 0; do
# for index in "${!resols[@]}"; do
    resol="${resols[${index}]}"
    # The 1.5deg boundary lives in its own native low-res dataset.
    boundary_root_dir="${era5_boundary_root}"
    if [ "${resol}" = "1.5" ]; then
        boundary_root_dir="${era5_boundary_root_15}"
    fi
    bd_out_dir="${root_dir}/era5_boundary_data_res${resol}_${bd_apply_mode}"

    if [ ! -f "${bd_out_dir}/MAE.csv" ]; then
        python eval_era5_boundary_forecast.py \
            --boundary_root_dir "${boundary_root_dir}" \
            --data_root_dir "/tmp3/yunye0121/era5_tw" \
            --start_date_hour "2020-03-01 00:00:00" \
            --end_date_hour "2020-04-01 00:00:00" \
            --batch_size 8 \
            --num_workers 1 \
            --csv_output_folder "${bd_out_dir}" \
            --eval_metric "MAE" \
            --boundary_resolution "${resol}" \
            --boundary_lowres_apply_mode "${bd_apply_mode}" \
            --gpus "0"
    fi

    csv_paths+=("${bd_out_dir}/${suffix}")
    legend_names+=("HRES ${resol} ${bd_apply_mode}")
    styles+=("linestyle=-,linewidth=2,color=${colors[${index}]}")
done

# Model rollouts driven by each boundary variant.
for index in "${!resols[@]}"; do
    for apply_mode in "direct" "interp"; do
        resol="${resols[${index}]}"
        if [ "${resol}" = 0.25 ] && [ "${apply_mode}" = "interp" ]; then
            continue
        fi
        csv_paths+=("${var_dir}/era5_boundary8_${smooth}_${interp}_${bd_position}_res${resol}_${apply_mode}/${suffix}")
        legend_names+=("Boundary 8 ${interp} ${smooth} ${bd_position} ${resol} ${apply_mode}")
        # legend_names+=("Boundary 8 ${interp} ${smooth} ${bd_position}")
        if [ "${apply_mode}" == "interp" ]; then
            styles+=("linestyle=--,linewidth=2,color=${colors[${index}]}")
        else
            styles+=("linestyle=:,linewidth=2,color=${colors[${index}]}")
        fi
    done
done

csv_paths+=("${var_dir}/aurora_boundary8_no_nearest_backbone_res0.25_direct/${suffix}")
legend_names+=("Aurora boundary 8")
styles+=("linestyle=-,linewidth=2,color=${colors[4]}")

# Aurora boundary forecast itself (no model), on its native 0.25deg grid.
aurora_boundary_root="/tmp3/b12902101/earth2/outputs"
aurora_bd_out_dir="${root_dir}/aurora_boundary_data_res0.25_direct"
if [ ! -f "${aurora_bd_out_dir}/MAE.csv" ]; then
    python eval_era5_boundary_forecast.py \
        --boundary_root_dir "${aurora_boundary_root}" \
        --boundary_source "aurora" \
        --data_root_dir "/tmp3/yunye0121/era5_tw" \
        --start_date_hour "2020-03-01 00:00:00" \
        --end_date_hour "2020-04-01 00:00:00" \
        --batch_size 8 \
        --num_workers 1 \
        --csv_output_folder "${aurora_bd_out_dir}" \
        --eval_metric "MAE" \
        --gpus "0"
fi
csv_paths+=("${aurora_bd_out_dir}/${suffix}")
legend_names+=("Aurora 0.25 direct")
styles+=("linestyle=-,linewidth=2,color=${colors[3]}")

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