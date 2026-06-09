# root_dir="../LAM_output_2mo"
root_dir="../LAM_output"
var_dir="../LAM_output"
suffix="MSE.csv"
csv_paths=("${root_dir}/era5_boundary0_inject-inside_no_exact_encoder/${suffix}")
styles=("linestyle=-,linewidth=2")
legend_names=("Baseline")

# if doesn't exist directory era5_boundary_data, run eval_era5_boundary_forecast.py
if [ ! -d "${root_dir}/era5_boundary_data" ]; then
    python eval_era5_boundary_forecast.py \
        --boundary_root_dir "/tmp3/b12902101/era5_tw_forecast_3d" \
        --data_root_dir "/tmp3/yunye0121/era5_tw" \
        --start_date_hour "2020-08-01 00:00:00" \
        --end_date_hour "2020-09-01 00:00:00" \
        --csv_output_folder ${root_dir}/era5_boundary_data \
        --gpus 7
fi

csv_paths+=("${root_dir}/era5_boundary_data/${suffix}")
legend_names+=("ERA5 boundary")
styles+=("linestyle=-,linewidth=2")

for bd_position in "backbone" "encoder"; do
    # for interp in "exact"; do
    for interp in "nearest"; do
        # for smooth in "no"; do
        for smooth in "no" "mean" "gaussian"; do
            csv_paths+=("${var_dir}/era5_boundary8_inject-inside_${smooth}_${interp}_${bd_position}/${suffix}")
            legend_names+=("Boundary 8 ${interp} ${smooth} ${bd_position}")
            if [ "${bd_position}" == "encoder" ]; then
                styles+=("linestyle=--,linewidth=2")
            else
                styles+=("linestyle=:,linewidth=2")
            fi
        done
    done
done


echo csv_paths[@]: "${csv_paths[@]}"

python draw_error_plots.py \
    --csv_paths "${csv_paths[@]}" \
    --legend_names "${legend_names[@]}" \
    --styles "${styles[@]}" \
    --output_dir plots --ext png --dpi 150