root_dir="../LAM_output"
var_dir="../LAM_output"
suffix="MAE.csv"
colors=("#FF0000" "#006000" "#00EC00" "#C07AB8")
smooth_modes=("no" "mean" "gaussian" "linear")
csv_paths=("${root_dir}/era5_boundary0_inject-inside_no_exact_encoder/${suffix}")
styles=("linestyle=-,linewidth=2")
legend_names=("Baseline")
# csv_paths=()
# styles=()
# legend_names=()
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# if doesn't exist era5_boundary_data MAE csv, run eval_era5_boundary_forecast.py
if [ ! -f "${root_dir}/era5_boundary_data/MAE.csv" ]; then
    python eval_era5_boundary_forecast.py \
        --boundary_root_dir "/tmp3/b12902101/era5_tw_forecast_0.25deg" \
        --data_root_dir "/tmp3/yunye0121/era5_tw" \
        --start_date_hour "2020-01-02 00:00:00" \
        --end_date_hour "2020-12-21 00:00:00" \
        --batch_size 8 \
        --num_workers 1 \
        --csv_output_folder ${root_dir}/era5_boundary_data \
        --eval_metric "MAE" \
        --gpus "0"
fi

csv_paths+=("${root_dir}/era5_boundary_data/${suffix}")
legend_names+=("HRES")
styles+=("linestyle=-,linewidth=2")

# for bd_position in "backbone" "encoder"; do
for bd_position in "backbone"; do
    for interp in "nearest"; do
        # for index in {0,3}; do
        for index in 0; do
            smooth=${smooth_modes[${index}]}
            csv_paths+=("${var_dir}/era5_boundary8_inject-inside_${smooth}_${interp}_${bd_position}/${suffix}")
            legend_names+=("Boundary 8 ${interp} ${smooth} ${bd_position}")
            if [ "${bd_position}" == "encoder" ]; then
                styles+=("linestyle=--,linewidth=2,color=${colors[${index}]}")
            else
                styles+=("linestyle=:,linewidth=2,color=${colors[${index}]}")
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