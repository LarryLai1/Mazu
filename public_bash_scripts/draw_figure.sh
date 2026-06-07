# root_dir="../LAM_output_2mo"
root_dir="../LAM_output"
var_dir="../LAM_output"
suffix="MSE.csv"
csv_paths=("${root_dir}/era5_boundary0_inject-inside_no_exact_encoder/${suffix}")
styles=("linestyle=-,linewidth=2")
legend_names=("Baseline")



for bd_position in "backbone" "encoder"; do
    # for interp in "exact"; do
    for interp in "nearest"; do
        # for smooth in "no"; do
        for smooth in "no" "mean" "gaussian"; do
            csv_paths+=("${var_dir}/era5_boundary8_inject-inside_${smooth}_${interp}_${bd_position}/${suffix}")
            legend_names+=("Boundary ${bw} ${interp} ${smooth} ${bd_position}")
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