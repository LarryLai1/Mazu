# root_dir="/tmp3/b12902101/Mazu/checkpoint-50"
root_dir="../LAM_output_archive"
suffix="MSE.csv"
csv_paths=("${root_dir}/era5_boundary0_inject-inside_smooth_no_interp_exact/${suffix}")
styles=("linestyle=-,linewidth=2")
legend_names=("Baseline")



for bw in 4 8; do
    # for interp in "exact"; do
    for interp in "nearest" "exact"; do
        # for smooth in "no"; do
        for smooth in "mean" "gaussian"; do
            csv_paths+=("${root_dir}/era5_boundary${bw}_inject-inside_smooth_${smooth}_interp_${interp}/${suffix}")
            legend_names+=("Boundary ${bw} ${interp} ${smooth}")
            if [ "${interp}" == "nearest" ]; then
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
    --output_dir plots --ext png --dpi 150 --zip