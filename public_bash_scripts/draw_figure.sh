# root_dir="/tmp3/b12902101/Mazu/checkpoint-50"
root_dir="~/LAM_output"
suffix="MSE.csv"
csv_paths=(
    "${root_dir}/era5_boundary0_inject-inside_poolingno/${suffix}"
    "${root_dir}/era5_boundary2_inject-inside_poolingno/${suffix}"
    "${root_dir}/era5_boundary4_inject-inside_poolingno/${suffix}"
    "${root_dir}/era5_boundary6_inject-inside_poolingno/${suffix}"
    "${root_dir}/era5_boundary8_inject-inside_poolingno/${suffix}"
)

echo csv_paths[@]: "${csv_paths[@]}"

python draw_error_plots.py \
    --csv_paths "${csv_paths[@]}" \
    --legend_names "Baseline" "Boundary 2" "Boundary 4" "Boundary 6" "Boundary 8" \
    --styles "linestyle=-,linewidth=2" "linestyle=--,linewidth=2" "linestyle=dashdot,linewidth=2" \
        "linestyle=--,linewidth=2" "linestyle=dashdot,linewidth=2" \
    --output_dir plots --ext png --dpi 150 --zip