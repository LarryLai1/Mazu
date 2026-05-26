root_dir="/tmp3/b12902101/Mazu/checkpoint-50"
suffix="MSE.csv"
csv_paths=(
    "${root_dir}/2020-07-01_2020-08-31_boundary0_inject-inside_poolingyes/${suffix}"
    "${root_dir}/2020-07-01_2020-08-31_boundary4_inject-inside_poolingno/${suffix}"
    "${root_dir}/2020-07-01_2020-08-31_boundary8_inject-inside_poolingno/${suffix}"
    "${root_dir}/2020-07-01_2020-08-31_boundary4_inject-inside_poolingyes/${suffix}"
    "${root_dir}/2020-07-01_2020-08-31_boundary8_inject-inside_poolingyes/${suffix}"
)

echo csv_paths[@]: "${csv_paths[@]}"

python draw_error_plots.py \
    --csv_paths "${csv_paths[@]}" \
    --legend_names "Run 1" "Run 2" "Run 3" "Run 4" "Run 5" \
    --styles "linestyle=-,linewidth=2" "linestyle=--,linewidth=2" "linestyle=dashdot,linewidth=2" "linestyle=--,linewidth=2" "linestyle=dashdot,linewidth=2" \
    --output_dir plots --ext png --dpi 150 --zip