python plot_residual.py --var_name msl --output_dir residual_plots_t-1/residual_plots_baseline \
    --preds_dir "/tmp3/b12902101/LAM_output_preds/era5_boundary0_inject-inside_no_exact_encoder/preds" \
    --plot_mode prediction --init_time "2020-07-10 01:00:00"


for interp in "nearest"; do
    for smooth in "no"; do
        for plot_mode in "prediction" "residual"; do
            # for bd_position in "backbone"; do
            for bd_position in "backbone"; do
                python plot_residual.py --var_name msl --output_dir residual_plots_t-1/residual_plots_8_${smooth}_${interp}_${bd_position} \
                    --preds_dir "/tmp3/b12902101/LAM_output_preds/era5_boundary8_inject-inside_${smooth}_${interp}_${bd_position}/preds/" \
                    --plot_mode "${plot_mode}" --init_time "2020-07-10 01:00:00"
            done
        done
    done
done