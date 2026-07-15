for plot_mode in "prediction" "residual"; do
    python plot_residual.py --var_name msl --output_dir residual_plots_t-1/residual_plots_era5 \
        --preds_dir "/tmp3/b12902101/era5_tw_forecast_0.25deg" \
        --plot_mode "${plot_mode}" --init_time "2020-03-01 01:00:00"
    python plot_residual.py --var_name msl --output_dir residual_plots_t-1/residual_plots_baseline \
        --preds_dir "/tmp3/b12902101/LAM_output_preds/era5_boundary0_inject-inside_no_exact_encoder_res0.25_interp/preds" \
        --plot_mode "${plot_mode}" --init_time "2020-03-01 01:00:00"
done


for interp in "nearest"; do
    for smooth in "no"; do
        for plot_mode in "prediction" "residual"; do
            for bd_position in "backbone"; do
                python plot_residual.py --var_name msl --output_dir residual_plots_t-1/residual_plots_8_${smooth}_${interp}_${bd_position} \
                    --preds_dir "/tmp3/b12902101/LAM_output_preds/era5_boundary8_inject-inside_${smooth}_${interp}_${bd_position}_res0.5_direct/preds/" \
                    --plot_mode "${plot_mode}" --init_time "2020-03-01 01:00:00"
            done
        done
    done
done