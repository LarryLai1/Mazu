# python plot_residual.py --var_name msl --output_dir residual_plots_typhoon/residual_plots \
#     --preds_dir "/tmp3/b12902101/LAM_output_preds/era5_boundary8_inject-inside_linear_nearest_backbone/preds" \
#     --plot_mode prediction


for interp in "nearest"; do
    for smooth in "no"; do
        for bd_position in "encoder" "backbone"; do
            python plot_residual.py --var_name msl --output_dir residual_plots_typhoon/residual_plots_8_${smooth}_${interp} \
                --preds_dir "/tmp3/b12902101/LAM_output_preds/era5_boundary8_inject-inside_${smooth}_${interp}_${bd_position}/preds/" \
                # --plot_mode prediction
        done
    done
done