# python plot_residual.py --var_name t2m --output_dir residual_plots/residual_plots_baseline \
#     --preds_dir "/tmp3/b12902101/LAM_output_preds/era5_boundary0_inject-inside_smooth_no_interp_exact/preds/" \
#     --plot_mode prediction


for bw in 8; do
    for interp in "nearest"; do
        for smooth in "mean"; do
            python plot_residual.py --var_name t2m --output_dir residual_plots_typhoon/residual_plots_${bw}_${smooth}_${interp} \
                --preds_dir "/tmp3/b12902101/LAM_output/era5_boundary${bw}_inject-inside_smooth_${smooth}_interp_${interp}/preds/" \
                --plot_mode prediction
        done
    done
done