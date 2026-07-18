var_name="msl"
init_time="2020-03-01 01:00:00"
out_root="residual_plots_t-1"

era5_boundary_root="/tmp3/b12902101/era5_tw_forecast_0.25deg"
era5_boundary_root_15="/tmp3/b12902101/era5_tw_forecast_1.5deg"
baseline_preds_dir="/tmp3/b12902101/LAM_output_preds/era5_boundary0_no_exact_encoder_res0.25_interp/preds"

smooth="no"
bd_position="backbone"
interp="nearest"
resols=(0.25 0.5 1.5)

for plot_mode in "prediction" "residual"; do
    python plot_residual.py --var_name "${var_name}" \
        --output_dir "${out_root}/residual_plots_baseline" \
        --preds_dir "${baseline_preds_dir}" \
        --plot_mode "${plot_mode}" --init_time "${init_time}"

    # HRES: the ERA5 boundary forecast itself, at each resolution.
    for resol in "${resols[@]}"; do
        # The 1.5deg boundary lives in its own native low-res dataset.
        era5_dir="${era5_boundary_root}"
        if [ "${resol}" = "1.5" ]; then
            era5_dir="${era5_boundary_root_15}"
        fi

        python plot_residual.py --var_name "${var_name}" \
            --output_dir "${out_root}/residual_plots_era5_res${resol}_direct" \
            --preds_dir "${era5_dir}" \
            --plot_mode "${plot_mode}" --init_time "${init_time}" \
            --boundary_resolution "${resol}" \
            --boundary_lowres_apply_mode "direct"
    done

    # Model rollouts driven by each boundary variant. The resolution is already baked into
    # these prediction files, so it only selects the directory here.
    for resol in "${resols[@]}"; do
        for apply_mode in "direct" "interp"; do
            if [ "${resol}" = "0.25" ] && [ "${apply_mode}" = "interp" ]; then
                continue
            fi

            python plot_residual.py --var_name "${var_name}" \
                --output_dir "${out_root}/residual_plots_8_${smooth}_${interp}_${bd_position}_res${resol}_${apply_mode}" \
                --preds_dir "/tmp3/b12902101/LAM_output_preds/era5_boundary8_${smooth}_${interp}_${bd_position}_res${resol}_${apply_mode}/preds/" \
                --plot_mode "${plot_mode}" --init_time "${init_time}"
        done
    done
done
