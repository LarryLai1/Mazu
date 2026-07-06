---
name: mazu-weather-prediction-and-soft
description: Guide to training, evaluating, and applying the Self-Output Fine-Tuning (SOFT) method on autoregressive regional weather models (Aurora/Pangu) in the Mazu repository.
---

# Mazu: Self-Output Fine-Tuning (SOFT) for Autoregressive Weather Prediction

This skill defines the workflows, architecture, and commands used in the **Mazu** repository to train, fine-tune, and evaluate regional weather forecasting models (specifically based on Microsoft Aurora and Pangu-Weather) using the **Self-Output Fine-Tuning (SOFT)** method.

---

## 1. Environment & Setup

### Conda/Mamba Setup
Install dependencies defined in [requirements.txt](file:///tmp3/b12902101/Mazu/requirements.txt):
```bash
pip install -r requirements.txt
```

### Dataset Download
The pipeline uses **ERA5** reanalysis data. Download static and main variables using the scripts under `download_era5_data`:
* **Static variables**:
  ```bash
  python download_era5_data/constant_download_era5.py --region tw
  ```
* **Main variables**:
  ```bash
  python download_era5_data/download_era5.py --region tw --start YYYY/MM/DD --end YYYY/MM/DD
  ```
* **Notes**: Regions supported include `tw` (Taiwan) and `eu` (Europe). Grid dimensions must be divisible by the patch size (default is **4**).

---

## 2. Baseline Model Training

Baseline models are trained without synthetic target injection.

### Running training:
* **AuroraSmallTW Baseline**:
  ```bash
  python train_AuroraSmallTW_onthefly.py \
      --data_root_dir /path/to/era5_tw \
      --output_dir AuroraTW_baseline \
      --train_start_date_hour "2020-01-01 00:00:00" \
      --train_end_date_hour "2020-06-30 18:00:00" \
      --val_start_date_hour "2020-07-01 00:00:00" \
      --val_end_date_hour "2020-07-31 18:00:00" \
      --upper_variables u v t q z \
      --surface_variables t2m u10 v10 msl \
      --static_variables lsm slt z \
      --levels 1000 925 850 700 500 300 150 50 \
      --latitude 39.75 5 --longitude 100 144.75 \
      --input_time_window 2 --rollout_step 1 --epochs 5
  ```
* Or execute the corresponding wrapper script:
  ```bash
  bash public_bash_scripts/train_AuroraSmallTW.sh
  ```

---

## 3. SOFT (Self-Output Fine-Tuning) Pipeline

Autoregressive models suffer from error accumulation over rollout steps. SOFT addresses this by generating synthetic prediction data and fine-tuning the model to handle its own prediction errors.

### Step 1: Generate Synthetic Predictions (Rollout)
Use the trained baseline model weights to roll out predictions over the training period:
```bash
python AuroraSmallTW_gen_eval_pipeline_custom_rollout.py \
    --data_root_dir /path/to/era5_tw \
    --checkpoint_path /path/to/baseline/model.safetensors \
    --start_date_hour "2020-01-01 00:00:00" \
    --end_date_hour "2020-06-30 18:00:00" \
    --pred true \
    --boundary_width 8 \
    --boundary_mode inject-inside \
    --boundary_source era5 \
    --csv_output_folder ./LAM_output_preds \
    --gen_result_folder ./LAM_output_preds/preds
```
* Or execute the wrapper:
  ```bash
  bash public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_custom_rollout_1hr_data_gen.sh
  ```

### Step 2: SOFT Fine-Tuning
Fine-tune the model using the synthetic predictions from Step 1 alongside ground truth data:
```bash
python train_AuroraSmallTW_with_AuroraPrediction.py \
    --data_root_dir /path/to/era5_tw \
    --checkpoint_path /path/to/baseline/model.safetensors \
    --Aurora_input_dir /path/to/LAM_output_preds/preds \
    --use_Aurora_input_len 1 \
    --train_start_date_hour "2020-01-01 00:00:00" \
    --train_end_date_hour "2020-06-30 18:00:00" \
    --val_start_date_hour "2020-07-01 00:00:00" \
    --val_end_date_hour "2020-07-31 18:00:00" \
    --upper_variables u v t q z \
    --surface_variables t2m u10 v10 msl \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 --longitude 100 144.75 \
    --input_time_window 2 --rollout_step 1 --epochs 5
```
* Point `--Aurora_input_dir` to the folder containing synthetic prediction outputs.
* `--use_Aurora_input_len` specifies how many input time window steps are replaced by model predictions.

---

## 4. Custom Rollout & Boundary Condition Injection

During regional inference, boundary conditions from a global model (e.g. HRES or ERA5) are injected at each step.

### Key Arguments:
- `--boundary_width`: The width of the boundary frame (e.g., `2` or `8` grid cells).
- `--boundary_mode`: The injection style (e.g., `inject-inside`).
- `--boundary_pooling`: Subsampling/pooling method for boundaries (e.g., `no`, `mean`).
- `--boundary_smooth_mode`: Border smoothing algorithm (`no`, `mean`, `gaussian`, `linear`).
- `--boundary_time_interp_mode`: Method of temporal alignment (`nearest`, `exact`, `interpolation`).
- `--replace_boundary_position`: Where boundary data replaces internal latents (`encoder` or `backbone`).
- `--boundary_source`: Boundary provider (`era5`, `ground_truth`, or `aurora`).

### Running Rollouts:
```bash
python AuroraSmallTW_gen_eval_pipeline_custom_rollout.py \
    --data_root_dir /path/to/era5_tw \
    --boundary_root_dir /path/to/boundary_data \
    --checkpoint_path /path/to/soft/model.safetensors \
    --boundary_width 8 \
    --boundary_mode inject-inside \
    --boundary_smooth_mode mean \
    --boundary_time_interp_mode nearest \
    --replace_boundary_position backbone \
    --rollout_step 72 \
    --csv_output_folder ./LAM_output
```

---

## 5. Evaluation & Metrics

The repository evaluates predictions against ground truth and analyzes the latent embeddings:

### Embedding Quality (FID & Linear Probing)
* **FID score for Backbone or Perceiver**:
  ```bash
  python AuroraSmallTW_calc_backbone_embedding_FID.py
  python AuroraSmallTW_calc_perceiver_embedding_FID.py
  ```
* **Linear Probe Evaluation**:
  ```bash
  python AuroraSmallTW_backbone_embedding_linear_probe.py
  python AuroraSmallTW_perceiver_embedding_linear_probe.py
  ```
* **Embedding Visualization**:
  ```bash
  python AuroraSmallTW_visualize_backbone_embedding.py
  python AuroraSmallTW_visualize_perceiver_embedding.py
  ```

### Meteorological Performance (MAE / MSE)
* Evaluate ERA5 boundary forecasts:
  ```bash
  python eval_era5_boundary_forecast.py \
      --boundary_root_dir /path/to/era5_tw_forecast_3d \
      --data_root_dir /path/to/era5_tw \
      --csv_output_folder ./LAM_output/era5_boundary_data
  ```
* Compare MAE CSV files:
  ```bash
  python compare_mae_csvs.py
  ```

---

## 6. Plotting & Figures

Visualizations are generated using scripts prefixed with `draw_`:
* **Plot error over time/rollouts**:
  ```bash
  python draw_error_plots.py --csv_paths path1.csv path2.csv --legend_names "Model A" "Model B" --output_dir plots
  ```
* **Error panel visualizations**:
  ```bash
  python draw_error_panel.py
  python draw_error_case_study_panel.py
  ```
* **Atmospheric field comparisons**:
  ```bash
  python draw_comparison_AuroraTW_era5.py
  ```
