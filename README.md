
# Mazu

This is the official implementation for the paper: **"Nipping the Butterfly Effect in the Bud: Self-Output Fine-Tuning for Autoregressive Weather Prediction"**.

## Preliminary

We assume your working directory is the `Mazu` folder. If you are running scripts from a different location, please adjust the commands accordingly.

### Environment Setting

We recommend using **Conda** or **Mamba** to manage your environment.

To install the required dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Download

Please use `download.py` to download the ERA5 data. We currently provide 2 region presets:

* `tw` (Taiwan)
* `eu` (Europe)

**Notes:**

* Start date and end date are required.
* You may select a custom range, but ensure your grid size is divisible by the patch size (default is **4**).

Run the following scripts to download the dataset:

```bash
# Download static variables
python download_era5_data/constant_download_era5.py --region tw

# Download main variables
python download_era5_data/download_era5.py --region tw --start YYYY/MM/DD --end YYYY/MM/DD
```

## Training

We provide both **Bash** (shell) and **Slurm** scripts for training.

### 1. Train Baseline Model

* **Bash:**

```bash
bash public_bash_scripts/train_AuroraSmallTW.sh

```

* **Slurm:**

```bash
sbatch public_slurm_scripts/train_AuroraTW.sh
```

### 2. Apply SOFT Method

To apply the SOFT (Self-Output Fine-Tuning) method, we recommend a 2-stage process: generating prediction data first, and then using a custom dataset loader for fine-tuning.

#### Step 1: Generate prediction data

Use your trained model checkpoint to generate the synthetic data.

* **Bash:**

```bash
bash public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_custom_rollout_1hr_data_gen.sh
```

* **Slurm:**

```bash
sbatch public_slurm_scripts/AuroraSmallTW_gen_eval_pipeline_1hr_syn.sh
```

#### Step 2: Apply custom training script

Fine-tune the model using the generated predictions.

* **Bash:**

```bash
bash public_bash_scripts/train_AuroraSmallTW_with_AuroraPrediction.sh
```

* **Slurm:**

```bash
sbatch public_slurm_scripts/train_AuroraSmallTW_with_AuroraPrediction.sh
```

## Inference

After training, you can run the inference pipeline using the following scripts:

* **Bash:**

```bash
bash public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_custom_rollout_1hr.sh
```

* **Slurm:**

```bash
# Please replace with your specific inference script name if different
sbatch public_slurm_scripts/AuroraSmallTW_gen_eval_pipeline.sh
```

