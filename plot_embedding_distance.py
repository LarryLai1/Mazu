#!/usr/bin/env python3
"""
Embedding distance / similarity between predictions and ERA5 ground truth, vs. lead time.

Both the prediction and the matching ERA5 state are pushed through the *encoder* of the very
Aurora model that produced the prediction (`Perceiver3DEncoder`, see aurora/model/encoder.py).
The resulting latents, shape (B, L', D), are then compared. This gives a model-internal view of
forecast error, complementary to the pixel-space MSE/MAE in `draw_error_plots.py`.

The encoder is fed a real 2-step history, as in training: for lead `t` the two history slots are
the states at `t - history_step` and `t`. Lead 0 has no prediction file and is therefore taken
from ERA5 ground truth, so the lead-1 prediction history is `[GT(init), pred(+1hr)]`.

Example:
  cd Mazu
  python plot_embedding_distance.py \
    --preds_dirs /tmp3/b12902101/LAM_output_preds/*/preds \
    --lead_times $(seq 1 24) \
    --init_times '2020-03-01 01:00:00'

Run from `Mazu/` so the vendored `aurora/` package shadows the pip-installed one.
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from safetensors.torch import load_file

from aurora.batch import Batch, Metadata
from aurora.model.aurora import AuroraSmall
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SURFACE_VARIABLES = ["t2m", "u10", "v10", "msl"]
UPPER_VARIABLES = ["u", "v", "t", "q", "z"]
STATIC_VARIABLES = ["lsm", "slt", "z"]
LEVELS = [1000, 925, 850, 700, 500, 300, 150, 50]
LATITUDE = [39.75, 5]
LONGITUDE = [100, 144.75]

# Okabe-Ito: colourblind-safe by construction, assigned in fixed order (never cycled).
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

METRICS = [
    ("cos_token", "Token-wise cosine similarity", "cosine similarity"),
    ("cos_pooled", "Pooled cosine similarity", "cosine similarity"),
    ("l2_token", "Token-wise L2 distance", "L2 distance"),
    ("l2_pooled", "Pooled L2 distance", "L2 distance"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--preds_dirs", nargs="+", required=True,
                   help="Directories holding <INIT>+<LEAD>hr.nc files, e.g. LAM_output_preds/*/preds")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Legend name per --preds_dirs entry. Defaults to the parent directory name.")
    p.add_argument("--init_times", nargs="+", default=["2020-03-01 01:00:00", "2020-03-01 02:00:00"],
                   help="Initialisation times. Results are averaged over them.")
    p.add_argument("--lead_times", nargs="+", type=int, default=list(range(1, 241)),
                   help="Forecast lead times in hours.")
    p.add_argument("--history_step", type=int, default=1,
                   help="Hours between the two history slots fed to the encoder.")

    p.add_argument("--data_root_dir", type=str, default="/tmp3/yunye0121/era5_tw")
    p.add_argument("--checkpoint_path", type=str,
                   default="/tmp2/yuanlim0919/lateral_smooth/model_weights/Aurora/model.safetensors")
    p.add_argument("--use_pretrained_weight", action="store_true",
                   help="Load Microsoft's pretrained checkpoint instead of --checkpoint_path.")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--stabilise_level_agg", action="store_true")
    p.add_argument("--timestep_hours", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--output_dir", type=str, default="embedding_distance_plots")
    p.add_argument("--ext", type=str, default="png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--width", type=float, default=8.0)
    p.add_argument("--height", type=float, default=5.0)

    args = p.parse_args()

    if args.labels is not None and len(args.labels) != len(args.preds_dirs):
        p.error(f"--labels has {len(args.labels)} entries but --preds_dirs has {len(args.preds_dirs)}.")
    return args


def default_label(preds_dir: str) -> str:
    """`.../hres_boundary8_..._res0.5_direct/preds` -> `hres_boundary8_..._res0.5_direct`."""
    path = Path(preds_dir.rstrip("/"))
    return path.parent.name if path.name == "preds" else path.name


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def create_model(args):
    """Mirrors create_model() in AuroraSmallTW_gen_eval_pipeline_custom_rollout.py.

    bf16 is deliberately off: distances between latents should not be quantisation-dominated.
    """
    model = AuroraSmall(
        use_lora=args.use_lora,
        bf16_mode=False,
        timestep=pd.Timedelta(hours=args.timestep_hours),
        stabilise_level_agg=args.stabilise_level_agg,
    )
    if args.use_pretrained_weight:
        logger.info("Loading pretrained weights provided by Microsoft Aurora...")
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt", strict=True)
    else:
        logger.info("Loading checkpoint: %s", args.checkpoint_path)
        model.load_state_dict(load_file(args.checkpoint_path))

    model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def encode(model, batch: Batch) -> torch.Tensor:
    """Encode a batch, returning the latent of shape (B, L', D).

    This is the `prepare_and_encode` sequence of `model_forward_with_latent_boundary` in
    AuroraSmallTW_gen_eval_pipeline_custom_rollout.py, which is the pre-encoder pipeline the
    real forward pass uses. Keep it in sync with that function.
    """
    import dataclasses

    p = next(model.parameters())

    batch = model.batch_transform_hook(batch)
    batch = batch.type(p.dtype)
    batch = batch.normalise(surf_stats=model.surf_stats)
    batch = batch.crop(patch_size=model.patch_size)
    batch = batch.to(p.device)

    B, T = next(iter(batch.surf_vars.values())).shape[:2]
    static_vars = {}
    for k, v in batch.static_vars.items():
        static_vars[k] = v[None, None].repeat(B, T, 1, 1) if v.ndim == 2 else v
    batch = dataclasses.replace(batch, static_vars=static_vars)

    transformed = batch
    if model.positive_surf_vars:
        transformed = dataclasses.replace(transformed, surf_vars={
            k: v.clamp(min=0) if k in model.positive_surf_vars else v
            for k, v in transformed.surf_vars.items()
        })
    if model.positive_atmos_vars:
        transformed = dataclasses.replace(transformed, atmos_vars={
            k: v.clamp(min=0) if k in model.positive_atmos_vars else v
            for k, v in transformed.atmos_vars.items()
        })
    transformed = model._pre_encoder_hook(transformed)

    # `model.timestep` (1h), not the forecast lead: the lead-time embedding must be a constant
    # offset shared by both batches, not a trend injected into the lead-time curve.
    return model.encoder(transformed, lead_time=model.timestep)


# --------------------------------------------------------------------------------------
# Snapshot loading
# --------------------------------------------------------------------------------------

class SnapshotLoader:
    """Loads single-timestep states as {"surf_vars": {(H,W)}, "atmos_vars": {(L,H,W)}}."""

    def __init__(self, ds_gt: ERA5TWDatasetforAurora):
        self.ds_gt = ds_gt
        self._cache: dict = {}

    def gt(self, time: pd.Timestamp) -> dict:
        key = ("gt", time)
        if key not in self._cache:
            upper_path, sfc_path = self.ds_gt._dt_to_path(time)
            with xr.open_dataset(upper_path) as upper_nc, xr.open_dataset(sfc_path) as sfc_nc:
                upper_nc.load()
                sfc_nc.load()
                self._cache[key] = self.ds_gt._nc_to_dict(upper_nc, sfc_nc)
        return self._cache[key]

    def pred(self, preds_dir: str, init: pd.Timestamp, lead: int) -> dict:
        # There is no +0hr prediction file: the analysis at init time *is* the ground truth.
        if lead == 0:
            return self.gt(init)

        key = ("pred", preds_dir, init, lead)
        if key not in self._cache:
            path = os.path.join(preds_dir, f"{init.strftime('%Y%m%d_%H%M%S')}+{lead}hr.nc")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with xr.open_dataset(path) as ds:
                ds.load()
                self._cache[key] = {
                    # Drop the length-1 `history` axis.
                    "surf_vars": {
                        self.ds_gt.map_var_name_for_Aurora(v): torch.tensor(ds[f"surf_{self.ds_gt.map_var_name_for_Aurora(v)}"].values[0])
                        for v in SURFACE_VARIABLES
                    },
                    "atmos_vars": {
                        v: torch.tensor(ds[f"atmos_{v}"].values[0]) for v in UPPER_VARIABLES
                    },
                }
        return self._cache[key]

    def clear(self):
        self._cache.clear()


def build_batch(snap_prev: dict, snap_curr: dict, static_vars: dict,
                lat: torch.Tensor, lon: torch.Tensor, valid_time: pd.Timestamp) -> Batch:
    """Two history slots -> a Batch with B=1, T=2, matching the shapes used at inference time."""
    return Batch(
        surf_vars={
            k: torch.stack([snap_prev["surf_vars"][k], snap_curr["surf_vars"][k]], dim=0)[None]
            for k in snap_curr["surf_vars"]
        },
        atmos_vars={
            k: torch.stack([snap_prev["atmos_vars"][k], snap_curr["atmos_vars"][k]], dim=0)[None]
            for k in snap_curr["atmos_vars"]
        },
        static_vars=static_vars,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(valid_time.to_pydatetime(),),
            atmos_levels=tuple(LEVELS),
        ),
    )


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def embedding_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Distance/similarity between two latents of shape (1, L', D). `b` is the reference."""
    a = a.float()
    b = b.float()

    ap, bp = a.mean(dim=1), b.mean(dim=1)
    diff_pooled = ap - bp
    l2_pooled = diff_pooled.norm(dim=-1)

    diff = a - b
    l2_per_token = diff.norm(dim=-1)
    ref_per_token = b.norm(dim=-1)

    return {
        "cos_pooled": torch.nn.functional.cosine_similarity(ap, bp, dim=-1).mean().item(),
        "l2_pooled": l2_pooled.mean().item(),
        "rel_l2_pooled": (l2_pooled / bp.norm(dim=-1)).mean().item(),
        "cos_token": torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item(),
        "l2_token": l2_per_token.mean().item(),
        "rel_l2_token": (l2_per_token / ref_per_token.clamp(min=1e-12)).mean().item(),
    }


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, labels, ax):
    for i, label in enumerate(labels):
        sub = df[df["label"] == label].groupby("lead_time", as_index=False)[metric].mean()
        if sub.empty:
            continue
        ax.plot(sub["lead_time"], sub[metric], label=label,
                color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("lead time (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def make_plots(df: pd.DataFrame, labels, args, output_dir: Path):
    for metric, title, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        plot_metric(df, metric, title, ylabel, labels, ax)
        if len(labels) >= 2:
            ax.legend(loc="best", frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.{args.ext}", dpi=args.dpi)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(args.width * 1.8, args.height * 1.8))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), METRICS):
        plot_metric(df, metric, title, ylabel, labels, ax)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if len(labels) >= 2 and handles:
        fig.legend(handles, legend_labels, loc="lower center",
                   ncol=min(3, len(labels)), frameon=False, fontsize=8)
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Aurora encoder embedding: prediction vs. ERA5", fontsize=14)
    fig.savefig(output_dir / f"embedding_summary.{args.ext}", dpi=args.dpi)
    plt.close(fig)


# --------------------------------------------------------------------------------------

def main():
    args = parse_args()

    labels = args.labels or [default_label(d) for d in args.preds_dirs]
    init_times = [pd.Timestamp(t) for t in args.init_times]
    lead_times = sorted(set(args.lead_times))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_time = max(init + pd.Timedelta(hours=max(lead_times)) for init in init_times)
    ds_gt = ERA5TWDatasetforAurora(
        data_root_dir=args.data_root_dir,
        start_date_hour=min(init_times),
        end_date_hour=max_time,
        upper_variables=UPPER_VARIABLES,
        surface_variables=SURFACE_VARIABLES,
        static_variables=STATIC_VARIABLES,
        levels=LEVELS,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        lead_time=1,
        input_time_window=1,
        rollout_step=1,
        sample_stride_hours=1,
    )

    lat, lon = ds_gt.get_latitude_longitude()
    static_vars = ds_gt.get_static_vars_ds()["static_vars"]
    logger.info("Grid: %d lat x %d lon", len(lat), len(lon))

    model = create_model(args)
    loader = SnapshotLoader(ds_gt)

    rows = []
    for init in init_times:
        # The ground-truth embedding at a given lead is shared by every prediction variant,
        # so compute it once per (init, lead).
        for lead in lead_times:
            prev_lead = lead - args.history_step
            if prev_lead < 0:
                logger.warning("Skipping lead %dh: history_step %dh reaches before init time.",
                               lead, args.history_step)
                continue

            valid_time = init + pd.Timedelta(hours=lead)
            try:
                gt_batch = build_batch(loader.gt(init + pd.Timedelta(hours=prev_lead)),
                                       loader.gt(valid_time), static_vars, lat, lon, valid_time)
            except (FileNotFoundError, KeyError) as e:
                logger.warning("Skipping init %s lead %dh: missing ground truth (%s).", init, lead, e)
                continue
            emb_gt = encode(model, gt_batch)

            for preds_dir, label in zip(args.preds_dirs, labels):
                try:
                    snap_prev = loader.pred(preds_dir, init, prev_lead)
                    snap_curr = loader.pred(preds_dir, init, lead)
                except FileNotFoundError as e:
                    logger.warning("Skipping %s init %s lead %dh: missing %s", label, init, lead, e)
                    continue

                pred_batch = build_batch(snap_prev, snap_curr, static_vars, lat, lon, valid_time)
                emb_pred = encode(model, pred_batch)
                assert emb_pred.shape == emb_gt.shape, f"{emb_pred.shape} != {emb_gt.shape}"

                rows.append({
                    "label": label,
                    "init_time": init.strftime("%Y-%m-%d %H:%M:%S"),
                    "lead_time": lead,
                    **embedding_metrics(emb_pred, emb_gt),
                })
                del emb_pred

            del emb_gt
            logger.info("init %s lead %3dh done (%d rows)", init.strftime("%Y%m%d_%H"), lead, len(rows))

        loader.clear()

    if not rows:
        logger.error("No results computed. Check --preds_dirs, --init_times and --lead_times.")
        return

    df = pd.DataFrame(rows)
    csv_path = output_dir / "embedding_distance.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(df))

    make_plots(df, labels, args, output_dir)
    logger.info("Wrote plots to %s", output_dir)


if __name__ == "__main__":
    main()
