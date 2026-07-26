#!/usr/bin/env python3
"""
t-SNE distribution of Aurora backbone embeddings for two predictions and ERA5 ground truth.

For a single init time and each hourly lead, the prediction snapshot (and its matching ERA5 state)
is pushed through the very Aurora model that produced the prediction; the embedding is the output of
the *last encoder layer of the Swin3D backbone* (the bottleneck latent, before any decoder runs). See
`utils/embedding.py`. Each `(1, L', D)` latent is flattened to a 1-D vector, and every vector across
all three series is embedded together with a single t-SNE so their coordinates are comparable.

The scatter shows the three series in three colour families (baseline = Blues, boundary replacement =
Oranges, ground truth = Greens). Colour depth encodes forecast time: dark near the init time, fading
brighter as the forecast advances (a point's brightness rises with days-from-init).

This complements the scalar cosine/L2 curves in `plot_embedding_distance.py` with a 2-D view of how
the trajectories spread apart in the model's latent space.

Example:
  cd Mazu
  python plot_embedding_tsne.py \
    --preds_dirs /tmp3/b12902101/LAM_output_preds/hres_boundary0_no_nearest_backbone_res0.25_direct/preds \
                 /tmp3/b12902101/LAM_output_preds/hres_boundary8_no_nearest_backbone_res0.25_direct/preds \
    --labels baseline "boundary replacement" \
    --init_time '2020-03-01 00:00:00' \
    --lead_times $(seq 1 240)

Run from `Mazu/` so the vendored `aurora/` package shadows the pip-installed one.
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.embedding import encode_batch
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
# Reuse the model/loader/batch machinery of the distance script (run from Mazu/).
from plot_embedding_distance import (
    SURFACE_VARIABLES,
    UPPER_VARIABLES,
    STATIC_VARIABLES,
    LEVELS,
    LATITUDE,
    LONGITUDE,
    create_model,
    SnapshotLoader,
    build_batch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GT_LABEL = "ground truth"

# One sequential colormap per series. Colour is taken dark (near init) -> light (later leads).
SERIES_CMAPS = ["Blues", "Oranges", "Greens"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--preds_dirs", nargs="+", required=True,
                   help="Directories holding <INIT>+<LEAD>hr.nc files (two entries expected).")
    p.add_argument("--labels", nargs="*", default=["baseline", "boundary replacement"],
                   help="Legend name per --preds_dirs entry.")
    p.add_argument("--init_time", type=str, default="2020-03-01 00:00:00",
                   help="Single initialisation time.")
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

    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--pca_dims", type=int, default=50,
                   help="PCA pre-reduction dims before t-SNE (0 to skip).")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--output_dir", type=str, default="embedding_tsne_plots")
    p.add_argument("--ext", type=str, default="png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--width", type=float, default=8.0)
    p.add_argument("--height", type=float, default=6.5)

    args = p.parse_args()
    if len(args.labels) != len(args.preds_dirs):
        p.error(f"--labels has {len(args.labels)} entries but --preds_dirs has {len(args.preds_dirs)}.")
    return args


def flatten_embedding(model, batch) -> np.ndarray:
    """Encode a batch and flatten its (1, L', D) latent to a 1-D float32 vector."""
    emb = encode_batch(model, batch)
    vec = emb.reshape(-1).float().cpu().numpy()
    del emb
    return vec


def collect_embeddings(args, model, loader, static_vars, lat, lon):
    """Return a DataFrame with columns [label, lead_time] and a parallel (N, dim) vector array."""
    init = pd.Timestamp(args.init_time)
    lead_times = sorted(set(args.lead_times))

    vectors = []
    meta = []
    for lead in lead_times:
        prev_lead = lead - args.history_step
        if prev_lead < 0:
            logger.warning("Skipping lead %dh: history_step %dh reaches before init time.",
                           lead, args.history_step)
            continue

        valid_time = init + pd.Timedelta(hours=lead)

        # Ground truth, shared reference series.
        try:
            gt_batch = build_batch(loader.gt(init + pd.Timedelta(hours=prev_lead)),
                                   loader.gt(valid_time), static_vars, lat, lon, valid_time)
        except (FileNotFoundError, KeyError) as e:
            logger.warning("Skipping init %s lead %dh: missing ground truth (%s).", init, lead, e)
            continue
        vectors.append(flatten_embedding(model, gt_batch))
        meta.append({"label": GT_LABEL, "lead_time": lead})

        # Each prediction variant.
        for preds_dir, label in zip(args.preds_dirs, args.labels):
            try:
                snap_prev = loader.pred(preds_dir, init, prev_lead)
                snap_curr = loader.pred(preds_dir, init, lead)
            except (FileNotFoundError, KeyError) as e:
                logger.warning("Skipping %s init %s lead %dh: missing %s", label, init, lead, e)
                continue
            pred_batch = build_batch(snap_prev, snap_curr, static_vars, lat, lon, valid_time)
            vectors.append(flatten_embedding(model, pred_batch))
            meta.append({"label": label, "lead_time": lead})

        logger.info("init %s lead %3dh done (%d vectors)", init.strftime("%Y%m%d_%H"), lead, len(vectors))

    loader.clear()
    return pd.DataFrame(meta), np.stack(vectors) if vectors else np.empty((0, 0))


def run_tsne(X: np.ndarray, args) -> np.ndarray:
    """PCA pre-reduction (optional) then a single 2-D t-SNE over all vectors."""
    pca_dims = min(args.pca_dims, X.shape[0], X.shape[1])
    if args.pca_dims and 0 < pca_dims < X.shape[1]:
        logger.info("PCA %d -> %d dims", X.shape[1], pca_dims)
        X = PCA(n_components=pca_dims, random_state=args.seed).fit_transform(X)
    perplexity = min(args.perplexity, max(5.0, (X.shape[0] - 1) / 3.0))
    logger.info("t-SNE on %d vectors (perplexity %.1f)", X.shape[0], perplexity)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=args.seed)
    return tsne.fit_transform(X)


def make_plot(df: pd.DataFrame, args, output_dir: Path):
    """Scatter the t-SNE coordinates: colour family per series, depth = forecast time."""
    labels = list(args.labels) + [GT_LABEL]
    max_lead = max(df["lead_time"]) if len(df) else 1
    max_days = max_lead / 24.0

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    legend_handles = []
    for i, label in enumerate(labels):
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        cmap = plt.get_cmap(SERIES_CMAPS[i % len(SERIES_CMAPS)])
        # Dark near init (small lead) -> light later; keep away from the near-white/near-black ends.
        shade = 0.15 + 0.75 * (sub["lead_time"].to_numpy() / max_lead)
        ax.scatter(sub["x"], sub["y"], c=cmap(shade), s=22, edgecolors="none")
        legend_handles.append(Patch(facecolor=cmap(0.7), edgecolor="none", label=label))

    ax.legend(handles=legend_handles, loc="best", frameon=True, fontsize=9)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of Aurora backbone embeddings — {args.init_time}, +1..{max_lead} h")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Grey colourbar explaining the depth gradient (dark = near init, light = later).
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=max_days), cmap="Greys_r")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f"days from init ({args.init_time})")

    fig.tight_layout()
    out = output_dir / f"embedding_tsne.{args.ext}"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    logger.info("Wrote %s", out)


def main():
    args = parse_args()
    print(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init = pd.Timestamp(args.init_time)
    max_time = init + pd.Timedelta(hours=max(args.lead_times))
    ds_gt = ERA5TWDatasetforAurora(
        data_root_dir=args.data_root_dir,
        start_date_hour=init,
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

    df, X = collect_embeddings(args, model, loader, static_vars, lat, lon)
    if len(df) == 0:
        logger.error("No embeddings computed. Check --preds_dirs, --init_time and --lead_times.")
        return
    logger.info("Collected %d vectors of dim %d", X.shape[0], X.shape[1])

    coords = run_tsne(X, args)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    csv_path = output_dir / "embedding_tsne.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(df))

    make_plot(df, args, output_dir)


if __name__ == "__main__":
    main()
