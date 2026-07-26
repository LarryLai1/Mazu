#!/usr/bin/env python3
"""
Per-lead-time t-SNE of the Swin3D bottleneck embeddings extracted by
AuroraSmallTW_gen_eval_pipeline_with_embeddings.py.

ONE FIGURE PER LEAD TIME. Passing embeddings saved at --embedding_save_steps 1 6 24 72
produces four independent figures (+1h, +6h, +24h, +72h), each with its own PCA->t-SNE fit
over only that lead time's vectors. This mirrors SOFT's own
AuroraSmall_bottleneck_probe_tsne.py (one plot per rollout step) and is the honest way to
do it: t-SNE coordinates from separate fits are not comparable, so a single shared fit split
across leads afterwards would invite reading distances that don't exist.

Downstream-only: never touches the model or a GPU, just reads the .npz files that pipeline
already wrote (each holding a flattened `pred_embedding`, the matching `gt_embedding`, and
`init_time`/`lead_time` metadata -- see its _save_embedding_npz()).

Within one figure every point shares the same lead time, so colour encodes SERIES IDENTITY
(each prediction config, plus the shared ERA5 ground truth) -- not lead time. Ground truth is
de-duplicated across input directories, since every run re-extracts the same ERA5 states.

Example:
  cd Mazu
  python plot_embedding_tsne_hooked.py \
    --embeddings_dirs /tmp3/b12902101/mazu_embedding_output/embeddings/baseline \
                      /tmp3/b12902101/mazu_embedding_output/embeddings/boundary8 \
    --labels baseline "boundary replacement" \
    --output_dir /tmp3/b12902101/mazu_embedding_output/tsne_plots
"""

import argparse
import glob
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GT_LABEL = "ground truth (ERA5)"

# Categorical slots 1-3 of the validated reference palette, in fixed order, never cycled, for
# the FORECAST series. A scatter puts every pair on screen at once, so it is held to the
# ALL-PAIRS gate, and only the first three slots clear it (worst pair CVD dE 9.2 light / 9.4
# dark, normal-vision 24.0 / 20.9). Slot 4 would put yellow next to orange and fail outright.
#
# ERA5 ground truth is deliberately NOT a fourth categorical hue: it is the reference every
# forecast is compared against, not another forecast, so it takes a neutral ink instead. That
# is also what makes four entities legal here -- an exhaustive search of all 70 four-slot
# subsets of the palette found none that clears the all-pairs target in both modes, whereas
# these neutrals sit clear of all three hues (CVD dE 9.2 light / 9.4 dark, normal-vision 21.8
# / 20.9) and above 3:1 on their own surface, needing no relief.
THEME = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "reference": "#52514e",
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#898781",
        "axis": "#c3c2b7",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70"],
        "reference": "#c3c2b7",
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#898781",
        "axis": "#383835",
    },
}
MAX_FORECAST_SERIES = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--embeddings_dirs", nargs="+", required=True,
                   help="One or more directories of <INIT>+<LEAD>hr.npz files, each produced by a "
                        "separate run of AuroraSmallTW_gen_eval_pipeline_with_embeddings.py.")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Legend name per --embeddings_dirs entry. Defaults to each dir's basename.")
    p.add_argument("--no_ground_truth", action="store_true",
                   help="Plot only the prediction series, omitting the shared ERA5 ground truth "
                        "(frees a colour slot when comparing 3 prediction configs).")

    p.add_argument("--lead_times", nargs="*", type=int, default=None,
                   help="Only plot these lead times (hours). Default: every lead time found.")
    p.add_argument("--init_days", type=float, default=None,
                   help="Keep only initialisation times within this many days of the earliest one "
                        "found (e.g. 5 -> the first 5 days). A month of hourly inits is ~744 points "
                        "per series per panel, which over-plots badly; a few days reads much better. "
                        "Default: every initialisation time.")
    p.add_argument("--max_points_per_series", type=int, default=2000,
                   help="Subsample each (series, lead time) group to at most this many points. "
                        "0 disables subsampling.")
    p.add_argument("--perplexity", type=float, default=30.0,
                   help="Upper bound on t-SNE perplexity; automatically reduced to fit small groups.")
    p.add_argument("--pca_dims", type=int, default=50,
                   help="PCA pre-reduction dims before t-SNE. Effectively mandatory here: the "
                        "flattened bottleneck vectors are ~885k-dim. 0 to skip anyway.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--theme", type=str, default="light", choices=["light", "dark"],
                   help="Selected (not auto-flipped) colour mode; each is validated against its "
                        "own surface.")
    p.add_argument("--output_dir", type=str, default="/tmp3/b12902101/mazu_embedding_output/tsne_plots")
    p.add_argument("--ext", type=str, default="png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--width", type=float, default=7.5)
    p.add_argument("--height", type=float, default=6.0)

    args = p.parse_args()
    if args.labels is not None and len(args.labels) != len(args.embeddings_dirs):
        p.error(f"--labels has {len(args.labels)} entries but --embeddings_dirs has "
                f"{len(args.embeddings_dirs)}.")

    if len(args.embeddings_dirs) > MAX_FORECAST_SERIES:
        p.error(
            f"{len(args.embeddings_dirs)} forecast directories given but a scatter is capped at "
            f"{MAX_FORECAST_SERIES}: only the first three categorical slots stay distinguishable "
            f"when every pair is on screen at once (colour-vision-deficiency separation), and no "
            f"four-hue subset of the palette clears that gate. Ground truth does not count "
            f"against this -- it is drawn in a neutral reference ink. Drop a directory, or split "
            f"the comparison across two runs."
        )
    return args


def _default_label(embeddings_dir):
    return Path(embeddings_dir.rstrip("/")).name


# <INIT>+<LEAD>hr.npz, as written by the pipeline's _save_embedding_npz().
_NAME_RE = re.compile(r"^(\d{8}_\d{6})\+(\d+)hr\.npz$")


def scan_dirs(args):
    """Index every .npz by (init time, lead time) from its FILENAME -- no array I/O.

    Reading the metadata out of each file just to discard most of them would mean paging in
    ~3.4 MB of embedding per rejected file; the name carries the same two fields, so the
    --init_days / --lead_times windows are applied before anything is loaded.

    Returns {dir: [(path, init_timestamp, lead_time), ...]}.
    """
    scanned = {}
    for embeddings_dir in args.embeddings_dirs:
        entries, unnamed = [], 0
        for path in sorted(glob.glob(os.path.join(embeddings_dir, "*.npz"))):
            m = _NAME_RE.match(os.path.basename(path))
            if not m:
                unnamed += 1
                continue
            entries.append((path, pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")),
                            int(m.group(2))))
        if unnamed:
            logger.warning("%s: skipped %d file(s) whose names are not <INIT>+<LEAD>hr.npz",
                           embeddings_dir, unnamed)
        if not entries:
            logger.warning("No usable .npz files found in %s", embeddings_dir)
        scanned[embeddings_dir] = entries
    return scanned


def apply_windows(scanned, args):
    """Drop entries outside --lead_times / --init_days. The init-day cutoff is measured from
    the earliest initialisation time across ALL directories, so every series is windowed
    identically and stays comparable."""
    all_inits = [init for entries in scanned.values() for _, init, _ in entries]
    if not all_inits:
        return scanned

    cutoff = None
    if args.init_days is not None:
        first = min(all_inits)
        cutoff = first + pd.Timedelta(days=args.init_days)
        logger.info("Initialisation window: %s .. %s (first %g day(s))",
                    first, cutoff, args.init_days)

    out = {}
    for embeddings_dir, entries in scanned.items():
        kept = [e for e in entries
                if (not args.lead_times or e[2] in args.lead_times)
                and (cutoff is None or e[1] < cutoff)]
        logger.info("%s: %d of %d file(s) inside the window", embeddings_dir, len(kept), len(entries))
        out[embeddings_dir] = kept
    return out


def load_embeddings(scanned, args, labels):
    """Load the windowed .npz files.

    Returns a DataFrame [label, lead_time, init_time] and a parallel (N, dim) float32 array.
    Ground-truth rows are de-duplicated on (init_time, lead_time): each run re-extracts the
    same ERA5 states, so pooling them unchanged would stack identical points on top of each
    other and overstate the ground-truth cluster's density.
    """
    rows, vecs = [], []
    seen_gt = set()

    for embeddings_dir, label in zip(args.embeddings_dirs, labels):
        n_pred = n_gt = no_gt_key = 0
        for path, _, _ in scanned[embeddings_dir]:
            with np.load(path, allow_pickle=False) as d:
                # Filenames drove the windowing; the in-file metadata stays authoritative
                # for what actually gets plotted.
                lead_time = int(d["lead_time"])
                init_time = str(d["init_time"])
                rows.append({"label": label, "lead_time": lead_time, "init_time": init_time})
                vecs.append(d["pred_embedding"].astype(np.float32))
                n_pred += 1

                # A directory extracted with --skip_gt omits the key entirely rather than
                # writing a placeholder, so its absence simply means "no ground truth here".
                if not args.no_ground_truth and "gt_embedding" in d.files:
                    key = (init_time, lead_time)
                    if key not in seen_gt:
                        seen_gt.add(key)
                        rows.append({"label": GT_LABEL, "lead_time": lead_time, "init_time": init_time})
                        vecs.append(d["gt_embedding"].astype(np.float32))
                        n_gt += 1
                elif not args.no_ground_truth:
                    no_gt_key += 1

        if no_gt_key:
            logger.info("%s: %d file(s) carry no gt_embedding (extracted with --skip_gt); ground "
                        "truth for those comes from the other directories.",
                        embeddings_dir, no_gt_key)
        logger.info("%s: %d prediction + %d new ground-truth vectors (label=%r)",
                    embeddings_dir, n_pred, n_gt, label)

    if not vecs:
        return pd.DataFrame(), np.empty((0, 0), dtype=np.float32)

    dims = {v.shape[0] for v in vecs}
    if len(dims) != 1:
        raise ValueError(f"Embedding vectors have inconsistent dimensionality: {sorted(dims)}")

    return pd.DataFrame(rows), np.stack(vecs)


def subsample(df, X, max_points, seed):
    """Cap each (label, lead_time) group independently, so no series can crowd out another."""
    if not max_points:
        return df, X
    rng = np.random.default_rng(seed)
    keep = []
    for (label, lead), grp in df.groupby(["label", "lead_time"], sort=False):
        idx = grp.index.to_numpy()
        if len(idx) > max_points:
            idx = np.sort(rng.choice(idx, size=max_points, replace=False))
            logger.info("lead %sh %s: subsampled %d -> %d", lead, label, len(grp), max_points)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return df.loc[keep].reset_index(drop=True), X[keep]


def run_tsne(X, args):
    """PCA pre-reduction then a 2-D t-SNE, fit on THIS lead time's vectors only."""
    n = X.shape[0]
    pca_dims = min(args.pca_dims, n, X.shape[1])
    if args.pca_dims and 0 < pca_dims < X.shape[1]:
        X = PCA(n_components=pca_dims, random_state=args.seed).fit_transform(X)
    # sklearn requires perplexity < n_samples; keep well under it for small groups.
    perplexity = max(1.0, min(args.perplexity, (n - 1) / 3.0))
    logger.info("  t-SNE on %d vectors, %d dims, perplexity %.2f", n, X.shape[1], perplexity)
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=args.seed).fit_transform(X)


def _mark_spec(n_points):
    """Marker size / ring width / alpha, scaled to how crowded the panel is.

    A big ringed mark is right for a sparse panel and wrong for a dense one, where the
    surface-coloured rings merge into pale worms and swallow the data.
    """
    if n_points <= 60:
        return 110, 2.0, 0.90
    if n_points <= 400:
        return 45, 1.0, 0.85
    return 16, 0.0, 0.60


def make_plot(sub, lead_time, series_order, args, theme, output_dir):
    """One figure for one lead time: colour = series identity (lead time is fixed here)."""
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    size, ring, alpha = _mark_spec(len(sub))
    # Forecast series take the categorical slots in CLI order; ground truth takes the neutral
    # reference ink and never consumes a slot (see THEME).
    forecasts = [lbl for lbl in series_order if lbl != GT_LABEL]
    colour_of = {label: theme["series"][slot] for slot, label in enumerate(forecasts)}
    colour_of[GT_LABEL] = theme["reference"]

    # Draw every series in ONE shuffled pass rather than series-by-series. Drawing
    # sequentially puts the last series permanently on top, which at short lead times --
    # where the predictions very nearly coincide with ERA5 -- hides entire series
    # underneath it. Interleaving the draw order lets each series show through in
    # proportion to how many points it actually has there.
    order = np.random.default_rng(args.seed).permutation(len(sub))
    shuffled = sub.iloc[order]
    ax.scatter(shuffled["x"], shuffled["y"], s=size,
               c=[colour_of[lbl] for lbl in shuffled["label"]], alpha=alpha,
               edgecolors=theme["surface"] if ring else "none", linewidths=ring, zorder=3)

    # Legend from proxy handles, since the single scatter above carries no per-series
    # label. Kept fully opaque so the swatch reads as the series' true colour.
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                      markerfacecolor=colour_of[label], markeredgecolor="none", label=label)
               for label in series_order if (sub["label"] == label).any()]

    n_init = sub["init_time"].nunique()
    inits = pd.to_datetime(sub["init_time"])
    span = f"{inits.min():%Y-%m-%d %H:%M} .. {inits.max():%Y-%m-%d %H:%M}"
    # Main title on the figure, muted subtitle on the axes: keeps them from overprinting
    # each other the way a single axes-title + manual text box does.
    fig.suptitle(f"Swin3D bottleneck embeddings, +{lead_time}h forecast",
                 color=theme["text"], fontsize=13, fontweight="600")
    ax.set_title(f"t-SNE of flattened embeddings · {n_init} initialisation times "
                 f"({span}) · {len(sub)} points",
                 color=theme["muted"], fontsize=9, pad=8)

    # t-SNE axes carry no units and no interpretable scale, so numeric ticks would only
    # invite over-reading; the axis names say what the dimensions are and nothing more.
    ax.set_xlabel("t-SNE dimension 1 (arbitrary units)", color=theme["muted"], fontsize=9)
    ax.set_ylabel("t-SNE dimension 2 (arbitrary units)", color=theme["muted"], fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])

    legend = ax.legend(handles=handles, loc="best", frameon=True, fontsize=9,
                       facecolor=theme["surface"], edgecolor=theme["axis"], framealpha=0.95)
    for text in legend.get_texts():
        text.set_color(theme["text"])

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for the suptitle
    out = output_dir / f"embedding_tsne_lead{lead_time}h.{args.ext}"
    fig.savefig(out, dpi=args.dpi, facecolor=theme["surface"])
    plt.close(fig)
    return out


def main():
    args = parse_args()
    theme = THEME[args.theme]
    labels = args.labels or [_default_label(d) for d in args.embeddings_dirs]
    series_order = list(labels) + ([] if args.no_ground_truth else [GT_LABEL])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scanned = apply_windows(scan_dirs(args), args)
    df, X = load_embeddings(scanned, args, labels)
    if len(df) == 0:
        logger.error("No embeddings found. Check --embeddings_dirs, --lead_times and --init_days.")
        return
    df, X = subsample(df, X, args.max_points_per_series, args.seed)
    logger.info("Collected %d vectors of dim %d across lead times %s",
                X.shape[0], X.shape[1], sorted(df["lead_time"].unique()))

    # One independent PCA -> t-SNE -> figure per lead time.
    written, coord_frames = [], []
    for lead_time in sorted(df["lead_time"].unique()):
        mask = (df["lead_time"] == lead_time).to_numpy()
        sub, X_lead = df.loc[mask].reset_index(drop=True), X[mask]
        logger.info("lead %sh: %d vectors (%s)", lead_time, len(sub),
                    ", ".join(f"{k}={v}" for k, v in sub["label"].value_counts().items()))
        if len(sub) < 3:
            logger.warning("lead %sh: only %d vectors; too few for t-SNE, skipping.",
                           lead_time, len(sub))
            continue

        coords = run_tsne(X_lead, args)
        sub = sub.assign(x=coords[:, 0], y=coords[:, 1])
        out = make_plot(sub, lead_time, series_order, args, theme, output_dir)
        logger.info("  wrote %s", out)
        written.append(out)
        coord_frames.append(sub)

    if not coord_frames:
        logger.error("No lead time had enough points to plot.")
        return

    # Table view: every plotted point with its coordinates, for the record and to satisfy
    # the "identity never colour-alone" requirement for anyone who cannot use the figure.
    csv_path = output_dir / "embedding_tsne_hooked.csv"
    pd.concat(coord_frames, ignore_index=True).to_csv(csv_path, index=False)

    logger.info("Wrote %d figure(s) to %s", len(written), output_dir)
    for out in written:
        logger.info("  - %s", out.name)
    logger.info("Wrote table view %s", csv_path)


if __name__ == "__main__":
    main()
