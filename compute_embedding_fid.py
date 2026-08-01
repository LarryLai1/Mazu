#!/usr/bin/env python3
"""
Frechet Inception Distance (FID) between each forecast series and ERA5, per lead time, over the
Swin3D bottleneck embeddings written by AuroraSmallTW_gen_eval_pipeline_with_embeddings.py and
extract_hres_embeddings.py. Produces one CSV and two line charts, both with forecast lead time on
the x axis and one line per series: `embedding_fid.png`, the FID on its own, and
`embedding_fid_terms.png`, three panels side by side holding the two terms the FID is the sum of
and the total -- `||Δμ||²` (labelled MSE: how far apart the clouds' centres sit),
`Tr(Σ₁+Σ₂−2√(Σ₁Σ₂))` (trace: how differently they spread), and FID. The split says whether a
series is drifting away from ERA5 or dispersing around it, which the total alone cannot
distinguish.

This is the numeric counterpart to plot_embedding_tsne_hooked.py. t-SNE shows THAT the series
separate, but its coordinates are arbitrary units from independent per-lead fits and cannot be
tabulated or compared across runs. FID gives a single number per (series, lead time) that can.

Downstream-only: reads .npz files, never loads the model or touches a GPU.


WHICH FEATURE IS SCORED
-----------------------
The .npz files hold the flattened bottleneck tokens, (L*D,) with L=432 tokens and D=1024 channels.
FID needs a sample covariance, so it is computed on the MEAN-POOLED feature -- vec.reshape(L, D)
followed by mean over tokens -- which reconstructs exactly the (D,) vector that both SOFT's
AuroraSmall_FID_extract_embeddings_bottleneck.py and Mazu's own
reference_artifact/AuroraSmallTW_calc_backbone_embedding_FID.py score. Same feature, same tap
point, recovered from files we already have rather than by re-running the model.


WHY PCA IS ON BY DEFAULT
------------------------
With a few hundred initialisation times per lead and D=1024, the sample covariance is rank
deficient (rank n-1 << 1024), so the Frechet distance degenerates into its trace terms: the
cross-covariance term can only ever account for n-1 directions. --pca_dim 64 projects onto a
basis small enough for the covariance to be full rank, which is what makes the number a distance
rather than a sum of variances. --pca_dim 0 reproduces Mazu's unreduced house definition and warns
when n <= d, the same warning SOFT prints.

The PCA basis is fit ONCE over every series and every lead time (--pca_scope global), so all the
points on the chart live in one space and the line is coherent along x. It is fit on the union of
ground truth and all forecasts, so no configuration is privileged.


READING THE RESULT
------------------
FID has no absolute scale, so by default the chart also carries a noise floor: the FID between two
random halves of ERA5 itself at the same lead time. A series sitting at that line is
indistinguishable from ERA5 at this sample size. The floor is estimated from half as many samples
as the series are, so it slightly OVER-states the floor -- a series below it is unambiguously fine.

Caveat on the HRES series: HRES is a different forecast system, not our model under a different
boundary condition. Its distance to ERA5 is small partly because its fields resemble ERA5
systematically, not only because of forecast skill.

Example:
  cd Mazu
  python compute_embedding_fid.py \
    --embeddings_dirs /tmp3/b12902101/mazu_embedding_output/embeddings/hres_boundary0_backbone_res0.25_direct \
                      /tmp3/b12902101/mazu_embedding_output/embeddings/hres_forecast_res0.25 \
    --labels baseline "hres boundary 0.25" \
    --output_dir /tmp3/b12902101/mazu_embedding_output/fid
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA

# The filename-level scan/windowing and the validated colour theme live in the t-SNE script; both
# tools read the same directories with the same --lead_times / --init_days semantics, so they are
# imported rather than restated. scan_dirs/apply_windows read args.embeddings_dirs,
# args.lead_times and args.init_days, all of which this parser also defines.
from plot_embedding_tsne_hooked import (
    GT_LABEL,
    THEME,
    _default_label,
    apply_windows,
    scan_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FLOOR_LABEL = "ERA5 self-split (noise floor)"

# AuroraSmall's Swin3D bottleneck width: embed_dim 256, doubled at each of the two patch merges.
DEFAULT_CHANNELS = 1024

MAX_FORECAST_SERIES = 3


# --------------------------------------------------------------------------------------------
# FID math. Copied verbatim from Mazu's own
# reference_artifact/AuroraSmallTW_calc_backbone_embedding_FID.py so the numbers this script
# produces are directly comparable with the ones that script already reports. (Imported would be
# cleaner, but that module pulls in torch and datasets.DiscriminatorDataset at import time, and
# nothing here needs either.) SOFT's AuroraSmall_FID_compute.py carries the same implementation.
# --------------------------------------------------------------------------------------------
def _covmean(sigma1, sigma2, eps=1e-6):
    """sqrtm(sigma1 . sigma2), with the house version's singularity handling.

    Factored out of calculate_frechet_distance so the term decomposition below shares the exact
    same arithmetic rather than a second copy of it.
    """
    # Product might be almost singular.
    # The house version calls sqrtm(..., disp=False); `disp` is deprecated and removed in SciPy
    # 1.18, so this uses SOFT's spelling of the identical step (tolerate either return shape)
    # rather than emitting a warning on every run. The arithmetic is unchanged.
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):  # older scipy returns (sqrtm, errest)
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    return covmean


def frechet_terms(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """The two halves the Frechet distance is the sum of.

        FID = ||mu1 - mu2||^2  +  Tr(sigma1 + sigma2 - 2 (sigma1 sigma2)^(1/2))
              ^^^ mean term ^^^    ^^^^^^^^^^^ trace term ^^^^^^^^^^^^^^^^^^^^

    The mean term is how far apart the two clouds' centres are: a systematic bias, the same
    offset applied to every sample. The trace term is how differently they are *spread*: it is
    zero only when the two covariances coincide, and it stays positive even for two clouds
    sharing a centre. Splitting them says whether a series is drifting away from ERA5 or
    dispersing around it -- a total FID alone cannot distinguish the two.

    Returns:
        tuple: `(mean_term, trace_term)`, both floats, summing to calculate_frechet_distance().
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean = _covmean(sigma1, sigma2, eps=eps)
    mean_term = float(diff.dot(diff))
    trace_term = float(np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
    return mean_term, trace_term


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mean_term, trace_term = frechet_terms(mu1, sigma1, mu2, sigma2, eps=eps)
    return mean_term + trace_term


def fid_between(real_arr, fake_arr, eps=1e-6):
    """FID between two (n, d) feature matrices."""
    return fid_terms_between(real_arr, fake_arr, eps=eps)["fid"]


def fid_terms_between(real_arr, fake_arr, eps=1e-6):
    """FID between two (n, d) feature matrices, split into its mean and trace terms."""
    mu_real, sigma_real = np.mean(real_arr, axis=0), np.cov(real_arr, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_arr, axis=0), np.cov(fake_arr, rowvar=False)
    mean_term, trace_term = frechet_terms(mu_real, sigma_real, mu_fake, sigma_fake, eps=eps)
    return {"fid": mean_term + trace_term, "mean_term": mean_term, "trace_term": trace_term}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--embeddings_dirs", nargs="+", required=True,
                   help="One or more directories of <INIT>+<LEAD>hr.npz files.")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Legend name per --embeddings_dirs entry. Defaults to each dir's basename.")

    p.add_argument("--lead_times", nargs="*", type=int, default=None,
                   help="Only score these lead times (hours). Default: every lead time found.")
    p.add_argument("--init_days", type=float, default=None,
                   help="Keep only initialisation times within this many days of the earliest one "
                        "found. Default: every initialisation time. Unlike the t-SNE plot there is "
                        "no over-plotting to avoid here, and MORE samples make FID better "
                        "conditioned, so leaving this unset is usually right.")
    p.add_argument("--no_common_inits", action="store_true",
                   help="Score each series on all of its own initialisation times instead of only "
                        "the ones every series shares. FID's small-sample bias grows as d/n, so "
                        "series with different n are not fairly comparable; only pass this if you "
                        "are looking at one series at a time.")

    p.add_argument("--channels", type=int, default=DEFAULT_CHANNELS,
                   help="Bottleneck channel count D. The saved vector is (L*D,) and is pooled over "
                        "its L tokens to recover the (D,) feature SOFT and Mazu's FID script use.")
    p.add_argument("--pca_dim", type=int, default=64,
                   help="Project the pooled feature to this many dims before computing FID, so the "
                        "covariance is full rank. 0 disables it (unreduced house definition).")
    p.add_argument("--pca_scope", choices=["global", "per_lead"], default="global",
                   help="global: one basis fit over every series and lead time, so values are "
                        "comparable along the x axis. per_lead: refit at each lead time.")
    p.add_argument("--no_noise_floor", action="store_true",
                   help="Omit the ERA5 self-split reference line.")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Regularisation added to the covariances if the matrix square root of "
                        "their product comes back non-finite.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--xscale", choices=["linear", "log"], default="linear",
                   help="Lead times are sampled sparsely and unevenly (1, 6, 24, 72, ...); log "
                        "spreads the short leads out, linear keeps distances on the axis true.")
    p.add_argument("--output_dir", type=str, default="/tmp3/b12902101/mazu_embedding_output/fid")
    p.add_argument("--ext", type=str, default="png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--width", type=float, default=8.0)
    p.add_argument("--height", type=float, default=5.5)

    args = p.parse_args()
    if args.labels is not None and len(args.labels) != len(args.embeddings_dirs):
        p.error(f"--labels has {len(args.labels)} entries but --embeddings_dirs has "
                f"{len(args.embeddings_dirs)}.")
    if len(args.embeddings_dirs) > MAX_FORECAST_SERIES:
        p.error(
            f"{len(args.embeddings_dirs)} forecast directories given but the chart is capped at "
            f"{MAX_FORECAST_SERIES}: only the first three categorical slots stay distinguishable "
            f"under colour-vision deficiency, and no four-hue subset of the palette clears that "
            f"gate. The noise floor does not count against this -- it is drawn in a neutral "
            f"reference ink. Drop a directory, or split across two runs."
        )
    return args


def pool(vec, channels):
    """(L*D,) flattened bottleneck tokens -> (D,) mean over tokens.

    This is the inverse of the pipeline's per-sample flatten of the (L, D) hook output, followed
    by SOFT's tokens.mean(dim=1). C-order reshape recovers (L, D) exactly.
    """
    if vec.shape[0] % channels:
        raise ValueError(
            f"Embedding dimension {vec.shape[0]} is not a multiple of --channels {channels}; the "
            f"flattened vector cannot be reshaped to (tokens, channels). Pass the D this run's "
            f"backbone actually used."
        )
    return vec.reshape(-1, channels).mean(axis=0)


def load_pooled(scanned, args, labels):
    """Load and pool every windowed .npz.

    Pooling happens per file, so only (D,) survives in memory -- holding the flattened vectors for
    a run this size would be tens of GB. Ground truth is de-duplicated on (init_time, lead_time)
    because every run re-extracts the same ERA5 states; they are bit-for-bit identical across
    directories, and double-counting them would distort the reference distribution.

    Returns a DataFrame [label, lead_time, init_time] and a parallel (N, D) float64 array.
    """
    rows, vecs = [], []
    seen_gt = set()

    for embeddings_dir, label in zip(args.embeddings_dirs, labels):
        n_pred = n_gt = no_gt_key = 0
        for path, _, _ in scanned[embeddings_dir]:
            with np.load(path, allow_pickle=False) as d:
                lead_time = int(d["lead_time"])
                init_time = str(d["init_time"])
                rows.append({"label": label, "lead_time": lead_time, "init_time": init_time})
                vecs.append(pool(d["pred_embedding"], args.channels))
                n_pred += 1

                if "gt_embedding" in d.files:
                    key = (init_time, lead_time)
                    if key not in seen_gt:
                        seen_gt.add(key)
                        rows.append({"label": GT_LABEL, "lead_time": lead_time,
                                     "init_time": init_time})
                        vecs.append(pool(d["gt_embedding"], args.channels))
                        n_gt += 1
                else:
                    no_gt_key += 1

        if no_gt_key:
            logger.info("%s: %d file(s) carry no gt_embedding (extracted with --skip_gt); ground "
                        "truth for those comes from the other directories.", embeddings_dir, no_gt_key)
        logger.info("%s: %d prediction + %d new ground-truth vectors (label=%r)",
                    embeddings_dir, n_pred, n_gt, label)

    if not vecs:
        return pd.DataFrame(), np.empty((0, 0))

    dims = {v.shape[0] for v in vecs}
    if len(dims) != 1:
        raise ValueError(f"Pooled vectors have inconsistent dimensionality: {sorted(dims)}")

    # float64 throughout: np.cov and the matrix square root are the numerically delicate part, and
    # the arrays are small once pooled.
    return pd.DataFrame(rows), np.stack(vecs).astype(np.float64)


def restrict_to_common_inits(df, X, labels):
    """Keep only the initialisation times every series has, independently at each lead time.

    FID's small-sample bias grows with d/n, so a series scored on more initialisation times gets a
    systematically lower value than one scored on fewer. Equalising n makes the gap between series
    attributable to the forecasts rather than to sample count, and it also means each series is
    scored against the same weather situations.
    """
    keep = []
    for lead, grp in df.groupby("lead_time", sort=True):
        per_label = {lbl: set(grp.loc[grp["label"] == lbl, "init_time"]) for lbl in labels}
        missing = [lbl for lbl, s in per_label.items() if not s]
        if missing:
            logger.warning("lead %sh: no vectors for %s; skipping this lead time.", lead, missing)
            continue

        common = set.intersection(*per_label.values())
        gt_inits = set(grp.loc[grp["label"] == GT_LABEL, "init_time"])
        common &= gt_inits

        dropped = {lbl: len(s) - len(common) for lbl, s in per_label.items() if len(s) != len(common)}
        logger.info("lead %sh: %d initialisation time(s) shared by every series%s",
                    lead, len(common),
                    "" if not dropped else f" (dropped {dropped})")
        keep.append(grp.index[grp["init_time"].isin(common)].to_numpy())

    if not keep:
        return df.iloc[:0], X[:0]
    keep = np.sort(np.concatenate(keep))
    return df.loc[keep].reset_index(drop=True), X[keep]


def reduce_dims(X, args, what):
    """PCA to --pca_dim, or pass through unchanged when it is 0."""
    if not args.pca_dim:
        return X
    n_components = min(args.pca_dim, X.shape[0], X.shape[1])
    if n_components >= X.shape[1]:
        return X
    Xr = PCA(n_components=n_components, random_state=args.seed).fit_transform(X)
    logger.info("PCA (%s): %d x %d -> %d dims", what, X.shape[0], X.shape[1], Xr.shape[1])
    return Xr


def _warn_if_singular(n, d, context):
    if n <= d:
        logger.warning("%s: n_samples=%d <= feature_dim=%d, so the sample covariance is singular "
                       "and this FID is dominated by its trace terms. Use --pca_dim to project "
                       "below n before comparing these values.", context, n, d)


def compute_fid_table(df, X, labels, args):
    """One FID per (series, lead time), plus the ERA5 self-split floor."""
    records = []
    for lead in sorted(df["lead_time"].unique()):
        mask = (df["lead_time"] == lead).to_numpy()
        sub, X_lead = df.loc[mask].reset_index(drop=True), X[mask]
        if args.pca_scope == "per_lead":
            X_lead = reduce_dims(X_lead, args, f"lead {lead}h")

        real = X_lead[(sub["label"] == GT_LABEL).to_numpy()]
        if real.shape[0] < 2:
            logger.warning("lead %sh: only %d ground-truth vector(s); skipping.", lead, real.shape[0])
            continue

        for label in labels:
            fake = X_lead[(sub["label"] == label).to_numpy()]
            if fake.shape[0] < 2:
                logger.warning("lead %sh %s: only %d vector(s); skipping.", lead, label, fake.shape[0])
                continue
            _warn_if_singular(min(real.shape[0], fake.shape[0]), X_lead.shape[1],
                              f"lead {lead}h {label}")
            records.append({"label": label, "lead_time": lead,
                            **fid_terms_between(real, fake, eps=args.eps),
                            "n_real": real.shape[0], "n_fake": fake.shape[0],
                            "feature_dim": X_lead.shape[1], "pca_dim": args.pca_dim})

        # Noise floor: ERA5 against itself. Two disjoint halves of the same distribution, so the
        # true FID is 0 and whatever comes out is the small-sample bias at this n and d -- the
        # scale below which a forecast is indistinguishable from ERA5 here.
        if not args.no_noise_floor and real.shape[0] >= 4:
            order = np.random.default_rng(args.seed).permutation(real.shape[0])
            half = real.shape[0] // 2
            a, b = real[order[:half]], real[order[half:2 * half]]
            records.append({"label": FLOOR_LABEL, "lead_time": lead,
                            **fid_terms_between(a, b, eps=args.eps),
                            "n_real": a.shape[0], "n_fake": b.shape[0],
                            "feature_dim": X_lead.shape[1], "pca_dim": args.pca_dim})

    return pd.DataFrame(records)


def _fmt(v):
    return f"{v:.3g}"


def make_plot(table, series_order, args, theme, output_dir):
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    # Forecast series take the categorical slots in CLI order and keep them regardless of how many
    # series are present. The noise floor is a reference, not a fourth forecast, so it takes the
    # neutral ink -- the same device ground truth uses in plot_embedding_tsne_hooked.py.
    forecasts = [lbl for lbl in series_order if lbl != FLOOR_LABEL]
    colour_of = {label: theme["series"][slot] for slot, label in enumerate(forecasts)}
    colour_of[FLOOR_LABEL] = theme["reference"]

    # Gridlines behind the data: horizontal only, hairline, solid, one step off the surface.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=theme["axis"], linewidth=1.0, linestyle="-", alpha=0.6)

    ends = []
    for label in series_order:
        grp = table[table["label"] == label].sort_values("lead_time")
        if grp.empty:
            continue
        ax.plot(grp["lead_time"], grp["fid"],
                color=colour_of[label], linewidth=2.0,
                linestyle="--" if label == FLOOR_LABEL else "-",
                solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=8,
                markeredgecolor=theme["surface"], markeredgewidth=2,
                label=label, zorder=3)
        ends.append((label, grp["lead_time"].iloc[-1], grp["fid"].iloc[-1]))

    # Bare hour numbers, with the unit carried by the axis label (draw_error_plots.py's
    # "forecast hour" convention). On a linear axis +1h and +6h are ~3% of the width apart, and
    # "+1h"/"+6h" labels overprint each other there; "1"/"6" fit.
    leads = sorted(table["lead_time"].unique())
    ax.set_xscale(args.xscale)
    ax.set_xticks(leads)
    ax.set_xticklabels([str(h) for h in leads])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)  # FID is a distance; a truncated baseline would exaggerate the gaps.
    ax.margins(x=0.12)

    # Selective direct labels: the value at the end of each line, and nothing else. Skipped
    # wholesale if any two would overlap -- nudging them apart detaches them from their lines and
    # the legend plus the CSV already carry identity.
    if ends:
        y_disp = sorted(ax.transData.transform([(0, y) for _, _, y in ends])[:, 1])
        min_gap = min(np.diff(y_disp)) if len(y_disp) > 1 else np.inf
        if min_gap >= 14:  # ~ one line of 9pt text at 150 dpi
            for _, x, y in ends:
                ax.annotate(_fmt(y), xy=(x, y), xytext=(7, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=9, color=theme["text"], zorder=4)
        else:
            logger.info("End labels would collide (%.1f px apart); relying on the legend and CSV.",
                        min_gap)

    dim = int(table["feature_dim"].iloc[0])
    fig.suptitle("Distance from ERA5 in Swin3D bottleneck space",
                 color=theme["text"], fontsize=13, fontweight="600")
    # Kept short enough to fit the figure width; the sample count and date span go in the
    # footnote, and "lower is better" is already on the y axis.
    ax.set_title(f"Frechet distance · pooled {args.channels}-d bottleneck feature"
                 f"{'' if not args.pca_dim else f' → PCA {dim}'}",
                 color=theme["muted"], fontsize=9, pad=8)

    ax.set_xlabel("forecast lead time (hours)", color=theme["muted"], fontsize=9)
    ax.set_ylabel("FID (lower is better)", color=theme["muted"], fontsize=9)
    ax.tick_params(colors=theme["muted"], labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])

    # handlelength 3 so the noise floor's dashed line is actually distinguishable from the solid
    # series in the key; at the default length every handle reads as line-marker-line.
    legend = ax.legend(loc="upper left", frameon=True, fontsize=9, handlelength=3.0,
                       facecolor=theme["surface"], edgecolor=theme["axis"], framealpha=0.95)
    for text in legend.get_texts():
        text.set_color(theme["text"])

    # The caveats that used to sit here as footnotes (equal sample counts, the floor being
    # estimated from half as many samples, HRES being a different forecast system) live in this
    # module's docstring and in public_bash_scripts/compute_embedding_fid.sh instead.
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = output_dir / f"embedding_fid.{args.ext}"
    fig.savefig(out, dpi=args.dpi, facecolor=theme["surface"])
    plt.close(fig)
    return out


def make_terms_plot(table, series_order, args, theme, output_dir):
    """The two halves of the Frechet distance and the total, side by side.

    Three panels rather than one frame with several y axes: the quantities carry different
    magnitudes, and a second scale on the same frame invites reading a crossing that is an
    artefact of the axes. Colour stays bound to the series across all three, so a line is the
    same configuration everywhere; only the quantity changes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(args.width * 2.6, args.height), sharex=True)
    fig.patch.set_facecolor(theme["surface"])

    forecasts = [lbl for lbl in series_order if lbl != FLOOR_LABEL]
    colour_of = {label: theme["series"][slot] for slot, label in enumerate(forecasts)}
    colour_of[FLOOR_LABEL] = theme["reference"]

    leads = sorted(table["lead_time"].unique())
    # Panel titles are the bare quantity names, as asked; the y label carries the formula, and
    # what each one means is in this module's docstring and in frechet_terms().
    panels = [
        ("mean_term", "‖Δμ‖²", "MSE"),
        ("trace_term", "Tr(Σ₁+Σ₂−2√(Σ₁Σ₂))", "trace"),
        ("fid", "MSE + trace", "FID"),
    ]

    for ax, (column, ylabel, panel_title) in zip(axes, panels):
        ax.set_facecolor(theme["surface"])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=theme["axis"], linewidth=1.0, linestyle="-", alpha=0.6)

        for label in series_order:
            grp = table[table["label"] == label].sort_values("lead_time")
            if grp.empty:
                continue
            ax.plot(grp["lead_time"], grp[column],
                    color=colour_of[label], linewidth=2.0,
                    linestyle="--" if label == FLOOR_LABEL else "-",
                    solid_capstyle="round", solid_joinstyle="round",
                    marker="o", markersize=8,
                    markeredgecolor=theme["surface"], markeredgewidth=2,
                    label=label, zorder=3)

        ax.set_xscale(args.xscale)
        ax.set_xticks(leads)
        ax.set_xticklabels([str(h) for h in leads])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)  # all three are non-negative; a cut baseline would mislead.
        ax.margins(x=0.12)
        ax.set_title(panel_title, color=theme["text"], fontsize=11, fontweight="600", pad=8)
        ax.set_xlabel("forecast lead time (hours)", color=theme["muted"], fontsize=9)
        ax.set_ylabel(ylabel, color=theme["muted"], fontsize=9)
        ax.tick_params(colors=theme["muted"], labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme["axis"])

    fig.suptitle("Frechet distance from ERA5: its two terms and the total",
                 color=theme["text"], fontsize=13, fontweight="600")

    # One key for all three panels: the series are identical, so a legend per panel would be the
    # same box three times. handlelength 3 keeps the noise floor's dashes readable.
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, legend_labels, loc="lower center", ncol=min(4, len(handles)),
                        frameon=False, fontsize=9, handlelength=3.0)
    for text in legend.get_texts():
        text.set_color(theme["text"])

    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    out = output_dir / f"embedding_fid_terms.{args.ext}"
    fig.savefig(out, dpi=args.dpi, facecolor=theme["surface"])
    plt.close(fig)
    return out


def main():
    args = parse_args()
    theme = THEME["light"]
    labels = args.labels or [_default_label(d) for d in args.embeddings_dirs]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, X = load_pooled(apply_windows(scan_dirs(args), args), args, labels)
    if len(df) == 0:
        logger.error("No embeddings found. Check --embeddings_dirs, --lead_times and --init_days.")
        return
    if (df["label"] == GT_LABEL).sum() == 0:
        logger.error("No gt_embedding found in any directory; there is nothing to measure the "
                     "forecasts against. Re-extract at least one series without --skip_gt.")
        return

    if not args.no_common_inits:
        df, X = restrict_to_common_inits(df, X, labels)
        if len(df) == 0:
            logger.error("No initialisation time is shared by every series. Pass --no_common_inits "
                         "to score each series on its own, accepting that unequal sample counts "
                         "make them unfair to compare.")
            return

    logger.info("Pooled %d vectors of dim %d across lead times %s",
                X.shape[0], X.shape[1], sorted(df["lead_time"].unique()))

    if args.pca_scope == "global":
        X = reduce_dims(X, args, "global")

    table = compute_fid_table(df, X, labels, args)
    if table.empty:
        logger.error("No lead time had enough samples to compute a FID.")
        return

    csv_path = output_dir / "embedding_fid.csv"
    table.to_csv(csv_path, index=False)

    wide = table.pivot(index="label", columns="lead_time", values="fid")
    wide.columns = [f"+{c}h" for c in wide.columns]
    print("\nFID vs ERA5, Swin3D bottleneck (pooled to "
          f"{args.channels}, {'no PCA' if not args.pca_dim else f'PCA {table.feature_dim.iloc[0]}'})")
    print("=" * 72)
    print(wide.to_string(float_format=lambda v: f"{v:.4f}"))
    print("=" * 72)

    for term, heading in (("mean_term", "Mean term ||dmu||^2"),
                          ("trace_term", "Trace term Tr(S1+S2-2(S1 S2)^1/2)")):
        wide_term = table.pivot(index="label", columns="lead_time", values=term)
        wide_term.columns = [f"+{c}h" for c in wide_term.columns]
        print(f"\n{heading}")
        print("-" * 72)
        print(wide_term.to_string(float_format=lambda v: f"{v:.4f}"))
    print("=" * 72)

    series_order = list(labels) + ([] if args.no_noise_floor else [FLOOR_LABEL])
    out = make_plot(table, series_order, args, theme, output_dir)
    out_terms = make_terms_plot(table, series_order, args, theme, output_dir)

    logger.info("Wrote %s", out)
    logger.info("Wrote %s", out_terms)
    logger.info("Wrote table view %s", csv_path)


if __name__ == "__main__":
    main()
