#!/usr/bin/env python3
"""Turn the boundary-study runs into the report's figures.

Two families, deliberately kept apart:
  * MAE + embedding distance + FID   -> fig_<factor>.svg
  * wavenumber spectrum              -> fig_<factor>_wavenumber.svg

Each figure carries both time scales: the 2020-03-01 day sweep on the top row, the 2020-03
month sweep on the bottom.
"""

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# Slot 4 is a light violet rather than the palette's yellow: with orange already in slot 2 the
# yellow read as a second orange on screen.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#7d6ad9"

# Every number in the report comes from this lead window, so the day and month sweeps are
# compared over identical lead composition.
MAX_LEAD = 48
INK, INK2, GRID = "#111827", "#52514e", "#9ca3af"
VARMAP = {INK: "var(--viz-ink)", INK2: "var(--viz-ink-2)", GRID: "var(--viz-grid)",
          S1: "var(--series-1)", S2: "var(--series-2)", S3: "var(--series-3)",
          S4: "var(--series-4)"}

# Baseline first in every factor, so the shared config keeps a stable colour.
FACTORS = {
    "resolution": [("base", "0.25° (baseline)", S1), ("res0.5", "0.5°", S2),
                   ("res1.5", "1.5°", S3)],
    "width":      [("w4", "width 4", S2), ("base", "width 8 (baseline)", S1),
                   ("w12", "width 12", S3), ("w16", "width 16", S4)],
    "source":     [("base", "HRES forecast, 6-hourly (baseline)", S1),
                   ("gt", "ERA5 truth, hourly", S3),
                   ("gt6", "ERA5 truth, 6-hourly", S4)],
}
SCALES = [("day", "2020-03-01, 24 init hours"),
          ("month", "2020-03, every hour of every day")]
BURST = (5, 11, 17, 23)

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.alpha": 0.35, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True, "lines.linewidth": 1.8,
})


def save(fig, name):
    if os.environ.get("PREVIEW"):
        fig.savefig(FIGS / f"{name}.png", format="png", dpi=110,
                    bbox_inches="tight", facecolor="white")
    p = FIGS / f"{name}.svg"
    fig.savefig(p, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    svg = p.read_text()
    for hexv, var in VARMAP.items():
        svg = re.sub(hexv, var, svg, flags=re.IGNORECASE)
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg, flags=re.IGNORECASE)
    svg = svg.replace("<svg ", '<svg class="fig" ', 1)
    p.write_text(svg)
    print("wrote", p)


def hour_axis(ax):
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlim(-0.6, 23.6)
    for h in BURST:
        ax.axvline(h, color=GRID, linestyle=":", linewidth=1.1, zorder=0)
    ax.grid(True, axis="y")


# ---------------------------------------------------------------------------------------------

_metrics_cache = {}


def metrics(scale, name):
    key = (scale, name)
    if key not in _metrics_cache:
        p = DATA / f"metrics_{scale}_{name}.csv"
        if not p.exists():
            return None
        d = pd.read_csv(p)
        d = d[d.lead <= MAX_LEAD].copy()
        d["dcos"] = 1 - d.cos_pooled
        for c in ("mae_int_msl", "mae_int_z", "mae_int_2t", "dcos"):
            d["r_" + c] = d[c] / d.groupby("lead")[c].transform("mean")
        _metrics_cache[key] = d
    return _metrics_cache[key]


_full_cache = {}


def metrics_full(scale, name):
    """Same table, no lead cap -- for the error-growth plot, which runs the whole 168 h."""
    key = (scale, name)
    if key not in _full_cache:
        p = DATA / f"metrics_{scale}_{name}.csv"
        if not p.exists():
            return None
        _full_cache[key] = pd.read_csv(p)
    return _full_cache[key]


# The nine panels of draw_error_plots.py's summary grid.
KEY_PANELS = [("2t", "t2m (K)"), ("msl", "msl (Pa)"),
              ("10u", "u10 (m/s)"), ("10v", "v10 (m/s)"),
              ("z_500", "z_500 (m²/s²)"), ("t_850", "t_850 (K)"),
              ("q_700", "q_700 (kg/kg)"), ("u_850", "u_850 (m/s)"),
              ("v_850", "v_850 (m/s)")]


def fig_mae_growth(factor, scale="month"):
    """MAE against lead time out to 168 h -- the plot draw_figure.sh produces, per factor."""
    series = FACTORS[factor]
    if all(metrics_full(scale, n) is None for n, _, _ in series):
        return
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.6))
    for ax, (col, ttl) in zip(axes.ravel(), KEY_PANELS):
        for n, lab, c in series:
            d = metrics_full(scale, n)
            if d is None or f"mae_kvint_{col}" not in d:
                continue
            g = d.groupby("lead")[f"mae_kvint_{col}"].mean()
            ax.plot(g.index, g.values, color=c, label=lab, linewidth=1.7)
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("lead time (h)")
        ax.set_xlim(0, 168)
        ax.set_xticks(range(0, 169, 24))
        ax.grid(True)
    axes[0, 0].set_ylabel("MAE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    scale_lab = dict(SCALES)[scale]
    fig.suptitle(f"Factor: {factor} — MAE vs lead time, out to 168 h ({scale_lab}). "
                 f"Common interior, 16 cells removed on every side.",
                 x=0.005, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(series), fontsize=9.5,
               bbox_to_anchor=(0.5, 0.004), columnspacing=2.4, handlelength=1.8)
    save(fig, f"fig_{factor}_mae")


def frechet(a, b, eps=1e-6):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


_fid_cache = {}


def fid_by_hour(scale, names):
    """FID per clock hour. One PCA basis over every config on this scale, so the numbers are
    comparable across lines within a figure."""
    key = (scale, tuple(sorted(names)))
    if key in _fid_cache:
        return _fid_cache[key]
    paths = {n: DATA / f"emb_{scale}_{n}.npz" for n in names}
    paths = {n: p for n, p in paths.items() if p.exists()}
    if not paths:
        return {}

    # Pass 1: fit the shared PCA basis on a random subsample. At month scale the full stack is
    # ~570k x 1024 vectors; a few thousand per config per side pins a 32-dim basis just as well
    # and keeps this out of swap.
    rng = np.random.default_rng(0)
    subs = []
    for n, p in paths.items():
        z = np.load(p)
        keep = z["lead"] <= MAX_LEAD
        for side in ("pred", "gt"):
            A = z[side][keep]
            take = min(len(A), 4000)
            subs.append(A[rng.choice(len(A), take, replace=False)])
        del z
    pca = PCA(n_components=32, random_state=0).fit(np.concatenate(subs))
    del subs

    # Pass 2: one config resident at a time.
    out = {}
    for n, p in paths.items():
        z = np.load(p)
        keep = z["lead"] <= MAX_LEAD
        P, G = pca.transform(z["pred"][keep]), pca.transform(z["gt"][keep])
        ch = z["clock_hour"][keep]
        del z
        s = {}
        for h in range(24):
            m = ch == h
            if m.sum() > 40:
                s[h] = frechet(G[m], P[m])
        out[n] = pd.Series(s)
    _fid_cache[key] = out
    return out


# ---------------------------------------------------------------------------------------------

def fig_factor(factor):
    series = FACTORS[factor]
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.2))
    fids = {sc: fid_by_hour(sc, [n for n, _, _ in series]) for sc, _ in SCALES}

    for r, (scale, scale_lab) in enumerate(SCALES):
        for n, lab, c in series:
            d = metrics(scale, n)
            if d is None:
                continue
            g = d.groupby("clock_hour")
            axes[r, 0].plot(g.mae_int_msl.mean(), color=c, label=lab)
            axes[r, 1].plot(g.r_mae_int_msl.mean(), color=c, label=lab)
            axes[r, 2].plot(g.dcos.mean(), color=c, label=lab)
            f = fids.get(scale, {}).get(n)
            if f is not None:
                axes[r, 3].plot(f.index, f.values, color=c, label=lab)

        axes[r, 1].axhline(1.0, color=GRID, linewidth=1)
        for cidx, ttl in enumerate(["MAE msl, interior (Pa)",
                                    "MAE msl ÷ same-lead mean",
                                    "embedding distance, 1 − cos",
                                    "FID (PCA-32 bottleneck)"]):
            ax = axes[r, cidx]
            ax.set_title(ttl if r == 0 else "", loc="left")
            ax.set_xlabel("valid time (UTC hour)" if r == 1 else "")
            hour_axis(ax)
        axes[r, 2].set_yscale("log")
        axes[r, 3].set_yscale("log")
        axes[r, 0].set_ylabel(f"{scale_lab}\n", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.suptitle(f"Factor: {factor} — dotted lines mark 05/11/17/23 UTC. "
                 f"Top row: single day. Bottom row: month.",
                 x=0.005, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(series), fontsize=9.5,
               bbox_to_anchor=(0.5, 0.004), columnspacing=2.4, handlelength=1.8)
    save(fig, f"fig_{factor}")


# ---------------------------------------------------------------------------------------------

def spec(scale, name):
    p = DATA / f"spec_{scale}_{name}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    keep = z["lead"] <= MAX_LEAD
    return {"pred": z["pred"][keep], "gt": z["gt"][keep],
            "clock_hour": z["clock_hour"][keep], "lead": z["lead"][keep]}


def fig_wavenumber(factor):
    series = FACTORS[factor]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))

    for r, (scale, scale_lab) in enumerate(SCALES):
        for n, lab, c in series:
            z = spec(scale, n)
            if z is None:
                continue
            k = np.arange(1, z["pred"].shape[-1] + 1)
            for vi, ax in enumerate(axes[r, :2]):
                ratio = (z["pred"][:, vi, :] / np.maximum(z["gt"][:, vi, :], 1e-12)).mean(axis=0)
                ax.plot(k, ratio, color=c, label=lab)
            # high-wavenumber energy vs clock hour: the spectrum tied back to the 6-hourly burst
            hi = (z["pred"][:, 0, -20:] / np.maximum(z["gt"][:, 0, -20:], 1e-12)).mean(axis=1)
            s = pd.Series(hi).groupby(z["clock_hour"]).mean()
            axes[r, 2].plot(s.index, s.values, color=c, label=lab)

        for ax, ttl in zip(axes[r, :2], ["z500 spectrum, forecast ÷ ERA5",
                                         "msl spectrum, forecast ÷ ERA5"]):
            ax.axhline(1.0, color=GRID, linewidth=1)
            ax.set_xscale("log")
            ax.set_title(ttl if r == 0 else "", loc="left")
            ax.set_xlabel("zonal wavenumber (cycles per interior domain)" if r == 1 else "")
            ax.grid(True, which="both", axis="y")
        axes[r, 2].axhline(1.0, color=GRID, linewidth=1)
        axes[r, 2].set_title("z500 high-k energy (top 20 k) ÷ ERA5" if r == 0 else "", loc="left")
        axes[r, 2].set_xlabel("valid time (UTC hour)" if r == 1 else "")
        hour_axis(axes[r, 2])
        axes[r, 0].set_ylabel(f"{scale_lab}\namplitude ratio", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.suptitle(f"Factor: {factor} — wavenumber spectrum, computed on the common interior "
                 f"(16 cells removed on every side)",
                 x=0.005, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(series), fontsize=9.5,
               bbox_to_anchor=(0.5, 0.004), columnspacing=2.4, handlelength=1.8)
    save(fig, f"fig_{factor}_wavenumber")


def main():
    summary = {}
    for factor in FACTORS:
        fig_factor(factor)
        fig_mae_growth(factor)
        fig_wavenumber(factor)

    for scale, _ in SCALES:
        for n in sorted({n for s in FACTORS.values() for n, _, _ in s}):
            d = metrics(scale, n)
            if d is None:
                continue
            g = d.groupby("clock_hour").r_mae_int_msl.mean()
            burst = g[list(BURST)].mean()
            other = g[[h for h in range(24) if h not in BURST]].mean()
            summary[f"{scale}/{n}"] = {
                "mae_int_msl_overall": round(float(d.mae_int_msl.mean()), 3),
                "mae_int_z_overall": round(float(d.mae_int_z.mean()), 3),
                "dcos_overall": round(float(d.dcos.mean()), 5),
                "burst_over_rest_msl": round(float(burst / other), 4),
                "clock_peak_to_trough_msl": round(float(g.max() / g.min()), 4),
            }
    # Wavenumber summary: how much small-scale energy each config carries relative to ERA5,
    # and whether that quantity itself follows the 6-hourly rhythm.
    for scale, _ in SCALES:
        for n in sorted({n for s in FACTORS.values() for n, _, _ in s}):
            z = spec(scale, n)
            if z is None:
                continue
            hi = (z["pred"][:, 0, -20:] / np.maximum(z["gt"][:, 0, -20:], 1e-12)).mean(axis=1)
            ser = pd.Series(hi).groupby(z["clock_hour"]).mean()
            b = ser[list(BURST)].mean()
            o = ser[[h for h in range(24) if h not in BURST]].mean()
            summary.setdefault(f"{scale}/{n}", {}).update({
                "z500_highk_ratio": round(float(hi.mean()), 4),
                "z500_highk_burst_over_rest": round(float(b / o), 4),
            })

    json.dump(summary, open(DATA / "summary.json", "w"), indent=1)
    print("wrote", DATA / "summary.json")

    # Data-driven readout for the wavenumber section of the report.
    rows = []
    for factor, series in FACTORS.items():
        bits = []
        for n, lab, _c in series:
            d = summary.get(f"day/{n}", {}).get("z500_highk_ratio")
            m = summary.get(f"month/{n}", {}).get("z500_highk_ratio")
            bd = summary.get(f"day/{n}", {}).get("z500_highk_burst_over_rest")
            bm = summary.get(f"month/{n}", {}).get("z500_highk_burst_over_rest")
            if d is None:
                continue
            f3 = lambda v: f"{v:.3f}" if v is not None else "—"
            bits.append(f'{lab} <span class="n">{f3(d)}</span>/<span class="n">{f3(m)}</span> '
                        f'(burst/rest <span class="n">{f3(bd)}</span>/'
                        f'<span class="n">{f3(bm)}</span>)')
        rows.append(f"<li><b>{factor}</b>：" + "，".join(bits) + "</li>")
    (HERE / "wavenumber_readout.html").write_text(
        "<p>z500 高波數（最高 20 個波數）能量相對 ERA5 的比值，"
        "格式為「單日 / 月平均」，括號內是同一個量的 burst/rest：</p>\n<ul>\n"
        + "\n".join(rows) + "\n</ul>")
    print("wrote", HERE / "wavenumber_readout.html")
    for k, v in summary.items():
        print(f"{k:18s} MAE {v['mae_int_msl_overall']:8.2f}  "
              f"burst/rest {v['burst_over_rest_msl']:.3f}  "
              f"peak/trough {v['clock_peak_to_trough_msl']:.3f}")


if __name__ == "__main__":
    main()
