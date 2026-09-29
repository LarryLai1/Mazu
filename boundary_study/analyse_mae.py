#!/usr/bin/env python3
"""MAE plots for the boundary sweep -- the 3x3 key-variable grid draw_error_plots.py produces.

Two views per factor, both out to 168 h:
    avg   averaged over the 24 init hours of 2020-03-01
    00z   the single rollout initialised 2020-03-01 00:00 UTC

Reads  data/metrics_day_<config>.csv
Writes figs/fig_<factor>_mae_<view>.svg  and  data/summary_mae.json
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

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#7d6ad9"
INK, INK2, GRID = "#111827", "#52514e", "#9ca3af"
VARMAP = {INK: "var(--viz-ink)", INK2: "var(--viz-ink-2)", GRID: "var(--viz-grid)",
          S1: "var(--series-1)", S2: "var(--series-2)", S3: "var(--series-3)",
          S4: "var(--series-4)"}

FACTORS = {
    "resolution": [("base", "0.25° (baseline)", S1), ("res0.5", "0.5°", S2),
                   ("res1.5", "1.5°", S3)],
    "width":      [("w4", "width 4", S2), ("base", "width 8 (baseline)", S1),
                   ("w12", "width 12", S3), ("w16", "width 16", S4)],
    "source":     [("base", "HRES forecast, 6-hourly (baseline)", S1),
                   ("gt", "ERA5 truth, hourly", S3),
                   ("gt6", "ERA5 truth, 6-hourly", S4)],
}
VIEWS = [("avg", "averaged over the 24 init hours of 2020-03-01"),
         ("00z", "single rollout, init 2020-03-01 00:00 UTC")]

# The nine panels of draw_error_plots.py's summary grid.
KEY_PANELS = [("2t", "t2m (K)"), ("msl", "msl (Pa)"),
              ("10u", "u10 (m/s)"), ("10v", "v10 (m/s)"),
              ("z_500", "z_500 (m²/s²)"), ("t_850", "t_850 (K)"),
              ("q_700", "q_700 (kg/kg)"), ("u_850", "u_850 (m/s)"),
              ("v_850", "v_850 (m/s)")]

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.alpha": 0.35, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True, "lines.linewidth": 1.7,
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


_cache = {}


def curves(name, view):
    """MAE against lead time for one config, as {column: Series indexed by lead}."""
    key = (name, view)
    if key in _cache:
        return _cache[key]
    p = DATA / f"metrics_day_{name}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if view == "00z":
        d = d[d.init_hour == 0]
    out = {col: d.groupby("lead")[f"mae_kvint_{col}"].mean() for col, _ in KEY_PANELS}
    _cache[key] = out
    return out


def fig_mae(factor, view, view_lab):
    series = FACTORS[factor]
    if all(curves(n, view) is None for n, _, _ in series):
        return
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.6))
    for ax, (col, ttl) in zip(axes.ravel(), KEY_PANELS):
        for n, lab, c in series:
            cv = curves(n, view)
            if cv is None:
                continue
            g = cv[col]
            ax.plot(g.index, g.values, color=c, label=lab)
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("lead time (h)")
        ax.set_xlim(0, 168)
        ax.set_xticks(range(0, 169, 24))
        ax.grid(True)
    axes[0, 0].set_ylabel("MAE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Factor: {factor} — MAE vs lead time, out to 168 h. {view_lab}. "
                 f"Common interior, 16 cells removed on every side.",
                 x=0.005, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(series), fontsize=9.5,
               bbox_to_anchor=(0.5, 0.004), columnspacing=2.4, handlelength=1.8)
    save(fig, f"fig_{factor}_mae_{view}")


def main():
    for factor in FACTORS:
        for view, view_lab in VIEWS:
            fig_mae(factor, view, view_lab)

    names = sorted({n for s in FACTORS.values() for n, _, _ in s})
    summary = {}
    for view, _ in VIEWS:
        for n in names:
            cv = curves(n, view)
            if cv is None:
                continue
            e = {}
            for col in ("msl", "z_500", "2t"):
                g = cv[col]
                for L in (24, 72, 168):
                    e[f"{col}@{L}h"] = round(float(g.loc[L]), 5)
                e[f"{col}_mean"] = round(float(g.mean()), 5)
            summary[f"{view}/{n}"] = e
    json.dump(summary, open(DATA / "summary_mae.json", "w"), indent=1)
    print("wrote", DATA / "summary_mae.json")

    for view, _ in VIEWS:
        print(f"\n=== {view} — MAE(msl) ===")
        print(f"{'config':>8}{'@24h':>10}{'@72h':>10}{'@168h':>10}{'mean':>10}")
        for n in names:
            k = f"{view}/{n}"
            if k in summary:
                e = summary[k]
                print(f"{n:>8}{e['msl@24h']:10.1f}{e['msl@72h']:10.1f}"
                      f"{e['msl@168h']:10.1f}{e['msl_mean']:10.1f}")


if __name__ == "__main__":
    main()
