#!/usr/bin/env python3
"""
Make one loss-vs-time plot per variable from one or more CSVs like:

  | <variable> | 1h | 2h | ... | 96h |

Each plot contains one series per CSV, so you can compare runs.

UPDATED (per your request):
- You can now specify a custom style PER INPUT CSV via --styles, using explicit
  matplotlib kwargs like:
    "marker=o,linestyle=--,linewidth=2,markersize=6,alpha=0.8"

Examples:
  python make_loss_scatter_plots.py \
    --csv_paths run1.csv run2.csv run3.csv \
    --legend_names "Run 1" "Run 2" "Run 3" \
    --styles "linestyle=-,linewidth=2" "linestyle=--,linewidth=2" "marker=*,linestyle=None,markersize=12" \
    --output_dir plots --ext png --dpi 150 --zip
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd


UPPER_BASE_VARS = ["u", "v", "t", "q", "z"]
UPPER_PRESSURES = [1000, 925, 850, 700, 500, 300, 150, 50]


def sanitize_filename(s: str) -> str:
    """Turn a variable name into a safe filename chunk."""
    s = str(s).strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)  # keep letters, numbers, _, -, .
    return s or "var"


def parse_args():
    p = argparse.ArgumentParser(description="Generate one plot per variable from multiple CSVs.")
    p.add_argument(
        "--csv_paths",
        nargs="+",
        required=True,
        help="Path(s) to the CSV file(s). Example: --csv_paths 96hrs_a.csv 96hrs_b.csv",
    )
    p.add_argument(
        "--legend_names",
        nargs="+",
        help="Optional legend_names for each CSV (same length and order as --csv_paths).",
    )
    p.add_argument(
        "--styles",
        nargs="+",
        help=(
            "Optional per-CSV matplotlib style kwargs (same length/order as --csv_paths).\n"
            "Format: 'key=val,key=val,...' e.g.\n"
            "  --styles 'linestyle=-,linewidth=2' 'marker=o,linestyle=None,markersize=6'\n"
            "Supported keys are whatever plt.plot accepts (marker, linestyle, linewidth, markersize, alpha, etc.)."
        ),
    )
    p.add_argument("--output_dir", default="err_plots", help="Directory to save images.")
    p.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg", "pdf", "svg"], help="Image format.")
    p.add_argument("--dpi", type=int, default=300, help="Image DPI.")
    p.add_argument("--width", type=float, default=6.0, help="Figure width (inches).")
    p.add_argument("--height", type=float, default=6.0, help="Figure height (inches).")
    p.add_argument("--alpha", type=float, default=0.9, help="Default alpha if not provided in --styles.")
    p.add_argument("--markersize", type=float, default=30.0, help="Default markersize if not provided in --styles.")
    p.add_argument("--zip", action="store_true", help="Zip all images after saving.")
    return p.parse_args()


def read_csv_with_time_cols(csv_path: Path):
    """Read a CSV and return (df, var_col, time_cols_sorted, hours)."""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path}: expected at least 2 columns (variable + time columns).")

    var_col = df.columns[0]

    # Keep only time columns shaped like '1h', '2h', ... and sort by hour
    time_cols = [c for c in df.columns[1:] if re.fullmatch(r"\d+h", str(c))]
    if not time_cols:
        raise ValueError(f"{csv_path}: no time columns like '1h', '2h', ... were found.")

    time_cols_sorted = sorted(time_cols, key=lambda c: int(str(c).replace("h", "")))
    hours = [int(str(c).replace("h", "")) for c in time_cols_sorted]

    return df, var_col, time_cols_sorted, hours


def collect_all_variables(frames: List[pd.DataFrame], var_cols: List[str]) -> List[str]:
    """Union of variable names across all CSVs (preserving rough order)."""
    seen = set()
    order = []
    for df, var_col in zip(frames, var_cols):
        for v in df[var_col].astype(str).tolist():
            if v not in seen:
                seen.add(v)
                order.append(v)
    return order


def run_label(path: Path) -> str:
    """Return the parent folder name (i.e. the second-to-last component) for a path.

    Example:
    /home/user/experiments/runA/metrics.csv -> "runA"
    """
    if len(path.parts) >= 3:
        return path.parts[-3]
    return path.stem


def _coerce_value(v: str) -> Any:
    """Best-effort coercion for style values."""
    v = v.strip()
    if v.lower() in {"none", "null"}:
        return None
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    # try int
    try:
        if re.fullmatch(r"[+-]?\d+", v):
            return int(v)
    except Exception:
        pass
    # try float
    try:
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", v):
            return float(v)
    except Exception:
        pass
    return v  # keep as string


def parse_style_kwargs(style_str: str) -> Dict[str, Any]:
    """
    Parse "key=val,key=val,..." into a dict for plt.plot(**kwargs).
    Example: "marker=o,linestyle=--,linewidth=2,markersize=6,alpha=0.8"
    """
    style_str = (style_str or "").strip()
    if not style_str:
        return {}

    out: Dict[str, Any] = {}
    parts = [p.strip() for p in style_str.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise ValueError(
                f"Bad --styles token '{style_str}'. "
                f"Expected 'key=val,key=val,...' (comma-separated). Problem part: '{part}'"
            )
        k, v = [x.strip() for x in part.split("=", 1)]
        out[k] = _coerce_value(v)
    return out


def split_upper_variable_name(name: str):
    """Return (base_var, pressure) for names like q_1000, or None otherwise."""
    m = re.fullmatch(r"([uvtqz])_(1000|925|850|700|500|300|150|50)", str(name))
    if not m:
        return None
    return m.group(1), int(m.group(2))


def plot_series(ax, xs, ys, label: str, style: Dict[str, Any], args):
    """Plot one CSV series with the script's style defaults applied."""
    plot_style = dict(style)
    plot_style.setdefault("alpha", args.alpha)
    if plot_style.get("marker", None) is not None:
        plot_style.setdefault("markersize", args.markersize)
    if "linestyle" not in plot_style and "marker" not in plot_style:
        plot_style["linestyle"] = "-"
        plot_style.setdefault("linewidth", 2.0)

    ax.plot(xs, ys, label=label, **plot_style)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in args.csv_paths]

    # Determine legend_names for each CSV
    if args.legend_names is not None:
        if len(args.legend_names) != len(csv_paths):
            raise ValueError(
                f"--legend_names must have the same length as --csv_paths "
                f"(got {len(args.legend_names)} legend_names for {len(csv_paths)} CSVs)."
            )
        label_by_file: Dict[Path, str] = {p: lbl for p, lbl in zip(csv_paths, args.legend_names)}
    else:
        label_by_file = {p: run_label(p) for p in csv_paths}

    # Determine style kwargs per CSV
    if args.styles is not None:
        if len(args.styles) != len(csv_paths):
            raise ValueError(
                f"--styles must have the same length as --csv_paths "
                f"(got {len(args.styles)} styles for {len(csv_paths)} CSVs)."
            )
        style_by_file: Dict[Path, Dict[str, Any]] = {
            p: parse_style_kwargs(s) for p, s in zip(csv_paths, args.styles)
        }
    else:
        style_by_file = {p: {} for p in csv_paths}

    # Read all CSVs and cache their time axes
    frames: List[pd.DataFrame] = []
    var_cols: List[str] = []
    time_cols_by_file: Dict[Path, List[str]] = {}
    hours_by_file: Dict[Path, List[int]] = {}

    for p in csv_paths:
        df, var_col, time_cols_sorted, hours = read_csv_with_time_cols(p)
        frames.append(df)
        var_cols.append(var_col)
        time_cols_by_file[p] = time_cols_sorted
        hours_by_file[p] = hours

    # Build union of all variable names present
    all_vars = collect_all_variables(frames, var_cols)

    upper_vars_present = {
        base
        for var_name in all_vars
        for split in [split_upper_variable_name(var_name)]
        if split is not None
        for base, _pressure in [split]
    }

    # For quick lookup by (file -> variable -> row)
    row_maps: Dict[Path, Dict[str, pd.Series]] = {}
    for p, df, var_col in zip(csv_paths, frames, var_cols):
        df_cols = df.columns.tolist()

        def tuple_to_series(tup):
            return pd.Series(tup, index=df_cols)

        row_maps[p] = {
            str(v): tuple_to_series(tup)
            for v, tup in zip(df[var_col].astype(str).tolist(), df.itertuples(index=False, name=None))
        }

    # Generate one combined figure per upper-variable base (u, v, t, q, z).
    for base_var in [v for v in UPPER_BASE_VARS if v in upper_vars_present]:
        fig, axes = plt.subplots(
            2,
            4,
            figsize=(args.width * 3.0, args.height * 2.0),
            sharex=True,
            sharey=False,
        )
        axes_flat = axes.flatten()
        any_data = False

        for ax, pressure in zip(axes_flat, UPPER_PRESSURES):
            var_name = f"{base_var}_{pressure}"
            has_any = False

            for p in csv_paths:
                if var_name not in row_maps[p]:
                    continue

                row = row_maps[p][var_name]
                time_cols = time_cols_by_file[p]
                hours = hours_by_file[p]

                y = pd.to_numeric(row[time_cols], errors="coerce").values
                mask = pd.notna(y)
                if not mask.any():
                    continue

                xs = [h for h, m in zip(hours, mask) if m]
                ys = [val for val, m in zip(y, mask) if m]
                if not xs:
                    continue

                plot_series(
                    ax,
                    xs,
                    ys,
                    label_by_file[p],
                    style_by_file.get(p, {}),
                    args,
                )
                has_any = True
                any_data = True

            ax.set_title(f"{pressure} hPa")
            ax.grid(True)
            if ax in axes_flat[::4]:
                ax.set_ylabel("loss value")
            if ax in axes_flat[-4:]:
                ax.set_xlabel("forecast hour")
            if has_any:
                ax.legend(loc="best", frameon=True)

        if any_data:
            fig.suptitle(base_var)
            fig.tight_layout()
            fig.savefig(output_dir / f"{base_var}_upper_levels.{args.ext}", dpi=args.dpi)
        plt.close(fig)

    # Generate one plot per variable, adding a series for each CSV that contains it
    for var_name in all_vars:
        if split_upper_variable_name(var_name) is not None:
            continue

        plt.figure(figsize=(args.width, args.height))

        has_any = False
        for p in csv_paths:
            if var_name not in row_maps[p]:
                continue  # this CSV doesn't have the variable

            row = row_maps[p][var_name]
            time_cols = time_cols_by_file[p]
            hours = hours_by_file[p]

            # pull y values, skip non-numeric safely
            y = pd.to_numeric(row[time_cols], errors="coerce").values
            mask = pd.notna(y)

            if not mask.any():
                continue

            xs = [h for h, m in zip(hours, mask) if m]
            ys = [val for val, m in zip(y, mask) if m]
            if not xs:
                continue

            plot_series(plt.gca(), xs, ys, label_by_file[p], style_by_file.get(p, {}), args)
            has_any = True

        if not has_any:
            plt.close()
            continue

        plt.xlabel("forecast hour")
        plt.ylabel("loss value")
        plt.tick_params(axis="x", pad=6)
        plt.tick_params(axis="y", pad=6)
        plt.title(f"{var_name}")
        plt.grid(True)
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()

        fname = f"{sanitize_filename(var_name)}.{args.ext}"
        plt.savefig(output_dir / fname, dpi=args.dpi)
        plt.close()

    # Generate a combined 3x3 plot of the 9 key variables
    # t2m (2t), msl, u10 (10u), v10 (10v), z_500, t_850, q_700, u_850, v_850
    key_vars = ["2t", "msl", "10u", "10v", "z_500", "t_850", "q_700", "u_850", "v_850"]
    key_vars_labels = {
        "2t": "t2m",
        "msl": "msl",
        "10u": "u10",
        "10v": "v10",
        "z_500": "z_500",
        "t_850": "t_850",
        "q_700": "q_700",
        "u_850": "u_850",
        "v_850": "v_850",
    }
    available_key_vars = [v for v in key_vars if any(v in row_maps[p] for p in csv_paths)]
    if available_key_vars:
        fig, axes = plt.subplots(
            3,
            3,
            figsize=(args.width * 3.0, args.height * 3.0),
            sharex=True,
            sharey=False,
        )
        axes_flat = axes.flatten()
        any_data = False

        for i, csv_var in enumerate(key_vars):
            ax = axes_flat[i]
            display_name = key_vars_labels[csv_var]
            has_any = False

            for p in csv_paths:
                if csv_var not in row_maps[p]:
                    continue

                row = row_maps[p][csv_var]
                time_cols = time_cols_by_file[p]
                hours = hours_by_file[p]

                y = pd.to_numeric(row[time_cols], errors="coerce").values
                mask = pd.notna(y)
                if not mask.any():
                    continue

                xs = [h for h, m in zip(hours, mask) if m]
                ys = [val for val, m in zip(y, mask) if m]
                if not xs:
                    continue

                plot_series(
                    ax,
                    xs,
                    ys,
                    label_by_file[p],
                    style_by_file.get(p, {}),
                    args,
                )
                has_any = True
                any_data = True

            ax.set_title(display_name)
            ax.grid(True)
            if i % 3 == 0:
                ax.set_ylabel("loss value")
            if i >= 6:
                ax.set_xlabel("forecast hour")
            if has_any:
                ax.legend(loc="best", frameon=True)

        if any_data:
            fig.suptitle("Key Variables Error Comparison", fontsize=16)
            fig.tight_layout()
            fig.savefig(output_dir / f"combined_key_vars.{args.ext}", dpi=args.dpi)
        plt.close(fig)

    # Optionally zip all images
    if args.zip:
        import shutil

        zip_base = output_dir.parent / f"{output_dir.name}"
        shutil.make_archive(str(zip_base), "zip", root_dir=output_dir)
        print(f"Zipped to: {zip_base}.zip")

    print(f"Saved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
