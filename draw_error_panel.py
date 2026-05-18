#!/usr/bin/env python3
"""
Generate loss-vs-time plots from one or more CSVs.

UPDATES:
- Added --var_map for custom variable titles.
- Added --xlabel / --ylabel configuration.
- Added --hide_xlabel / --hide_ylabel to toggle visibility.
"""

import argparse
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------- Utilities ---------------------------

def sanitize_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s or "var"


def parse_args():
    p = argparse.ArgumentParser(description="Generate loss-vs-time plots from multiple CSVs.")
    p.add_argument("--csv_paths", nargs="+", required=True,
                   help="Path(s) to the CSV file(s).")
    p.add_argument("--output_dir", default="err_plots", help="Directory to save images.")
    p.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg", "pdf", "svg"], help="Image format.")
    p.add_argument("--dpi", type=int, default=300, help="Image DPI.")
    p.add_argument("--zip", action="store_true", help="Zip all images after saving.")

    # --- Sizing & Layout ---
    p.add_argument("--width", type=float, default=10.0, 
                   help="Figure width (inches).")
    p.add_argument("--height", type=float, default=6.0, 
                   help="Figure height (inches).")
    p.add_argument("--subplot_width", type=float, default=5.0,
                   help="Width of ONE subplot in panel mode.")
    p.add_argument("--subplot_height", type=float, default=3.5,
                   help="Height of ONE subplot in panel mode.")
    
    # --- Spacing & Positioning ---
    p.add_argument("--top_margin", type=float, default=1,
                   help="Top margin limit for subplots (0.0 to 1.0).")
    p.add_argument("--title_y", type=float, default=None,
                   help="Vertical position of the main Figure super-title.")
    p.add_argument("--legend_y", type=float, default=0.98,
                   help="Vertical position (anchor) of the top legend.")

    # --- Aesthetics ---
    p.add_argument("--theme", type=str, default=None,
                   help="Matplotlib style theme.")
    p.add_argument("--bg_color", type=str, default=None,
                   help="Manual background color override.")
    p.add_argument("--alpha", type=float, default=0.9, help="Default alpha.")
    p.add_argument("--markersize", type=float, default=30.0, help="Default markersize.")

    # --- Fonts & Text ---
    p.add_argument("--font_family", type=str, default="sans-serif",
                   help="Font family.")
    p.add_argument("--base_size", type=float, default=12.0,
                   help="Base font size.")
    p.add_argument("--title_size", type=float, default=14.0,
                   help="Font size for subplot titles.")
    p.add_argument("--label_size", type=float, default=12.0,
                   help="Font size for axis labels.")
    p.add_argument("--bold_title", action="store_true", help="Bold subplot titles.")
    p.add_argument("--bold_labels", action="store_true", help="Bold axis labels.")

    # ### NEW ADDITION 1: Axis Labels & Visibility ###
    p.add_argument("--ylabel", type=str, default="MAE", 
                   help="Label for Y axis (default: MAE).")
    p.add_argument("--xlabel", type=str, default="hours", 
                   help="Label for X axis (default: hours).")
    p.add_argument("--hide_ylabel", action="store_true", 
                   help="If set, hides the Y axis label.")
    p.add_argument("--hide_xlabel", action="store_true", 
                   help="If set, hides the X axis label.")

    # selection & layouts
    p.add_argument("--vars", nargs="+", default=None, help="Only include these variables.")
    p.add_argument("--combine", action="store_true", help="Combine variables in ONE axes.")
    p.add_argument("--legend_top", action="store_true", help="Shared legend at the top.")
    p.add_argument("--legend_cols", type=int, default=3, help="Columns for top legend.")

    # panel (grid)
    p.add_argument("--panel", action="store_true", help="Create a panel grid.")
    p.add_argument("--panel_cols", type=int, default=3, help="Columns in panel grid.")
    p.add_argument("--panel_title", default=None, help="Optional overall figure title.")
    p.add_argument("--sharey", action="store_true", help="Share Y axis.")

    # styling
    p.add_argument("--legend_names", nargs="+", default=None, help="Custom legend labels.")
    p.add_argument("--styles", nargs="+", default=None, help="Per-CSV style kwargs.")

    # Title Mapping
    p.add_argument("--var_map", action="append", default=[], 
                   help="Map var names to titles: --var_map loss_val='Validation Loss'")

    return p.parse_args()


def read_csv_with_time_cols(csv_path: Path) -> Tuple[pd.DataFrame, str, List[str], List[int]]:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path}: expected at least 2 columns.")
    var_col = df.columns[0]
    time_cols = [c for c in df.columns[1:] if re.fullmatch(r"\d+h", str(c))]
    if not time_cols:
        raise ValueError(f"{csv_path}: no time columns like '1h' found.")
    time_cols_sorted = sorted(time_cols, key=lambda c: int(str(c).replace("h", "")))
    hours = [int(str(c).replace("h", "")) for c in time_cols_sorted]
    return df, var_col, time_cols_sorted, hours


def collect_all_variables(frames: List[pd.DataFrame], var_cols: List[str]) -> List[str]:
    seen, order = set(), []
    for df, var_col in zip(frames, var_cols):
        for v in df[var_col].astype(str).tolist():
            if v not in seen:
                seen.add(v)
                order.append(v)
    return order


def label_for_index(i: int, path: Path, custom_names: Optional[List[str]]) -> str:
    if custom_names and i < len(custom_names):
        return str(custom_names[i])
    return path.parts[-2] if len(path.parts) >= 2 else path.stem


def parse_style_kwargs(style_str: str) -> Dict[str, Any]:
    style_str = (style_str or "").strip()
    if not style_str: return {}
    out = {}
    parts = [p.strip() for p in style_str.split(",") if p.strip()]
    for part in parts:
        if "=" not in part: continue
        k, v = [x.strip() for x in part.split("=", 1)]
        if v.lower() in {"none", "null"}: v_coerced = None
        elif v.lower() == "true": v_coerced = True
        elif v.lower() == "false": v_coerced = False
        elif re.fullmatch(r"[+-]?\d+", v): v_coerced = int(v)
        elif re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", v): v_coerced = float(v)
        else: v_coerced = v
        out[k] = v_coerced
    return out


def styles_by_csv(csv_paths: List[Path], styles: Optional[List[str]]) -> Dict[Path, Dict[str, Any]]:
    if styles is None: return {p: {} for p in csv_paths}
    return {p: parse_style_kwargs(s) for p, s in zip(csv_paths, styles)}


def apply_theme_and_color(args, fig, ax_list):
    if args.bg_color:
        fig.patch.set_facecolor(args.bg_color)
        for ax in ax_list:
            ax.set_facecolor(args.bg_color)


def plot_one_variable(ax, var_name, csv_paths, row_maps, time_cols_by_file, hours_by_file, 
                      alpha, markersize, legend_names, style_by_file):
    has_any = False
    for i, p in enumerate(csv_paths):
        if var_name not in row_maps[p]: continue
        row = row_maps[p][var_name]
        time_cols = time_cols_by_file[p]
        hours = hours_by_file[p]
        y = pd.to_numeric(row[time_cols], errors="coerce").values
        mask = pd.notna(y)
        if not mask.any(): continue
        xs = [h for h, m in zip(hours, mask) if m]
        ys = [val for val, m in zip(y, mask) if m]
        if not xs: continue

        label = label_for_index(i, p, legend_names)
        style = dict(style_by_file.get(p, {}))
        style.setdefault("alpha", alpha)
        if style.get("marker", None) is not None:
            style.setdefault("markersize", markersize)
        if "linestyle" not in style and "marker" not in style:
            style["linestyle"] = "-"
            style.setdefault("linewidth", 2.0)

        ax.plot(xs, ys, label=label, **style)
        has_any = True
    return has_any


def dedup_legend_from_axes(axes: List[plt.Axes]):
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    seen = {}
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen[lab] = h
    new_labels = list(seen.keys())
    new_handles = [seen[k] for k in new_labels]
    return new_handles, new_labels


# --------------------------- Main ---------------------------

def main():
    args = parse_args()

    # Parse Title Map
    var_mapping = {}
    for m in args.var_map:
        if "=" in m:
            k, v = m.split("=", 1)
            var_mapping[k.strip()] = v.strip()
    
    # Helper to get display name
    get_title = lambda v: var_mapping.get(v, v)

    # 1. Apply Theme & Fonts
    if args.theme:
        try: plt.style.use(args.theme)
        except OSError: print(f"Warning: Style '{args.theme}' not found.")

    plt.rcParams['font.family'] = args.font_family
    plt.rcParams['font.size'] = args.base_size
    plt.rcParams['axes.titlesize'] = args.title_size
    plt.rcParams['axes.labelsize'] = args.label_size
    plt.rcParams['xtick.labelsize'] = args.base_size
    plt.rcParams['ytick.labelsize'] = args.base_size
    plt.rcParams['legend.fontsize'] = args.base_size
    if args.bold_title: plt.rcParams['axes.titleweight'] = 'bold'
    if args.bold_labels: plt.rcParams['axes.labelweight'] = 'bold'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [Path(p) for p in args.csv_paths]
    style_by_file = styles_by_csv(csv_paths, args.styles)

    frames, var_cols, time_cols_by_file, hours_by_file = [], [], {}, {}
    for p in csv_paths:
        df, var_col, time_cols_sorted, hours = read_csv_with_time_cols(p)
        frames.append(df)
        var_cols.append(var_col)
        time_cols_by_file[p] = time_cols_sorted
        hours_by_file[p] = hours

    all_vars = collect_all_variables(frames, var_cols)
    if args.vars:
        # Create a set of existing variables for fast lookup
        existing_vars_set = set(all_vars)
        # Iterate through YOUR list (args.vars) to preserve your order
        # Only keep variables that actually exist in the CSVs
        plot_vars = [v for v in args.vars if v in existing_vars_set]
    else:
        plot_vars = all_vars

    row_maps = {}
    for p, df, var_col in zip(csv_paths, frames, var_cols):
        cols = df.columns.tolist()
        to_series = lambda tup: pd.Series(tup, index=cols)
        row_maps[p] = {str(v): to_series(t) for v, t in zip(df[var_col].astype(str), df.itertuples(index=False))}

    # --- Calculation for Panel/Combined ---
    if args.panel and plot_vars:
        n = len(plot_vars)
        ncols = max(1, args.panel_cols)
        nrows = math.ceil(n / ncols)
        total_w = args.subplot_width * ncols
        total_h = args.subplot_height * nrows
        
        if args.legend_top: total_h += 0.5 

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(total_w, total_h),
                                 sharey=args.sharey, squeeze=False)
        flat_axes = [ax for row in axes for ax in row]
        apply_theme_and_color(args, fig, flat_axes)
        
        any_plotted = False
        for i, var_name in enumerate(plot_vars):
            ax = flat_axes[i]
            if plot_one_variable(ax, var_name, csv_paths, row_maps, time_cols_by_file, hours_by_file,
                                 args.alpha, args.markersize, args.legend_names, style_by_file):
                any_plotted = True
                
                # ### NEW ADDITION 2a: Labels & visibility ###
                
                ax.set_title(get_title(str(var_name)))
                
                if not args.hide_xlabel:
                    ax.set_xlabel(args.xlabel)
                if not args.hide_ylabel:
                    ax.set_ylabel(args.ylabel)
                    
                ax.grid(True)
            else:
                ax.set_visible(False)
        for j in range(len(plot_vars), len(flat_axes)):
            flat_axes[j].set_visible(False)

        if any_plotted:
            save_kwargs = {'dpi': args.dpi, 'bbox_inches': 'tight'}
            if args.bg_color: save_kwargs['facecolor'] = args.bg_color

            if args.legend_top:
                h, l = dedup_legend_from_axes([ax for ax in flat_axes if ax.get_visible()])
                fig.legend(handles=h, labels=l, loc="lower center", 
                           bbox_to_anchor=(0.5, args.legend_y), 
                           ncol=args.legend_cols, frameon=True)
                plt.tight_layout(rect=[0, 0, 1, args.top_margin])
            else:
                for ax in reversed(flat_axes):
                    if ax.get_visible():
                        ax.legend(loc="best")
                        break
                plt.tight_layout()

            if args.panel_title:
                y_title = args.title_y if args.title_y else (args.legend_y + 0.05)
                fig.suptitle(args.panel_title, y=y_title, fontsize=args.title_size+4)

            var_concat = "-".join([sanitize_filename(v) for v in plot_vars])
            fig.savefig(output_dir / f"panel-[{var_concat}_vars].{args.ext}", **save_kwargs)
            plt.close(fig)

        print(f"Saved panel to: {output_dir.resolve()}")
        return

    # --- Combined or Single ---
    if args.combine and plot_vars:
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        apply_theme_and_color(args, fig, [ax])
        any_plotted = False
        for var_name in plot_vars:
            did = plot_one_variable(ax, var_name, csv_paths, row_maps, time_cols_by_file, hours_by_file,
                                    args.alpha, args.markersize, args.legend_names, style_by_file)
            any_plotted = any_plotted or did

        if any_plotted:
            # ### NEW ADDITION 2b: Labels & visibility ###
            if not args.hide_xlabel:
                ax.set_xlabel(args.xlabel)
            if not args.hide_ylabel:
                ax.set_ylabel(args.ylabel)

            ax.set_title(f"Combined: {len(plot_vars)} variables")
            ax.grid(True)
            
            save_kwargs = {'dpi': args.dpi, 'bbox_inches': 'tight'}
            if args.bg_color: save_kwargs['facecolor'] = args.bg_color

            if args.legend_top:
                h, l = ax.get_legend_handles_labels()
                if ax.get_legend(): ax.get_legend().remove()
                fig.legend(handles=h, labels=l, loc="lower center", 
                           bbox_to_anchor=(0.5, args.legend_y),
                           ncol=args.legend_cols)
                plt.tight_layout(rect=[0, 0, 1, args.top_margin])
            else:
                ax.legend(loc="best")
                plt.tight_layout()
            
            fig.savefig(output_dir / f"combined.{args.ext}", **save_kwargs)
            plt.close(fig)

    else:
        # Per variable
        for var_name in plot_vars:
            fig, ax = plt.subplots(figsize=(args.width, args.height))
            apply_theme_and_color(args, fig, [ax])
            if plot_one_variable(ax, var_name, csv_paths, row_maps, time_cols_by_file, hours_by_file,
                                 args.alpha, args.markersize, args.legend_names, style_by_file):
                
                # ### NEW ADDITION 2c: Labels & visibility ###
                if not args.hide_xlabel:
                    ax.set_xlabel(args.xlabel)
                if not args.hide_ylabel:
                    ax.set_ylabel(args.ylabel)

                ax.set_title(get_title(str(var_name)))
                ax.grid(True)
                
                save_kwargs = {'dpi': args.dpi, 'bbox_inches': 'tight'}
                if args.bg_color: save_kwargs['facecolor'] = args.bg_color

                if args.legend_top:
                    h, l = ax.get_legend_handles_labels()
                    if ax.get_legend(): ax.get_legend().remove()
                    fig.legend(handles=h, labels=l, loc="lower center", 
                               bbox_to_anchor=(0.5, args.legend_y),
                               ncol=args.legend_cols)
                    plt.tight_layout(rect=[0, 0, 1, args.top_margin])
                else:
                    ax.legend(loc="best")
                    plt.tight_layout()

                fig.savefig(output_dir / f"{sanitize_filename(var_name)}.{args.ext}", **save_kwargs)
            plt.close(fig)

    if args.zip:
        import shutil
        shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
    print(f"Saved plots to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()