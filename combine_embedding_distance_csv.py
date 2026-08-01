#!/usr/bin/env python3
"""
Overlay several `embedding_distance.csv` files (one per run configuration) in one figure set.

Each CSV is what AuroraSmallTW_gen_eval_pipeline_with_embeddings.py writes in its
--embedding_metrics_output_dir mode: one row per lead time, already averaged over init times.
A rollout can only carry one boundary configuration, so comparing configurations means running
the pipeline once per configuration and combining the resulting CSVs here.

Plotting itself is plot_embedding_distance.make_plots, the same function the single-run export
uses, so the combined figures are drawn exactly like the per-run ones -- one line per label.

Example:
  cd Mazu
  python combine_embedding_distance_csv.py \
    --inputs /tmp3/b12902101/mazu_embedding_output/embedding_distance/hres_boundary*_backbone_res0.25_direct \
    --output_dir /tmp3/b12902101/mazu_embedding_output/embedding_distance/combined
"""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from plot_embedding_distance import METRICS, make_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CSV_NAME = "embedding_distance.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help=f"Per-run {CSV_NAME} files, or directories holding one.")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Legend name per --inputs entry. Defaults to the directory name.")
    p.add_argument("--output_dir", required=True,
                   help="Where the combined PNGs (and combined CSV) are written.")
    p.add_argument("--title", type=str, default="",
                   help="Text for the summary figure's title.")
    p.add_argument("--ext", type=str, default="png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--width", type=float, default=8.0)
    p.add_argument("--height", type=float, default=5.0)

    args = p.parse_args()
    if args.labels is not None and len(args.labels) != len(args.inputs):
        p.error(f"--labels has {len(args.labels)} entries but --inputs has {len(args.inputs)}.")
    return args


def resolve_csv(entry: str) -> Path:
    path = Path(entry)
    return path / CSV_NAME if path.is_dir() else path


def default_label(csv_path: Path) -> str:
    """`.../hres_boundary8_backbone_res0.25_direct/embedding_distance.csv` -> the config name."""
    return csv_path.parent.name


def main():
    args = parse_args()

    frames = []
    labels = []
    for i, entry in enumerate(args.inputs):
        csv_path = resolve_csv(entry)
        if not csv_path.exists():
            logger.warning("Skipping %s: no %s", entry, csv_path)
            continue
        label = args.labels[i] if args.labels else default_label(csv_path)
        df = pd.read_csv(csv_path)
        df["label"] = label
        frames.append(df)
        labels.append(label)
        logger.info("Read %s (%d lead times) as '%s'", csv_path, len(df), label)

    if not frames:
        logger.error("No usable CSV found in --inputs.")
        return

    df = pd.concat(frames, ignore_index=True)
    missing = [m for m, _, _ in METRICS if m not in df.columns]
    if missing:
        logger.error("Input CSVs are missing the metric column(s) %s; nothing to plot.", missing)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = output_dir / "embedding_distance_combined.csv"
    df.to_csv(combined_csv, index=False)
    logger.info("Wrote %s (%d rows, %d labels)", combined_csv, len(df), len(labels))

    # make_plots reads only these four attributes off `args` and draws one line per label.
    plot_args = SimpleNamespace(width=args.width, height=args.height, ext=args.ext, dpi=args.dpi)
    make_plots(df, labels, plot_args, output_dir, args.title)
    logger.info("Wrote combined plots to %s", output_dir)


if __name__ == "__main__":
    main()
