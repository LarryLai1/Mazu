"""Compare several MAE/MSE CSV files cell-by-cell and report which file wins.

Input CSVs have:
  - first row: empty first cell, then lead-hour headers (e.g. "1h,2h,...,168h")
  - each data row: variable name (e.g. "10u", "2t") followed by numeric values

Usage:
  python compare_mae_csvs.py FILE [FILE ...] [--out-dir DIR] [--ties {first,all}]

Optionally label a file by passing "LABEL=PATH" instead of just "PATH".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_inputs(items: list[str]) -> list[tuple[str, Path]]:
    """Parse positional args into (label, path) pairs. Auto-label if not given."""
    parsed: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for item in items:
        if "=" in item and not os.path.exists(item):
            label, _, raw_path = item.partition("=")
            path = Path(raw_path)
        else:
            path = Path(item)
            label = path.stem
        # Disambiguate duplicate labels by appending a suffix.
        base = label
        i = 2
        while label in seen_labels:
            label = f"{base}#{i}"
            i += 1
        seen_labels.add(label)
        parsed.append((label, path))
    return parsed


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def compare(
    dfs: dict[str, pd.DataFrame],
    ties: str = "first",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (winners_df, win_counts_df, wins_per_variable_df, wins_per_hour_df).

    winners_df: same shape as inputs, each cell = winning label (or "A|B" on ties
                when ties="all").
    win_counts_df: one row per file with total wins and win-rate.
    wins_per_variable_df: for each variable (row), how many times each file wins
                          across all lead hours. Index=variable, columns=files.
    wins_per_hour_df: for each lead hour (row), how many variables each file wins.
                      Index=lead hour, columns=files.
    """
    labels = list(dfs.keys())
    first = dfs[labels[0]]
    shape_ref = (tuple(first.index), tuple(first.columns))
    for lbl, df in dfs.items():
        if (tuple(df.index), tuple(df.columns)) != shape_ref:
            raise ValueError(
                f"File '{lbl}' has a different shape/index/columns than the first file."
            )

    # Stack into (file, var, lead) array.
    stack = np.stack([dfs[lbl].to_numpy(dtype=float) for lbl in labels], axis=0)
    mins = np.nanmin(stack, axis=0, keepdims=True)
    is_min = stack == mins  # shape (F, V, L); True where cell equals the column min.

    if ties == "first":
        winner_idx = np.argmax(is_min, axis=0)  # first True index
        winner_labels_arr = np.array(labels)[winner_idx]
    elif ties == "all":
        labels_arr = np.array(labels, dtype=object)
        # Build "A|B" strings where multiple files tie.
        V, L = is_min.shape[1], is_min.shape[2]
        winner_labels_arr = np.empty((V, L), dtype=object)
        for v in range(V):
            for l in range(L):
                winner_labels_arr[v, l] = "|".join(labels_arr[is_min[:, v, l]])
    else:
        raise ValueError(f"Unknown ties mode: {ties!r}")

    winners_df = pd.DataFrame(
        winner_labels_arr, index=first.index, columns=first.columns
    )

    # Win counts: for fairness under ties, credit every tied file with 1/k of a win.
    tie_counts = is_min.sum(axis=0)  # (V, L)
    credit = is_min.astype(float) / tie_counts  # broadcasts: each cell sums to 1
    wins_per_file_per_var = credit.sum(axis=2)  # (F, V); sum over lead hours
    wins_per_file_per_hour = credit.sum(axis=1)  # (F, L); sum over variables
    total_wins = wins_per_file_per_var.sum(axis=1)  # (F,)
    total_cells = first.size
    win_counts_df = pd.DataFrame(
        {
            "file": labels,
            "wins": total_wins,
            "win_rate": total_wins / total_cells,
        }
    ).set_index("file")

    wins_per_variable_df = pd.DataFrame(
        wins_per_file_per_var.T, index=first.index, columns=labels
    )
    wins_per_variable_df.index.name = "variable"

    wins_per_hour_df = pd.DataFrame(
        wins_per_file_per_hour.T, index=first.columns, columns=labels
    )
    wins_per_hour_df.index.name = "lead_hour"

    return winners_df, win_counts_df, wins_per_variable_df, wins_per_hour_df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "files",
        nargs="+",
        help="CSV paths, optionally prefixed with 'LABEL=' to override the label.",
    )
    p.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write winners.csv / win_counts.csv / wins_per_variable.csv / wins_per_hour.csv.",
    )
    p.add_argument(
        "--ties",
        choices=["first", "all"],
        default="first",
        help="On ties, record only the first file ('first') or all tied files joined by '|' ('all'). "
        "Win counts always split credit equally across tied files.",
    )
    args = p.parse_args(argv)

    if len(args.files) < 2:
        p.error("Need at least 2 CSV files to compare.")

    pairs = parse_inputs(args.files)
    dfs = {lbl: load_csv(path) for lbl, path in pairs}

    winners_df, win_counts_df, wins_per_variable_df, wins_per_hour_df = compare(
        dfs, ties=args.ties
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    winners_df.to_csv(out / "winners.csv")
    win_counts_df.to_csv(out / "win_counts.csv")
    wins_per_variable_df.to_csv(out / "wins_per_variable.csv")
    wins_per_hour_df.to_csv(out / "wins_per_hour.csv")

    total_cells = next(iter(dfs.values())).size
    n_vars, n_hours = winners_df.shape
    print(f"Compared {len(dfs)} files on {total_cells} cells "
          f"({n_vars} variables x {n_hours} lead hours).")
    print()
    print("Win totals (ties split equally):")
    print(win_counts_df.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"Wins per variable (rows=variable, cols=file; each row sums to {n_hours}):")
    print(wins_per_variable_df.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"Wins per lead hour (rows=hour, cols=file; each row sums to {n_vars}):")
    print(wins_per_hour_df.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"Wrote: {out/'winners.csv'}")
    print(f"Wrote: {out/'win_counts.csv'}")
    print(f"Wrote: {out/'wins_per_variable.csv'}")
    print(f"Wrote: {out/'wins_per_hour.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
