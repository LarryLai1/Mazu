#!/usr/bin/env python3
"""
Summarize 6h errors from two MSE CSV files and print a table.

Uses the same CSV reading logic as draw_error_plots.py:
- First column is variable name
- Time columns are like "1h", "2h", ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def read_csv_with_time_cols(csv_path: Path):
	"""Read a CSV and return (df, var_col, time_cols_sorted, hours)."""
	df = pd.read_csv(csv_path)
	if df.shape[1] < 2:
		raise ValueError(f"{csv_path}: expected at least 2 columns (variable + time columns).")

	var_col = df.columns[0]

	# Keep only time columns shaped like '1h', '2h', ... and sort by hour
	time_cols = [c for c in df.columns[1:] if pd.Series([c]).astype(str).str.fullmatch(r"\d+h")[0]]
	if not time_cols:
		raise ValueError(f"{csv_path}: no time columns like '1h', '2h', ... were found.")

	time_cols_sorted = sorted(time_cols, key=lambda c: int(str(c).replace("h", "")))
	hours = [int(str(c).replace("h", "")) for c in time_cols_sorted]

	return df, var_col, time_cols_sorted, hours


def build_table(csv_paths: List[Path], hour: int) -> pd.DataFrame:
	"""Return a merged table with the selected hour column for each CSV."""
	tables = []
	for p in csv_paths:
		df, var_col, time_cols_sorted, hours = read_csv_with_time_cols(p)
		target_col = f"{hour}h"
		if target_col not in time_cols_sorted:
			raise ValueError(f"{p}: missing {target_col} column.")

		tmp = df[[var_col, target_col]].copy()
		tmp.columns = ["variable", p.name]
		tables.append(tmp)

	# Outer-join on variable to keep all variables from both CSVs
	merged = tables[0]
	for t in tables[1:]:
		merged = merged.merge(t, on="variable", how="outer")

	return merged


def parse_args():
	p = argparse.ArgumentParser(description="Summarize 6h error table from multiple MSE CSVs.")
	p.add_argument(
		"--csv_paths",
		nargs="+",
		required=True,
		help="Path(s) to the CSV file(s).",
	)
	p.add_argument("--hour", type=int, default=6, help="Forecast hour to summarize.")
	p.add_argument(
		"--sort_by",
		default=None,
		help="Optional column name to sort by (e.g., MSE.csv from a specific path name).",
	)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	csv_paths = [Path(p) for p in args.csv_paths]

	table = build_table(csv_paths, args.hour)
	base_col = table.columns[1]
	other_cols = list(table.columns[2:])

	# Build difference columns: base - each other CSV
	diff_cols = []
	for col in other_cols:
		diff_col = f"{base_col}_minus_{col}"
		table[diff_col] = table[base_col] - table[col]
		diff_cols.append(diff_col)

	# Keep only variable + diffs
	table = table[["variable"] + diff_cols]

	if args.sort_by is not None:
		if args.sort_by not in table.columns:
			raise ValueError(f"--sort_by must be one of: {', '.join(table.columns)}")
		table = table.sort_values(by=args.sort_by, ascending=False)

	# Print a clean, aligned table
	with pd.option_context("display.max_rows", None, "display.max_columns", None):
		print(table.to_string(index=False))


if __name__ == "__main__":
	main()
