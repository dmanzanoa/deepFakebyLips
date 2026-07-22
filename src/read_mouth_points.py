from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_csv_path(csv_arg: str | None, outputs_dir: Path) -> Path:
	if csv_arg:
		candidate = Path(csv_arg)
		if not candidate.is_absolute():
			candidate = outputs_dir / candidate
		return candidate

	csv_files = sorted(outputs_dir.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError(f"No .csv files found in {outputs_dir}")
	return csv_files[0]


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Read a CSV from the outputs folder with pandas"
	)
	parser.add_argument(
		"csv_file",
		nargs="?",
		default=None,
		help="CSV file name inside outputs (example: real_kennedy_mouth_points.csv)",
	)
	parser.add_argument(
		"--rows",
		type=int,
		default=5,
		help="How many rows to preview (default: 5)",
	)
	args = parser.parse_args()

	project_root = Path(__file__).resolve().parents[1]
	outputs_dir = project_root / "outputs"
	csv_path = resolve_csv_path(args.csv_file, outputs_dir)

	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = pd.read_csv(csv_path)

	print(f"Loaded: {csv_path}")
	print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
	print("\nColumns:")
	print(df.columns.tolist())
	print(f"\nFirst {args.rows} rows:")
	print(df.head(args.rows))


if __name__ == "__main__":
	main()
