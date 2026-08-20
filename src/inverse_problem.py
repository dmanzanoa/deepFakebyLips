from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8
SPATIAL_NORMALIZATION_CHOICES = ("none", "center_scale", "center_scale_rotate")


def read_wide_mouth_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
	df = pd.read_csv(csv_path)

	required = {"timestamp_ms", "detected"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

	value_columns = [
		col
		for col in df.columns
		if col.startswith("x_") or col.startswith("y_")
	]
	if not value_columns:
		raise ValueError(
			f"No mouth coordinate columns found in {csv_path}. "
			"Expected columns like x_0, y_0, x_13, y_13, ..."
		)

	# Keep only rows where landmarks were detected and coordinates are numeric.
	detected_df = df[df["detected"] == 1].copy()
	for col in value_columns:
		detected_df[col] = pd.to_numeric(detected_df[col], errors="coerce")
	detected_df = detected_df.dropna(subset=["timestamp_ms", *value_columns])

	if len(detected_df) < 5:
		raise ValueError(
			"Not enough valid detected rows to fit oscillator model. "
			f"Found {len(detected_df)} rows."
		)

	t = detected_df["timestamp_ms"].to_numpy(dtype=float) / 1000.0
	y = detected_df[value_columns].to_numpy(dtype=float)

	# Normalize time origin for better numerical stability.
	t = t - t[0]
	return t, y, value_columns


def list_wide_mouth_csvs(csv_dir: Path) -> list[Path]:
	if not csv_dir.exists():
		raise ValueError(f"Directory does not exist: {csv_dir}")
	if not csv_dir.is_dir():
		raise ValueError(f"Expected a directory path, got: {csv_dir}")

	paths = [p for p in csv_dir.glob("*.csv") if p.is_file()]
	if not paths:
		raise ValueError(f"No CSV files found in directory: {csv_dir}")

	def _sort_key(path: Path) -> tuple[int, int | str]:
		stem = path.stem
		if stem.isdigit():
			return (0, int(stem))
		return (1, stem)

	return sorted(paths, key=_sort_key)


def load_training_sequences(
	csv_paths: list[Path],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
	if not csv_paths:
		raise ValueError("No training CSV paths were provided.")

	sequences: list[tuple[np.ndarray, np.ndarray]] = []
	feature_names: list[str] | None = None

	for csv_path in csv_paths:
		time_s, values, current_features = read_wide_mouth_csv(csv_path)
		if feature_names is None:
			feature_names = current_features
		elif current_features != feature_names:
			raise ValueError(
				"Feature mismatch across training CSV files. "
				f"File {csv_path} has a different set/order of columns."
			)
		sequences.append((time_s, values))

	if feature_names is None:
		raise ValueError("Unable to load feature names from training CSV files.")

	return sequences, feature_names


def _xy_pair_indices(feature_names: list[str]) -> tuple[list[int], list[int]]:
	x_by_id: dict[str, int] = {}
	y_by_id: dict[str, int] = {}

	for idx, name in enumerate(feature_names):
		if name.startswith("x_"):
			x_by_id[name[2:]] = idx
		elif name.startswith("y_"):
			y_by_id[name[2:]] = idx

	common_ids = sorted(set(x_by_id) & set(y_by_id), key=lambda s: (not s.isdigit(), s if not s.isdigit() else int(s)))
	x_idx = [x_by_id[k] for k in common_ids]
	y_idx = [y_by_id[k] for k in common_ids]
	return x_idx, y_idx


def spatially_normalize_values(
	values: np.ndarray,
	feature_names: list[str],
	mode: str,
) -> np.ndarray:
	if mode == "none":
		return values

	if mode not in SPATIAL_NORMALIZATION_CHOICES:
		raise ValueError(
			f"Unsupported spatial normalization mode: {mode}. "
			f"Expected one of: {SPATIAL_NORMALIZATION_CHOICES}."
		)

	x_idx, y_idx = _xy_pair_indices(feature_names)
	if len(x_idx) < 2:
		raise ValueError(
			"Spatial normalization requires at least two x/y landmark pairs. "
			f"Found {len(x_idx)} valid pairs."
		)

	norm = values.copy()
	x = norm[:, x_idx]
	y = norm[:, y_idx]

	center_x = np.mean(x, axis=1, keepdims=True)
	center_y = np.mean(y, axis=1, keepdims=True)
	x = x - center_x
	y = y - center_y

	scale = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)
	scale = np.maximum(scale, EPS)
	x = x / scale
	y = y / scale

	if mode == "center_scale_rotate":
		left_idx = np.argmin(x, axis=1)
		right_idx = np.argmax(x, axis=1)
		row_idx = np.arange(x.shape[0])
		vx = x[row_idx, right_idx] - x[row_idx, left_idx]
		vy = y[row_idx, right_idx] - y[row_idx, left_idx]
		angles = np.arctan2(vy, vx)
		cos_a = np.cos(angles)[:, None]
		sin_a = np.sin(angles)[:, None]

		x_rot = cos_a * x + sin_a * y
		y_rot = -sin_a * x + cos_a * y
		x, y = x_rot, y_rot

	norm[:, x_idx] = x
	norm[:, y_idx] = y
	return norm


def fit_sho_coefficients(time_s: np.ndarray, signal: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
	design = np.column_stack(
		[
			np.cos(omega * time_s),
			np.sin(omega * time_s),
			np.ones_like(time_s),
		]
	)
	coef, _, _, _ = np.linalg.lstsq(design, signal, rcond=None)
	reconstruction = design @ coef
	return coef, reconstruction


def mean_rms_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def finite_diff_second_derivative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
	first = np.gradient(values, time_s, axis=0)
	second = np.gradient(first, time_s, axis=0)
	return second


def normalized_physics_residual(values: np.ndarray, time_s: np.ndarray, omega: float) -> np.ndarray:
	centered = values - np.mean(values, axis=0, keepdims=True)
	second = finite_diff_second_derivative(centered, time_s)
	residual = second + (omega**2) * centered
	scale = np.std(centered, axis=0, keepdims=True) + EPS
	return residual / scale


def search_best_omega(
	time_s: np.ndarray,
	values: np.ndarray,
	freq_min_hz: float,
	freq_max_hz: float,
	n_grid: int,
) -> float:
	freqs = np.linspace(freq_min_hz, freq_max_hz, n_grid)
	best_freq = freqs[0]
	best_error = float("inf")

	for freq in freqs:
		omega = 2.0 * np.pi * freq
		preds = np.zeros_like(values)
		for j in range(values.shape[1]):
			_, rec = fit_sho_coefficients(time_s, values[:, j], omega)
			preds[:, j] = rec
		error = mean_rms_error(values, preds)
		if error < best_error:
			best_error = error
			best_freq = freq

	return float(2.0 * np.pi * best_freq)


def search_best_omega_for_sequences(
	sequences: list[tuple[np.ndarray, np.ndarray]],
	freq_min_hz: float,
	freq_max_hz: float,
	n_grid: int,
) -> float:
	freqs = np.linspace(freq_min_hz, freq_max_hz, n_grid)
	best_freq = freqs[0]
	best_error = float("inf")

	for freq in freqs:
		omega = 2.0 * np.pi * freq
		total_squared_error = 0.0
		total_values = 0

		for time_s, values in sequences:
			preds = np.zeros_like(values)
			for j in range(values.shape[1]):
				_, rec = fit_sho_coefficients(time_s, values[:, j], omega)
				preds[:, j] = rec

			diff = values - preds
			total_squared_error += float(np.sum(diff**2))
			total_values += diff.size

		error = float(np.sqrt(total_squared_error / max(total_values, 1)))
		if error < best_error:
			best_error = error
			best_freq = freq

	return float(2.0 * np.pi * best_freq)


@dataclass
class SHOMouthModel:
	omega: float
	feature_names: list[str]
	recon_threshold: float
	physics_threshold: float
	normalization_mode: str = "none"

	@property
	def frequency_hz(self) -> float:
		return self.omega / (2.0 * np.pi)

	def to_dict(self) -> dict:
		return {
			"omega": self.omega,
			"frequency_hz": self.frequency_hz,
			"feature_names": self.feature_names,
			"recon_threshold": self.recon_threshold,
			"physics_threshold": self.physics_threshold,
			"normalization_mode": self.normalization_mode,
		}

	@classmethod
	def from_dict(cls, payload: dict) -> "SHOMouthModel":
		return cls(
			omega=float(payload["omega"]),
			feature_names=list(payload["feature_names"]),
			recon_threshold=float(payload["recon_threshold"]),
			physics_threshold=float(payload["physics_threshold"]),
			normalization_mode=str(payload.get("normalization_mode", "none")),
		)


def fit_model(
	csv_path: Path,
	freq_min_hz: float,
	freq_max_hz: float,
	n_grid: int,
	percentile: float,
	threshold_scale: float,
	normalization_mode: str,
) -> SHOMouthModel:
	return fit_model_from_csv_paths(
		csv_paths=[csv_path],
		freq_min_hz=freq_min_hz,
		freq_max_hz=freq_max_hz,
		n_grid=n_grid,
		percentile=percentile,
		threshold_scale=threshold_scale,
		normalization_mode=normalization_mode,
	)


def fit_model_from_csv_paths(
	csv_paths: list[Path],
	freq_min_hz: float,
	freq_max_hz: float,
	n_grid: int,
	percentile: float,
	threshold_scale: float,
	normalization_mode: str,
) -> SHOMouthModel:
	sequences, feature_names = load_training_sequences(csv_paths)
	normalized_sequences = [
		(time_s, spatially_normalize_values(values, feature_names, normalization_mode))
		for time_s, values in sequences
	]
	omega = search_best_omega_for_sequences(
		normalized_sequences,
		freq_min_hz=freq_min_hz,
		freq_max_hz=freq_max_hz,
		n_grid=n_grid,
	)

	recon_error_parts: list[np.ndarray] = []
	physics_norm_parts: list[np.ndarray] = []

	for time_s, values in normalized_sequences:
		reconstructed = np.zeros_like(values)
		for j in range(values.shape[1]):
			_, rec = fit_sho_coefficients(time_s, values[:, j], omega)
			reconstructed[:, j] = rec

		recon_errors = np.sqrt(np.mean((values - reconstructed) ** 2, axis=1))
		phys_res = normalized_physics_residual(values, time_s, omega)
		phys_norm = np.sqrt(np.mean(phys_res**2, axis=1))

		recon_error_parts.append(recon_errors)
		physics_norm_parts.append(phys_norm)

	all_recon_errors = np.concatenate(recon_error_parts, axis=0)
	all_physics_norm = np.concatenate(physics_norm_parts, axis=0)

	recon_threshold = float(np.percentile(all_recon_errors, percentile) * threshold_scale)
	physics_threshold = float(np.percentile(all_physics_norm, percentile) * threshold_scale)

	return SHOMouthModel(
		omega=omega,
		feature_names=feature_names,
		recon_threshold=recon_threshold,
		physics_threshold=physics_threshold,
		normalization_mode=normalization_mode,
	)


def evaluate_sequence(model: SHOMouthModel, csv_path: Path) -> dict:
	time_s, values, feature_names = read_wide_mouth_csv(csv_path)

	if feature_names != model.feature_names:
		raise ValueError(
			"Feature mismatch between model and CSV. "
			"Ensure both files use the same extractor format and landmark columns."
		)

	values = spatially_normalize_values(values, feature_names, model.normalization_mode)

	reconstructed = np.zeros_like(values)
	for j in range(values.shape[1]):
		_, rec = fit_sho_coefficients(time_s, values[:, j], model.omega)
		reconstructed[:, j] = rec

	recon_errors = np.sqrt(np.mean((values - reconstructed) ** 2, axis=1))
	phys_res = normalized_physics_residual(values, time_s, model.omega)
	phys_norm = np.sqrt(np.mean(phys_res**2, axis=1))

	recon_mean = float(np.mean(recon_errors))
	recon_p95 = float(np.percentile(recon_errors, 95))
	physics_mean = float(np.mean(phys_norm))
	physics_p95 = float(np.percentile(phys_norm, 95))

	is_physical = (
		recon_mean <= model.recon_threshold and physics_mean <= model.physics_threshold
	)

	return {
		"is_physical": bool(is_physical),
		"reconstruction": {
			"mean": recon_mean,
			"p95": recon_p95,
			"threshold": model.recon_threshold,
		},
		"physics": {
			"mean": physics_mean,
			"p95": physics_p95,
			"threshold": model.physics_threshold,
		},
	}


def save_model(model: SHOMouthModel, model_path: Path) -> None:
	model_path.parent.mkdir(parents=True, exist_ok=True)
	with model_path.open("w", encoding="utf-8") as handle:
		json.dump(model.to_dict(), handle, indent=2)


def load_model(model_path: Path) -> SHOMouthModel:
	with model_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	return SHOMouthModel.from_dict(payload)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Physics-informed mouth trajectory model using simple harmonic "
			"oscillator constraints."
		)
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	train_parser = subparsers.add_parser("train", help="Fit SHO model and save it.")
	train_parser.add_argument(
		"--train-csv",
		type=Path,
		default=None,
		help="Training mouth CSV in wide format (x_*, y_* columns).",
	)
	train_parser.add_argument(
		"--train-dir",
		type=Path,
		default=None,
		help=(
			"Directory with training CSV files in wide format. "
			"All *.csv files are used to fit one generalized model."
		),
	)
	train_parser.add_argument(
		"--model-out",
		type=Path,
		default=Path("outputs/sho_mouth_model.json"),
		help="Path to write trained model JSON.",
	)
	train_parser.add_argument(
		"--freq-min-hz",
		type=float,
		default=0.5,
		help="Minimum oscillator frequency (Hz) for grid search.",
	)
	train_parser.add_argument(
		"--freq-max-hz",
		type=float,
		default=10.0,
		help="Maximum oscillator frequency (Hz) for grid search.",
	)
	train_parser.add_argument(
		"--n-grid",
		type=int,
		default=100,
		help="Number of frequency candidates in grid search.",
	)
	train_parser.add_argument(
		"--threshold-percentile",
		type=float,
		default=95.0,
		help="Percentile used to estimate training consistency thresholds.",
	)
	train_parser.add_argument(
		"--threshold-scale",
		type=float,
		default=1.1,
		help="Safety scaling factor for thresholds.",
	)
	train_parser.add_argument(
		"--spatial-normalization",
		type=str,
		default="none",
		choices=SPATIAL_NORMALIZATION_CHOICES,
		help=(
			"Spatial normalization mode applied before model fitting. "
			"Use center_scale or center_scale_rotate for better cross-video generalization."
		),
	)

	eval_parser = subparsers.add_parser(
		"check", help="Evaluate if test mouth points are physically consistent."
	)
	eval_parser.add_argument(
		"--model",
		type=Path,
		required=True,
		help="Path to trained model JSON.",
	)
	eval_parser.add_argument(
		"--test-csv",
		type=Path,
		required=True,
		help="Test mouth CSV in wide format.",
	)
	eval_parser.add_argument(
		"--report-out",
		type=Path,
		default=None,
		help="Optional JSON report output path.",
	)

	args = parser.parse_args()

	if args.command == "train":
		if (args.train_csv is None) == (args.train_dir is None):
			raise ValueError("Provide exactly one of --train-csv or --train-dir.")

		if args.train_csv is not None:
			train_csv_paths = [args.train_csv]
		else:
			train_csv_paths = list_wide_mouth_csvs(args.train_dir)

		model = fit_model_from_csv_paths(
			csv_paths=train_csv_paths,
			freq_min_hz=args.freq_min_hz,
			freq_max_hz=args.freq_max_hz,
			n_grid=args.n_grid,
			percentile=args.threshold_percentile,
			threshold_scale=args.threshold_scale,
			normalization_mode=args.spatial_normalization,
		)
		save_model(model, args.model_out)
		print("Model trained and saved")
		print(f"- training files: {len(train_csv_paths)}")
		print(f"- model path: {args.model_out}")
		print(f"- spatial normalization: {model.normalization_mode}")
		print(f"- omega: {model.omega:.6f} rad/s")
		print(f"- frequency: {model.frequency_hz:.6f} Hz")
		print(f"- reconstruction threshold: {model.recon_threshold:.6f}")
		print(f"- physics threshold: {model.physics_threshold:.6f}")
		return

	if args.command == "check":
		model = load_model(args.model)
		report = evaluate_sequence(model, args.test_csv)
		print("Physical consistency check")
		print(f"- model: {args.model}")
		print(f"- test csv: {args.test_csv}")
		print(f"- spatial normalization: {model.normalization_mode}")
		print(f"- physically consistent: {report['is_physical']}")
		print(
			"- reconstruction mean/p95/threshold: "
			f"{report['reconstruction']['mean']:.6f} / "
			f"{report['reconstruction']['p95']:.6f} / "
			f"{report['reconstruction']['threshold']:.6f}"
		)
		print(
			"- physics mean/p95/threshold: "
			f"{report['physics']['mean']:.6f} / "
			f"{report['physics']['p95']:.6f} / "
			f"{report['physics']['threshold']:.6f}"
		)

		if args.report_out is not None:
			args.report_out.parent.mkdir(parents=True, exist_ok=True)
			with args.report_out.open("w", encoding="utf-8") as handle:
				json.dump(report, handle, indent=2)
			print(f"- report path: {args.report_out}")


if __name__ == "__main__":
	main()
