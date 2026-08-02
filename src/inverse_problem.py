from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8


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


@dataclass
class SHOMouthModel:
	omega: float
	feature_names: list[str]
	recon_threshold: float
	physics_threshold: float

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
		}

	@classmethod
	def from_dict(cls, payload: dict) -> "SHOMouthModel":
		return cls(
			omega=float(payload["omega"]),
			feature_names=list(payload["feature_names"]),
			recon_threshold=float(payload["recon_threshold"]),
			physics_threshold=float(payload["physics_threshold"]),
		)


def fit_model(
	csv_path: Path,
	freq_min_hz: float,
	freq_max_hz: float,
	n_grid: int,
	percentile: float,
	threshold_scale: float,
) -> SHOMouthModel:
	time_s, values, feature_names = read_wide_mouth_csv(csv_path)
	omega = search_best_omega(time_s, values, freq_min_hz, freq_max_hz, n_grid)

	reconstructed = np.zeros_like(values)
	for j in range(values.shape[1]):
		_, rec = fit_sho_coefficients(time_s, values[:, j], omega)
		reconstructed[:, j] = rec

	recon_errors = np.sqrt(np.mean((values - reconstructed) ** 2, axis=1))
	phys_res = normalized_physics_residual(values, time_s, omega)
	phys_norm = np.sqrt(np.mean(phys_res**2, axis=1))

	recon_threshold = float(np.percentile(recon_errors, percentile) * threshold_scale)
	physics_threshold = float(np.percentile(phys_norm, percentile) * threshold_scale)

	return SHOMouthModel(
		omega=omega,
		feature_names=feature_names,
		recon_threshold=recon_threshold,
		physics_threshold=physics_threshold,
	)


def evaluate_sequence(model: SHOMouthModel, csv_path: Path) -> dict:
	time_s, values, feature_names = read_wide_mouth_csv(csv_path)

	if feature_names != model.feature_names:
		raise ValueError(
			"Feature mismatch between model and CSV. "
			"Ensure both files use the same extractor format and landmark columns."
		)

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
		required=True,
		help="Training mouth CSV in wide format (x_*, y_* columns).",
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
		model = fit_model(
			csv_path=args.train_csv,
			freq_min_hz=args.freq_min_hz,
			freq_max_hz=args.freq_max_hz,
			n_grid=args.n_grid,
			percentile=args.threshold_percentile,
			threshold_scale=args.threshold_scale,
		)
		save_model(model, args.model_out)
		print("Model trained and saved")
		print(f"- model path: {args.model_out}")
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
