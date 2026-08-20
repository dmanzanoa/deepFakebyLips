from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-8
NORMALIZATION_MODES = ("none", "center_scale", "center_scale_rotate")


def coordinate_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
	x_by_id = {column[2:]: column for column in frame if column.startswith("x_")}
	y_by_id = {column[2:]: column for column in frame if column.startswith("y_")}
	landmark_ids = sorted(
		set(x_by_id) & set(y_by_id),
		key=lambda landmark_id: (not landmark_id.isdigit(), int(landmark_id) if landmark_id.isdigit() else landmark_id),
	)
	if len(landmark_ids) < 2:
		raise ValueError("Expected at least two paired x_<id>, y_<id> landmark columns.")
	return landmark_ids, [x_by_id[landmark_id] for landmark_id in landmark_ids], [y_by_id[landmark_id] for landmark_id in landmark_ids]


def normalize_landmarks(values: np.ndarray, mode: str) -> np.ndarray:
	if mode == "none":
		return values

	x = values[:, :, 0]
	y = values[:, :, 1]
	center_x = x.mean(axis=1, keepdims=True)
	center_y = y.mean(axis=1, keepdims=True)
	x = x - center_x
	y = y - center_y

	width = np.maximum(x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True), EPS)
	x = x / width
	y = y / width

	if mode == "center_scale_rotate":
		left = np.argmin(x, axis=1)
		right = np.argmax(x, axis=1)
		rows = np.arange(len(x))
		angle = np.arctan2(y[rows, right] - y[rows, left], x[rows, right] - x[rows, left])
		cos_angle = np.cos(angle)[:, None]
		sin_angle = np.sin(angle)[:, None]
		x, y = cos_angle * x + sin_angle * y, -sin_angle * x + cos_angle * y

	return np.stack((x, y), axis=-1)


def load_video(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
	frame = pd.read_csv(csv_path)
	required_columns = {"timestamp_ms", "detected"}
	missing = required_columns - set(frame.columns)
	if missing:
		raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

	landmark_ids, x_columns, y_columns = coordinate_columns(frame)
	columns = ["timestamp_ms", *x_columns, *y_columns]
	detected = frame[frame["detected"] == 1].copy()
	detected[columns] = detected[columns].apply(pd.to_numeric, errors="coerce")
	detected = detected.dropna(subset=columns)
	if detected.empty:
		raise ValueError(f"{csv_path} has no valid detected landmark rows.")

	x = detected[x_columns].to_numpy(dtype=float)
	y = detected[y_columns].to_numpy(dtype=float)
	return detected["timestamp_ms"].to_numpy(dtype=float) / 1000.0, np.stack((x, y), axis=-1), landmark_ids


def dominant_motion_frequency(time_s: np.ndarray, values: np.ndarray, max_frequency_hz: float) -> float | None:
	if len(time_s) < 8:
		return None
	dt = float(np.median(np.diff(time_s)))
	if dt <= 0:
		return None

	sample_rate = 1.0 / dt
	sample_index = np.arange(len(time_s), dtype=float)
	signal = values.reshape(len(values), -1)
	trend = np.column_stack((sample_index, np.ones(len(sample_index))))
	coefficients, _, _, _ = np.linalg.lstsq(trend, signal, rcond=None)
	signal = signal - trend @ coefficients
	signal /= np.std(signal, axis=0, keepdims=True) + EPS

	spectrum = np.mean(np.abs(np.fft.rfft(signal, axis=0)) ** 2, axis=1)
	frequencies = np.fft.rfftfreq(len(signal), d=dt)
	valid = (frequencies > 0) & (frequencies <= min(max_frequency_hz, sample_rate / 2.0))
	if not np.any(valid):
		return None
	return float(frequencies[valid][np.argmax(spectrum[valid])])


def summarize_landmarks(samples: dict[str, list[np.ndarray]]) -> pd.DataFrame:
	rows: list[dict[str, float | int | str]] = []
	for landmark_id, chunks in samples.items():
		values = np.concatenate(chunks, axis=0)
		x, y = values[:, 0], values[:, 1]
		radius = np.hypot(x, y)
		rows.append(
			{
				"landmark_id": landmark_id,
				"sample_count": len(values),
				"x_mean": float(np.mean(x)),
				"x_std": float(np.std(x)),
				"x_q05": float(np.quantile(x, 0.05)),
				"x_median": float(np.median(x)),
				"x_q95": float(np.quantile(x, 0.95)),
				"y_mean": float(np.mean(y)),
				"y_std": float(np.std(y)),
				"y_q05": float(np.quantile(y, 0.05)),
				"y_median": float(np.median(y)),
				"y_q95": float(np.quantile(y, 0.95)),
				"xy_correlation": float(np.corrcoef(x, y)[0, 1]) if np.std(x) > EPS and np.std(y) > EPS else 0.0,
				"radius_mean": float(np.mean(radius)),
				"radius_std": float(np.std(radius)),
			}
		)
	return pd.DataFrame(rows).sort_values("landmark_id", key=lambda series: series.astype(int))


def write_landmark_histograms(samples: dict[str, list[np.ndarray]], output_dir: Path, bins: int) -> None:
	histogram_dir = output_dir / "histograms"
	histogram_dir.mkdir(exist_ok=True)
	for landmark_id, chunks in samples.items():
		values = np.concatenate(chunks, axis=0)
		figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
		for axis, coordinate, color in zip(axes, ("x", "y"), ("#207398", "#d06c3b")):
			axis.hist(values[:, 0 if coordinate == "x" else 1], bins=bins, color=color, edgecolor="white")
			axis.set_title(f"Normalized {coordinate}")
			axis.set_xlabel(f"{coordinate} / mouth width")
			axis.grid(axis="y", alpha=0.25)
		axes[0].set_ylabel("Frames")
		figure.suptitle(f"Landmark {landmark_id}")
		figure.tight_layout()
		figure.savefig(histogram_dir / f"landmark_{landmark_id}.png", dpi=150)
		plt.close(figure)


def analyze_directory(input_dir: Path, output_dir: Path, normalization: str, max_frequency_hz: float) -> None:
	csv_paths = sorted(input_dir.glob("*.csv"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)
	if not csv_paths:
		raise ValueError(f"No CSV files found in {input_dir}")

	landmark_samples: dict[str, list[np.ndarray]] = {}
	frequency_rows: list[dict[str, float | int | str | None]] = []
	expected_ids: list[str] | None = None
	skipped: dict[str, str] = {}

	for csv_path in csv_paths:
		try:
			time_s, values, landmark_ids = load_video(csv_path)
			if expected_ids is None:
				expected_ids = landmark_ids
			elif landmark_ids != expected_ids:
				raise ValueError("landmark columns differ from the first CSV")
			normalized = normalize_landmarks(values, normalization)
			for index, landmark_id in enumerate(landmark_ids):
				landmark_samples.setdefault(landmark_id, []).append(normalized[:, index, :])
			frequency_rows.append(
				{
					"video": csv_path.name,
					"detected_frames": len(time_s),
					"duration_s": float(time_s[-1] - time_s[0]),
					"dominant_motion_frequency_hz": dominant_motion_frequency(time_s, normalized, max_frequency_hz),
				}
			)
		except (ValueError, pd.errors.ParserError) as error:
			skipped[csv_path.name] = str(error)

	if not landmark_samples:
		raise ValueError("No usable videos were found.")

	output_dir.mkdir(parents=True, exist_ok=True)
	summarize_landmarks(landmark_samples).to_csv(output_dir / "landmark_distribution.csv", index=False)
	pd.DataFrame(frequency_rows).to_csv(output_dir / "video_motion_frequency.csv", index=False)
	write_landmark_histograms(landmark_samples, output_dir, bins=50)
	(output_dir / "analysis_summary.json").write_text(
		json.dumps(
			{
				"input_directory": str(input_dir),
				"normalization": normalization,
				"videos_found": len(csv_paths),
				"videos_analyzed": len(frequency_rows),
				"videos_skipped": skipped,
				"landmarks_analyzed": len(landmark_samples),
				"notes": [
					"Coordinates are normalized independently in every frame, so distributions describe mouth shape rather than image position or scale.",
					"The frequency is a per-video dominant peak from detrended normalized landmark motion; it is descriptive and not a speaking-rate estimate.",
				],
			},
			indent=2,
		)
		+ "\n"
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Analyze normalized landmark distributions across mouth-landmark CSV videos.")
	parser.add_argument("input_dir", type=Path, help="Directory containing wide-format mouth-landmark CSV files")
	parser.add_argument("--output-dir", type=Path, default=Path("outputs/landmark_statistics"))
	parser.add_argument("--normalization", choices=NORMALIZATION_MODES, default="center_scale_rotate")
	parser.add_argument("--max-frequency-hz", type=float, default=12.0)
	args = parser.parse_args()

	if args.max_frequency_hz <= 0:
		parser.error("--max-frequency-hz must be positive")
	analyze_directory(args.input_dir, args.output_dir, args.normalization, args.max_frequency_hz)


if __name__ == "__main__":
	main()
