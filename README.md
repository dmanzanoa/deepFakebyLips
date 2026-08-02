# deepFakebyLips

Extract MediaPipe mouth landmarks from a video into a CSV file and generate an
annotated review video with the detected mouth points drawn over the frames.

## What This Repo Includes

- `src/extract_mouth_points_video.py`: command-line extractor
- `models/face_landmarker.task`: MediaPipe Face Landmarker model
- `examples/Real_Kennedy.mp4`: small public sample video from the LIPINC-V2 repo
- `outputs/`: suggested location for generated CSV and overlay videos

## Setup

```bash
git clone https://github.com/dmanzanoa/deepFakebyLips.git
cd deepFakebyLips

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On some Linux machines, MediaPipe may also require:

```bash
sudo apt-get update
sudo apt-get install -y libgles2
```

## Run The Example

```bash
python src/extract_mouth_points_video.py \
  examples/Real_Kennedy.mp4 \
  --output-csv outputs/real_kennedy_mouth_points.csv
```

This creates:

```text
outputs/real_kennedy_mouth_points.csv
outputs/real_kennedy_mouth_points_overlay.mp4
```

The CSV has one row per processed frame:

```text
frame,timestamp_ms,detected,x_0,y_0,x_13,y_13,...
```

Each `x_<id>,y_<id>` pair is a MediaPipe lip landmark in pixel coordinates.

The overlay video contains:

```text
red dots = detected mouth points
yellow lines = mouth/lip connections
```

## Use Your Own Video

```bash
python src/extract_mouth_points_video.py \
  /path/to/video.mp4 \
  --output-csv outputs/my_video_mouth_points.csv
```

Choose a custom annotated video name:

```bash
python src/extract_mouth_points_video.py \
  /path/to/video.mp4 \
  --output-csv outputs/my_video_mouth_points.csv \
  --annotated-video outputs/my_video_review.mp4
```

Process only the first 20 frames:

```bash
python src/extract_mouth_points_video.py \
  examples/Real_Kennedy.mp4 \
  --output-csv outputs/test_20_frames.csv \
  --max-frames 20
```

Write normalized MediaPipe coordinates instead of pixel coordinates:

```bash
python src/extract_mouth_points_video.py \
  examples/Real_Kennedy.mp4 \
  --output-csv outputs/normalized_mouth_points.csv \
  --normalized
```

Write one row per landmark per frame:

```bash
python src/extract_mouth_points_video.py \
  examples/Real_Kennedy.mp4 \
  --output-csv outputs/real_kennedy_mouth_points_long.csv \
  --long-format
```

## Notes

This utility is intended as a preprocessing and manual-review helper for fake
lip-sync or deepfake detection workflows. It does not train a detector by
itself; it extracts mouth trajectories that can be used by downstream models.

## Physics-Informed Mouth Consistency Model

The file [src/inverse_problem.py](src/inverse_problem.py) implements a
physics-informed model that constrains each mouth coordinate trajectory with a
simple harmonic oscillator (SHO) equation:

$$
\ddot{q}(t) + \omega^2\left(q(t)-c\right)=0
$$

Where $q(t)$ is each landmark coordinate over time, $\omega$ is a global mouth
oscillation frequency learned from training data, and $c$ is a learned offset.

Train the model from a mouth CSV:

```bash
python src/inverse_problem.py train \
  --train-csv outputs/real_kennedy_mouth_points.csv \
  --model-out outputs/sho_mouth_model.json
```

Check a new test sequence for physical consistency:

```bash
python src/inverse_problem.py check \
  --model outputs/sho_mouth_model.json \
  --test-csv outputs/real_kennedy_mouth_points.csv \
  --report-out outputs/sho_consistency_report.json
```

The check command returns `physically consistent: True/False` using
training-calibrated thresholds for:

- reconstruction error under SHO-constrained fitting
- residual error from the SHO physics equation

