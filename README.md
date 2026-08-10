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

## Process A Directory

Recommended approach: create one CSV and one overlay video per input video, plus
a `manifest.csv` that lists every result.

```bash
python src/extract_mouth_points_video.py \
  "/home/mzo/fake_lip_sync_detection/data/raw/LipSyncTimit Dataset/Original Size/RealVideo" \
  --output-dir outputs/real_videos
```

For each input video, this creates:

```text
outputs/real_videos/<video_name>.csv
outputs/real_videos/<video_name>_overlay.mp4
```

It also creates:

```text
outputs/real_videos/manifest.csv
```

The manifest has:

```text
video_path,csv_path,annotated_video_path,status,processed_frames,error
```

If you also want one large CSV with all frame rows from all videos, add
`--consolidated-csv`:

```bash
python src/extract_mouth_points_video.py \
  "/home/mzo/fake_lip_sync_detection/data/raw/LipSyncTimit Dataset/Original Size/RealVideo" \
  --output-dir outputs/real_videos \
  --consolidated-csv outputs/real_videos_all.csv
```

The consolidated CSV starts with a `video_path` column so each row can be traced
back to its source video. This file can become large for big datasets.

By default, directory mode searches recursively. To process only videos directly
inside the directory, add:

```bash
--no-recursive
```

To customize which extensions are processed:

```bash
--video-extensions .mp4,.avi,.mov
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
