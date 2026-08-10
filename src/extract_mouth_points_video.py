from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LIP_CONNECTIONS = tuple(vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def get_lip_indices() -> list[int]:
    lip_indices: set[int] = set()
    for connection in LIP_CONNECTIONS:
        lip_indices.add(connection.start)
        lip_indices.add(connection.end)
    return sorted(lip_indices)


def make_wide_header(lip_indices: list[int], include_z: bool) -> list[str]:
    header = ["frame", "timestamp_ms", "detected"]
    for index in lip_indices:
        header.extend([f"x_{index}", f"y_{index}"])
        if include_z:
            header.append(f"z_{index}")
    return header


def landmark_to_row_values(landmark, width: int, height: int, normalized: bool, include_z: bool) -> list[float | int]:
    if normalized:
        values: list[float | int] = [landmark.x, landmark.y]
    else:
        values = [round(landmark.x * width), round(landmark.y * height)]

    if include_z:
        values.append(landmark.z if normalized else landmark.z * width)
    return values


def landmark_to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    return round(landmark.x * width), round(landmark.y * height)


def draw_mouth_overlay(frame, face_landmarks, lip_indices: list[int]) -> None:
    height, width = frame.shape[:2]
    points = {
        index: landmark_to_pixel(face_landmarks[index], width, height)
        for index in lip_indices
    }

    for connection in LIP_CONNECTIONS:
        start = points.get(connection.start)
        end = points.get(connection.end)
        if start is not None and end is not None:
            cv2.line(frame, start, end, (0, 255, 255), 1, cv2.LINE_AA)

    for index, point in points.items():
        cv2.circle(frame, point, 2, (0, 0, 255), -1, cv2.LINE_AA)


def write_mouth_points(
    video_path: Path,
    output_csv: Path,
    model_path: Path,
    annotated_video: Path,
    max_frames: int | None,
    frame_step: int,
    normalized: bool,
    include_z: bool,
    long_format: bool,
) -> int:
    if frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    lip_indices = get_lip_indices()

    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}. "
            "Download face_landmarker.task and pass it with --model-path."
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    annotated_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = fps / frame_step
    writer_video = cv2.VideoWriter(
        str(annotated_video),
        fourcc,
        output_fps,
        (frame_width, frame_height),
    )
    if not writer_video.isOpened():
        raise RuntimeError(f"Could not create annotated video: {annotated_video}")

    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        if long_format:
            header = ["frame", "timestamp_ms", "detected", "landmark_id", "x", "y"]
            if include_z:
                header.append("z")
            writer.writerow(header)
        else:
            writer.writerow(make_wide_header(lip_indices, include_z))

        frame_number = 0
        processed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_number % frame_step != 0:
                    frame_number += 1
                    continue

                timestamp_ms = int(round(frame_number * 1000.0 / fps))
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = detector.detect_for_video(image, timestamp_ms)

                if result.face_landmarks:
                    face_landmarks = result.face_landmarks[0]
                    draw_mouth_overlay(frame, face_landmarks, lip_indices)

                    if long_format:
                        for index in lip_indices:
                            values = landmark_to_row_values(
                                face_landmarks[index],
                                width,
                                height,
                                normalized,
                                include_z,
                            )
                            writer.writerow([frame_number, timestamp_ms, 1, index, *values])
                    else:
                        row: list[float | int] = [frame_number, timestamp_ms, 1]
                        for index in lip_indices:
                            row.extend(
                                landmark_to_row_values(
                                    face_landmarks[index],
                                    width,
                                    height,
                                    normalized,
                                    include_z,
                                )
                            )
                        writer.writerow(row)
                else:
                    if long_format:
                        writer.writerow([frame_number, timestamp_ms, 0, "", "", ""])
                    else:
                        empty_count = len(lip_indices) * (3 if include_z else 2)
                        writer.writerow([frame_number, timestamp_ms, 0, *([""] * empty_count)])

                writer_video.write(frame)

                processed += 1
                frame_number += 1
                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            cap.release()
            writer_video.release()
            detector.close()
    return processed


def iter_video_files(input_dir: Path, recursive: bool, extensions: tuple[str, ...]) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    videos = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(videos)


def default_csv_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}.csv"


def default_annotated_video_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_overlay.mp4")


def directory_output_csv_path(video_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = video_path.relative_to(input_dir)
    return (output_dir / relative).with_suffix(".csv")


def append_to_consolidated_csv(source_csv: Path, video_path: Path, consolidated_csv: Path) -> None:
    consolidated_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = consolidated_csv.exists()

    with source_csv.open(newline="") as source_handle:
        reader = csv.reader(source_handle)
        source_header = next(reader, None)
        if source_header is None:
            return

        with consolidated_csv.open("a", newline="") as output_handle:
            writer = csv.writer(output_handle)
            if not file_exists:
                writer.writerow(["video_path", *source_header])
            for row in reader:
                writer.writerow([str(video_path), *row])


def write_manifest(manifest_path: Path, rows: list[dict[str, str | int]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_path",
        "csv_path",
        "annotated_video_path",
        "status",
        "processed_frames",
        "error",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe mouth/lip landmarks from one video or every video in a directory."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input video file or directory containing videos.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path for single-video mode. Defaults to OUTPUT_DIR/<video_stem>.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs",
        help="Output directory for generated CSV files, overlay videos, and manifest.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_DIR / "models" / "face_landmarker.task",
        help="Path to MediaPipe face_landmarker.task model.",
    )
    parser.add_argument(
        "--annotated-video",
        type=Path,
        default=None,
        help="Optional annotated output video path for single-video mode. Defaults to CSV stem plus _overlay.mp4.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Manifest CSV path for directory mode. Defaults to OUTPUT_DIR/manifest.csv.",
    )
    parser.add_argument(
        "--consolidated-csv",
        type=Path,
        default=None,
        help="Optional single CSV containing rows from all processed videos. Can become very large.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="For directory mode, only process videos directly inside the input directory.",
    )
    parser.add_argument(
        "--video-extensions",
        default=",".join(DEFAULT_VIDEO_EXTENSIONS),
        help="Comma-separated video extensions to process in directory mode.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Write normalized MediaPipe coordinates instead of pixel coordinates.",
    )
    parser.add_argument("--include-z", action="store_true", help="Also write MediaPipe z values.")
    parser.add_argument(
        "--long-format",
        action="store_true",
        help="Write one row per frame and landmark: frame, timestamp_ms, detected, landmark_id, x, y.",
    )
    args = parser.parse_args()
    extensions = tuple(
        extension.strip().lower()
        for extension in args.video_extensions.split(",")
        if extension.strip()
    )
    extensions = tuple(
        extension if extension.startswith(".") else f".{extension}"
        for extension in extensions
    )

    if args.input_path.is_file():
        output_csv = args.output_csv or default_csv_path(args.input_path, args.output_dir)
        annotated_video = args.annotated_video or default_annotated_video_path(output_csv)
        processed_frames = write_mouth_points(
            video_path=args.input_path,
            output_csv=output_csv,
            model_path=args.model_path,
            annotated_video=annotated_video,
            max_frames=args.max_frames,
            frame_step=args.frame_step,
            normalized=args.normalized,
            include_z=args.include_z,
            long_format=args.long_format,
        )
        if args.consolidated_csv is not None:
            append_to_consolidated_csv(output_csv, args.input_path, args.consolidated_csv)
        print(f"Processed {args.input_path}")
        print(f"  frames: {processed_frames}")
        print(f"  csv: {output_csv}")
        print(f"  annotated_video: {annotated_video}")
        return

    if not args.input_path.is_dir():
        raise RuntimeError(f"Input path is not a file or directory: {args.input_path}")

    if args.output_csv is not None:
        raise RuntimeError("--output-csv is only valid when processing a single video file.")
    if args.annotated_video is not None:
        raise RuntimeError("--annotated-video is only valid when processing a single video file.")

    videos = iter_video_files(
        input_dir=args.input_path,
        recursive=not args.no_recursive,
        extensions=extensions,
    )
    if not videos:
        raise RuntimeError(f"No videos found in directory: {args.input_path}")

    manifest_path = args.manifest_csv or args.output_dir / "manifest.csv"
    manifest_rows: list[dict[str, str | int]] = []

    for index, video_path in enumerate(videos, start=1):
        output_csv = directory_output_csv_path(video_path, args.input_path, args.output_dir)
        annotated_video = default_annotated_video_path(output_csv)
        print(f"[{index}/{len(videos)}] {video_path}")
        try:
            processed_frames = write_mouth_points(
                video_path=video_path,
                output_csv=output_csv,
                model_path=args.model_path,
                annotated_video=annotated_video,
                max_frames=args.max_frames,
                frame_step=args.frame_step,
                normalized=args.normalized,
                include_z=args.include_z,
                long_format=args.long_format,
            )
            if args.consolidated_csv is not None:
                append_to_consolidated_csv(output_csv, video_path, args.consolidated_csv)
            manifest_rows.append(
                {
                    "video_path": str(video_path),
                    "csv_path": str(output_csv),
                    "annotated_video_path": str(annotated_video),
                    "status": "ok",
                    "processed_frames": processed_frames,
                    "error": "",
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "video_path": str(video_path),
                    "csv_path": str(output_csv),
                    "annotated_video_path": str(annotated_video),
                    "status": "error",
                    "processed_frames": 0,
                    "error": str(exc),
                }
            )
            print(f"  error: {exc}")

    write_manifest(manifest_path, manifest_rows)
    ok_count = sum(1 for row in manifest_rows if row["status"] == "ok")
    print(f"Processed {ok_count}/{len(videos)} videos")
    print(f"Manifest: {manifest_path}")
    if args.consolidated_csv is not None:
        print(f"Consolidated CSV: {args.consolidated_csv}")


if __name__ == "__main__":
    main()
