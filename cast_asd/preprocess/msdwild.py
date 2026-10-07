"""Pack MSDWild into v4 shards of per-speaker face tracks.

    python -m cast_asd.preprocess.msdwild --input <MSDWild>/videos --split few.train
    python -m cast_asd.preprocess.msdwild --input <MSDWild>/videos --split few.val
    python -m cast_asd.preprocess.msdwild --input <MSDWild>/videos --split many.val

`--input` holds `<video>.mp4` with its bounding-box CSV `<video>.csv` beside it.
Speaking labels come from `--rttm` (default `<input>/../all.rttm`); a split is
the set of video ids in `--split-rttm` (default `<input>/../<split>.rttm`, the
official split RTTMs). Output defaults to `<input>/../packed/<split>`, which is
what `packed_asd.{train,val}` in `configs/msdwild` point at.

Each video is cut into consecutive `--clip-seconds` windows from its first
annotated frame (the tail window is kept if at least `--min-clip-seconds`).
Within a window, every speaker whose face is annotated on at least
`--min-face-ratio` of the frames becomes one sample `<video>_<window>_<speaker>`;
all samples of a window share its audio, so `<video>_<window>` is the scene.
Frames where the speaker has no box get a mid-grey face. A frame is labelled
speaking when its centre falls inside one of that speaker's RTTM intervals.
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from cast_asd.data.media import FACE_SIZE, SAMPLE_RATE, load_waveform
from cast_asd.preprocess.writer import ShardWriterV4

Box = tuple[float, float, float, float]


def read_annotations(path: Path) -> dict[int, dict[str, Box]]:
    """`frame, _, speaker, x1, y1, x2, y2, _` rows -> {frame: {speaker: box}}."""
    annotations: dict[int, dict[str, Box]] = defaultdict(dict)
    with path.open(newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 8:
                raise ValueError(f"{path}:{row_number}: expected 8 columns")
            try:
                frame = int(row[0])
                speaker_id = row[2]
                bbox = tuple(float(value) for value in row[3:7])
                int(row[7])
            except ValueError as error:
                raise ValueError(f"{path}:{row_number}: invalid annotation") from error
            annotations[frame][speaker_id] = bbox
    return dict(annotations)


def read_rttm(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """{video: {speaker: sorted [(start, end), ...]}}."""
    activity: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 8 or parts[0] != "SPEAKER":
                raise ValueError(f"{path}:{line_number}: invalid RTTM row")
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid time") from error
            activity[parts[1]][parts[7]].append((start, start + duration))
    return {
        video_id: {speaker: sorted(intervals) for speaker, intervals in speakers.items()}
        for video_id, speakers in activity.items()
    }


def retained_speakers(
    annotations: dict[int, dict[str, Box]],
    start_frame: int,
    frame_count: int,
    min_face_ratio: float,
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for frame in range(start_frame, start_frame + frame_count):
        for speaker_id in annotations.get(frame, {}):
            counts[speaker_id] += 1
    minimum = min_face_ratio * frame_count
    return sorted(speaker_id for speaker_id, count in counts.items() if count >= minimum)


def crop_face(frame: np.ndarray, bbox: Box | None) -> np.ndarray:
    """Square grayscale crop centred on the box, grey-padded past the image edge."""
    blank = np.full((FACE_SIZE, FACE_SIZE), 128, dtype=np.uint8)
    if bbox is None:
        return blank
    x1, y1, x2, y2 = (int(value) for value in bbox)
    if x1 >= x2 or y1 >= y2:
        return blank

    h, w = frame.shape[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    size = max(x2 - x1, y2 - y1)
    half = size // 2
    x1s, y1s = cx - half, cy - half
    x2s, y2s = x1s + size, y1s + size
    x1c, y1c = max(0, x1s), max(0, y1s)
    x2c, y2c = min(w, x2s), min(h, y2s)
    if x1c >= x2c or y1c >= y2c:
        return blank

    cropped = cv2.cvtColor(frame[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2GRAY)
    pad_top, pad_left = y1c - y1s, x1c - x1s
    pad_bottom, pad_right = y2s - y2c, x2s - x2c
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=128,
        )
    return cv2.resize(cropped, (FACE_SIZE, FACE_SIZE))


def video_metadata(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    return fps, frame_count


def read_frames(path: Path, start_frame: int, count: int) -> tuple[list, float]:
    """Up to `count` BGR frames from `start_frame`, at the native frame rate."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise ValueError(f"video has an invalid frame rate: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def process_video(
    video_path: Path,
    annotation_path: Path,
    activity: dict[str, list[tuple[float, float]]],
    writer: ShardWriterV4,
    clip_seconds: float,
    min_face_ratio: float,
    min_clip_seconds: float,
) -> int:
    """Write every retained (window, speaker) track of one video; returns the count."""
    annotations = read_annotations(annotation_path)
    if not annotations:
        return 0

    fps, video_frames = video_metadata(video_path)
    clip_frames = round(clip_seconds * fps)
    min_clip_frames = max(1, round(min_clip_seconds * fps))
    if clip_frames <= 0:
        raise ValueError("a clip must contain at least one frame")
    waveform = load_waveform(video_path)
    first_face = min(annotations)
    last_face = min(max(annotations), video_frames - 1)
    written = 0

    for segment_id, start_frame in enumerate(
        range(first_face, last_face + 1, clip_frames)
    ):
        window_frames = min(clip_frames, video_frames - start_frame)
        if window_frames < min_clip_frames:
            break
        speakers = retained_speakers(
            annotations, start_frame, window_frames, min_face_ratio
        )
        if not speakers:
            continue

        frames, decoded_fps = read_frames(video_path, start_frame, window_frames)
        if not np.isclose(decoded_fps, fps):
            raise ValueError(f"frame-rate changed while decoding {video_path}")
        if len(frames) < min_clip_frames:
            continue
        window_frames = len(frames)

        start_time = start_frame / fps
        start_sample = round(start_time * SAMPLE_RATE)
        target_samples = round(window_frames / fps * SAMPLE_RATE)
        audio = waveform[:, start_sample : start_sample + target_samples]

        for speaker_id in speakers:
            faces = []
            labels = np.zeros(window_frames, dtype=np.uint8)
            for offset, frame in enumerate(frames):
                bbox = annotations.get(start_frame + offset, {}).get(speaker_id)
                faces.append(crop_face(frame, bbox))
            intervals = activity.get(speaker_id, [])
            for offset in range(window_frames):
                timestamp = start_time + (offset + 0.5) / fps
                labels[offset] = any(start <= timestamp < end for start, end in intervals)
            writer.add_arrays(
                dataset="msdwild",
                sample_id=f"{video_path.stem}_{segment_id:05d}_{speaker_id}",
                faces=np.stack(faces).astype(np.uint8, copy=False),
                waveform=audio,
                labels=labels,
                source_fps=fps,
                start_sec=start_time,
            )
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack MSDWild videos into per-speaker ASD shards"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="directory of <video>.mp4 + <video>.csv")
    parser.add_argument("--rttm", type=Path, help="default: <input>/../all.rttm")
    parser.add_argument("--output", type=Path,
                        help="default: <input>/../packed/<split>")
    parser.add_argument("--split", default="all",
                        help="all, few.train, few.val or many.val")
    parser.add_argument("--split-rttm", type=Path,
                        help="RTTM whose video ids define the split; "
                        "default: <input>/../<split>.rttm")
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    parser.add_argument("--min-clip-seconds", type=float, default=2.0)
    parser.add_argument("--min-face-ratio", type=float, default=0.9)
    parser.add_argument("--max-shard-gb", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.clip_seconds <= 0:
        raise ValueError("clip-seconds must be positive")
    if not 0 < args.min_face_ratio <= 1:
        raise ValueError("min-face-ratio must be in (0, 1]")
    if args.max_shard_gb <= 0:
        raise ValueError("max-shard-gb must be positive")
    if not 0 < args.min_clip_seconds <= args.clip_seconds:
        raise ValueError("min-clip-seconds must be in (0, clip-seconds]")
    if args.output is None:
        args.output = args.input.parent / "packed" / args.split
        print(f"packing into {args.output}")
    if args.output.exists():
        if not args.overwrite and any(args.output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {args.output}")
        if args.overwrite:
            shutil.rmtree(args.output)

    args.output.mkdir(parents=True)
    videos = sorted(args.input.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no MP4 files found in {args.input}")

    rttm_path = args.rttm or args.input.parent / "all.rttm"
    if not rttm_path.exists():
        raise FileNotFoundError(f"missing RTTM labels: {rttm_path}")
    activity = read_rttm(rttm_path)

    if args.split != "all":
        split_path = args.split_rttm or args.input.parent / f"{args.split}.rttm"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split list: {split_path}")
        wanted = set(read_rttm(split_path))
        videos = [video for video in videos if video.stem in wanted]
        if not videos:
            raise ValueError(f"no videos matched split {args.split}")
        print(f"{args.split}: {len(videos)} videos")

    total = 0
    writer = ShardWriterV4(
        args.output, max_shard_bytes=round(args.max_shard_gb * 1_000_000_000)
    )
    try:
        for index, video_path in enumerate(videos, start=1):
            annotation_path = video_path.with_suffix(".csv")
            if not annotation_path.exists():
                raise FileNotFoundError(f"missing annotations: {annotation_path}")
            total += process_video(
                video_path,
                annotation_path,
                activity.get(video_path.stem, {}),
                writer,
                args.clip_seconds,
                args.min_face_ratio,
                args.min_clip_seconds,
            )
            print(f"[{index}/{len(videos)}] {video_path.stem}: {total} samples")
    finally:
        writer.close()
    print(f"Wrote {total} samples to {args.output}")


if __name__ == "__main__":
    main()
