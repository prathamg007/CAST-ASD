"""Pack AVA-ActiveSpeaker into v3 shards.

    python -m cast_asd.preprocess.ava --input <AVA> --output <AVA>/packed/val --split val

`<AVA>` holds `csv/{split}_loader.csv` (TalkNet's loader TSV),
`csv/{split}_orig.csv`, `clips_videos/{split}/<video>/<entity>/*.jpg` and
`clips_audios/{split}/<video>/<entity>.wav`.
"""

import argparse
import ast
import csv
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

from cast_asd.preprocess.schema import SourceRecord
from cast_asd.preprocess.writer import ShardWriter, build_sample


def _labels(text):
    labels = [int(value) for value in ast.literal_eval(text)]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("labels must contain only 0 and 1")
    return labels


def ava_records(input_dir: Path, split: str) -> Iterable[SourceRecord]:
    """One record per `{split}_loader.csv` row, in file order.

    That order matches the entity blocks of `{split}_orig.csv`, which is what
    lines the packed frames up with the official ground-truth rows.
    """
    loader = input_dir / "csv" / f"{split}_loader.csv"
    visual_root = input_dir / "clips_videos" / split
    audio_root = input_dir / "clips_audios" / split
    faces = {path.name: path for path in visual_root.glob("*/*") if path.is_dir()}
    audio = {path.stem: path for path in audio_root.glob("*/*.wav")}
    with loader.open(newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) < 4:
                raise ValueError(f"{loader}:{row_number}: expected at least 4 columns")
            sample_id, frame_count, fps, label_text = row[:4]
            if sample_id not in faces or sample_id not in audio:
                raise FileNotFoundError(f"{sample_id}: missing face directory or WAV")
            labels = _labels(label_text)
            if len(labels) != int(frame_count):
                raise ValueError(
                    f"{sample_id}: {len(labels)} labels for {frame_count} frames"
                )
            yield SourceRecord(
                sample_id=sample_id,
                visual_path=faces[sample_id],
                audio_path=audio[sample_id],
                source_fps=float(fps),
                labels=labels,
            )


def _groundtruth_rows(input_dir: Path, split: str) -> int:
    path = input_dir / "csv" / f"{split}_orig.csv"
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main():
    parser = argparse.ArgumentParser(
        description="Pack AVA faces, audio, and labels for ASD training"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="AVA root containing csv, clips_videos, and clips_audios",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--max-shard-gb", type=float, default=2.0)
    parser.add_argument(
        "--workers", type=int, default=8, help="decode processes; 0 runs inline"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after N entities (smoke tests)"
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")

    records = ava_records(args.input, args.split)
    if args.limit is not None:
        records = (record for index, record in enumerate(records) if index < args.limit)

    rows = _groundtruth_rows(args.input, args.split) if args.limit is None else None
    writer = ShardWriter(args.output, round(args.max_shard_gb * 1_000_000_000))

    if args.workers > 0:
        # imap preserves loader-CSV order; see `ava_records`.
        with Pool(args.workers) as pool:
            for sample in pool.imap(build_sample, records, chunksize=16):
                writer.add_sample(sample)
    else:
        for record in records:
            writer.add(record)

    writer.close(rows)
    print(
        f"Wrote {writer.samples_written} AVA {args.split} samples "
        f"({writer.frames_written} frames) to {args.output}"
    )
    if rows and writer.frames_written != rows:
        raise SystemExit(
            f"frame/groundtruth mismatch: packed {writer.frames_written}, "
            f"{args.split}_orig.csv has {rows}"
        )


if __name__ == "__main__":
    main()
