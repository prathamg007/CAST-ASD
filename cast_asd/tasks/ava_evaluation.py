"""Export AVA predictions and score them with the official ActivityNet metric.

The prediction CSV echoes the ground-truth `frame_timestamp` and box columns as
the *original raw strings*. That matters: the official script joins on
`str(frame_timestamp) + ":" + entity_id` and checks boxes with `eq(a, b, 1e-9)`,
so byte-identical text makes both the merge and the box check pass by
construction, whatever dtype pandas happens to infer.
"""

import contextlib
import csv
import io
from dataclasses import dataclass
from pathlib import Path

import torch

from cast_asd.tasks.asd_metrics import ava_active_speaker_map

GROUNDTRUTH_FIELDS = [
    "video_id",
    "frame_timestamp",
    "entity_box_x1",
    "entity_box_y1",
    "entity_box_x2",
    "entity_box_y2",
    "label",
    "entity_id",
]
PREDICTION_FIELDS = GROUNDTRUTH_FIELDS + ["score"]

#: AVA timestamps carry at most two decimals, so integer centiseconds are an
#: exact key.
TIMESTAMP_SCALE = 100


def timestamp_key(text: str) -> int:
    """Exact centisecond key. '960', '960.0' and '960.00' all map to 96000."""
    value = float(text)
    key = round(value * TIMESTAMP_SCALE)
    if abs(value * TIMESTAMP_SCALE - key) > 1e-6:
        raise ValueError(f"AVA frame_timestamp is not a whole centisecond: {text!r}")
    return key


@dataclass(frozen=True)
class ExportResult:
    rows: int
    average_precision: float


def export_ava_predictions(
    groundtruth_path, output_path, scores_by_key, score_format="{:.9f}"
) -> ExportResult:
    """Write the official prediction CSV in ground-truth row order.

    `scores_by_key` maps `(entity_id, timestamp_centiseconds)` to a float score;
    every ground-truth row must have exactly one. Also returns the in-repo AP.
    """
    groundtruth_path = Path(groundtruth_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, int]] = set()
    missing: list[tuple[str, int]] = []
    labels: list[bool] = []
    scores: list[float] = []

    with groundtruth_path.open(newline="") as source, output_path.open(
        "w", newline=""
    ) as sink:
        reader = csv.DictReader(source)
        absent = set(GROUNDTRUTH_FIELDS).difference(reader.fieldnames or ())
        if absent:
            raise ValueError(f"groundtruth CSV is missing columns: {sorted(absent)}")

        writer = csv.writer(sink)
        writer.writerow(PREDICTION_FIELDS)
        for row in reader:
            key = (row["entity_id"], timestamp_key(row["frame_timestamp"]))
            if key in seen:
                raise ValueError(f"duplicate groundtruth row: {key}")
            seen.add(key)

            score = scores_by_key.get(key)
            if score is None:
                missing.append(key)
                continue

            score = float(score)
            writer.writerow(
                # Columns 0..5 verbatim so the official box check is exact.
                [row[field] for field in GROUNDTRUTH_FIELDS[:6]]
                + ["SPEAKING_AUDIBLE", row["entity_id"], score_format.format(score)]
            )
            labels.append(row["label"] == "SPEAKING_AUDIBLE")
            scores.append(score)

    extra = set(scores_by_key).difference(seen)
    if missing or extra:
        raise ValueError(
            f"AVA row mismatch: {len(missing)} groundtruth rows have no score "
            f"(e.g. {missing[:5]}); {len(extra)} scores match no groundtruth row "
            f"(e.g. {sorted(extra)[:5]})"
        )

    return ExportResult(
        rows=len(scores),
        average_precision=ava_active_speaker_map(
            torch.tensor(scores), torch.tensor(labels)
        ),
    )


def official_ava_map(groundtruth_path, predictions_path) -> float:
    """Run the vendored ActivityNet script in-process."""
    from cast_asd.tasks.ava_official.get_ava_active_speaker_performance import run_evaluation

    with contextlib.redirect_stdout(io.StringIO()):
        return float(run_evaluation(str(groundtruth_path), str(predictions_path)))
