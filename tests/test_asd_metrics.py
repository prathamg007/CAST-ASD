import csv
import random

import pytest

from cast_asd.tasks.asd_metrics import ava_active_speaker_map
from cast_asd.tasks.ava_evaluation import (
    GROUNDTRUTH_FIELDS,
    export_ava_predictions,
    official_ava_map,
    timestamp_key,
)


def test_map_is_reported_as_percentage():
    assert ava_active_speaker_map([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 100.0


def test_map_uses_voc_precision_envelope():
    result = ava_active_speaker_map([0.9, 0.8, 0.7], [1, 0, 1])
    assert result == pytest.approx(83.3333333333)


def test_in_repo_map_agrees_with_the_official_scorer(tmp_path):
    """The in-repo AP must reproduce the vendored ActivityNet scorer."""
    rng = random.Random(0)
    groundtruth = tmp_path / "gt.csv"
    scores = {}
    with groundtruth.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUNDTRUTH_FIELDS)
        writer.writeheader()
        for entity in range(6):
            for frame in range(50):
                stamp = f"{900 + frame * 0.04:.2f}"
                label = rng.choice(["SPEAKING_AUDIBLE", "NOT_SPEAKING", "SPEAKING_NOT_AUDIBLE"])
                writer.writerow({
                    "video_id": f"v{entity % 2}", "frame_timestamp": stamp,
                    "entity_box_x1": "0.1", "entity_box_y1": "0.2",
                    "entity_box_x2": "0.3", "entity_box_y2": "0.4",
                    "label": label, "entity_id": f"v{entity % 2}:{entity}",
                })
                # Noisy but informative; untied, since the two break ties differently.
                bonus = 0.3 if label == "SPEAKING_AUDIBLE" else 0.0
                scores[(f"v{entity % 2}:{entity}", timestamp_key(stamp))] = (
                    rng.random() * 0.7 + bonus
                )

    predictions = tmp_path / "pred.csv"
    in_repo = export_ava_predictions(groundtruth, predictions, scores).average_precision
    official = official_ava_map(groundtruth, predictions)
    assert 50.0 < in_repo < 100.0
    assert in_repo == pytest.approx(official, abs=1e-4)
