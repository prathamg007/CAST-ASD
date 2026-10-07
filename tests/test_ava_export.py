import csv

import pytest

from cast_asd.tasks.ava_evaluation import export_ava_predictions, timestamp_key

FIELDS = ["video_id", "frame_timestamp", "entity_box_x1", "entity_box_y1",
          "entity_box_x2", "entity_box_y2", "label", "entity_id"]


def _groundtruth(tmp_path, rows):
    path = tmp_path / "gt.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(timestamp, label, entity="face"):
    return {"video_id": "v", "frame_timestamp": timestamp, "entity_box_x1": "0.1",
            "entity_box_y1": "0.2", "entity_box_x2": "0.3", "entity_box_y2": "0.4",
            "label": label, "entity_id": entity}


def test_mixed_timestamp_formatting_resolves_to_one_key():
    assert timestamp_key("960") == timestamp_key("960.0") == timestamp_key("960.00")


def test_export_matches_rows_and_echoes_columns_verbatim(tmp_path):
    rows = [_row("960", "SPEAKING_AUDIBLE"), _row("960.0", "NOT_SPEAKING"),
            _row("960.08", "SPEAKING_AUDIBLE")]
    # Distinct timestamps; the first two differ only in formatting of 960 vs 960.0
    rows[1]["frame_timestamp"] = "960.04"
    groundtruth = _groundtruth(tmp_path, rows)
    scores = {("face", 96000): 0.9, ("face", 96004): 0.1, ("face", 96008): 0.8}

    result = export_ava_predictions(groundtruth, tmp_path / "pred.csv", scores)
    assert result.rows == 3
    assert result.average_precision == 100.0

    exported = list(csv.DictReader((tmp_path / "pred.csv").open()))
    assert [r["frame_timestamp"] for r in exported] == ["960", "960.04", "960.08"]
    assert all(r["entity_box_x1"] == "0.1" for r in exported)
    assert all(r["label"] == "SPEAKING_AUDIBLE" for r in exported)


def test_missing_and_extra_scores_are_reported(tmp_path):
    groundtruth = _groundtruth(tmp_path, [_row("960", "SPEAKING_AUDIBLE")])
    with pytest.raises(ValueError, match="no score"):
        export_ava_predictions(groundtruth, tmp_path / "p.csv", {})
    with pytest.raises(ValueError, match="match no groundtruth"):
        export_ava_predictions(
            groundtruth, tmp_path / "p.csv",
            {("face", 96000): 0.5, ("ghost", 1): 0.5},
        )


def test_speaking_not_audible_counts_as_negative(tmp_path):
    rows = [_row("960", "SPEAKING_AUDIBLE"), _row("960.04", "SPEAKING_NOT_AUDIBLE")]
    groundtruth = _groundtruth(tmp_path, rows)
    result = export_ava_predictions(
        groundtruth, tmp_path / "p.csv",
        {("face", 96000): 0.9, ("face", 96004): 0.1},
    )
    assert result.average_precision == 100.0
