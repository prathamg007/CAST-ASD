"""Hand-checked scene-conditioned metrics and oracle DER/JER."""

import math

import pytest
import torch

from cast_asd.tasks.scene_metrics import SceneDiagnostics, scene_active_counts


def _one_scene():
    """One scene, two tracks, four frames, every population non-empty.

    frame:            0     1     2     3
    track 0 label     1     1     0     0
    track 1 label     0     1     1     0
    speakers          1     2     1     0
                    solo  OVERLAP      quiet
    """
    labels = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    scores = torch.tensor([[0.9, 0.9, 0.8, 0.1], [0.6, 0.9, 0.9, 0.2]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    scene_index = torch.tensor([0, 0])
    diagnostics = SceneDiagnostics()
    diagnostics.update(["s_0", "s_1"], scores, labels, mask, scene_index, scenes=1)
    return diagnostics


def test_scene_active_counts_sums_the_tracks_of_each_scene():
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    counts = scene_active_counts(labels, torch.tensor([0, 0, 1]), scenes=2)
    assert counts.tolist() == [[2.0, 1.0], [0.0, 1.0]]


def test_others_speaking_excludes_the_track_itself():
    diagnostics = _one_scene()
    _, labels, others = diagnostics.flat()
    # track 0 then track 1, in sample-id order.
    assert others.tolist() == [0, 1, 1, 0, 1, 1, 0, 0]
    # At frame 1 both speak, so each sees exactly one *other* speaker -- not two.
    assert labels[1] == 1 and others[1] == 1


def test_the_breakdown_matches_the_hand_count():
    metrics = _one_scene().compute()

    assert metrics["frames"] == 8
    assert metrics["positive_rate"] == 0.5
    # 2 of the 4 negatives sit next to a speaking face.
    assert metrics["competing_rate"] == 0.5
    # At 0.5: 4 true positives, 2 false positives (both competing), no misses.
    assert metrics["precision"] == pytest.approx(4 / 6)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.8)
    # The 90%-recall cut lands at 0.9, above both false positives.
    assert metrics["threshold@90r"] == pytest.approx(0.9)
    assert metrics["fa_competing@90r"] == 0.0
    assert metrics["fa_quiet@90r"] == 0.0
    assert metrics["recall_overlap@90r"] == 1.0
    assert "oracle_der" in metrics


def test_threshold_is_honoured():
    """The cut moves the threshold metrics and nothing else."""
    diagnostics = _one_scene()

    diagnostics.threshold = 0.85
    metrics = diagnostics.compute()
    # Both false positives (0.8, 0.6) drop out; every positive is at 0.9.
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0

    diagnostics.threshold = 0.95
    metrics = diagnostics.compute()
    assert math.isnan(metrics["precision"])
    assert metrics["recall"] == 0.0
    # AP is threshold-free.
    assert metrics["ap"] == _one_scene().compute()["ap"]


def test_subset_ap_is_nan_when_the_subset_has_no_positive():
    """NaN, not 0.0: an unmeasurable population must not read as a bad one."""
    labels = torch.tensor([[1.0, 1.0]])            # nobody is ever silent
    scores = torch.tensor([[0.9, 0.8]])
    diagnostics = SceneDiagnostics()
    diagnostics.update(
        ["s_0"], scores, labels, torch.ones_like(labels, dtype=torch.bool),
        torch.tensor([0]), scenes=1,
    )
    metrics = diagnostics.compute()
    assert math.isnan(metrics["fa@90r"])           # no negatives at all
    assert math.isnan(metrics["ap_competing"])     # single track, never competing
    assert metrics["ap_quiet"] == metrics["ap"]


def test_padding_is_excluded_from_the_scene_count_and_the_metrics():
    labels = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    scores = torch.tensor([[0.9, 0.9], [0.1, 0.9]])
    mask = torch.tensor([[True, False], [True, False]])   # frame 1 is padding
    diagnostics = SceneDiagnostics()
    diagnostics.update(["s_0", "s_1"], scores, labels, mask, torch.tensor([0, 0]), 1)
    scores_flat, labels_flat, others = diagnostics.flat()

    assert len(scores_flat) == 2
    assert labels_flat.tolist() == [1.0, 0.0]
    # The padded frame's label must not count towards frame 0's scene total.
    assert others.tolist() == [0.0, 1.0]


def test_merge_deduplicates_by_sample_id():
    """Ranks see the same unsharded val set; the union must not double-count."""
    first, second = _one_scene(), _one_scene()
    first.merge(second.state())
    assert first.compute()["frames"] == 8


# ----------------------------------------------------------------- diarization


def test_der_is_missed_plus_false_alarm_under_oracle_tracking():
    """Speaker error is zero by construction, so DER is exactly MS + FA."""
    import numpy as np

    from cast_asd.tasks.diarization_metrics import score_recording

    # 100 frames at 25 fps = 4 s. Speaker 0 talks the first half, speaker 1 the
    # second; the system shifts speaker 1 four frames late.
    reference = np.zeros((2, 100), dtype=bool)
    reference[0, :50] = True
    reference[1, 50:] = True
    hypothesis = reference.copy()
    hypothesis[1, 50:54] = False

    # collar=0 so the shifted boundary is actually charged rather than forgiven.
    result = score_recording(reference, hypothesis, fps=25.0, collar=0.0)
    assert result.missed == pytest.approx(4 / 25)
    assert result.false_alarm == 0.0
    assert result.reference_speech == pytest.approx(100 / 25)
    assert result.der == pytest.approx(100 * (4 / 25) / (100 / 25))


def test_the_collar_forgives_exactly_the_boundary_region():
    """A 0.25 s collar at 25 fps is 6 frames each side of every boundary."""
    import numpy as np

    from cast_asd.tasks.diarization_metrics import collar_scored_mask

    reference = np.zeros((1, 100), dtype=bool)
    reference[0, 40:60] = True
    scored = collar_scored_mask(reference, fps=25.0, collar=0.25)

    assert not scored[40]           # the turn's start boundary
    assert not scored[59]           # inside the end boundary's window
    assert scored[0] and scored[99]  # far from any boundary
    # Two boundaries, 12 frames each, and they do not overlap here.
    assert (~scored).sum() == 24


def test_jer_is_one_minus_jaccard_per_speaker():
    import numpy as np

    from cast_asd.tasks.diarization_metrics import score_recording

    reference = np.zeros((2, 100), dtype=bool)
    reference[0, :50] = True         # speaker 0: 50 frames
    reference[1, :20] = True         # speaker 1: 20 frames
    hypothesis = np.zeros((2, 100), dtype=bool)
    hypothesis[0, :50] = True        # perfect
    hypothesis[1, :40] = True        # 20 frames of false alarm

    result = score_recording(reference, hypothesis, fps=25.0, collar=0.0)
    # speaker 0: intersection 50, union 50 -> 0.  speaker 1: 20/40 -> 0.5.
    assert result.jaccard == pytest.approx([0.0, 0.5])
    assert result.jer == pytest.approx(25.0)


def test_a_speaker_who_never_speaks_is_skipped_not_scored_perfect():
    import numpy as np

    from cast_asd.tasks.diarization_metrics import score_recording

    reference = np.zeros((2, 40), dtype=bool)
    reference[0, :20] = True
    hypothesis = np.zeros((2, 40), dtype=bool)
    hypothesis[0, :20] = True

    result = score_recording(reference, hypothesis, fps=25.0, collar=0.0)
    assert len(result.jaccard) == 1        # the silent speaker contributes nothing
    assert result.jer == pytest.approx(0.0)


def test_matched_recall_removes_the_calibration_advantage():
    """Two models with identical ranking tie on fa@90r, whatever the scale."""
    labels = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    sharp = torch.tensor([[0.9, 0.8, 0.7, 0.1], [0.6, 0.95, 0.85, 0.2]])
    squashed = sharp * 0.4          # same order, everything below 0.5
    mask = torch.ones_like(labels, dtype=torch.bool)

    def diagnose(scores):
        d = SceneDiagnostics()
        d.update(["s_0", "s_1"], scores, labels, mask, torch.tensor([0, 0]), 1)
        return d.compute()

    a, b = diagnose(sharp), diagnose(squashed)
    assert a["recall"] != b["recall"]                # the fixed cut is fooled
    assert a["fa@90r"] == pytest.approx(b["fa@90r"])  # the matched point is not
    assert a["fa_competing@90r"] == pytest.approx(b["fa_competing@90r"])
    assert a["ap"] == pytest.approx(b["ap"])


def test_rttm_reference_charges_offscreen_speech_the_packed_labels_hide(tmp_path):
    """The reason to score against the RTTM rather than the packed labels.

    A speaker with no face track is invisible to the pack, so scoring against
    packed labels gives their speech away for free. The RTTM has them, and the
    same speech becomes missed speech -- which is what a deployed system pays.
    """
    import numpy as np

    from cast_asd.tasks.diarization_metrics import (
        align_to_reference,
        load_rttm_dir,
        rttm_reference,
        score_recording,
    )

    # Video "vid": speaker 0 is on camera for 0-2 s; speaker 9 is off camera and
    # talks 2-4 s. The pack only ever saw speaker 0.
    rttm = tmp_path / "vid.rttm"
    rttm.write_text(
        "SPEAKER vid 1 0.00 2.00 <NA> <NA> 0 <NA> <NA>\n"
        "SPEAKER vid 1 2.00 2.00 <NA> <NA> 9 <NA> <NA>\n"
    )
    loaded = load_rttm_dir(tmp_path)
    assert set(loaded["vid"]) == {"0", "9"}

    frames, fps = 100, 25.0            # 4 s
    reference, reference_speakers = rttm_reference(loaded, "vid", 0.0, frames, fps)
    assert reference_speakers == ["0", "9"]
    assert reference[0, :50].all() and not reference[0, 50:].any()
    assert reference[1, 50:].all() and not reference[1, :50].any()

    # The model gets speaker 0 exactly right and cannot say anything about 9.
    hypothesis = np.zeros((1, frames), dtype=bool)
    hypothesis[0, :50] = True

    aligned_reference, aligned_hypothesis = align_to_reference(
        hypothesis, ["0"], reference, reference_speakers
    )
    assert aligned_reference.shape == aligned_hypothesis.shape == (2, frames)

    result = score_recording(aligned_reference, aligned_hypothesis, fps, collar=0.0)
    # Speaker 9's whole 2 s is missed; nothing is a false alarm.
    assert result.missed == pytest.approx(2.0)
    assert result.false_alarm == 0.0
    assert result.der == pytest.approx(50.0)      # 2 s missed of 4 s reference

    # Against the packed labels (speaker 0 only) the same system looks perfect.
    packed = score_recording(
        aligned_reference[:1], aligned_hypothesis[:1], fps, collar=0.0
    )
    assert packed.der == pytest.approx(0.0)


def test_a_tracked_speaker_absent_from_the_rttm_is_charged_not_dropped():
    import numpy as np

    from cast_asd.tasks.diarization_metrics import align_to_reference, score_recording

    reference = np.zeros((1, 50), dtype=bool)
    reference[0, :25] = True
    hypothesis = np.ones((1, 50), dtype=bool)          # a track the RTTM lacks

    aligned_reference, aligned_hypothesis = align_to_reference(
        hypothesis, ["ghost"], reference, ["0"]
    )
    assert aligned_reference.shape == (2, 50)
    result = score_recording(aligned_reference, aligned_hypothesis, 25.0, collar=0.0)
    assert result.false_alarm == pytest.approx(50 / 25)   # every frame of "ghost"
    assert result.missed == pytest.approx(25 / 25)        # speaker 0 never claimed


def test_align_to_reference_does_not_binarise_scores():
    """The sweep aligns once and re-thresholds; bool coercion would break it."""
    import numpy as np

    from cast_asd.tasks.diarization_metrics import align_to_reference

    reference = np.zeros((1, 4), dtype=bool)
    scores = np.array([[0.1, 0.4, 0.6, 0.9]], dtype=np.float32)
    _, aligned = align_to_reference(scores, ["0"], reference, ["0"])
    assert aligned.dtype == np.float32
    assert aligned[0].tolist() == pytest.approx([0.1, 0.4, 0.6, 0.9])


def test_occupancy_buckets_split_by_faces_per_scene():
    """The per-face-count breakdown."""
    labels, scores, ids = [], [], []
    # one 2-face scene, one 3-face scene, one 5-face scene
    for scene, k in (("v_0", 2), ("v_1", 3), ("v_2", 5)):
        for track in range(k):
            labels.append([1.0, 0.0, 1.0, 0.0])
            scores.append([0.9, 0.6, 0.8, 0.1])
            ids.append(f"{scene}_{track}")
    labels = torch.tensor(labels)
    scores = torch.tensor(scores)
    mask = torch.ones_like(labels, dtype=torch.bool)
    scene_index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])

    d = SceneDiagnostics()
    d.update(ids, scores, labels, mask, scene_index, scenes=3)
    m = d.compute()

    assert m["frames_k2"] == 2 * 4
    assert m["frames_k3"] == 3 * 4
    assert m["frames_k5plus"] == 5 * 4      # 5 falls in the open-ended bucket
    assert "frames_k4" not in m             # no 4-face scene here
    for tag in ("k2", "k3", "k5plus"):
        assert f"ap_{tag}" in m and f"fa_competing_{tag}" in m


def test_diagnostics_score_der_against_the_rttm(tmp_path):
    """Off-screen speech in the RTTM is charged as missed; the collar is applied."""
    from cast_asd.tasks.diarization_metrics import load_rttm_dir

    (tmp_path / "v.rttm").write_text(
        "SPEAKER 00007 1 0.00 1.00 <NA> <NA> 0 <NA> <NA>\n"
        "SPEAKER 00007 1 2.00 1.00 <NA> <NA> h0 <NA> <NA>\n"
    )
    labels = torch.zeros(1, 100)
    labels[0, :25] = 1.0                       # tracked speaker 0 speaks 0-1 s
    diagnostics = SceneDiagnostics(rttm=load_rttm_dir(tmp_path))
    diagnostics.update(
        ["00007_00000_0"], labels.clone(), labels, torch.ones(1, 100, dtype=torch.bool),
        torch.tensor([0]), 1, start_sec=[0.0], source_fps=[25.0],
    )
    metrics = diagnostics.diarization()
    # After the 6-frame collar each speaker keeps 13 scored frames; h0's are missed.
    assert metrics["oracle_fa"] == 0.0
    assert metrics["oracle_der"] == pytest.approx(50.0)
