"""DER and JER with oracle tracking and clustering (MSDWild scoring: 0.25 s collar, overlap scored).

Each track carries its MSDWild speaker id, so the model only decides *when* each
known speaker talks: speaker error is zero and DER = missed + false alarm. Keys
are prefixed `oracle_` to keep this apart from MSDWild's end-to-end DER. With an
RTTM the reference includes off-screen speakers, whose speech counts as missed.
Scores are micro-aggregated over the packed segments of a split.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLLAR_SECONDS = 0.25


def runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges `[start, end)` where a boolean frame array is True."""
    if not active.any():
        return []
    padded = np.concatenate(([False], active.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def collar_scored_mask(
    reference: np.ndarray, fps: float, collar: float = COLLAR_SECONDS
) -> np.ndarray:
    """`[frames]`: True outside a `+-collar` window of any reference boundary.

    As in md-eval, the collar around any speaker's turn boundary excludes those
    frames for every speaker of the recording.
    """
    frames = reference.shape[1]
    scored = np.ones(frames, dtype=bool)
    width = int(round(collar * fps))
    if width <= 0:
        return scored
    for speaker in reference:
        for start, end in runs(speaker):
            for boundary in (start, end):
                scored[max(0, boundary - width) : min(frames, boundary + width)] = False
    return scored


@dataclass
class DiarizationResult:
    missed: float
    false_alarm: float
    reference_speech: float
    overlap_error: float
    overlap_reference: float
    jaccard: list[float]
    scored_seconds: float

    @property
    def der(self) -> float:
        if self.reference_speech <= 0:
            return float("nan")
        return 100.0 * (self.missed + self.false_alarm) / self.reference_speech

    @property
    def jer(self) -> float:
        return 100.0 * float(np.mean(self.jaccard)) if self.jaccard else float("nan")

    def as_dict(self) -> dict[str, float]:
        reference = max(self.reference_speech, 1e-9)
        return {
            "oracle_der": self.der,
            "oracle_jer": self.jer,
            "oracle_ms": 100.0 * self.missed / reference,
            "oracle_fa": 100.0 * self.false_alarm / reference,
            # Error on the overlapped frames only.
            "oracle_ovl": (
                100.0 * self.overlap_error / self.overlap_reference
                if self.overlap_reference > 0
                else float("nan")
            ),
            "oracle_speech_seconds": self.reference_speech,
        }


def _score_one(reference: np.ndarray, hypothesis: np.ndarray, fps: float) -> DiarizationResult:
    """One collar-masked recording; `[speakers, frames]` bool, rows aligned by speaker."""
    step = 1.0 / fps
    overlapped = reference.sum(axis=0) >= 2
    jaccard = []
    for speaker_reference, speaker_hypothesis in zip(reference, hypothesis):
        union = float((speaker_reference | speaker_hypothesis).sum())
        if union == 0:
            # Silent and never claimed: no evidence either way (DIHARD's JER skips these).
            continue
        intersection = float((speaker_reference & speaker_hypothesis).sum())
        jaccard.append(1.0 - intersection / union)
    return DiarizationResult(
        missed=float((reference & ~hypothesis).sum()) * step,
        false_alarm=float((~reference & hypothesis).sum()) * step,
        reference_speech=float(reference.sum()) * step,
        overlap_error=float((reference[:, overlapped] ^ hypothesis[:, overlapped]).sum()) * step,
        overlap_reference=float(reference[:, overlapped].sum()) * step,
        jaccard=jaccard,
        scored_seconds=float(reference.shape[1]) * step,
    )


def combine(results: list[DiarizationResult]) -> DiarizationResult:
    """Micro-aggregate: total error time over total reference time."""
    return DiarizationResult(
        missed=sum(r.missed for r in results),
        false_alarm=sum(r.false_alarm for r in results),
        reference_speech=sum(r.reference_speech for r in results),
        overlap_error=sum(r.overlap_error for r in results),
        overlap_reference=sum(r.overlap_reference for r in results),
        # JER averages over speakers, so the lists concatenate.
        jaccard=[value for r in results for value in r.jaccard],
        scored_seconds=sum(r.scored_seconds for r in results),
    )


def score_recording(
    reference: np.ndarray,
    hypothesis: np.ndarray,
    fps: float,
    collar: float = COLLAR_SECONDS,
) -> DiarizationResult:
    """One recording. `reference`/`hypothesis` are `[speakers, frames]` boolean."""
    if reference.shape != hypothesis.shape:
        raise ValueError("reference and hypothesis must have the same shape")
    scored = collar_scored_mask(reference, fps, collar)
    return _score_one(
        reference[:, scored].astype(bool), hypothesis[:, scored].astype(bool), fps
    )


def prepare(scenes, fps: float = 25.0, collar: float = COLLAR_SECONDS):
    """`(scores, labels)` per scene -> collar-masked `(reference, scores)`.

    The collar depends only on the reference, so it is applied once and a
    threshold sweep is then pure boolean arithmetic.
    """
    prepared = []
    for scores, labels in scenes:
        reference = labels.astype(bool)
        scored = collar_scored_mask(reference, fps, collar)
        prepared.append((reference[:, scored], np.asarray(scores)[:, scored]))
    return prepared


def score_prepared(prepared, threshold: float, fps: float) -> DiarizationResult:
    return combine(
        [_score_one(reference, scores >= threshold, fps) for reference, scores in prepared]
    )


def best_threshold(prepared, fps: float, grid=None) -> tuple[float, float]:
    """The threshold minimising DER over `grid`, and that DER (NaN, NaN if none)."""
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else np.asarray(grid)
    scored = [(float(t), score_prepared(prepared, float(t), fps).der) for t in grid]
    scored = [(t, d) for t, d in scored if not np.isnan(d)]
    if not scored:
        return float("nan"), float("nan")
    return min(scored, key=lambda pair: pair[1])


# --------------------------------------------------------------- RTTM reference


def load_rttm_dir(path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """`{video_id: {speaker: [(start, end)]}}` from a directory of RTTMs or one file.

    The video id is read from the RTTM field, not the filename.
    """
    path = Path(path)
    files = sorted(path.glob("*.rttm")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no .rttm files under {path}")

    activity: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for file in files:
        for number, line in enumerate(file.open(), start=1):
            parts = line.split()
            if not parts:
                continue
            if parts[0] != "SPEAKER":
                raise ValueError(f"{file}:{number}: not an RTTM SPEAKER row")
            if len(parts) < 8:
                raise ValueError(f"{file}:{number}: RTTM row has {len(parts)} fields")
            start, duration = float(parts[3]), float(parts[4])
            activity[parts[1]][parts[7]].append((start, start + duration))
    return {
        video: {speaker: sorted(spans) for speaker, spans in speakers.items()}
        for video, speakers in activity.items()
    }


def rttm_reference(
    rttm: dict[str, dict[str, list[tuple[float, float]]]],
    video_id: str,
    start_sec: float,
    frames: int,
    fps: float,
) -> tuple[np.ndarray, list[str]]:
    """`[speakers, frames]` reference for one packed segment, plus speaker ids.

    Includes speakers with no face track. Frame `n` belongs to a speaker when its
    midpoint `start_sec + (n + 0.5) / fps` falls inside one of their turns.
    """
    speakers = sorted(rttm.get(video_id, {}))
    reference = np.zeros((len(speakers), frames), dtype=bool)
    if not speakers:
        return reference, speakers

    midpoints = start_sec + (np.arange(frames) + 0.5) / fps
    for row, speaker in enumerate(speakers):
        for begin, end in rttm[video_id][speaker]:
            if end <= midpoints[0] or begin >= midpoints[-1]:
                continue
            reference[row] |= (midpoints >= begin) & (midpoints < end)
    return reference, speakers


def align_to_reference(
    hypothesis: np.ndarray,
    hypothesis_speakers: list[str],
    reference: np.ndarray,
    reference_speakers: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Put system and reference rows on one speaker axis.

    An untracked reference speaker gets an all-silent hypothesis row (missed
    speech); a tracked speaker absent from the RTTM gets an all-silent reference
    row (false alarm). The hypothesis keeps its dtype so raw scores can be
    re-thresholded afterwards.
    """
    speakers = list(reference_speakers) + [
        s for s in hypothesis_speakers if s not in set(reference_speakers)
    ]
    frames = reference.shape[1] if reference.size else hypothesis.shape[1]
    aligned_reference = np.zeros((len(speakers), frames), dtype=bool)
    aligned_hypothesis = np.zeros((len(speakers), frames), dtype=hypothesis.dtype)

    index = {speaker: row for row, speaker in enumerate(speakers)}
    for row, speaker in enumerate(reference_speakers):
        aligned_reference[index[speaker]] = reference[row]
    for row, speaker in enumerate(hypothesis_speakers):
        aligned_hypothesis[index[speaker]] = hypothesis[row]
    return aligned_reference, aligned_hypothesis
