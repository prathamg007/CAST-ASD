"""Scene-conditioned validation metrics for MSDWild.

Frames are split by what the rest of the scene is doing at the same instant:
*competing* frames have another tracked face speaking (hard negatives for a
silent face, overlapped speech for a speaking one); *quiet* frames do not.
Rows are keyed by `sample_id`, so merging the same validation set gathered
from several ranks deduplicates.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from cast_asd.data.dataloaders.scene_packed import scene_key
from cast_asd.tasks.asd_metrics import ava_active_speaker_map
from cast_asd.tasks.diarization_metrics import (
    align_to_reference,
    best_threshold,
    collar_scored_mask,
    prepare,
    rttm_reference,
    score_prepared,
)


def scene_active_counts(
    labels: torch.Tensor, scene_index: torch.Tensor, scenes: int
) -> torch.Tensor:
    """`[scenes, time]`: how many tracks of each scene are speaking at each frame."""
    counts = labels.new_zeros(scenes, labels.shape[1])
    return counts.index_add_(0, scene_index, labels)


def _rate(numerator, denominator) -> float:
    denominator = int(denominator)
    return float("nan") if denominator == 0 else float(numerator) / denominator


def _subset_ap(scores, labels, keep) -> float:
    """AP over a subset; NaN when the subset holds no positive to rank."""
    if not bool(keep.any()) or not bool(labels[keep].any()):
        return float("nan")
    return ava_active_speaker_map(scores[keep], labels[keep])


@dataclass
class SceneDiagnostics:
    """Accumulates one validation split's frames, then reports the metrics."""

    #: Cut for precision/recall/F1 and DER.
    threshold: float = 0.5
    #: Recall at which the false-alarm rates are read (FA@90R), so models are
    #: compared at the same operating point regardless of calibration.
    recall_target: float = 0.90
    fps: float = 25.0
    #: sample_id -> (scores, labels, others_speaking), each `[valid frames]`.
    rows: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    #: sample_id -> (start_sec, fps); places a segment in its video for RTTM scoring.
    placement: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: `{video: {speaker: [(start, end)]}}`; when set, it is the DER reference
    #: instead of the packed on-screen labels.
    rttm: dict | None = None

    def update(
        self,
        sample_ids,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        frame_mask: torch.Tensor,
        scene_index: torch.Tensor,
        scenes: int,
        start_sec=None,
        source_fps=None,
    ) -> None:
        """`probabilities`/`labels`/`frame_mask` are `[tracks, time]`."""
        labels = labels.float()
        active = scene_active_counts(labels * frame_mask, scene_index, scenes)
        others = active[scene_index] - labels
        probabilities = probabilities.detach().float().cpu()
        labels = labels.detach().cpu()
        others = others.detach().float().cpu()
        frame_mask = frame_mask.detach().cpu()
        for row, sample_id in enumerate(sample_ids):
            valid = frame_mask[row]
            self.rows[sample_id] = (
                probabilities[row][valid],
                labels[row][valid],
                others[row][valid],
            )
            if start_sec is not None:
                self.placement[sample_id] = (
                    float(start_sec[row]),
                    float(source_fps[row]) if source_fps is not None else self.fps,
                )

    def merge(self, other: dict) -> None:
        """Merge a `state()` from another rank."""
        self.rows.update(other["rows"])
        self.placement.update(other["placement"])

    def state(self) -> dict:
        return {"rows": self.rows, "placement": self.placement}

    def reset(self) -> None:
        self.rows.clear()
        self.placement.clear()

    def flat(self):
        """`(scores, labels, others_speaking)` over all frames, in sorted sample-id order."""
        if not self.rows:
            empty = torch.zeros(0)
            return empty, empty, empty
        ordered = [self.rows[key] for key in sorted(self.rows)]
        return tuple(torch.cat([row[i] for row in ordered]) for i in range(3))

    def _scenes(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for sample_id in sorted(self.rows):
            grouped[scene_key(sample_id)].append(sample_id)
        return grouped

    def threshold_at_recall(self, scores, labels, target: float) -> float:
        """Lowest cut whose recall is at least `target`; NaN if unreachable."""
        positive = labels > 0.5
        total = int(positive.sum())
        if total == 0:
            return float("nan")
        ranked = torch.sort(scores[positive], descending=True).values
        needed = int(np.ceil(target * total))
        if needed > len(ranked):
            return float("nan")
        return float(ranked[needed - 1])

    def at_recall(self, scores, labels, competing, target: float) -> dict[str, float]:
        """False-alarm rates at the cut that reaches `target` recall."""
        cut = self.threshold_at_recall(scores, labels, target)
        if not np.isfinite(cut):
            return {}
        positive = labels > 0.5
        negative = ~positive
        quiet = ~competing
        predicted = scores >= cut
        tag = f"{int(round(100 * target))}r"
        return {
            f"threshold@{tag}": cut,
            f"recall@{tag}": _rate((predicted & positive).sum(), positive.sum()),
            f"fa@{tag}": _rate((predicted & negative).sum(), negative.sum()),
            f"fa_competing@{tag}": _rate(
                (predicted & negative & competing).sum(), (negative & competing).sum()
            ),
            f"fa_quiet@{tag}": _rate(
                (predicted & negative & quiet).sum(), (negative & quiet).sum()
            ),
            f"recall_overlap@{tag}": _rate(
                (predicted & positive & competing).sum(), (positive & competing).sum()
            ),
        }

    def by_occupancy(self, buckets=(2, 3, 4)) -> dict[str, float]:
        """AP and competing false-alarm rate (at the matched-recall cut) per faces-in-scene.

        The last bucket is open-ended (`k5plus`).
        """
        occupancy = {
            member: len(members) for members in self._scenes().values() for member in members
        }
        sizes = np.concatenate(
            [np.full(len(self.rows[key][0]), occupancy[key]) for key in sorted(self.rows)]
        )
        scores, labels, others = self.flat()
        competing = others > 0.5
        negative = labels <= 0.5

        results: dict[str, float] = {}
        cut = self.threshold_at_recall(scores, labels, self.recall_target)
        predicted = scores >= cut if np.isfinite(cut) else scores >= self.threshold
        for bucket in list(buckets) + ["+"]:
            if bucket == "+":
                keep = torch.from_numpy(sizes > max(buckets))
                tag = f"k{max(buckets) + 1}plus"
            else:
                keep = torch.from_numpy(sizes == bucket)
                tag = f"k{bucket}"
            if not bool(keep.any()):
                continue
            results[f"ap_{tag}"] = _subset_ap(scores, labels, keep)
            results[f"fa_competing_{tag}"] = _rate(
                (keep & negative & competing & predicted).sum(),
                (keep & negative & competing).sum(),
            )
            results[f"frames_{tag}"] = float(int(keep.sum()))
        return results

    def _prepared_scenes(self):
        """Collar-masked `(reference, scores)` per scene for DER scoring."""
        grouped = self._scenes()
        if self.rttm is None:
            # Every track of a scene shares its length, so the rows stack.
            scenes = []
            for members in grouped.values():
                if len({len(self.rows[m][0]) for m in members}) != 1:
                    raise ValueError(f"scene {scene_key(members[0])}: ragged tracks")
                scenes.append((
                    np.stack([self.rows[m][0].numpy() for m in members]),
                    np.stack([self.rows[m][1].numpy() for m in members]),
                ))
            return prepare(scenes, self.fps)

        prepared = []
        for scene, members in grouped.items():
            # Scene key `<video>_<segment>`; sample id `<scene>_<speaker>`.
            video_id = scene.rpartition("_")[0] or scene
            start_sec, fps = self.placement.get(members[0], (0.0, self.fps))
            frames = len(self.rows[members[0]][0])
            reference, reference_speakers = rttm_reference(
                self.rttm, video_id, start_sec, frames, fps
            )
            hypothesis = np.stack([self.rows[m][0].numpy() for m in members])
            aligned_reference, aligned_scores = align_to_reference(
                hypothesis, [m.rpartition("_")[2] for m in members],
                reference, reference_speakers,
            )
            scored = collar_scored_mask(aligned_reference, self.fps)
            prepared.append((aligned_reference[:, scored], aligned_scores[:, scored]))
        return prepared

    def diarization(self) -> dict[str, float]:
        """Oracle DER/JER at `threshold`, plus the DER-minimising threshold and its DER."""
        prepared = self._prepared_scenes()
        if not prepared:
            return {}
        results = score_prepared(prepared, self.threshold, self.fps).as_dict()
        cut, der = best_threshold(prepared, self.fps)
        results["oracle_der_tuned"] = der
        results["oracle_der_threshold"] = cut
        return results

    def compute(self) -> dict[str, float]:
        scores, labels, others = self.flat()
        if not len(scores):
            return {}

        positive = labels > 0.5
        negative = ~positive
        competing = others > 0.5           # another tracked face is speaking
        quiet = ~competing

        predicted = scores >= self.threshold
        true_positive = (predicted & positive).sum()
        false_positive = (predicted & negative).sum()
        false_negative = ((~predicted) & positive).sum()

        results = {
            "ap": ava_active_speaker_map(scores, labels),
            # mAP_c: AP restricted to competing frames.
            "ap_competing": _subset_ap(scores, labels, competing),
            "ap_quiet": _subset_ap(scores, labels, quiet),
            "precision": _rate(true_positive, true_positive + false_positive),
            "recall": _rate(true_positive, true_positive + false_negative),
            "f1": _rate(
                2 * true_positive, 2 * true_positive + false_positive + false_negative
            ),
            "frames": float(len(scores)),
            "positive_rate": _rate(positive.sum(), len(scores)),
            "competing_rate": _rate((negative & competing).sum(), negative.sum()),
        }
        results.update(self.at_recall(scores, labels, competing, self.recall_target))
        results.update(self.by_occupancy())
        results.update(self.diarization())
        return results
