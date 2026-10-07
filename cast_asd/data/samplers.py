"""Batching scenes under a bound on padded track-frames."""

import random

from torch.utils.data import Sampler


class ScenePaddedBatchSampler(Sampler[list[int]]):
    """Batch scenes so that `width(max(frames)) x sum(tracks) <= frames_per_batch`.

    A batch is padded to its longest scene times its total track count, so
    bucketing on each scene's own `frames x tracks` does not bound it. Scenes
    are taken in descending frame count, so the first scene of a batch fixes
    its width and a scene joins only while the padded product stays in budget.
    A scene over budget on its own forms a batch alone.
    """

    def __init__(
        self,
        frames,
        tracks,
        frames_per_batch: int = 12000,
        max_batch_size: int = 24,
        shuffle: bool = True,
        frame_quantum: int = 0,
    ):
        if frames_per_batch <= 0:
            raise ValueError("frames_per_batch must be positive")
        if frame_quantum < 0:
            raise ValueError("frame_quantum cannot be negative")
        if len(frames) != len(tracks):
            raise ValueError("frames and tracks must describe the same scenes")
        self.frames = list(frames)
        self.tracks = list(tracks)
        self._longest = max(self.frames) if self.frames else 0
        self.frames_per_batch = frames_per_batch
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        # Rounding batch width up to a multiple of this keeps the number of
        # distinct tensor shapes small (each new shape costs a cuDNN autotune).
        self.frame_quantum = frame_quantum
        self.epoch = 0
        self._cached = self._build(0)

    def width(self, frames: int) -> int:
        """The padded frame count a batch of this length is rounded up to."""
        if self.frame_quantum <= 1:
            return frames
        quantised = -(-frames // self.frame_quantum) * self.frame_quantum
        # Never wider than the longest scene: the tail would be pure padding.
        return min(quantised, self._longest)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._cached = self._build(epoch)

    def _build(self, epoch: int) -> list[list[int]]:
        rng = random.Random(epoch)
        # Random tiebreak within equal frame counts (most MSDWild scenes are
        # exactly 30 s), so shuffling changes which scenes share a batch.
        order = sorted(
            range(len(self.frames)),
            key=lambda index: (self.frames[index], rng.random()),
            reverse=True,
        )

        batches: list[list[int]] = []
        batch: list[int] = []
        width = 0          # padded frame count of this batch
        rows = 0           # track rows accumulated
        for index in order:
            candidate_width = self.width(max(width, self.frames[index]))
            candidate_rows = rows + self.tracks[index]
            if batch and (
                candidate_width * candidate_rows > self.frames_per_batch
                or len(batch) >= self.max_batch_size
            ):
                batches.append(batch)
                batch, width, rows = [], 0, 0
                candidate_width = self.width(self.frames[index])
                candidate_rows = self.tracks[index]
            batch.append(index)
            width, rows = candidate_width, candidate_rows
        if batch:
            batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._cached

    def __len__(self) -> int:
        return len(self._cached)
