"""Scene-grouped view over a packed MSDWild dataset.

A scene is the set of face tracks in the same window of a video; they share
one soundtrack. MSDWild sample ids are `<video>_<window>_<speaker>`, so
the scene key is everything before the last underscore. One waveform is
carried per scene, which also keeps SNR augmentation coherent across its tracks.
"""

from collections import defaultdict
from typing import Sequence

import random

import torch

from cast_asd.data.augment import mix_audio_overlap
from cast_asd.data.dataloaders.packed import PackedASDDataset


def scene_key(sample_id: str) -> str:
    """``00001_00000_2`` -> ``00001_00000``; an id with no track suffix is its own scene."""
    head, sep, _ = sample_id.rpartition("_")
    return head if sep else sample_id


class SceneASDDataset(torch.utils.data.Dataset):
    """One item per scene: its shared waveform and every track in it."""

    def __init__(self, roots, augment: bool = False, min_frames: int = 0):
        self.tracks = PackedASDDataset(roots, augment=augment)

        grouped: dict[str, list[int]] = defaultdict(list)
        for index, entry in enumerate(self.tracks.entries):
            grouped[scene_key(entry.metadata["sample_id"])].append(index)

        # Sorted by key, and by sample id within a scene, so the track order is
        # independent of shard iteration.
        self.scenes: list[list[int]] = [
            sorted(grouped[key], key=lambda i: self.tracks.entries[i].metadata["sample_id"])
            for key in sorted(grouped)
        ]

        # Train only: dropping short scenes from a validation split would score
        # a subset of it.
        if min_frames:
            self.scenes = [
                members
                for members in self.scenes
                if self.tracks.entries[members[0]].metadata["frames"] >= min_frames
            ]
            if not self.scenes:
                raise ValueError(
                    f"min_frames={min_frames} discarded every scene in {roots}"
                )

        for members in self.scenes:
            metas = [self.tracks.entries[i].metadata for i in members]
            if len({m["frames"] for m in metas}) > 1:
                raise ValueError(
                    f"scene {scene_key(metas[0]['sample_id'])}: tracks disagree on "
                    "frame count, so they cannot share a time axis"
                )
            if len({m["audio_samples"] for m in metas}) > 1:
                raise ValueError(
                    f"scene {scene_key(metas[0]['sample_id'])}: tracks disagree on "
                    "audio length, so they cannot share a waveform"
                )

    @property
    def scene_frames(self) -> list[int]:
        return [self.tracks.entries[m[0]].metadata["frames"] for m in self.scenes]

    @property
    def track_counts(self) -> list[int]:
        return [len(members) for members in self.scenes]

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int):
        members = self.scenes[index]
        # Every track of a scene stores the same PCM.
        waveform = self.tracks.load_waveform(members[0])
        visual = [self.tracks.load_visual(i) for i in members]
        labels = [self.tracks.load_labels(i) for i in members]
        metadata = [self.tracks.entries[i].metadata for i in members]
        return waveform, visual, labels, metadata


class SceneASDCollator:
    """Pad a batch of scenes into flat track rows plus a scene index.

    Rows are tracks (``[tracks, time, ...]``); ``scene_index`` / ``track_slot``
    place each row in its scene. ``waveform``, ``audio_lengths`` and the
    ``scene_*`` fields have one row per scene; `AudioVisualASD` expands the
    audio features through ``scene_index`` after the encoder.

    Visual padding replicates the last real face: zero pixels normalise far
    from the data and would skew the visual frontend's BatchNorm statistics.
    """

    def __init__(
        self,
        audio_augment: bool = False,
        rng: random.Random | None = None,
        frame_quantum: int = 0,
        max_frames: int | None = None,
    ):
        self.audio_augment = audio_augment
        # None -> the `random` module, looked up per call (a module attribute
        # cannot be pickled into spawned workers).
        self.rng = rng
        # Pad to a multiple of this (capped at `max_frames`), matching
        # `ScenePaddedBatchSampler.width`.
        self.frame_quantum = frame_quantum
        self.max_frames = max_frames

    def __call__(self, batch: Sequence[tuple]):
        scene_frames = torch.tensor([item[1][0].shape[0] for item in batch])
        scene_samples = torch.tensor([item[0].shape[-1] for item in batch])
        max_frames = int(scene_frames.max())
        if self.frame_quantum > 1:
            widened = -(-max_frames // self.frame_quantum) * self.frame_quantum
            max_frames = min(widened, self.max_frames or widened)
        max_samples = int(scene_samples.max())
        tracks_per_scene = [len(item[1]) for item in batch]
        total_tracks = sum(tracks_per_scene)

        waveform = torch.zeros(len(batch), 1, max_samples)
        visual = torch.empty(
            total_tracks, max_frames, *batch[0][1][0].shape[1:], dtype=torch.uint8
        )
        labels = torch.zeros(total_tracks, max_frames)
        scene_index = torch.empty(total_tracks, dtype=torch.long)
        track_slot = torch.empty(total_tracks, dtype=torch.long)
        frame_lengths = torch.empty(total_tracks, dtype=torch.long)
        scene_fps = torch.tensor([float(item[3][0]["source_fps"]) for item in batch])
        source_fps: list[float] = []
        start_sec: list[float] = []
        sample_ids: list[str] = []

        rng = self.rng or random
        row = 0
        for scene, (scene_audio, scene_visual, scene_labels, scene_meta) in enumerate(
            batch
        ):
            # One draw per scene: every track hears the same augmented mix.
            if self.audio_augment and len(batch) > 1 and rng.randint(0, 1) == 1:
                donor = rng.choice(
                    [other for other in range(len(batch)) if other != scene]
                )
                scene_audio = mix_audio_overlap(scene_audio, batch[donor][0], rng)
            waveform[scene, :, : scene_audio.shape[-1]] = scene_audio

            for slot, (track_visual, track_labels, meta) in enumerate(
                zip(scene_visual, scene_labels, scene_meta)
            ):
                frames = track_visual.shape[0]
                visual[row, :frames] = track_visual
                visual[row, frames:] = track_visual[-1]
                labels[row, :frames] = track_labels
                scene_index[row] = scene
                track_slot[row] = slot
                frame_lengths[row] = frames
                source_fps.append(float(meta["source_fps"]))
                # Offset in the source video, for RTTM scoring.
                start_sec.append(float(meta["start_sec"]))
                sample_ids.append(meta["sample_id"])
                row += 1

        positions = torch.arange(max_frames)
        frame_mask = positions.unsqueeze(0) < frame_lengths.unsqueeze(1)
        track_mask = torch.zeros(len(batch), max(tracks_per_scene), dtype=torch.bool)
        for scene, count in enumerate(tracks_per_scene):
            track_mask[scene, :count] = True

        return {
            "waveform": waveform,
            "visual": visual,
            "labels": labels,
            "audio_lengths": scene_samples,
            "scene_frame_lengths": scene_frames,
            "scene_source_fps": scene_fps,
            "frame_lengths": frame_lengths,
            "frame_mask": frame_mask,
            "scene_index": scene_index,
            "track_slot": track_slot,
            "track_mask": track_mask,
            "source_fps": torch.tensor(source_fps),
            "start_sec": start_sec,
            "sample_ids": sample_ids,
        }
