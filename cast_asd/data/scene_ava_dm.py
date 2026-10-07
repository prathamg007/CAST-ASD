"""Lightning data module for scene-grouped AVA (`data_backend: scene_ava`).

`collate_scenes` puts all target rows first and lists only targets in
`sample_ids`, which is the row order `ASDTask.validation_step` relies on;
context rows follow. Context rows are trained on but masked out in validation,
so no face is scored twice.
"""

import random
from functools import partial

from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader

from cast_asd.data.augment import mix_audio_overlap
from cast_asd.data.dataloaders.scene_ava import SceneAVADataset


def collate_scenes(items, score_targets_only: bool, audio_augment=False, rng=None):
    """Variable-slot scene items -> one flat batch, targets first.

    `score_targets_only` (validation) masks context rows out of the loss and
    metrics. `audio_augment` gives each scene a 50% chance of SNR-mixed audio
    from another scene in the batch, one draw per scene since its faces share
    one waveform.
    """
    rng = rng or random
    frames = max(int(i["frames"]) for i in items)
    samples = max(int(i["waveform"].shape[-1]) for i in items)
    crop = items[0]["visual"].shape[-1]
    slots = [int(i["track_mask"].shape[0]) for i in items]
    total = sum(slots)

    visual = torch.zeros(total, frames, crop, crop)
    labels = torch.zeros(total, frames)
    frame_mask = torch.zeros(total, frames, dtype=torch.bool)
    scene_index = torch.zeros(total, dtype=torch.long)
    track_slot = torch.zeros(total, dtype=torch.long)
    track_mask = torch.zeros(len(items), max(slots), dtype=torch.bool)
    # One waveform per scene, as in `SceneASDCollator`.
    waveform = torch.zeros(len(items), 1, samples)
    frame_lengths = torch.zeros(total, dtype=torch.long)
    scene_frames = torch.zeros(len(items), dtype=torch.long)
    scene_audio = torch.zeros(len(items), dtype=torch.long)
    fps = torch.zeros(total)
    scene_fps = torch.zeros(len(items))

    # Targets occupy rows 0..len(items)-1; context rows follow.
    row_of = []
    cursor = len(items)
    for scene, item in enumerate(items):
        row_of.append(scene)
        for slot in range(1, slots[scene]):
            row_of.append(cursor)
            cursor += 1

    at = 0
    for scene, item in enumerate(items):
        n = int(item["frames"])
        s = int(item["waveform"].shape[-1])
        clip = item["waveform"].reshape(1, -1)[:, :s]
        if audio_augment and len(items) > 1 and rng.randint(0, 1) == 1:
            donor = rng.choice([o for o in range(len(items)) if o != scene])
            clip = mix_audio_overlap(clip, items[donor]["waveform"].reshape(1, -1), rng)
        waveform[scene, 0, :s] = clip.reshape(-1)[:s]
        scene_frames[scene] = n
        scene_audio[scene] = s
        scene_fps[scene] = item["source_fps"]
        for slot in range(slots[scene]):
            row = row_of[at]; at += 1
            visual[row, :n] = item["visual"][slot, :n].float()
            labels[row, :n] = item["labels"][slot, :n]
            present = item["frame_mask"][slot, :n]
            frame_mask[row, :n] = present if (slot == 0 or not score_targets_only) \
                else torch.zeros_like(present)
            scene_index[row] = scene
            track_slot[row] = slot
            track_mask[scene, slot] = True
            frame_lengths[row] = n
            fps[row] = item["source_fps"]

    return {
        "sample_ids": [i["entity_id"] for i in items],   # targets only, in row order
        "waveform": waveform,
        "visual": visual,
        "labels": labels,
        "frame_mask": frame_mask,
        "frame_lengths": frame_lengths,
        "audio_lengths": scene_audio,
        "source_fps": fps,
        "scene_index": scene_index,
        "track_slot": track_slot,
        "track_mask": track_mask,
        "scene_frame_lengths": scene_frames,
        "scene_source_fps": scene_fps,
    }


class EqualShapeBatches(torch.utils.data.Sampler):
    """Validation batches of entities with identical (frames, fps).

    Such batches need no padding, so scores match batch size 1 (offline models
    would otherwise attend to padded frames). Order is deterministic and every
    entity appears exactly once.
    """

    def __init__(self, dataset, batch_size: int):
        self.batches = []
        buckets = {}
        for index in range(len(dataset)):
            key = (int(dataset.frames[index]), round(float(dataset.fps[index]), 3))
            buckets.setdefault(key, []).append(index)
        for key in sorted(buckets):
            group = buckets[key]
            for start in range(0, len(group), batch_size):
                self.batches.append(group[start:start + batch_size])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


class SceneAVADataModule(LightningDataModule):
    """AVA batched by target entity with its scene-mates."""

    def __init__(self, cfg: dict):
        super().__init__()
        asd = cfg["asd"]
        self.root = cfg["ava_asd"]["root"]
        self.max_tracks = int((asd.get("scene") or {}).get("max_tracks") or 3)
        self.min_overlap = float(asd.get("scene_min_overlap", 0.5))
        self.num_workers = int(asd.get("num_workers", 8))
        self.train_scenes = int(asd.get("scenes_per_batch", 4))
        # Upper bound per equal-shape bucket; see `EqualShapeBatches`.
        self.val_scenes = int(asd.get("val_scenes_per_batch", 8))
        # Drives both the per-face visual augmentation and the per-scene mix.
        self.augment = bool(asd.get("augment", False))

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = SceneAVADataset(
                self.root, "train", self.max_tracks, self.min_overlap,
                augment=self.augment
            )
        if stage in ("fit", "validate", None):
            self.val_dataset = SceneAVADataset(
                self.root, "val", self.max_tracks, self.min_overlap, augment=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.train_scenes, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            # A lambda would not pickle into spawned workers (Windows, macOS).
            collate_fn=partial(
                collate_scenes, score_targets_only=False, audio_augment=self.augment
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=EqualShapeBatches(self.val_dataset, self.val_scenes),
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=partial(collate_scenes, score_targets_only=True),
        )
