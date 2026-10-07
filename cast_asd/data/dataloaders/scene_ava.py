"""Scene-grouped AVA over the packed v3 shards, target-anchored like LoCoNet.

An item is one target entity over its full timeline plus up to
`max_tracks - 1` scene-mates (faces in the same `<video>_<start>_<end>`
segment), aligned onto the target's timestamps and absent elsewhere. Every
ground-truth row of every entity is therefore scored exactly once, as the
official evaluator requires.

Annotations are parsed and contexts resolved once in `__init__`, into flat
numpy arrays rather than per-target Python lists, so forked DataLoader workers
share them copy-on-write. `__getitem__` only slices memmaps.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cast_asd.data.dataloaders.packed import PackedASDDataset


def scene_of(entity_id: str) -> str:
    """``HV0H6oc4Kvs_0960_1020:1`` -> ``HV0H6oc4Kvs_0960_1020``.

    AVA separates entity from scene with ``:``; `scene_packed.scene_key`
    (MSDWild, ``_``) would return the wrong scene here.
    """
    head, sep, _ = entity_id.rpartition(":")
    return head if sep else entity_id


def timestamp_centiseconds(text: str) -> int:
    """`960.03` -> 96003. Integer keys so float formatting cannot split a frame."""
    return int(round(float(text) * 100))


def read_timelines(ava_root, split: str):
    """`{split}_orig.csv` -> (entity timelines, faces present per instant).

    Returns
        entity_ts : entity_id -> ascending int64 array of centiseconds
        ts_faces  : (scene, centisecond) -> [entity_id, ...]
    """
    path = Path(ava_root) / "csv" / f"{split}_orig.csv"
    if not path.exists():
        raise FileNotFoundError(f"AVA scene grouping needs {path}")
    seen = defaultdict(list)
    ts_faces = defaultdict(list)
    with open(path, newline="") as source:
        for row in csv.DictReader(source):
            entity = row["entity_id"]
            cs = timestamp_centiseconds(row["frame_timestamp"])
            seen[entity].append(cs)
            ts_faces[(scene_of(entity), cs)].append(entity)
    timelines = {}
    for entity, stamps in seen.items():
        stamps = np.asarray(sorted(stamps), dtype=np.int64)
        if len(np.unique(stamps)) != len(stamps):
            raise ValueError(f"{entity}: duplicate timestamps in {path.name}")
        timelines[entity] = stamps
    return timelines, ts_faces


class SceneAVADataset(torch.utils.data.Dataset):
    """One item per target entity: its own clip plus its aligned scene-mates.

    `max_tracks` counts the target. A scene-mate qualifies when it shares more
    than `min_overlap` of the target's frames; the ones sharing the most win
    (ties by id), so the grouping is deterministic.
    """

    def __init__(self, ava_root, split, max_tracks=3, min_overlap=0.5,
                 augment=False):
        root = Path(ava_root)
        packed = root / "packed" / split
        if not packed.is_dir():
            raise FileNotFoundError(f"no packed AVA shards at {packed}")
        self.packed = PackedASDDataset(packed, augment=augment)
        self.row_of = {
            entry.metadata["sample_id"]: i
            for i, entry in enumerate(self.packed.entries)
        }
        self.max_tracks = int(max_tracks)
        self.min_overlap = float(min_overlap)

        timelines, ts_faces = read_timelines(root, split)
        self.targets = [e for e in sorted(self.row_of) if e in timelines]
        if not self.targets:
            raise ValueError(f"no usable AVA entities for split {split!r}")
        self.frames = np.array(
            [len(timelines[e]) for e in self.targets], dtype=np.int64
        )
        self.fps = np.array(
            [float(self.packed.entries[self.row_of[e]].metadata["source_fps"])
             for e in self.targets], dtype=np.float64
        )
        self._check_native(timelines)
        # `ts_faces` is the largest structure here and is not needed afterwards.
        self._build_context(timelines, ts_faces)
        del ts_faces

    # -- build ------------------------------------------------------------

    def _check_native(self, timelines):
        """Packed frame count must equal the ground-truth row count."""
        bad = []
        for entity in self.targets:
            packed_frames = self.packed.entries[self.row_of[entity]].metadata["frames"]
            if packed_frames != len(timelines[entity]):
                bad.append((entity, packed_frames, len(timelines[entity])))
        if bad:
            raise ValueError(
                f"{len(bad)} entities where packed frames != groundtruth rows, "
                f"e.g. {bad[:3]}; the pack is not on the native timeline"
            )

    def _build_context(self, timelines, ts_faces):
        """Resolve every target's context and alignment into flat arrays.

        Sets `ctx_ptr` (target -> slice of the slot table), `ctx_row` (the
        packed row each slot reads), and `ctx_iptr` / `ctx_src` / `ctx_dst`
        (slot -> its slice of the concatenated (context frame, target frame)
        index pairs).
        """
        ptr = [0]
        rows, iptr, src_all, dst_all = [], [0], [], []
        for entity in self.targets:
            own = timelines[entity]
            position = {int(cs): i for i, cs in enumerate(own)}
            scene = scene_of(entity)
            counts = defaultdict(int)
            for cs in own:
                for other in ts_faces.get((scene, int(cs)), ()):
                    if other != entity and other in self.row_of:
                        counts[other] += 1
            need = self.min_overlap * len(own)
            ranked = sorted(
                (e for e, n in counts.items() if n > need),
                key=lambda e: (-counts[e], e),
            )[: self.max_tracks - 1]

            for other in ranked:
                other_ts = timelines[other]
                limit = self.packed.entries[self.row_of[other]].metadata["frames"]
                src, dst = [], []
                for j, cs in enumerate(other_ts):
                    if j >= limit:
                        break
                    i = position.get(int(cs))
                    if i is not None:
                        src.append(j)
                        dst.append(i)
                if src:
                    rows.append(self.row_of[other])
                    src_all.extend(src)
                    dst_all.extend(dst)
                    iptr.append(len(src_all))
            ptr.append(len(rows))

        self.ctx_ptr = np.asarray(ptr, dtype=np.int64)
        self.ctx_row = np.asarray(rows, dtype=np.int64)
        self.ctx_iptr = np.asarray(iptr, dtype=np.int64)
        self.ctx_src = np.asarray(src_all, dtype=np.int32)
        self.ctx_dst = np.asarray(dst_all, dtype=np.int32)

    # -- access -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        entity = self.targets[index]
        row = self.row_of[entity]
        frames = int(self.frames[index])

        faces = self.packed.load_visual(row)
        labels = self.packed.load_labels(row)
        waveform = self.packed.load_waveform(row)

        lo, hi = int(self.ctx_ptr[index]), int(self.ctx_ptr[index + 1])
        n_slots = hi - lo
        visual = faces.new_zeros((1 + n_slots, frames, *faces.shape[1:]))
        label_stack = torch.zeros(1 + n_slots, frames)
        frame_mask = torch.zeros(1 + n_slots, frames, dtype=torch.bool)
        visual[0], label_stack[0], frame_mask[0] = faces, labels, True

        for slot, k in enumerate(range(lo, hi), start=1):
            a, b = int(self.ctx_iptr[k]), int(self.ctx_iptr[k + 1])
            src_t = torch.from_numpy(self.ctx_src[a:b].astype(np.int64))
            dst_t = torch.from_numpy(self.ctx_dst[a:b].astype(np.int64))
            other_row = int(self.ctx_row[k])
            visual[slot, dst_t] = self.packed.load_visual(other_row)[src_t]
            label_stack[slot, dst_t] = self.packed.load_labels(other_row)[src_t]
            frame_mask[slot, dst_t] = True

        return {
            "entity_id": entity,
            "waveform": waveform,               # the target's; slots share it
            "visual": visual,                   # [slots, frames, H, W]
            "labels": label_stack,
            "frame_mask": frame_mask,           # per-slot, per-frame presence
            "track_mask": torch.ones(1 + n_slots, dtype=torch.bool),
            "frames": frames,
            # Native rate; the model aligns audio features to frames with it.
            "source_fps": float(self.fps[index]),
        }
