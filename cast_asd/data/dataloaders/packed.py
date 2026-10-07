"""Memory-mapped reader for packed ASD shards (v3 and v4, native timeline).

Audio is returned at its native duration, never resampled or time-warped onto
a frame grid; audio features are aligned to the video frames inside the model
(`models/modules/resample.py`).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from cast_asd.data.augment import augment_visual
from cast_asd.data.media import (
    ASD_MAX_GAIN_DB,
    ASD_PEAK_LIMIT,
    ASD_TARGET_RMS_DBFS,
    SAMPLE_RATE,
    normalize_waveform,
)
from cast_asd.preprocess.schema import (
    FORMAT_NAME,
    FORMAT_VERSION,
    TIMELINE,
    V4_FORMAT_NAME,
    V4_FORMAT_VERSION,
)

#: Accepted formats -> index-field renames onto the v3 names.
_SUPPORTED_FORMATS = {
    (FORMAT_NAME, FORMAT_VERSION): {},
    (V4_FORMAT_NAME, V4_FORMAT_VERSION): {
        "source_frames": "frames",
        "face_offset": "frame_offset",
    },
}

_EXPECTED_NORMALIZATION = {
    "remove_dc": True,
    "target_rms_dbfs": ASD_TARGET_RMS_DBFS,
    "max_gain_db": ASD_MAX_GAIN_DB,
    "peak_limit": ASD_PEAK_LIMIT,
}


@dataclass(frozen=True)
class _Entry:
    root: Path
    shard: str
    metadata: dict
    #: v4 shards store source-gain PCM, normalised on load to match v3.
    normalize: bool = False


class PackedASDDataset(torch.utils.data.Dataset):
    """Random-access per-track dataset over OS-page-cached NumPy memmaps."""

    def __init__(self, roots: str | Path | Sequence[str | Path], augment: bool = False):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.augment = augment
        self.entries: list[_Entry] = []
        self._arrays: dict[tuple[Path, str], tuple[np.ndarray, ...]] = {}
        self.groundtruth_rows = 0

        for raw_root in roots:
            root = Path(raw_root)
            with (root / "dataset.json").open() as handle:
                manifest = json.load(handle)
            key = (manifest.get("format"), manifest.get("version"))
            if key not in _SUPPORTED_FORMATS or manifest.get("timeline") != TIMELINE:
                raise ValueError(f"unsupported packed dataset format: {root}")
            renames = _SUPPORTED_FORMATS[key]
            if manifest.get("audio_sample_rate") != SAMPLE_RATE:
                raise ValueError(f"incompatible audio sample rate: {root}")
            stored_normalization = manifest.get("audio_normalization")
            if stored_normalization not in (None, _EXPECTED_NORMALIZATION):
                raise ValueError(f"incompatible audio normalization: {root}")
            root_normalize = stored_normalization is None

            root_frames = 0
            for shard_dir in sorted(root.glob("shard-*")):
                with (shard_dir / "index.json").open() as handle:
                    for metadata in json.load(handle):
                        for old, new in renames.items():
                            metadata[new] = metadata.pop(old)
                        self.entries.append(
                            _Entry(root, shard_dir.name, metadata, root_normalize)
                        )
                        root_frames += metadata["frames"]

            rows = manifest.get("groundtruth_rows")
            if rows:
                # Fail at construction rather than after a full validation epoch.
                if root_frames != rows:
                    raise ValueError(
                        f"{root}: packed {root_frames} frames but the ground-truth "
                        f"CSV has {rows} rows; official evaluation cannot align"
                    )
                self.groundtruth_rows += rows

    def __len__(self) -> int:
        return len(self.entries)

    def _open(self, entry: _Entry):
        key = (entry.root, entry.shard)
        arrays = self._arrays.get(key)
        if arrays is None:
            shard_dir = entry.root / entry.shard
            arrays = tuple(
                np.load(shard_dir / name, mmap_mode="r", allow_pickle=False)
                for name in ("faces.npy", "audio.npy", "labels.npy")
            )
            self._arrays[key] = arrays
        return arrays

    def load_visual(self, index: int) -> torch.Tensor:
        """[frames, 112, 112] uint8, augmented when the dataset is in train mode."""
        entry = self.entries[index]
        faces, _, _ = self._open(entry)
        meta = entry.metadata
        frame_start = meta["frame_offset"]
        visual_array = np.array(
            faces[frame_start : frame_start + meta["frames"]], copy=True
        )
        if self.augment:
            visual_array = augment_visual(visual_array)
        return torch.from_numpy(visual_array)

    def load_waveform(self, index: int) -> torch.Tensor:
        """[1, samples] float32, loudness-normalized."""
        entry = self.entries[index]
        _, audio, _ = self._open(entry)
        meta = entry.metadata
        audio_start = meta["audio_offset"]
        waveform = (
            torch.from_numpy(
                np.array(
                    audio[audio_start : audio_start + meta["audio_samples"]], copy=True
                )
            )
            .to(torch.float32)
            .div_(32768.0)
            .unsqueeze(0)
        )
        if entry.normalize:
            waveform, _ = normalize_waveform(waveform)
        return waveform

    def load_labels(self, index: int) -> torch.Tensor:
        entry = self.entries[index]
        _, _, labels = self._open(entry)
        meta = entry.metadata
        frame_start = meta["frame_offset"]
        return torch.from_numpy(
            np.array(labels[frame_start : frame_start + meta["frames"]], copy=True)
        ).float()

    def __getitem__(self, index: int):
        return (
            self.load_waveform(index),
            self.load_visual(index),
            self.load_labels(index),
            self.entries[index].metadata,
        )
