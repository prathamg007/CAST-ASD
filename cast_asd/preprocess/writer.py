"""Pack native-timeline samples into memory-mapped NumPy shards.

A pack is `dataset.json` plus `shard-NNNNNN/{faces,audio,labels}.npy` and an
`index.json` of per-sample offsets. `ShardWriter` writes v3 (AVA, normalized
PCM); `ShardWriterV4` writes v4 (MSDWild, source-gain PCM). See `schema`.
"""

import json
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cast_asd.data.media import (
    ASD_MAX_GAIN_DB,
    ASD_PEAK_LIMIT,
    ASD_TARGET_RMS_DBFS,
    FACE_SIZE,
    SAMPLE_RATE,
    load_waveform,
    normalize_waveform,
    validate_visual,
)
from cast_asd.preprocess.schema import (
    FORMAT_NAME,
    FORMAT_VERSION,
    TIMELINE,
    V4_FORMAT_NAME,
    V4_FORMAT_VERSION,
    SourceRecord,
)


def load_faces(record: SourceRecord) -> np.ndarray:
    """Decode an entity's face crops in timestamp order."""
    paths = sorted(record.visual_path.glob("*.jpg"), key=lambda path: float(path.stem))
    if not paths:
        raise FileNotFoundError(f"{record.sample_id}: no JPEG faces")
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"{record.sample_id}: could not decode {path}")
        frames.append(
            cv2.resize(frame, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
        )
    return np.stack(frames).astype(np.uint8, copy=False)


def _pcm16(waveform) -> np.ndarray:
    values = waveform.squeeze(0).clamp(-1.0, 1.0)
    return (values * 32767.0).round().to(torch.int16).cpu().numpy()


def build_sample(record: SourceRecord) -> dict:
    """Decode one AVA entity into the arrays and metadata a v3 shard needs."""
    faces = validate_visual(load_faces(record))
    labels = np.asarray(record.labels, dtype=np.uint8)
    frames = len(faces)
    if frames != len(labels):
        raise ValueError(
            f"{record.sample_id}: {frames} faces but {len(labels)} labels"
        )

    waveform = load_waveform(record.audio_path)

    # The WAV spans the entity's annotated range, (N-1)/fps for almost every
    # entity; those with dropped frames run slightly longer. It is stored
    # verbatim and the audio front end aligns features to the frame count.
    span_samples = round((frames - 1) / record.source_fps * SAMPLE_RATE)
    drift_sec = (waveform.shape[-1] - span_samples) / SAMPLE_RATE
    if abs(drift_sec) > 2.0:
        raise ValueError(
            f"{record.sample_id}: audio is {waveform.shape[-1]} samples but the "
            f"{frames}-frame span at {record.source_fps} fps implies {span_samples} "
            f"({drift_sec:+.2f}s drift); the entity looks corrupt"
        )

    source_audio = waveform.squeeze(0)
    source_peak = source_audio.abs().max().item() if source_audio.numel() else 0.0
    source_rms = (
        source_audio.square().mean().sqrt().item() if source_audio.numel() else 0.0
    )
    waveform, gain = normalize_waveform(waveform)
    audio = _pcm16(waveform)
    audio_float = waveform.squeeze(0)

    return {
        "faces": faces,
        "audio": audio,
        "labels": labels,
        "metadata": {
            "dataset": "ava",
            "sample_id": record.sample_id,
            "source_fps": record.source_fps,
            "start_sec": 0.0,
            "frames": frames,
            "audio_samples": int(len(audio)),
            "audio_drift_sec": drift_sec,
            "audio_peak": audio_float.abs().max().item() if audio_float.numel() else 0.0,
            "audio_rms": audio_float.square().mean().sqrt().item()
            if audio_float.numel()
            else 0.0,
            "audio_mean": audio_float.mean().item() if audio_float.numel() else 0.0,
            "source_audio_peak": source_peak,
            "source_audio_rms": source_rms,
            "audio_gain_db": 20 * math.log10(gain) if gain > 0 else 0.0,
            "speaking_ratio": float(labels.mean()) if len(labels) else 0.0,
        },
    }


class _ShardBuffer:
    """Accumulates samples and flushes a shard once it would exceed the cap."""

    def __init__(self, output_dir: Path, max_shard_bytes: int):
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self.shard_index = 0
        self.samples_written = 0
        self.frames_written = 0
        self.audio_samples_written = 0
        self._reset()
        output_dir.mkdir(parents=True, exist_ok=True)

    def _reset(self) -> None:
        self.faces: list[np.ndarray] = []
        self.audio: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.entries: list[dict] = []
        self.byte_count = 0
        self.frame_count = 0
        self.audio_count = 0

    def _append(
        self,
        faces: np.ndarray,
        audio: np.ndarray,
        labels: np.ndarray,
        entry: Callable[[int, int], dict],
    ) -> None:
        """`entry(frame_offset, audio_offset)` builds the index row, after any flush."""
        sample_bytes = faces.nbytes + audio.nbytes + labels.nbytes
        if self.entries and self.byte_count + sample_bytes > self.max_shard_bytes:
            self.flush()

        self.entries.append(entry(self.frame_count, self.audio_count))
        self.faces.append(faces)
        self.audio.append(audio)
        self.labels.append(labels)
        self.byte_count += sample_bytes
        self.frame_count += len(faces)
        self.audio_count += len(audio)

    def flush(self) -> None:
        if not self.entries:
            return
        shard_dir = self.output_dir / f"shard-{self.shard_index:06d}"
        shard_dir.mkdir(parents=False, exist_ok=False)
        np.save(shard_dir / "faces.npy", np.concatenate(self.faces), allow_pickle=False)
        np.save(shard_dir / "audio.npy", np.concatenate(self.audio), allow_pickle=False)
        np.save(
            shard_dir / "labels.npy", np.concatenate(self.labels), allow_pickle=False
        )
        with (shard_dir / "index.json").open("w") as handle:
            json.dump(self.entries, handle, separators=(",", ":"))
        self.samples_written += len(self.entries)
        self.frames_written += self.frame_count
        self.audio_samples_written += self.audio_count
        self.shard_index += 1
        self._reset()

    def _write_manifest(self, manifest: dict) -> None:
        with (self.output_dir / "dataset.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)


class ShardWriter(_ShardBuffer):
    """v3 writer for AVA."""

    def __init__(self, output_dir: Path, max_shard_bytes: int = 2_000_000_000):
        super().__init__(output_dir, max_shard_bytes)

    def add_sample(self, sample: dict) -> None:
        metadata = sample["metadata"]
        self._append(
            sample["faces"],
            sample["audio"],
            sample["labels"],
            lambda frame, audio: {
                **metadata, "frame_offset": frame, "audio_offset": audio
            },
        )

    def add(self, record: SourceRecord) -> None:
        self.add_sample(build_sample(record))

    def close(self, groundtruth_rows: int | None = None) -> None:
        self.flush()
        manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "timeline": TIMELINE,
            "audio_sample_rate": SAMPLE_RATE,
            "audio_storage": "unwarped_source",
            "face_shape": [FACE_SIZE, FACE_SIZE],
            "face_dtype": "uint8",
            "audio_dtype": "int16",
            "audio_normalization": {
                "remove_dc": True,
                "target_rms_dbfs": ASD_TARGET_RMS_DBFS,
                "max_gain_db": ASD_MAX_GAIN_DB,
                "peak_limit": ASD_PEAK_LIMIT,
            },
            "label_dtype": "uint8",
            "label_semantics": "1 == SPEAKING_AUDIBLE",
            "shards": self.shard_index,
            "samples": self.samples_written,
            "frames": self.frames_written,
            "audio_samples": self.audio_samples_written,
        }
        if groundtruth_rows is not None:
            manifest["groundtruth_rows"] = groundtruth_rows
        self._write_manifest(manifest)


class ShardWriterV4(_ShardBuffer):
    """v4 (`muvap-asd-mmap`) writer for MSDWild: source-gain PCM.

    `audio_mean/rms/peak` are the sample's level after DC removal; the reader
    recomputes its gain from the PCM and does not read them.
    """

    def __init__(self, output_dir: Path, max_shard_bytes: int = 1_000_000_000):
        super().__init__(output_dir, max_shard_bytes)

    def add_arrays(
        self,
        *,
        dataset: str,
        sample_id: str,
        faces: np.ndarray,
        waveform: torch.Tensor,
        labels,
        source_fps: float,
        start_sec: float = 0.0,
    ) -> None:
        faces = validate_visual(faces)
        labels = np.asarray(labels, dtype=np.uint8)
        if len(faces) != len(labels):
            raise ValueError(
                f"{sample_id}: {len(faces)} faces but {len(labels)} labels"
            )
        source_frames = len(labels)
        if source_frames == 0:
            raise ValueError(f"{sample_id}: sample is empty")

        # Crop or right-pad to exactly the span of the stored frames.
        samples = round(source_frames / source_fps * SAMPLE_RATE)
        waveform = waveform[..., :samples]
        waveform = F.pad(waveform, (0, max(0, samples - waveform.shape[-1]))).contiguous()
        mean = waveform.mean().item() if waveform.numel() else 0.0
        centred = waveform - mean
        rms = centred.square().mean().sqrt().item() if centred.numel() else 0.0
        peak = centred.abs().max().item() if centred.numel() else 0.0
        audio = _pcm16(waveform)

        self._append(
            faces,
            audio,
            labels,
            lambda frame, audio_offset: {
                "dataset": dataset,
                "sample_id": sample_id,
                "source_fps": source_fps,
                "source_frames": source_frames,
                "start_sec": start_sec,
                # Frame count on a 25 Hz grid; informational, unread.
                "model_frames": int(np.floor((source_frames / source_fps) * 25.0 + 1e-9)),
                "face_offset": frame,
                "audio_offset": audio_offset,
                "audio_samples": len(audio),
                "label_offset": frame,
                "audio_mean": mean,
                "audio_rms": rms,
                "audio_peak": peak,
                "speaking_ratio": float(labels.mean()),
            },
        )

    def close(self) -> None:
        self.flush()
        self._write_manifest(
            {
                "format": V4_FORMAT_NAME,
                "version": V4_FORMAT_VERSION,
                "timeline": TIMELINE,
                "audio_sample_rate": SAMPLE_RATE,
                "audio_normalization": None,
                "face_shape": [FACE_SIZE, FACE_SIZE],
                "face_color": "grayscale",
                "face_dtype": "uint8",
                "audio_dtype": "int16",
                "label_dtype": "uint8",
                "shards": self.shard_index,
                "samples": self.samples_written,
            }
        )
