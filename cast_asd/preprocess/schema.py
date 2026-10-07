"""Packed-shard formats and the AVA source record.

Both formats store faces, labels and audio on the source video's native
timeline: one packed frame per annotated frame, audio never time-warped.

* v3 `ava-asd-mmap` (AVA, `preprocess.ava`): PCM stored loudness-normalized.
* v4 `muvap-asd-mmap` (MSDWild, `preprocess.msdwild`): PCM at source gain;
  the reader normalizes on load. Index fields `source_frames`/`face_offset`
  are v3's `frames`/`frame_offset`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FORMAT_NAME = "ava-asd-mmap"
FORMAT_VERSION = 3
V4_FORMAT_NAME = "muvap-asd-mmap"
V4_FORMAT_VERSION = 4
TIMELINE = "native"


@dataclass(frozen=True)
class SourceRecord:
    """One AVA entity: a directory of `<timestamp>.jpg` face crops and a WAV."""

    sample_id: str
    visual_path: Path
    audio_path: Path
    source_fps: float
    labels: Sequence[int]

    def __post_init__(self) -> None:
        if self.source_fps <= 0:
            raise ValueError(f"{self.sample_id}: source_fps must be positive")
