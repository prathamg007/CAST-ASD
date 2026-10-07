"""Pack writers (v3 AVA, v4 MSDWild) round-trip through `PackedASDDataset`."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile
import torch
import torchaudio

from cast_asd.data.dataloaders.packed import PackedASDDataset
from cast_asd.data.media import normalize_waveform
from cast_asd.preprocess.schema import SourceRecord
from cast_asd.preprocess.writer import ShardWriter, ShardWriterV4


@pytest.fixture(autouse=True)
def wav_loader(monkeypatch):
    """torchaudio.load needs torchcodec, which the test env may lack."""
    def load(path, normalize=True):
        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), rate

    monkeypatch.setattr(torchaudio, "load", load)


def _record(tmp_path: Path, frames: int, fps: float, name="sample") -> SourceRecord:
    faces = tmp_path / name
    faces.mkdir()
    for index in range(frames):
        cv2.imwrite(str(faces / f"{index / fps:.3f}.jpg"), np.full((90, 90), 100, np.uint8))
    audio = tmp_path / f"{name}.wav"
    samples = round((frames - 1) / fps * 16_000)
    soundfile.write(audio, np.random.default_rng(0).standard_normal(samples) * 0.1, 16_000)
    return SourceRecord(
        sample_id=name,
        visual_path=faces,
        audio_path=audio,
        source_fps=fps,
        labels=[0] * (frames // 2) + [1] * (frames - frames // 2),
    )


def _write_v3(records, output: Path, groundtruth_rows=None) -> Path:
    writer = ShardWriter(output)
    for record in records:
        writer.add(record)
    writer.close(groundtruth_rows)
    return output


@pytest.mark.parametrize("fps", [23.976, 25.0, 30.0])
def test_v3_keeps_native_frames_and_unwarped_audio(tmp_path: Path, fps: float):
    frames = 30
    dataset = PackedASDDataset(_write_v3([_record(tmp_path, frames, fps)], tmp_path / "p"))
    waveform, visual, labels, metadata = dataset[0]

    assert metadata["frames"] == frames
    assert visual.shape == (frames, 112, 112) and visual.dtype == torch.uint8
    assert labels.tolist() == [0.0] * 15 + [1.0] * 15
    # Audio keeps its true duration at the native fps; it is not frames * 640.
    assert waveform.shape == (1, round((frames - 1) / fps * 16_000))
    assert waveform.abs().max() <= 1.0


def test_groundtruth_row_mismatch_fails_at_construction(tmp_path: Path):
    output = _write_v3([_record(tmp_path, 20, 25.0)], tmp_path / "p", groundtruth_rows=21)
    with pytest.raises(ValueError, match="ground-truth"):
        PackedASDDataset(output)


def test_v4_is_normalized_on_load(tmp_path: Path):
    """A v4 pack stores source gain; the reader applies the v3 loudness contract."""
    waveform = torch.from_numpy(
        np.random.default_rng(1).standard_normal((1, 16_000)).astype(np.float32) * 0.01
    )
    writer = ShardWriterV4(tmp_path / "p")
    writer.add_arrays(
        dataset="msdwild", sample_id="v_00000_1",
        faces=np.zeros((25, 112, 112), np.uint8), waveform=waveform,
        labels=np.ones(25), source_fps=25.0, start_sec=3.0,
    )
    writer.close()
    manifest = json.loads((tmp_path / "p" / "dataset.json").read_text())
    assert (manifest["format"], manifest["version"]) == ("muvap-asd-mmap", 4)

    loaded, _, labels, metadata = PackedASDDataset(tmp_path / "p")[0]
    assert metadata["frames"] == 25 and metadata["start_sec"] == 3.0
    stored = torch.from_numpy(
        (waveform.clamp(-1, 1) * 32767.0).round().to(torch.int16).numpy()
    ).float().div(32768.0)
    assert torch.equal(loaded, normalize_waveform(stored)[0])
    assert labels.sum() == 25
