"""`preprocess.msdwild` end to end on a tiny synthetic video."""

from pathlib import Path

import cv2
import numpy as np
import soundfile
import torch
import torchaudio

from cast_asd.data.dataloaders.packed import PackedASDDataset
from cast_asd.data.dataloaders.scene_packed import SceneASDCollator, SceneASDDataset
from cast_asd.preprocess import msdwild

FPS = 25
FRAMES = 30          # 1.2 s: one full 1 s window plus a 0.2 s tail


def _fake_msdwild(root: Path) -> Path:
    videos = root / "videos"
    videos.mkdir(parents=True)
    writer = cv2.VideoWriter(str(videos / "00001.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (160, 120))
    for index in range(FRAMES):
        frame = np.full((120, 160, 3), 40, np.uint8)
        frame[20:60, 20:60] = 200                          # speaker 0's face
        frame[60:100, 100:140] = 80 + 4 * index            # speaker 1's face
        writer.write(frame)
    writer.release()
    # The mp4 has no audio track; the test's torchaudio.load reads this instead.
    soundfile.write(videos / "00001.wav",
                    np.random.default_rng(0).standard_normal(FRAMES * 640) * 0.05, 16_000)

    # frame, _, speaker, x1, y1, x2, y2, _ ; speaker 2 appears too briefly to keep.
    with (videos / "00001.csv").open("w") as handle:
        for frame in range(FRAMES):
            handle.write(f"{frame},0,0,20,20,60,60,1\n")
            handle.write(f"{frame},0,1,100,60,140,100,1\n")
            if frame < 3:
                handle.write(f"{frame},0,2,0,0,10,10,1\n")
    (root / "all.rttm").write_text(
        "SPEAKER 00001 1 0.00 0.50 <NA> <NA> 0 <NA> <NA>\n"
        "SPEAKER 00001 1 0.40 0.80 <NA> <NA> 1 <NA> <NA>\n"
        "SPEAKER 00002 1 0.00 1.00 <NA> <NA> 0 <NA> <NA>\n"
    )
    (root / "few.val.rttm").write_text("SPEAKER 00001 1 0.00 0.50 <NA> <NA> 0 <NA> <NA>\n")
    return videos


def test_prep_writes_a_pack_the_scene_reader_reads(tmp_path: Path, monkeypatch):
    def load(path, normalize=True):
        data, rate = soundfile.read(str(Path(path).with_suffix(".wav")),
                                    dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), rate

    monkeypatch.setattr(torchaudio, "load", load)
    videos = _fake_msdwild(tmp_path)
    msdwild.main(["--input", str(videos), "--split", "few.val",
                  "--clip-seconds", "1.0", "--min-clip-seconds", "0.2"])
    output = tmp_path / "packed" / "few.val"

    tracks = PackedASDDataset(output)
    ids = [entry.metadata["sample_id"] for entry in tracks.entries]
    assert ids == ["00001_00000_0", "00001_00000_1", "00001_00001_0", "00001_00001_1"]
    meta = tracks.entries[1].metadata
    assert (meta["frames"], meta["source_fps"], meta["start_sec"]) == (25, 25.0, 0.0)
    assert tracks.entries[2].metadata["frames"] == 5
    assert tracks.entries[2].metadata["start_sec"] == 1.0

    waveform, visual, labels, _ = tracks[1]
    assert waveform.shape == (1, 16_000)
    assert visual.shape == (25, 112, 112)
    # Speaker 1 speaks over [0.4, 1.2): frame centres from 0.42 s on.
    assert labels.tolist() == [0.0] * 10 + [1.0] * 15
    assert tracks[0][2].tolist() == [1.0] * 12 + [0.0] * 13
    # The face crop is the box, not the background.
    assert abs(float(tracks[0][1].float().mean()) - 200) < 10

    scenes = SceneASDDataset(output)
    assert scenes.track_counts == [2, 2]
    batch = SceneASDCollator()([scenes[0], scenes[1]])
    assert batch["visual"].shape == (4, 25, 112, 112)
    assert batch["start_sec"] == [0.0, 0.0, 1.0, 1.0]
