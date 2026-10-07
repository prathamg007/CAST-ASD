"""Scene-grouped MSDWild reader and collator (`data_backend: scene_packed`)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from cast_asd.config import read_config
from cast_asd.data.dataloaders.packed import PackedASDDataset
from cast_asd.data.dataloaders.scene_packed import (
    SceneASDCollator,
    SceneASDDataset,
    scene_key,
)
from cast_asd.models.asd import AudioVisualASD
from cast_asd.preprocess.writer import ShardWriterV4

FPS = 25.0
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _pack(tmp_path: Path, layout: dict[str, int], frames: int = 24, name="packed",
          ragged: dict[str, int] | None = None) -> Path:
    """One v4 track per (scene, track); the tracks of a scene share one waveform."""
    output = tmp_path / name
    writer = ShardWriterV4(output)
    for scene, tracks in layout.items():
        rng = np.random.default_rng(abs(hash(scene)) % 2**32)
        waveform = torch.from_numpy(
            rng.standard_normal((1, round(frames / FPS * 16_000))).astype(np.float32) * 0.1
        )
        for track in range(tracks):
            n = (ragged or {}).get(f"{scene}_{track}", frames)
            writer.add_arrays(
                dataset="msdwild",
                sample_id=f"{scene}_{track}",
                faces=rng.integers(0, 255, (n, 112, 112), dtype=np.uint8),
                waveform=waveform,
                labels=[(track + index) % 2 for index in range(n)],
                source_fps=FPS,
                start_sec=1.5,
            )
    writer.close()
    return output


def _scene(batch):
    return {
        "index": batch["scene_index"],
        "slot": batch["track_slot"],
        "mask": batch["track_mask"],
        "audio_lengths": batch["audio_lengths"],
        "frame_lengths": batch["scene_frame_lengths"],
        "source_fps": batch["scene_source_fps"],
    }


def test_scene_key_splits_on_the_track_suffix():
    assert scene_key("00001_00000_2") == "00001_00000"
    assert scene_key("00001_00000_10") == "00001_00000"
    assert scene_key("solo") == "solo"


def test_scenes_group_tracks_in_sample_id_order(tmp_path: Path):
    dataset = SceneASDDataset(_pack(tmp_path, {"a_00000": 3, "b_00000": 2}))

    assert len(dataset) == 2
    assert dataset.track_counts == [3, 2]
    assert dataset.scene_frames == [24, 24]
    waveform, visual, labels, metadata = dataset[0]
    assert waveform.shape[0] == 1
    assert len(visual) == len(labels) == len(metadata) == 3
    assert [m["sample_id"] for m in metadata] == ["a_00000_0", "a_00000_1", "a_00000_2"]


def test_min_frames_drops_short_scenes(tmp_path: Path):
    _pack(tmp_path, {"a_00000": 2}, frames=24, name="p")
    long = _pack(tmp_path, {"b_00000": 2}, frames=30, name="q")
    roots = [tmp_path / "p", long]
    assert len(SceneASDDataset(roots)) == 2
    kept = SceneASDDataset(roots, min_frames=30)
    assert [scene_key(kept[0][3][0]["sample_id"])] == ["b_00000"]
    with pytest.raises(ValueError, match="discarded every scene"):
        SceneASDDataset(roots, min_frames=31)


def test_collator_shapes_and_indices(tmp_path: Path):
    dataset = SceneASDDataset(_pack(tmp_path, {"a_00000": 3, "b_00000": 2}))
    batch = SceneASDCollator()([dataset[0], dataset[1]])

    assert batch["waveform"].shape[0] == 2          # one row per scene
    assert batch["visual"].shape[0] == 5            # one row per track
    assert batch["scene_index"].tolist() == [0, 0, 0, 1, 1]
    assert batch["track_slot"].tolist() == [0, 1, 2, 0, 1]
    assert batch["track_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["scene_frame_lengths"].tolist() == [24, 24]
    assert batch["frame_lengths"].tolist() == [24] * 5
    assert batch["start_sec"] == [1.5] * 5
    for row, scene in enumerate(batch["scene_index"].tolist()):
        assert batch["sample_ids"][row].startswith("a_" if scene == 0 else "b_")


def test_quantum_pads_with_the_last_face(tmp_path: Path):
    dataset = SceneASDDataset(_pack(tmp_path, {"a_00000": 2}))
    batch = SceneASDCollator(frame_quantum=16, max_frames=64)([dataset[0]])
    assert batch["visual"].shape[1] == 32
    assert not batch["frame_mask"][:, 24:].any()
    assert torch.equal(batch["visual"][:, 31], batch["visual"][:, 23])


def test_scene_waveform_is_every_tracks_waveform(tmp_path: Path):
    pack = _pack(tmp_path, {"a_00000": 3})
    per_track = PackedASDDataset(pack)
    scene_waveform = SceneASDDataset(pack)[0][0]
    for index in range(len(per_track)):
        assert torch.equal(per_track.load_waveform(index), scene_waveform)


def test_scene_rejects_tracks_that_cannot_share_a_time_axis(tmp_path: Path):
    pack = _pack(tmp_path, {"a_00000": 2}, ragged={"a_00000_1": 30})
    with pytest.raises(ValueError, match="cannot share a time axis"):
        SceneASDDataset(pack)


def _model() -> AudioVisualASD:
    cfg = read_config(CONFIGS / "msdwild" / "talknet.yaml")["asd"]
    return AudioVisualASD(cfg).eval()


def _run(model, batch, scene):
    with torch.no_grad():
        return model(
            batch["waveform"],
            batch["visual"].float(),
            batch["audio_lengths"],
            batch["frame_lengths"],
            batch["frame_mask"],
            source_fps=batch["source_fps"],
            scene=scene,
        )[0]


def test_shared_audio_pass_equals_encoding_every_track(tmp_path: Path):
    """Encoding one waveform per scene and expanding it == encoding it per track."""
    torch.manual_seed(0)
    dataset = SceneASDDataset(_pack(tmp_path, {"a_00000": 3, "b_00000": 2}))
    batch = SceneASDCollator()([dataset[0], dataset[1]])
    model = _model()
    grouped = _run(model, batch, _scene(batch))

    flat = dict(batch)
    flat["waveform"] = batch["waveform"][batch["scene_index"]]
    flat["audio_lengths"] = batch["audio_lengths"][batch["scene_index"]]
    assert torch.allclose(grouped, _run(model, flat, None), atol=1e-5)


def test_widening_the_batch_does_not_move_real_frames(tmp_path: Path):
    """Causal model: frames appended after a clip cannot reach back into it."""
    torch.manual_seed(0)
    dataset = SceneASDDataset(_pack(tmp_path, {"a_00000": 3, "b_00000": 2}))
    model = _model()

    def run(quantum):
        batch = SceneASDCollator(frame_quantum=quantum, max_frames=64)(
            [dataset[0], dataset[1]]
        )
        return batch, _run(model, batch, _scene(batch))

    tight, logits_tight = run(0)
    _, logits_wide = run(16)
    real = tight["frame_mask"]
    # Not bit-exact: the attention softmax reduces over a longer key axis.
    assert (logits_tight[real] - logits_wide[:, :24][real]).abs().max() < 1e-5


def test_the_per_face_model_does_not_couple_tracks(tmp_path: Path):
    torch.manual_seed(0)
    dataset = SceneASDDataset(_pack(tmp_path, {"a_00000": 3, "b_00000": 2}))
    batch = SceneASDCollator()([dataset[0], dataset[1]])
    model = _model()
    reference = _run(model, batch, _scene(batch))

    scrambled = dict(batch)
    scrambled["visual"] = batch["visual"].clone()
    scrambled["visual"][0] = torch.randint_like(batch["visual"][0], 0, 255)
    perturbed = _run(model, scrambled, _scene(scrambled))

    assert not torch.equal(perturbed[0], reference[0])
    assert torch.equal(perturbed[1:], reference[1:])
