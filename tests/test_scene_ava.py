"""Scene-grouped AVA (`data_backend: scene_ava`) on a synthetic pack.

The failure modes asserted here run silently otherwise: grouping faces across
scenes, and a scored-frame count that does not match the ground-truth rows.
"""

import csv
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from cast_asd.config import read_config
from cast_asd.data.dataloaders.scene_ava import (
    SceneAVADataset,
    read_timelines,
    scene_of,
    timestamp_centiseconds,
)
from cast_asd.data.dataloaders.scene_packed import scene_key
from cast_asd.data.scene_ava_dm import EqualShapeBatches, SceneAVADataModule, collate_scenes
from cast_asd.preprocess.writer import ShardWriter

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

#: entity -> (fps, first frame, frame count) on its scene's frame grid.
ENTITIES = {
    "vidA_0900_0960:1": (25.0, 0, 20),
    "vidA_0900_0960:2": (25.0, 5, 20),    # overlaps :1 on 15 of its 20 frames
    "vidA_0900_0960:3": (25.0, 18, 2),    # too short to be :1's context
    "vidB_0900_0960:1": (30.0, 0, 20),
    "vidC_0900_0960:1": (25.0, 0, 20),
}


@pytest.fixture(scope="module")
def ava_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ava")
    (root / "csv").mkdir()
    rng = np.random.default_rng(0)
    writer = ShardWriter(root / "packed" / "val")
    rows = []
    for entity, (fps, first, count) in ENTITIES.items():
        for frame in range(first, first + count):
            rows.append([entity.split("_")[0], f"{900 + frame / fps:.2f}",
                         0, 0, 1, 1, "SPEAKING_AUDIBLE", entity])
        audio = (rng.standard_normal(round((count - 1) / fps * 16_000)) * 3000)
        writer.add_sample({
            "faces": rng.integers(0, 255, (count, 112, 112), dtype=np.uint8),
            "audio": audio.astype(np.int16),
            "labels": (np.arange(count) % 2).astype(np.uint8),
            "metadata": {
                "dataset": "ava", "sample_id": entity, "source_fps": fps,
                "start_sec": 0.0, "frames": count,
                "audio_samples": len(audio),
            },
        })
    writer.close(groundtruth_rows=len(rows))
    with (root / "csv" / "val_orig.csv").open("w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["video_id", "frame_timestamp", "entity_box_x1", "entity_box_y1",
                      "entity_box_x2", "entity_box_y2", "label", "entity_id"])
        out.writerows(rows)
    return root


@pytest.fixture(scope="module")
def dataset(ava_root):
    return SceneAVADataset(ava_root, "val", max_tracks=3)


def _item(dataset, entity):
    return dataset[dataset.targets.index(entity)]


def test_ava_ids_need_their_own_scene_key():
    entity = "HV0H6oc4Kvs_0960_1020:1"
    assert scene_of(entity) == "HV0H6oc4Kvs_0960_1020"
    assert scene_key(entity) == "HV0H6oc4Kvs_0960"       # MSDWild's key: wrong here


def test_timestamps_are_integers():
    assert timestamp_centiseconds("960.03") == 96003
    assert timestamp_centiseconds("1080") == 108000


def test_scenes_never_mix_segments(ava_root):
    _, ts_faces = read_timelines(ava_root, "val")
    for (scene, _), faces in ts_faces.items():
        assert all(scene_of(face) == scene for face in faces)


def test_context_is_aligned_scene_mates_with_enough_overlap(dataset):
    item = _item(dataset, "vidA_0900_0960:1")
    assert item["track_mask"].shape[0] == 2                    # :3 overlaps too little
    assert item["frame_mask"][0].all()
    assert item["frame_mask"][1].tolist() == [False] * 5 + [True] * 15
    mate = dataset.packed.load_visual(dataset.row_of["vidA_0900_0960:2"])
    assert torch.equal(item["visual"][1, 5:], mate[:15])

    short = _item(dataset, "vidA_0900_0960:3")
    assert short["track_mask"].shape[0] == 3


def test_scored_frames_match_groundtruth_rows(dataset):
    for index, entity in enumerate(dataset.targets):
        batch = collate_scenes([dataset[index]], score_targets_only=True)
        assert int(batch["frame_mask"][0].sum()) == ENTITIES[entity][2]


def test_targets_first_and_only_targets_scored_in_validation(dataset):
    items = [dataset[i] for i in range(3)]
    batch = collate_scenes(items, score_targets_only=True)
    n = len(items)
    assert batch["sample_ids"] == [i["entity_id"] for i in items]
    assert batch["track_slot"][:n].tolist() == [0] * n
    scored = batch["frame_mask"].any(1)
    assert scored[:n].all() and not scored[n:].any()
    train = collate_scenes(items, score_targets_only=False)
    assert train["frame_mask"].any(1)[n:].all()


def test_native_fps_and_per_scene_audio_reach_the_batch(dataset):
    items = [_item(dataset, "vidA_0900_0960:1"), _item(dataset, "vidB_0900_0960:1")]
    batch = collate_scenes(items, score_targets_only=True)
    assert batch["scene_source_fps"].tolist() == [25.0, 30.0]
    assert batch["source_fps"].shape[0] == batch["visual"].shape[0]
    assert batch["waveform"].shape[0] == batch["audio_lengths"].shape[0] == 2
    assert batch["audio_lengths"].tolist() == [round(19 / 25 * 16_000), round(19 / 30 * 16_000)]


def test_audio_augmentation_is_one_draw_per_scene(dataset):
    items = [dataset[i] for i in range(4)]
    plain = collate_scenes(items, score_targets_only=False)
    mixed = collate_scenes(items, score_targets_only=False, audio_augment=True,
                           rng=random.Random(0))
    assert mixed["waveform"].shape == plain["waveform"].shape
    assert not torch.equal(mixed["waveform"], plain["waveform"])


def test_visual_augmentation_reaches_the_faces(ava_root, dataset):
    random.seed(0)
    augmented = SceneAVADataset(ava_root, "val", max_tracks=3, augment=True)
    changed = sum(
        not torch.equal(dataset[i]["visual"][0], augmented[i]["visual"][0])
        for i in range(len(dataset))
        for _ in range(3)
    )
    assert changed


def test_equal_shape_batches_cover_every_entity_once_without_padding(dataset):
    sampler = EqualShapeBatches(dataset, 8)
    assert sorted(i for b in sampler for i in b) == list(range(len(dataset)))
    for batch in sampler:
        assert len({(int(dataset.frames[i]), float(dataset.fps[i])) for i in batch}) == 1
    # vidA:1, vidA:2 and vidC:1 share (20 frames, 25 fps).
    assert max(len(b) for b in sampler) == 3


def test_datamodule_reads_augment(ava_root):
    cfg = {"ava_asd": {"root": str(ava_root)}, "asd": {"augment": True, "num_workers": 0}}
    assert SceneAVADataModule(cfg).augment is True


def test_batched_validation_reproduces_one_at_a_time(dataset):
    from cast_asd.tasks.asd_task import ASDTask

    batch_indices = max(EqualShapeBatches(dataset, 8), key=len)
    items = [dataset[i] for i in batch_indices]
    torch.manual_seed(0)
    task = ASDTask(read_config(CONFIGS / "ava" / "talknet.yaml")).eval()

    with torch.no_grad():
        for item in items:
            task.validation_step(collate_scenes([item], score_targets_only=True), 0)
        solo = dict(task.validation_entities)
        task.validation_entities = []
        task.validation_step(collate_scenes(items, score_targets_only=True), 0)
        grouped = dict(task.validation_entities)

    assert set(solo) == set(grouped)
    for sid in solo:
        # fp32 reduction order only; misalignment from padding is ~1e-3.
        assert (solo[sid] - grouped[sid]).abs().max() < 1e-6
