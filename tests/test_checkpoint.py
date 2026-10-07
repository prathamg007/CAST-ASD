"""Checkpoints drop the frozen audio encoder on save and restore it on load."""

import torch
from lightning.pytorch import seed_everything

from cast_asd.config import load_config
from cast_asd.tasks.scene_asd_task import SceneASDTask

PREFIX = "model.audio_frontend.encoder."


def _task(seed):
    cfg = load_config("configs/msdwild/cast.yaml", "asd")
    cfg["evaluation"] = {"val_splits": ["few"]}
    seed_everything(seed)
    return SceneASDTask(cfg)


def test_frozen_encoder_is_stripped_and_restored():
    task = _task(0)
    checkpoint = {"state_dict": task.state_dict()}
    assert any(key.startswith(PREFIX) for key in checkpoint["state_dict"])
    task.on_save_checkpoint(checkpoint)
    assert not any(key.startswith(PREFIX) for key in checkpoint["state_dict"])

    other = _task(1)
    other.on_load_checkpoint(checkpoint)
    other.load_state_dict(checkpoint["state_dict"], strict=True)
    for key, value in task.state_dict().items():
        assert torch.equal(value, other.state_dict()[key]), key
