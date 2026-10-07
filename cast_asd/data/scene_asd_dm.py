"""Lightning data module for scene-grouped MSDWild packs (`data_backend: scene_packed`)."""

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from cast_asd.data.dataloaders.scene_packed import SceneASDCollator, SceneASDDataset
from cast_asd.data.samplers import ScenePaddedBatchSampler


def named_val_roots(packed: dict) -> dict[str, list]:
    """`packed_asd.val` as `{name: roots}`; a bare list is the single split "val".

    Named splits (MSDWild's `few`, `many`) are scored and reported separately.
    """
    val = packed["val"]
    if isinstance(val, dict):
        return {name: roots for name, roots in val.items()}
    return {"val": val}


class SceneASDDataModule(LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        asd = cfg["asd"]
        self.num_workers = asd["num_workers"]
        self.frames_per_batch = asd.get("frames_per_batch", 2500)
        self.val_frames_per_batch = asd.get("val_frames_per_batch", self.frames_per_batch)
        self.max_batch_size = asd.get("max_batch_size", 32)
        self.augment = asd.get("augment", True)
        self.prefetch_factor = asd.get("prefetch_factor", 4)
        self.frame_quantum = asd.get("frame_quantum", 0)
        # Train only; see `SceneASDDataset`.
        self.train_min_frames = asd.get("train_min_frames", 0)
        self.val_names = list(named_val_roots(cfg["packed_asd"]))

    def setup(self, stage: str | None = None):
        packed = self.cfg["packed_asd"]
        if stage in ("fit", None):
            self.train_dataset = SceneASDDataset(
                packed["train"], augment=self.augment, min_frames=self.train_min_frames
            )
        if stage in ("fit", "validate", "test", None):
            self.val_datasets = [
                SceneASDDataset(roots, augment=False)
                for roots in named_val_roots(packed).values()
            ]

    def _loader(self, dataset, shuffle: bool, frames_per_batch: int, audio_augment: bool):
        frames = dataset.scene_frames
        sampler = ScenePaddedBatchSampler(
            frames,
            dataset.track_counts,
            frames_per_batch=frames_per_batch,
            max_batch_size=self.max_batch_size,
            shuffle=shuffle,
            frame_quantum=self.frame_quantum,
        )
        workers = self.num_workers
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=workers,
            collate_fn=SceneASDCollator(
                audio_augment=audio_augment,
                frame_quantum=self.frame_quantum,
                max_frames=max(frames),
            ),
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=self.prefetch_factor if workers > 0 else None,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, True, self.frames_per_batch, self.augment)

    def val_dataloader(self):
        # Unshuffled and unaugmented, so every track is scored on its real clip.
        loaders = [
            self._loader(dataset, False, self.val_frames_per_batch, False)
            for dataset in self.val_datasets
        ]
        return loaders[0] if len(loaders) == 1 else loaders
