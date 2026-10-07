"""Lightning task for AVA-ActiveSpeaker, scored with the official ActivityNet metric."""

import csv
from collections import defaultdict

import torch
import torch.distributed as dist
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_warn

from cast_asd.models.asd import AudioVisualASD
from cast_asd.models.modules.audio import build_audio_encoder
from cast_asd.models.modules.masking import masked_bce, masked_cross_entropy
from cast_asd.tasks.ava_evaluation import export_ava_predictions, official_ava_map, timestamp_key
from cast_asd.tasks.optim import get_cosine_warmup_scheduler, get_step_decay_scheduler


class ASDTask(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        arch = (cfg["asd"].get("arch") or "talknet").lower()
        if arch == "lightasd":
            from cast_asd.models.lightasd.adapter import LightASDModel

            self.model = LightASDModel(cfg["asd"])
        elif arch == "talknet":
            self.model = AudioVisualASD(cfg["asd"])
        else:
            raise ValueError(f"unknown asd.arch {arch!r}; choose talknet or lightasd")
        self.head_type = cfg["asd"].get("head", "binary")
        self.loss_cfg = cfg["asd"].get("loss", {})
        self.validation_entities: list[tuple[str, torch.Tensor]] = []
        evaluation = cfg.get("evaluation", {})
        self.ava_groundtruth = evaluation.get("ava_groundtruth")
        self.ava_predictions = evaluation.get("ava_predictions")
        self._entity_timestamps: dict[str, list[int]] | None = None

    def forward(
        self,
        waveform,
        visual,
        audio_lengths,
        frame_lengths,
        key_padding_mask=None,
        source_fps=None,
        scene=None,
    ):
        return self.model(
            waveform,
            visual,
            audio_lengths,
            frame_lengths,
            key_padding_mask,
            return_embeddings=True,
            source_fps=source_fps,
            scene=scene,
        )

    @staticmethod
    def scene_of(batch):
        """The scene grouping both scene collates add to a batch."""
        return {
            "index": batch["scene_index"],
            "slot": batch["track_slot"],
            "mask": batch["track_mask"],
            # Per scene, not per track: `waveform` has one row per scene.
            "audio_lengths": batch["audio_lengths"],
            "frame_lengths": batch["scene_frame_lengths"],
            "source_fps": batch["scene_source_fps"],
        }

    def probabilities(self, logits):
        return logits.softmax(dim=-1)[..., 1] if self.head_type == "talknet" else logits.sigmoid()

    def _shared_step(self, batch):
        mask = batch["frame_mask"]
        labels = batch["labels"]
        av, audio, visual, _, _, _ = self(
            batch["waveform"],
            batch["visual"].float(),
            batch["audio_lengths"],
            batch["frame_lengths"],
            mask,
            batch["source_fps"],
            self.scene_of(batch),
        )

        loss_fn = masked_cross_entropy if self.head_type == "talknet" else masked_bce
        av_loss = loss_fn(av, labels, mask)
        audio_loss = loss_fn(audio, labels, mask)
        visual_loss = loss_fn(visual, labels, mask)
        # TalkNet's weighting: lossAV + 0.4 * lossA + 0.4 * lossV.
        loss = (
            av_loss
            + float(self.loss_cfg.get("audio_weight", 0.4)) * audio_loss
            + float(self.loss_cfg.get("visual_weight", 0.4)) * visual_loss
        )
        return av, mask, {
            "loss": loss,
            "av_loss": av_loss,
            "audio_loss": audio_loss,
            "visual_loss": visual_loss,
        }

    def training_step(self, batch, batch_idx):
        _, _, losses = self._shared_step(batch)
        # A scene batch has one waveform row per scene but one label row per
        # track; Lightning would infer the former, so pass the track count.
        tracks = int(batch["labels"].shape[0])
        for name, value in losses.items():
            self.log(
                f"train/{name}",
                value,
                prog_bar=name == "loss",
                sync_dist=True,
                batch_size=tracks,
            )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        av, mask, losses = self._shared_step(batch)
        probabilities = self.probabilities(av)
        # `sample_ids` lists only the target rows, which the collate puts first.
        for index, sample_id in enumerate(batch["sample_ids"]):
            valid = mask[index]
            self.validation_entities.append(
                (sample_id, probabilities[index][valid].detach().float().cpu())
            )
        for name, value in losses.items():
            self.log(f"val/{name}", value, prog_bar=name == "loss", sync_dist=True)

    def _load_entity_timestamps(self):
        """Map each entity to its ground-truth timestamps, in file order."""
        timestamps = defaultdict(list)
        with open(self.ava_groundtruth, newline="") as handle:
            for row in csv.DictReader(handle):
                timestamps[row["entity_id"]].append(timestamp_key(row["frame_timestamp"]))
        return timestamps

    def on_validation_epoch_end(self):
        entities = self.validation_entities
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, entities)
            entities = [item for shard in gathered for item in shard]

        if (
            self.ava_groundtruth
            and self.ava_predictions
            and self.trainer.is_global_zero
            and not self.trainer.sanity_checking
        ):
            if self._entity_timestamps is None:
                self._entity_timestamps = self._load_entity_timestamps()
            scores_by_key = {}
            for sample_id, scores in dict(entities).items():
                stamps = self._entity_timestamps.get(sample_id)
                if stamps is None or len(stamps) != len(scores):
                    raise ValueError(
                        f"{sample_id}: {len(scores)} scores for "
                        f"{0 if stamps is None else len(stamps)} groundtruth rows"
                    )
                for stamp, score in zip(stamps, scores.tolist()):
                    scores_by_key[(sample_id, stamp)] = score

            result = export_ava_predictions(
                self.ava_groundtruth, self.ava_predictions, scores_by_key
            )
            self.log("val/mAP", result.average_precision, prog_bar=True, rank_zero_only=True)
            official = official_ava_map(self.ava_groundtruth, self.ava_predictions)
            self.log("val/mAP_official", official, prog_bar=True, rank_zero_only=True)
            if abs(official - result.average_precision) > 0.05:
                rank_zero_warn(
                    f"mAP disagreement: official {official:.4f} vs "
                    f"in-repo {result.average_precision:.4f}"
                )
        self.validation_entities.clear()

    def _reproducible_encoder_keys(self) -> set[str]:
        """Frozen audio-encoder keys whose tensors a fresh build reproduces.

        Equality with a rebuilt encoder is the test, not `requires_grad`: a
        frozen tensor that cannot be rebuilt must stay in the checkpoint.
        """
        # Light-ASD has no pretrained audio encoder.
        frontend = getattr(self.model, "audio_frontend", None)
        if frontend is None or not frontend.encoder.frozen:
            return set()
        try:
            # Keep the encoder's random init off the training RNG stream.
            with torch.random.fork_rng(devices=[]):
                rebuilt = build_audio_encoder(dict(self.hparams["asd"]["audio"])).state_dict()
        except Exception:  # pragma: no cover - never lose a checkpoint over this
            rank_zero_warn("could not rebuild the audio encoder; keeping its tensors")
            return set()

        prefix = "model.audio_frontend.encoder."
        keys = set()
        for name, current in frontend.encoder.state_dict().items():
            reference = rebuilt.get(name)
            if reference is not None and torch.equal(
                reference.to(current.dtype).cpu(), current.cpu()
            ):
                keys.add(prefix + name)
        return keys

    def on_fit_start(self):
        # Computed once, before DataLoader workers start: the encoder is frozen,
        # and rebuilding it at every save mid-run is slow and can crash workers.
        self._strippable_encoder_keys = self._reproducible_encoder_keys()

    def on_save_checkpoint(self, checkpoint):
        """Drop frozen audio-encoder tensors; `on_load_checkpoint` restores them."""
        keys = getattr(self, "_strippable_encoder_keys", None)
        if keys is None:  # saved outside `fit`
            keys = self._reproducible_encoder_keys()
        state = checkpoint["state_dict"]
        for key in keys:
            state.pop(key, None)

    def on_load_checkpoint(self, checkpoint):
        """Refill only the frozen-encoder tensors `on_save_checkpoint` dropped.

        Any other missing key means the config does not match the checkpoint,
        and the strict load should fail.
        """
        prefix = "model.audio_frontend.encoder."
        state = checkpoint["state_dict"]
        for key, value in self.state_dict().items():
            if key.startswith(prefix) and key not in state:
                state[key] = value

    def configure_optimizers(self):
        cfg = self.hparams["asd"]
        # Frozen parameters stay out of AdamW, which would allocate moments for them.
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
        )
        scheduler_type = cfg.get("scheduler", "cosine_warmup")
        if scheduler_type == "step_decay":
            scheduler = get_step_decay_scheduler(optimizer, float(cfg.get("lr_decay", 0.95)))
            interval = "epoch"
        elif scheduler_type == "cosine_warmup":
            scheduler = get_cosine_warmup_scheduler(
                optimizer, self.trainer.estimated_stepping_batches, float(cfg["warmup_ratio"])
            )
            interval = "step"
        else:
            raise ValueError(f"unknown asd.scheduler {scheduler_type!r}")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": interval}}
