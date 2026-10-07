"""MSDWild task: several named validation splits scored by `SceneDiagnostics`.

Labels live in the packs, so AP is computed in-repo from the scored frames; the
splits (`few`, `many`) are reported side by side as `val/<split>/<metric>`.
"""

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.utilities import rank_zero_info

from cast_asd.tasks.asd_task import ASDTask
from cast_asd.tasks.diarization_metrics import load_rttm_dir
from cast_asd.tasks.scene_metrics import SceneDiagnostics

#: Shown in the progress bar for the first split.
_PROGRESS = ("ap", "ap_competing", "fa_competing@90r")


class SceneASDTask(ASDTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        evaluation = cfg.get("evaluation", {})
        self.val_names: list[str] = list(evaluation.get("val_splits") or ["val"])
        self.score_dump = evaluation.get("score_dump")
        threshold = float(cfg["asd"].get("decision_threshold", 0.5))
        recall_target = float(cfg["asd"].get("recall_target", 0.90))
        rttm_dir = evaluation.get("rttm_dir")
        self.diagnostics = []
        for name in self.val_names:
            # `{split}` in the path selects a per-split reference; MSDWild video
            # ids are unique across splits, so one shared directory also works.
            rttm = load_rttm_dir(str(rttm_dir).format(split=name)) if rttm_dir else None
            self.diagnostics.append(
                SceneDiagnostics(threshold, recall_target=recall_target, rttm=rttm)
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        av, mask, losses = self._shared_step(batch)
        self.diagnostics[dataloader_idx].update(
            batch["sample_ids"],
            self.probabilities(av),
            batch["labels"],
            mask,
            batch["scene_index"],
            int(batch["track_mask"].shape[0]),
            start_sec=batch.get("start_sec"),
            source_fps=batch.get("source_fps"),
        )
        name = self.val_names[dataloader_idx]
        tracks = int(batch["labels"].shape[0])
        for key, value in losses.items():
            self.log(
                f"val/{name}/{key}",
                value,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=tracks,
            )

    def on_validation_epoch_end(self):
        for index, name in enumerate(self.val_names):
            diagnostics = self.diagnostics[index]
            if dist.is_available() and dist.is_initialized():
                # Validation is unsharded, so ranks hold the same rows; the
                # sample_id-keyed merge is idempotent.
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, diagnostics.state())
                for shard in gathered:
                    diagnostics.merge(shard)

            metrics = diagnostics.compute()
            for key, value in metrics.items():
                self.log(
                    f"val/{name}/{key}",
                    torch.tensor(float(value)),
                    prog_bar=index == 0 and key in _PROGRESS,
                    rank_zero_only=True,
                    add_dataloader_idx=False,
                )

            if metrics and self.trainer.is_global_zero and not self.trainer.sanity_checking:
                nan = float("nan")
                rank_zero_info(
                    f"[{name}] mAP {metrics['ap']:.2f} | mAP_c {metrics['ap_competing']:.2f} | "
                    f"FA_c@90R {100 * metrics.get('fa_competing@90r', nan):.1f}% | "
                    f"DER* {metrics.get('oracle_der', nan):.2f}"
                )
                self._dump_scores(name, diagnostics)

            diagnostics.reset()

    def _dump_scores(self, name: str, diagnostics: SceneDiagnostics) -> None:
        """Per-frame scores of one split, for offline analysis (`score_dump` path)."""
        if not self.score_dump:
            return
        path = Path(self.score_dump.format(split=name))
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(diagnostics.rows)
        scores, labels, others = diagnostics.flat()
        np.savez_compressed(
            path,
            sample_ids=np.array(keys),
            frames=np.array([len(diagnostics.rows[k][0]) for k in keys]),
            scores=scores.numpy().astype("float32"),
            labels=labels.numpy().astype("uint8"),
            others_speaking=others.numpy().astype("uint8"),
        )
