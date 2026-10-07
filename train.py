"""Train or evaluate a CAST-ASD config on AVA-ActiveSpeaker or MSDWild.

    python train.py --config configs/msdwild/cast.yaml [--audio mimi|vggish|talknet] [--seed N]
    python train.py --config configs/ava/cast.yaml --test --checkpoint best.ckpt

The dataset follows `asd.data_backend`: `scene_ava` scores AVA with the official
ActivityNet evaluator (`val/mAP_official`); `scene_packed` reports every MSDWild
split in `packed_asd.val` as `val/<split>/*`.
"""

import argparse
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.utilities import rank_zero_info

from cast_asd.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="YAML under configs/")
    parser.add_argument(
        "--audio", choices=("mimi", "vggish", "talknet"), help="override asd.audio.name"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="validate --checkpoint only")
    parser.add_argument("--checkpoint")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="CAST-ASD")
    parser.add_argument(
        "--run-name",
        help="output directory under checkpoint_dir (default: <config>-seed<N>[-test])",
    )
    parser.add_argument(
        "--devices", type=int, default=1,
        help="GPUs to use; the batch samplers are not rank-aware, so keep 1",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.test and not args.checkpoint:
        raise SystemExit("--test needs --checkpoint")

    cfg = load_config(args.config, "asd")
    asd = cfg["asd"]
    if args.audio is not None:
        asd["audio"]["name"] = args.audio
    seed_everything(args.seed, workers=True)

    logger = None
    run_name = args.run_name or (
        f"{Path(args.config).stem}-seed{args.seed}" + ("-test" if args.test else "")
    )
    run_id = run_name
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger

        # Uploads the `best` and `last` checkpoints at the end of training.
        logger = WandbLogger(project=args.wandb_project, name=run_name, log_model=True)
        if not args.run_name:
            run_id = logger.experiment.id
    run_dir = Path(cfg["checkpoint_dir"]) / run_id
    evaluation = cfg.setdefault("evaluation", {})

    backend = asd["data_backend"]
    if backend == "scene_ava":
        from cast_asd.data.scene_ava_dm import SceneAVADataModule
        from cast_asd.tasks.asd_task import ASDTask

        # Per run: the official scorer re-reads this file, so a shared path
        # would make concurrent runs score each other's predictions.
        evaluation["ava_predictions"] = str(run_dir / "predictions.csv")
        data, model = SceneAVADataModule(cfg), ASDTask(cfg)
        monitor = asd.get("monitor", "val/mAP_official")
    elif backend == "scene_packed":
        from cast_asd.data.scene_asd_dm import SceneASDDataModule, named_val_roots
        from cast_asd.tasks.scene_asd_task import SceneASDTask

        # The task builds one accumulator per split, before the datamodule runs.
        evaluation["val_splits"] = list(named_val_roots(cfg["packed_asd"]))
        if evaluation.get("dump_scores", True):
            evaluation["score_dump"] = str(run_dir / "scores-{split}.npz")
        data, model = SceneASDDataModule(cfg), SceneASDTask(cfg)
        monitor = asd.get("monitor", f"val/{evaluation['val_splits'][0]}/ap")
        # MSDWild scenes are mostly the same length, so cuDNN autotuning pays off.
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError(f"unknown asd.data_backend {backend!r}; choose scene_ava or scene_packed")

    # A cosine schedule is defined over the whole run, so it runs to completion
    # (the final epoch is reported); step decay uses early stopping.
    early_stopping = []
    if asd.get("scheduler", "cosine_warmup") == "cosine_warmup":
        rank_zero_info(f"cosine_warmup: early stopping disabled, {asd['max_epochs']} epochs")
    else:
        early_stopping.append(
            EarlyStopping(
                monitor=monitor, mode="max", patience=asd.get("early_stopping_patience", 3)
            )
        )

    torch.set_float32_matmul_precision("high")
    precision = asd.get("precision", "16-mixed")
    if precision == "16-mixed" and not torch.cuda.is_available():
        # fp16 autocast is GPU-only; on CPU fall back to full precision.
        rank_zero_info("no CUDA device: precision 16-mixed -> 32-true")
        precision = "32-true"
    trainer = Trainer(
        max_epochs=asd["max_epochs"],
        precision=precision,
        use_distributed_sampler=False,
        logger=logger,
        # Per run, so the fallback CSVLogger's version_N folders cannot collide.
        default_root_dir=run_dir,
        callbacks=[
            ModelCheckpoint(
                monitor=monitor,
                mode="max",
                save_top_k=1,
                dirpath=run_dir,
                filename="best",
                save_last=True,
            ),
            *early_stopping,
            RichProgressBar(),
            LearningRateMonitor("step"),
        ],
        log_every_n_steps=100,
        accelerator="auto",
        devices=args.devices,
    )
    if args.test:
        trainer.validate(model, datamodule=data, ckpt_path=args.checkpoint)
    else:
        trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
