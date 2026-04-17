"""Training entry point for basic Voynich entropy model fine-tuning.

Run from the src/ directory:
    cd /workspace/src && uv run python -m fine_tune.train
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime

os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")

import lightning as L
from clearml import Task
from lightning.pytorch.callbacks import ModelCheckpoint

from voynpy.corpora import vms_unicode

from fine_tune.clearml_logger import ClearMLLogger
from fine_tune.config import FineTuneConfig
from fine_tune.dataset import VoynichEntropyDataset, folio_split, make_dataloader
from fine_tune.module import VoynichEntropyFineTune

_CLEARML_ENV_VARS = (
    "CLEARML_API_HOST",
    "CLEARML_WEB_HOST",
    "CLEARML_FILES_HOST",
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
)


def _configure_clearml_credentials() -> None:
    """Wire the ClearML server credentials from environment variables."""
    missing = [name for name in _CLEARML_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required ClearML env vars: {', '.join(missing)}"
        )
    Task.set_credentials(
        api_host=os.environ["CLEARML_API_HOST"],
        web_host=os.environ["CLEARML_WEB_HOST"],
        files_host=os.environ["CLEARML_FILES_HOST"],
        key=os.environ["CLEARML_API_ACCESS_KEY"],
        secret=os.environ["CLEARML_API_SECRET_KEY"],
    )


def _git_commit() -> str:
    """Get the current git commit hash, or 'unknown' if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def make_run_id() -> str:
    """Generate a timestamp-based run ID."""
    return datetime.now().strftime("bft-%Y%m%d-%H%M")


def main() -> None:
    run_id = make_run_id()
    config = FineTuneConfig(run_id=run_id)

    # Folio-level train/val split
    df = vms_unicode.df
    train_folios, val_folios = folio_split(df, config.split_seed)
    config.train_folios = train_folios
    config.val_folios = val_folios

    # Datasets and DataLoaders
    train_ds = VoynichEntropyDataset(df, train_folios, config)
    val_ds = VoynichEntropyDataset(df, val_folios, config)
    train_dl = make_dataloader(train_ds, shuffle=True)
    val_dl = make_dataloader(val_ds, shuffle=False)

    # Compute total steps for LR scheduler
    steps_per_epoch = len(train_ds)
    total_steps = steps_per_epoch * config.epochs

    # Create output directories
    run_dir = config.run_dir
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config at run start
    config.to_yaml(run_dir / "config.yaml")

    # ClearML experiment tracking
    _configure_clearml_credentials()
    task = Task.init(
        project_name=config.clearml_project,
        task_name=run_id,
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch_lightning": False},
        output_uri=False,
    )
    run_hparams = {
        "run_id": config.run_id,
        "split_seed": config.split_seed,
        "train_folios": config.train_folios,
        "val_folios": config.val_folios,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
        "warmup_steps": config.warmup_steps,
        "epochs": config.epochs,
        "max_seq_len": config.max_seq_len,
        "hf_repo": config.hf_repo,
        "train_chunks": len(train_ds),
        "val_chunks": len(val_ds),
        "total_steps": total_steps,
        "git_commit": _git_commit(),
    }
    task.connect(run_hparams, name="run")
    clearml_logger = ClearMLLogger(task)

    # Checkpoint callbacks
    periodic_ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch={epoch:03d}",
        every_n_epochs=config.checkpoint_every_n_epochs,
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Lightning module
    module = VoynichEntropyFineTune(config, total_steps)

    # Trainer
    trainer = L.Trainer(
        max_epochs=config.epochs,
        precision="bf16-mixed",
        gradient_clip_val=config.grad_clip,
        callbacks=[periodic_ckpt],
        logger=clearml_logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        default_root_dir=str(run_dir),
    )

    trainer.fit(module, train_dl, val_dl)

    # Final validation metrics
    val_results = trainer.validate(module, val_dl)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    with open(eval_dir / "val_metrics.json", "w") as f:
        json.dump(val_results, f, indent=2)

    task.close()


if __name__ == "__main__":
    main()
