"""Training entry point for replay-based Voynich entropy fine-tuning.

Run from the src/ directory:
    cd /workspace/src && uv run python -m fine_tune.replay_train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")

import lightning as L
from clearml import Task
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from voynpy.corpora import vms_unicode

from fine_tune.clearml_logger import ClearMLLogger
from fine_tune.dataset import (VoynichEntropyDataset, folio_split,
                               make_dataloader)
from fine_tune.replay_config import ReplayFineTuneConfig
from fine_tune.replay_dataset import (DCLMReplayDataset,
                                      load_or_fetch_replay_pool)
from fine_tune.replay_loader import (BatchScheduler, MixedDataLoader,
                                     MixedLoaderEpochCallback)
from fine_tune.replay_module import ReplayEntropyFineTune

_CLEARML_ENV_VARS = (
    "CLEARML_API_HOST",
    "CLEARML_WEB_HOST",
    "CLEARML_FILES_HOST",
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
)


def _configure_clearml_credentials() -> None:
    missing = [name for name in _CLEARML_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required ClearML env vars: {', '.join(missing)}")
    Task.set_credentials(
        api_host=os.environ["CLEARML_API_HOST"],
        web_host=os.environ["CLEARML_WEB_HOST"],
        files_host=os.environ["CLEARML_FILES_HOST"],
        key=os.environ["CLEARML_API_ACCESS_KEY"],
        secret=os.environ["CLEARML_API_SECRET_KEY"],
    )


def _git_commit() -> str:
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


def make_run_id(ratio: float) -> str:
    """Generate a timestamp-based run ID including the replay ratio."""
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    return f"rft-{ts}-r{ratio}"


def _replay_pool_hash(docs: list[str]) -> str:
    h = hashlib.sha256()
    for doc in docs:
        h.update(doc.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def _replay_cache_dir(config: ReplayFineTuneConfig) -> Path:
    return Path(config.experiments_dir) / "_replay_cache"


def parse_args() -> ReplayFineTuneConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--replay-pool-size", type=int, default=1000)
    parser.add_argument("--replay-val-pool-size", type=int, default=100)
    parser.add_argument("--replay-seed", type=int, default=42)
    parser.add_argument("--replay-schedule-seed", type=int, default=43)
    parser.add_argument("--replay-source", type=str, default="DCLM")
    parser.add_argument("--warmup-fraction", type=float, default=0.04)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    run_id = make_run_id(args.replay_ratio)
    return ReplayFineTuneConfig(
        run_id=run_id,
        replay_ratio=args.replay_ratio,
        replay_pool_size=args.replay_pool_size,
        replay_val_pool_size=args.replay_val_pool_size,
        replay_seed=args.replay_seed,
        replay_schedule_seed=args.replay_schedule_seed,
        replay_source=args.replay_source,
        warmup_fraction=args.warmup_fraction,
        epochs=args.epochs,
    )


def main() -> None:
    config = parse_args()

    df = vms_unicode.df
    train_folios, val_folios = folio_split(df, config.split_seed)
    config.train_folios = train_folios
    config.val_folios = val_folios

    # Voynich datasets
    voynich_train_ds = VoynichEntropyDataset(df, train_folios, config)
    voynich_val_ds = VoynichEntropyDataset(df, val_folios, config)

    # Replay pools (training + held-out validation)
    cache_dir = _replay_cache_dir(config)
    train_docs = load_or_fetch_replay_pool(
        config.replay_source,
        seed=config.replay_seed,
        pool_size=config.replay_pool_size,
        cache_dir=cache_dir,
        max_bytes=config.max_seq_len,
    )
    val_docs = load_or_fetch_replay_pool(
        config.replay_source,
        seed=config.replay_seed + 1,
        pool_size=config.replay_val_pool_size,
        cache_dir=cache_dir,
        max_bytes=config.max_seq_len,
    )
    replay_train_ds = DCLMReplayDataset(train_docs, config.max_seq_len)
    replay_val_ds = DCLMReplayDataset(val_docs, config.max_seq_len)

    # Scheduler + mixed train loader
    scheduler = BatchScheduler(
        num_voynich=len(voynich_train_ds),
        ratio=config.replay_ratio,
        seed=config.replay_schedule_seed,
    )
    train_loader = MixedDataLoader(
        voynich_ds=voynich_train_ds,
        replay_ds=replay_train_ds,
        scheduler=scheduler,
        voynich_shuffle_seed=config.split_seed,
        replay_shuffle_seed=config.replay_schedule_seed + 7,
    )

    # Two validation dataloaders (order matters: voynich then replay)
    voynich_val_dl = make_dataloader(voynich_val_ds, shuffle=False)
    replay_val_dl = DataLoader(
        replay_val_ds, batch_size=1, shuffle=False, num_workers=0
    )
    val_dls = [voynich_val_dl, replay_val_dl]

    # LR schedule length must match the mixed step count
    total_steps = scheduler.total_steps * config.epochs

    # Output directories and config snapshot
    run_dir = config.run_dir
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(run_dir / "config.yaml")

    # ClearML tracking
    _configure_clearml_credentials()
    task = Task.init(
        project_name=config.clearml_project,
        task_name=config.run_id,
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch_lightning": False},
        output_uri=False,
    )
    run_hparams = {
        "run_id": config.run_id,
        "replay_ratio": config.replay_ratio,
        "replay_source": config.replay_source,
        "replay_pool_size": config.replay_pool_size,
        "replay_val_pool_size": config.replay_val_pool_size,
        "replay_seed": config.replay_seed,
        "replay_schedule_seed": config.replay_schedule_seed,
        "replay_docs_fetched": len(train_docs),
        "replay_pool_hash": _replay_pool_hash(train_docs),
        "split_seed": config.split_seed,
        "train_folios": config.train_folios,
        "val_folios": config.val_folios,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
        "warmup_fraction": config.warmup_fraction,
        "epochs": config.epochs,
        "max_seq_len": config.max_seq_len,
        "hf_repo": config.hf_repo,
        "voynich_train_chunks": len(voynich_train_ds),
        "voynich_val_chunks": len(voynich_val_ds),
        "replay_train_chunks": len(replay_train_ds),
        "replay_val_chunks": len(replay_val_ds),
        "steps_per_epoch": scheduler.total_steps,
        "total_steps": total_steps,
        "git_commit": _git_commit(),
    }
    task.connect(run_hparams, name="run")
    clearml_logger = ClearMLLogger(task)

    periodic_ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch={epoch:03d}",
        every_n_epochs=config.checkpoint_every_n_epochs,
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    module = ReplayEntropyFineTune(config, total_steps)

    epoch_sync = MixedLoaderEpochCallback(train_loader)
    trainer = L.Trainer(
        max_epochs=config.epochs,
        precision="bf16-mixed",
        gradient_clip_val=config.grad_clip,
        callbacks=[periodic_ckpt, epoch_sync],
        logger=clearml_logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        default_root_dir=str(run_dir),
    )

    trainer.fit(module, train_loader, val_dls)

    # Final validation: both dataloaders in one call (Lightning routes by
    # dataloader_idx, giving one metrics dict per source). Write each to
    # its own eval file so the filesystem mirrors the per-source split.
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    val_results = trainer.validate(module, val_dls)
    voynich_metrics = val_results[0] if len(val_results) > 0 else {}
    replay_metrics = val_results[1] if len(val_results) > 1 else {}
    with open(eval_dir / "val_voynich_metrics.json", "w") as f:
        json.dump(voynich_metrics, f, indent=2)
    with open(eval_dir / "val_replay_metrics.json", "w") as f:
        json.dump(replay_metrics, f, indent=2)

    # Log realized replay ratio at run end (averaged across epochs)
    if train_loader.last_realized_ratio is not None:
        task.connect(
            {"realized_replay_ratio_final_epoch": train_loader.last_realized_ratio},
            name="realized",
        )

    task.close()


if __name__ == "__main__":
    main()
