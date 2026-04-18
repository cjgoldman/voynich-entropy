"""Configuration for replay-based Voynich entropy model fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from fine_tune.config import FineTuneConfig


@dataclass
class ReplayFineTuneConfig(FineTuneConfig):
    """Full configuration for a replay fine-tuning run.

    Inherits every field from :class:`FineTuneConfig` and adds replay-specific
    knobs. ``warmup_fraction`` supersedes ``warmup_steps`` so warmup scales
    proportionally with the mixed step count.
    """

    # --- Replay data ---
    replay_source: str = "DCLM"
    replay_pool_size: int = 1000
    replay_val_pool_size: int = 100
    replay_seed: int = 42
    replay_schedule_seed: int = 43

    # --- Batch mixing ---
    replay_ratio: float = 1.0

    # --- Schedule ---
    warmup_fraction: float = 0.04

    # --- ClearML ---
    clearml_project: str = "voynich-replay-fine-tune"

    @classmethod
    def from_yaml(cls, path: Path) -> ReplayFineTuneConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        data["experiments_dir"] = Path(data["experiments_dir"])
        return cls(**data)
