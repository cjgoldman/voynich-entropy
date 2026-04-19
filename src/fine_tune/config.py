"""Configuration for basic Voynich entropy model fine-tuning."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HF_REPO = "facebook/blt-entropy"
BLT_BYTE_OFFSET = 4
PAD_ID = 2
VOCAB_SIZE = 260
MAX_SEQ_LEN = 8192
SLIDING_WINDOW = 512


@dataclass
class FineTuneConfig:
    """Full configuration for a basic fine-tuning run."""

    # --- Run identity ---
    run_id: str
    experiments_dir: Path = Path("../data/experiments")

    # --- Model ---
    hf_repo: str = HF_REPO

    # --- Data ---
    split_seed: int = 42
    train_folios: list[str] = field(default_factory=list)
    val_folios: list[str] = field(default_factory=list)
    max_seq_len: int = MAX_SEQ_LEN
    pad_id: int = PAD_ID
    blt_byte_offset: int = BLT_BYTE_OFFSET
    vocab_size: int = VOCAB_SIZE
    sliding_window: int = SLIDING_WINDOW

    # --- Optimizer ---
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 10.0

    # --- Schedule ---
    warmup_steps: int = 50
    epochs: int = 10

    # --- Checkpointing ---
    checkpoint_every_n_epochs: int = 10

    # --- ClearML ---
    clearml_project: str = "voynich-fine-tune"

    @property
    def run_dir(self) -> Path:
        return Path(self.experiments_dir) / self.run_id

    def to_yaml(self, path: Path) -> None:
        """Serialize config to a YAML file."""
        data = dataclasses.asdict(self)
        # Convert Path objects to strings for YAML serialization
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> FineTuneConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        data["experiments_dir"] = Path(data["experiments_dir"])
        return cls(**data)
