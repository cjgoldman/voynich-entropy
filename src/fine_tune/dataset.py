"""Voynich entropy fine-tuning dataset and data utilities."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import vms_uprep
from fine_tune.config import BLT_BYTE_OFFSET, MAX_SEQ_LEN, PAD_ID, FineTuneConfig


def build_sliding_window_causal_mask(
    seq_len: int,
    window_size: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a sliding window causal attention mask for sdpa.

    Returns a float tensor of shape (seq_len, seq_len) with 0.0 for attended
    positions and -inf for masked positions.
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = (rows >= cols) & (rows - cols < window_size)
    attn_mask = torch.where(mask, 0.0, float("-inf"))
    return attn_mask


def folio_split(
    df: pd.DataFrame,
    seed: int,
    train_frac: float = 0.8,
) -> tuple[list[str], list[str]]:
    """Split folio identifiers into train/val sets.

    Args:
        df: Voynich DataFrame with a 'folio' column.
        seed: Random seed for reproducible shuffling.
        train_frac: Fraction of folios to assign to training.

    Returns:
        Tuple of (train_folios, val_folios).
    """
    folios = sorted(df["folio"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(folios)
    split_idx = int(len(folios) * train_frac)
    return folios[:split_idx], folios[split_idx:]


def _encode_chunk(text: str, max_seq_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """Encode a text chunk to BLT token IDs, right-padded to max_seq_len."""
    raw_bytes = text.encode("utf-8")
    n = min(len(raw_bytes), max_seq_len)
    out = np.full(max_seq_len, PAD_ID, dtype=np.int64)
    if n > 0:
        out[:n] = (
            np.frombuffer(raw_bytes, dtype=np.uint8, count=n).astype(np.int64)
            + BLT_BYTE_OFFSET
        )
    return torch.from_numpy(out)


class VoynichEntropyDataset(Dataset):
    """PyTorch Dataset for Voynich fine-tuning chunks.

    Each item is a LongTensor of shape (max_seq_len,) containing BLT token IDs,
    right-padded with PAD_ID.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        folios: list[str],
        config: FineTuneConfig,
    ) -> None:
        filtered_df = df[df["folio"].isin(folios)]
        lines = vms_uprep.prepare(filtered_df, max_bytes=config.max_seq_len * 10)
        chunks = vms_uprep.stack_lines(lines, max_bytes=config.max_seq_len)
        self.tokens = [_encode_chunk(chunk, config.max_seq_len) for chunk in chunks]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]


def make_dataloader(dataset: VoynichEntropyDataset, shuffle: bool) -> DataLoader:
    """Create a DataLoader for the dataset with batch_size=1, num_workers=0."""
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
