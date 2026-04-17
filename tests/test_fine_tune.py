"""Tests for the fine_tune package.

Tests cover config serialization, dataset construction with folio splitting,
token encoding, padding, and the sliding window mask utility. Model loading
is mocked to avoid requiring GPU/HuggingFace access.
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, "src")

import pandas as pd

from fine_tune.config import (
    BLT_BYTE_OFFSET,
    MAX_SEQ_LEN,
    PAD_ID,
    VOCAB_SIZE,
    FineTuneConfig,
)
from fine_tune.dataset import (
    VoynichEntropyDataset,
    _encode_chunk,
    build_sliding_window_causal_mask,
    folio_split,
    make_dataloader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_df(n_folios=10, lines_per_folio=3):
    """Create a minimal DataFrame matching the vms_unicode schema."""
    rows = []
    for i in range(n_folios):
        folio = f"{i + 1}r"
        for line in range(1, lines_per_folio + 1):
            rows.append(
                {
                    "folio": folio,
                    "par": 1,
                    "line": line,
                    "t1": "\uf8d0\uf8d1",  # PUA chars like Voynich glyphs
                    "t2": "\uf8d2",
                    "t3": "$",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestFineTuneConfig:
    def test_defaults(self):
        cfg = FineTuneConfig(run_id="test-001")
        assert cfg.learning_rate == 1e-4
        assert cfg.epochs == 100
        assert cfg.pad_id == PAD_ID
        assert cfg.vocab_size == VOCAB_SIZE
        assert cfg.max_seq_len == MAX_SEQ_LEN

    def test_run_dir(self):
        cfg = FineTuneConfig(run_id="bft-test", experiments_dir=Path("/tmp/exp"))
        assert cfg.run_dir == Path("/tmp/exp/bft-test")

    def test_yaml_roundtrip(self):
        cfg = FineTuneConfig(
            run_id="bft-test",
            train_folios=["1r", "2r"],
            val_folios=["3r"],
            learning_rate=5e-5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            cfg.to_yaml(path)
            loaded = FineTuneConfig.from_yaml(path)
            assert loaded.run_id == "bft-test"
            assert loaded.train_folios == ["1r", "2r"]
            assert loaded.val_folios == ["3r"]
            assert loaded.learning_rate == 5e-5
            assert isinstance(loaded.experiments_dir, Path)


# ---------------------------------------------------------------------------
# Folio split
# ---------------------------------------------------------------------------


class TestFolioSplit:
    def test_split_sizes(self):
        df = _make_test_df(n_folios=10)
        train, val = folio_split(df, seed=42)
        assert len(train) == 8
        assert len(val) == 2
        assert set(train) & set(val) == set()  # no overlap

    def test_deterministic(self):
        df = _make_test_df(n_folios=20)
        train1, val1 = folio_split(df, seed=99)
        train2, val2 = folio_split(df, seed=99)
        assert train1 == train2
        assert val1 == val2

    def test_different_seeds(self):
        df = _make_test_df(n_folios=20)
        train1, _ = folio_split(df, seed=1)
        train2, _ = folio_split(df, seed=2)
        assert train1 != train2


# ---------------------------------------------------------------------------
# Token encoding
# ---------------------------------------------------------------------------


class TestEncodeChunk:
    def test_basic_encoding(self):
        text = "abc"
        tokens = _encode_chunk(text, max_seq_len=10)
        assert tokens.dtype == torch.long
        assert tokens.shape == (10,)
        # 'a'=97, token=97+4=101
        assert tokens[0].item() == ord("a") + BLT_BYTE_OFFSET
        assert tokens[1].item() == ord("b") + BLT_BYTE_OFFSET
        assert tokens[2].item() == ord("c") + BLT_BYTE_OFFSET
        # Remaining positions are PAD
        for i in range(3, 10):
            assert tokens[i].item() == PAD_ID

    def test_multibyte_encoding(self):
        text = "\u00b6"  # pilcrow, 2 bytes in UTF-8: 0xC2 0xB6
        tokens = _encode_chunk(text, max_seq_len=5)
        assert tokens[0].item() == 0xC2 + BLT_BYTE_OFFSET
        assert tokens[1].item() == 0xB6 + BLT_BYTE_OFFSET
        assert tokens[2].item() == PAD_ID

    def test_max_length(self):
        tokens = _encode_chunk("a" * 100, max_seq_len=10)
        assert tokens.shape == (10,)
        assert all(t.item() != PAD_ID for t in tokens)


# ---------------------------------------------------------------------------
# Sliding window mask
# ---------------------------------------------------------------------------


class TestSlidingWindowMask:
    def test_shape(self):
        mask = build_sliding_window_causal_mask(16, 4)
        assert mask.shape == (16, 16)

    def test_causal(self):
        mask = build_sliding_window_causal_mask(8, 8)
        # Upper triangle should be -inf (future positions)
        for i in range(8):
            for j in range(i + 1, 8):
                assert mask[i, j].item() == float("-inf")

    def test_window_limit(self):
        mask = build_sliding_window_causal_mask(8, 3)
        # Position 5 should attend to positions 3, 4, 5 only
        assert mask[5, 5].item() == 0.0
        assert mask[5, 4].item() == 0.0
        assert mask[5, 3].item() == 0.0
        assert mask[5, 2].item() == float("-inf")

    def test_first_position(self):
        mask = build_sliding_window_causal_mask(8, 3)
        # Position 0 only attends to itself
        assert mask[0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# VoynichEntropyDataset
# ---------------------------------------------------------------------------


class TestVoynichEntropyDataset:
    def test_construction_and_length(self):
        df = _make_test_df(n_folios=5)
        cfg = FineTuneConfig(run_id="test", max_seq_len=8192)
        train_folios, _ = folio_split(df, cfg.split_seed)
        ds = VoynichEntropyDataset(df, train_folios, cfg)
        assert len(ds) > 0

    def test_item_shape_and_dtype(self):
        df = _make_test_df(n_folios=5)
        cfg = FineTuneConfig(run_id="test", max_seq_len=256)
        train_folios, _ = folio_split(df, cfg.split_seed)
        ds = VoynichEntropyDataset(df, train_folios, cfg)
        item = ds[0]
        assert item.shape == (256,)
        assert item.dtype == torch.long

    def test_token_values_in_range(self):
        df = _make_test_df(n_folios=5)
        cfg = FineTuneConfig(run_id="test", max_seq_len=256)
        train_folios, _ = folio_split(df, cfg.split_seed)
        ds = VoynichEntropyDataset(df, train_folios, cfg)
        item = ds[0]
        for val in item.tolist():
            assert val == PAD_ID or (BLT_BYTE_OFFSET <= val < VOCAB_SIZE)


class TestMakeDataloader:
    def test_creates_dataloader(self):
        df = _make_test_df(n_folios=5)
        cfg = FineTuneConfig(run_id="test", max_seq_len=256)
        train_folios, _ = folio_split(df, cfg.split_seed)
        ds = VoynichEntropyDataset(df, train_folios, cfg)
        dl = make_dataloader(ds, shuffle=False)
        batch = next(iter(dl))
        assert batch.shape == (1, 256)
