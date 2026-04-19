"""Tests for the replay fine-tune package.

Tests cover config, the replay pool cache, DCLMReplayDataset, the batch
scheduler, and MixedDataLoader. Model loading and HF network access are
mocked so the suite runs on CPU without credentials.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, "src")

import hf_data_samp
from fine_tune.config import BLT_BYTE_OFFSET, PAD_ID, VOCAB_SIZE
from fine_tune.replay_config import ReplayFineTuneConfig
from fine_tune.replay_dataset import (DCLMReplayDataset, _cache_path,
                                      load_or_fetch_replay_pool,
                                      load_replay_dataset_with_min_chunks)
from fine_tune.replay_loader import (REPLAY, VOYNICH, BatchScheduler,
                                     MixedDataLoader, MixedLoaderEpochCallback,
                                     _is_integer_ratio, replay_batch_count)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_samples(texts):
    return [
        hf_data_samp.HFSample(
            text=t,
            doc_index=i,
            dataset_id=hf_data_samp.DCLM.repo_id,
            byte_length=len(t.encode("utf-8")),
            truncated=False,
        )
        for i, t in enumerate(texts)
    ]


class _TensorDataset:
    """Minimal dataset of fixed-length LongTensors for loader tests."""

    def __init__(self, values, seq_len=4):
        self._items = [torch.full((seq_len,), v, dtype=torch.long) for v in values]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


# ---------------------------------------------------------------------------
# ReplayFineTuneConfig
# ---------------------------------------------------------------------------


class TestReplayFineTuneConfig:
    def test_defaults(self):
        cfg = ReplayFineTuneConfig(run_id="rft-test")
        assert cfg.replay_ratio == 1.0
        assert cfg.replay_pool_size == 1000
        assert cfg.replay_val_pool_size == 100
        assert cfg.replay_seed == 42
        assert cfg.replay_schedule_seed == 43
        assert cfg.replay_source == "DCLM"
        assert cfg.warmup_fraction == 0.04
        assert cfg.clearml_project == "voynich-replay-fine-tune"
        # Inherits basic fine-tune fields
        assert cfg.epochs == 100
        assert cfg.max_seq_len == 8192

    def test_yaml_roundtrip(self):
        cfg = ReplayFineTuneConfig(
            run_id="rft-test",
            replay_ratio=2.0,
            replay_pool_size=10,
            warmup_fraction=0.08,
            train_folios=["1r"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            cfg.to_yaml(path)
            loaded = ReplayFineTuneConfig.from_yaml(path)
            assert loaded.replay_ratio == 2.0
            assert loaded.replay_pool_size == 10
            assert loaded.warmup_fraction == 0.08
            assert loaded.train_folios == ["1r"]


# ---------------------------------------------------------------------------
# Replay pool cache
# ---------------------------------------------------------------------------


class TestReplayPoolCache:
    def test_cache_path(self):
        path = _cache_path(Path("/tmp/cache"), "DCLM", 42, 100)
        assert path == Path("/tmp/cache/dclm-seed42-n100.jsonl")

    def test_fetch_writes_cache(self):
        samples = _make_fake_samples(["doc a", "doc b", "doc c"])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            with patch.object(
                hf_data_samp,
                "sample_with_metadata",
                return_value=samples,
            ) as mock_fn:
                docs = load_or_fetch_replay_pool(
                    "DCLM", seed=42, pool_size=3, cache_dir=cache_dir
                )
            assert docs == ["doc a", "doc b", "doc c"]
            mock_fn.assert_called_once()

            cache_file = _cache_path(cache_dir, "DCLM", 42, 3)
            assert cache_file.exists()
            lines = [json.loads(line) for line in cache_file.read_text().splitlines()]
            assert [line["text"] for line in lines] == ["doc a", "doc b", "doc c"]

    def test_cache_hit_skips_fetch(self):
        samples = _make_fake_samples(["cached doc"])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            with patch.object(
                hf_data_samp, "sample_with_metadata", return_value=samples
            ) as mock_fn:
                load_or_fetch_replay_pool(
                    "DCLM", seed=7, pool_size=1, cache_dir=cache_dir
                )
                assert mock_fn.call_count == 1
                # Second call should hit cache
                docs = load_or_fetch_replay_pool(
                    "DCLM", seed=7, pool_size=1, cache_dir=cache_dir
                )
                assert docs == ["cached doc"]
                assert mock_fn.call_count == 1

    def test_unknown_source_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown replay source"):
                load_or_fetch_replay_pool(
                    "NOPE", seed=1, pool_size=1, cache_dir=Path(tmpdir)
                )

    def test_truncated_cache_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            path = _cache_path(cache_dir, "DCLM", seed=5, pool_size=10)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Simulate a partially-written cache (3 of 10)
            with open(path, "w") as f:
                for i in range(3):
                    f.write(
                        json.dumps(
                            {"text": f"partial {i}", "doc_index": i, "truncated": False}
                        )
                        + "\n"
                    )
            with pytest.raises(RuntimeError, match="truncated"):
                load_or_fetch_replay_pool(
                    "DCLM", seed=5, pool_size=10, cache_dir=cache_dir
                )


# ---------------------------------------------------------------------------
# DCLMReplayDataset
# ---------------------------------------------------------------------------


class TestDCLMReplayDataset:
    def test_basic_construction(self):
        docs = ["hello world", "lorem ipsum", "more text"]
        ds = DCLMReplayDataset(docs, max_seq_len=64)
        assert len(ds) >= 1
        item = ds[0]
        assert item.shape == (64,)
        assert item.dtype == torch.long

    def test_token_values_in_range(self):
        docs = ["alpha beta gamma"]
        ds = DCLMReplayDataset(docs, max_seq_len=128)
        for val in ds[0].tolist():
            assert val == PAD_ID or (BLT_BYTE_OFFSET <= val < VOCAB_SIZE)

    def test_byte_offset_applied(self):
        docs = ["abc"]
        ds = DCLMReplayDataset(docs, max_seq_len=16)
        tokens = ds[0]
        assert tokens[0].item() == ord("a") + BLT_BYTE_OFFSET
        assert tokens[1].item() == ord("b") + BLT_BYTE_OFFSET
        assert tokens[2].item() == ord("c") + BLT_BYTE_OFFSET
        assert tokens[3].item() == PAD_ID


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------


class TestIntegerRatioDetection:
    def test_integers(self):
        assert _is_integer_ratio(0.0)
        assert _is_integer_ratio(1.0)
        assert _is_integer_ratio(4.0)

    def test_non_integers(self):
        assert not _is_integer_ratio(0.5)
        assert not _is_integer_ratio(0.25)
        assert not _is_integer_ratio(2.5)


class TestReplayBatchCount:
    def test_integer_ratio(self):
        assert replay_batch_count(10, 0.0) == 0
        assert replay_batch_count(10, 1.0) == 10
        assert replay_batch_count(10, 4.0) == 40

    def test_fractional_ratio_uses_ceil(self):
        assert replay_batch_count(10, 0.5) == 5
        assert replay_batch_count(10, 0.25) == 3
        assert replay_batch_count(7, 0.5) == 4


# ---------------------------------------------------------------------------
# BatchScheduler
# ---------------------------------------------------------------------------


class TestBatchScheduler:
    def test_zero_ratio_all_voynich(self):
        sched = BatchScheduler(num_voynich=5, ratio=0.0, seed=43)
        seq = sched.schedule(epoch=0)
        assert seq == [VOYNICH] * 5
        assert sched.total_steps == 5

    def test_integer_ratio_deterministic_pattern(self):
        sched = BatchScheduler(num_voynich=3, ratio=2.0, seed=43)
        seq = sched.schedule(epoch=0)
        assert seq == [VOYNICH, REPLAY, REPLAY] * 3
        assert sched.total_steps == 9
        # Integer schedule is independent of epoch
        assert sched.schedule(epoch=7) == seq

    def test_ratio_one_alternates(self):
        sched = BatchScheduler(num_voynich=4, ratio=1.0, seed=43)
        seq = sched.schedule(epoch=0)
        assert seq == [VOYNICH, REPLAY] * 4

    def test_fractional_ratio_exact_counts(self):
        sched = BatchScheduler(num_voynich=10, ratio=0.5, seed=43)
        seq = sched.schedule(epoch=0)
        assert seq.count(VOYNICH) == 10
        assert seq.count(REPLAY) == 5
        assert len(seq) == 15

    def test_fractional_ratio_shuffled_per_epoch(self):
        sched = BatchScheduler(num_voynich=10, ratio=0.5, seed=43)
        s0 = sched.schedule(epoch=0)
        s1 = sched.schedule(epoch=1)
        # Counts stable, order differs
        assert s0.count(VOYNICH) == s1.count(VOYNICH) == 10
        assert s0 != s1

    def test_fractional_ratio_deterministic_per_epoch(self):
        s1 = BatchScheduler(num_voynich=8, ratio=0.5, seed=43).schedule(epoch=3)
        s2 = BatchScheduler(num_voynich=8, ratio=0.5, seed=43).schedule(epoch=3)
        assert s1 == s2

    def test_rejects_negative_ratio(self):
        with pytest.raises(ValueError):
            BatchScheduler(num_voynich=1, ratio=-0.1, seed=43)

    def test_rejects_zero_num_voynich(self):
        with pytest.raises(ValueError):
            BatchScheduler(num_voynich=0, ratio=1.0, seed=43)


# ---------------------------------------------------------------------------
# MixedDataLoader
# ---------------------------------------------------------------------------


class TestMixedDataLoader:
    def test_yields_correct_count_and_tags(self):
        voynich = _TensorDataset([10, 20, 30], seq_len=4)
        replay = _TensorDataset([1, 2, 3, 4], seq_len=4)
        sched = BatchScheduler(num_voynich=3, ratio=1.0, seed=43)
        loader = MixedDataLoader(
            voynich_ds=voynich,
            replay_ds=replay,
            scheduler=sched,
            voynich_shuffle_seed=42,
            replay_shuffle_seed=50,
            total_epochs=1,
        )
        assert len(loader) == 6
        batches = list(loader)
        assert len(batches) == 6
        v = sum(1 for b in batches if b["source"] == VOYNICH)
        r = sum(1 for b in batches if b["source"] == REPLAY)
        assert v == 3 and r == 3
        for b in batches:
            assert b["tokens"].shape == (1, 4)
            assert b["tokens"].dtype == torch.long

    def test_pin_memory_flag_is_no_op_without_cuda(self):
        # pin_memory=True must be safe on CPU-only hosts: the loader guards
        # the call with torch.cuda.is_available() so iteration still works
        # and yields ordinary (un-pinned) tensors.
        voynich = _TensorDataset([10, 20], seq_len=4)
        replay = _TensorDataset([1, 2], seq_len=4)
        sched = BatchScheduler(num_voynich=2, ratio=1.0, seed=43)
        loader = MixedDataLoader(
            voynich_ds=voynich,
            replay_ds=replay,
            scheduler=sched,
            voynich_shuffle_seed=42,
            replay_shuffle_seed=50,
            total_epochs=1,
            pin_memory=True,
        )
        assert loader.pin_memory == torch.cuda.is_available()
        batches = list(loader)
        assert len(batches) == 4
        for b in batches:
            assert b["tokens"].shape == (1, 4)
            if torch.cuda.is_available():
                assert b["tokens"].is_pinned()

    def test_voynich_samples_not_repeated_within_epoch(self):
        voynich = _TensorDataset([10, 20, 30], seq_len=2)
        replay = _TensorDataset([1], seq_len=2)
        sched = BatchScheduler(num_voynich=3, ratio=0.0, seed=43)
        loader = MixedDataLoader(
            voynich_ds=voynich,
            replay_ds=replay,
            scheduler=sched,
            voynich_shuffle_seed=42,
            replay_shuffle_seed=50,
            total_epochs=1,
        )
        seen = set()
        for batch in loader:
            seen.add(int(batch["tokens"][0, 0].item()))
        assert seen == {10, 20, 30}

    def test_realized_ratio_recorded(self):
        voynich = _TensorDataset([10, 20], seq_len=2)
        replay = _TensorDataset([1, 2], seq_len=2)
        sched = BatchScheduler(num_voynich=2, ratio=1.0, seed=43)
        loader = MixedDataLoader(
            voynich_ds=voynich,
            replay_ds=replay,
            scheduler=sched,
            voynich_shuffle_seed=42,
            replay_shuffle_seed=50,
            total_epochs=1,
        )
        assert loader.last_realized_ratio is None
        list(loader)
        assert loader.last_realized_ratio == pytest.approx(0.5)

    def test_epoch_counter_advances(self):
        voynich = _TensorDataset([10, 20, 30], seq_len=2)
        # Need >= 2 replay per epoch * 2 epochs = 4 chunks
        replay = _TensorDataset([1, 2, 3, 4], seq_len=2)
        sched = BatchScheduler(num_voynich=3, ratio=0.5, seed=43)
        loader = MixedDataLoader(
            voynich_ds=voynich,
            replay_ds=replay,
            scheduler=sched,
            voynich_shuffle_seed=42,
            replay_shuffle_seed=50,
            total_epochs=2,
        )
        order_epoch0 = [b["source"] for b in loader]
        order_epoch1 = [b["source"] for b in loader]
        # Non-integer ratio should produce different orders across epochs
        assert order_epoch0 != order_epoch1

    def test_epoch_sync_callback_overrides_internal_counter(self):
        voynich = _TensorDataset([10, 20, 30], seq_len=2)
        replay = _TensorDataset([1, 2, 3, 4], seq_len=2)
        sched = BatchScheduler(num_voynich=3, ratio=0.5, seed=43)
        loader = MixedDataLoader(
            voynich_ds=voynich,
            replay_ds=replay,
            scheduler=sched,
            voynich_shuffle_seed=42,
            replay_shuffle_seed=50,
            total_epochs=2,
        )

        class _FakeTrainer:
            current_epoch = 7

        cb = MixedLoaderEpochCallback(loader)
        cb.on_train_epoch_start(_FakeTrainer(), pl_module=None)

        order_after_sync = [b["source"] for b in loader]
        # A fresh loader set to the same epoch must yield the same schedule
        loader2 = MixedDataLoader(
            voynich_ds=voynich,
            replay_ds=replay,
            scheduler=sched,
            voynich_shuffle_seed=42,
            replay_shuffle_seed=50,
            total_epochs=2,
        )
        loader2.set_epoch(7)
        order_reference = [b["source"] for b in loader2]
        assert order_after_sync == order_reference

    def test_mismatched_sizes_raises(self):
        voynich = _TensorDataset([10, 20], seq_len=2)
        replay = _TensorDataset([1], seq_len=2)
        sched = BatchScheduler(num_voynich=3, ratio=1.0, seed=43)
        with pytest.raises(ValueError, match="voynich chunks"):
            MixedDataLoader(
                voynich_ds=voynich,
                replay_ds=replay,
                scheduler=sched,
                voynich_shuffle_seed=42,
                replay_shuffle_seed=50,
                total_epochs=1,
            )

    def test_replay_globally_unique_across_epochs(self):
        voynich = _TensorDataset([10, 20], seq_len=2)
        # ratio=2.0 → 4 replay per epoch, 3 epochs → need 12 unique replay
        replay = _TensorDataset(list(range(100, 112)), seq_len=2)
        sched = BatchScheduler(num_voynich=2, ratio=2.0, seed=43)
        loader = MixedDataLoader(
            voynich_ds=voynich,
            replay_ds=replay,
            scheduler=sched,
            voynich_shuffle_seed=42,
            replay_shuffle_seed=50,
            total_epochs=3,
        )
        seen_replay_values = []
        for _ in range(3):
            for b in loader:
                if b["source"] == REPLAY:
                    seen_replay_values.append(int(b["tokens"][0, 0].item()))
        assert len(seen_replay_values) == 12
        assert len(set(seen_replay_values)) == 12
        assert set(seen_replay_values) == set(range(100, 112))

    def test_init_raises_when_replay_too_small(self):
        voynich = _TensorDataset([10, 20], seq_len=2)
        # num_voynich=2, ratio=1.0, epochs=5 → 10 replay needed, only 5 given
        replay = _TensorDataset([1, 2, 3, 4, 5], seq_len=2)
        sched = BatchScheduler(num_voynich=2, ratio=1.0, seed=43)
        with pytest.raises(ValueError, match="globally-unique"):
            MixedDataLoader(
                voynich_ds=voynich,
                replay_ds=replay,
                scheduler=sched,
                voynich_shuffle_seed=42,
                replay_shuffle_seed=50,
                total_epochs=5,
            )

    def test_replay_cursor_persists_across_epochs(self):
        voynich = _TensorDataset([10, 20], seq_len=2)
        # ratio=1.0 → 2 replay per epoch; 3 epochs → need 6 replay
        replay = _TensorDataset(list(range(200, 206)), seq_len=2)
        sched = BatchScheduler(num_voynich=2, ratio=1.0, seed=43)
        loader = MixedDataLoader(
            voynich_ds=voynich,
            replay_ds=replay,
            scheduler=sched,
            voynich_shuffle_seed=42,
            replay_shuffle_seed=50,
            total_epochs=3,
        )
        assert loader._replay_cursor == 0
        list(loader)
        assert loader._replay_cursor == sched.num_replay
        list(loader)
        assert loader._replay_cursor == 2 * sched.num_replay
        list(loader)
        assert loader._replay_cursor == 3 * sched.num_replay


class TestLoadReplayDatasetWithMinChunks:
    def test_grows_pool_until_enough_chunks(self):
        """Helper doubles pool_size until the built dataset meets min_chunks."""
        call_sizes = []

        # Each doc is short enough that it packs into exactly one chunk.
        # stack_lines with max_bytes will concatenate small lines together,
        # so we mock DCLMReplayDataset directly to make the chunks count
        # deterministic.
        class _FakeDS:
            def __init__(self, docs, max_seq_len):
                # 70% packing efficiency: n docs → n // 2 chunks
                self._n = max(1, len(docs) // 2)

            def __len__(self):
                return self._n

        def fake_load(source, seed, pool_size, cache_dir, max_bytes):
            call_sizes.append(pool_size)
            return [f"doc-{i}" for i in range(pool_size)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "fine_tune.replay_dataset.load_or_fetch_replay_pool",
                    side_effect=fake_load,
                ),
                patch(
                    "fine_tune.replay_dataset.DCLMReplayDataset",
                    _FakeDS,
                ),
            ):
                docs, ds = load_replay_dataset_with_min_chunks(
                    source="DCLM",
                    seed=1,
                    min_chunks=20,
                    max_seq_len=64,
                    cache_dir=Path(tmpdir),
                    initial_pool_size=10,
                )
        # initial_pool_size=10 < min_chunks=20 → start at 20 (min_chunks),
        # yields 10 chunks, double to 40 (yields 20) → sufficient.
        assert call_sizes == [20, 40]
        assert len(ds) >= 20
        assert len(docs) == 40

    def test_raises_when_cannot_grow_enough(self):
        class _FakeDS:
            def __init__(self, docs, max_seq_len):
                self._n = 1  # always yields 1 chunk

            def __len__(self):
                return self._n

        def fake_load(source, seed, pool_size, cache_dir, max_bytes):
            return [f"doc-{i}" for i in range(pool_size)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "fine_tune.replay_dataset.load_or_fetch_replay_pool",
                    side_effect=fake_load,
                ),
                patch(
                    "fine_tune.replay_dataset.DCLMReplayDataset",
                    _FakeDS,
                ),
            ):
                with pytest.raises(RuntimeError, match="Unable to build"):
                    load_replay_dataset_with_min_chunks(
                        source="DCLM",
                        seed=1,
                        min_chunks=5,
                        max_seq_len=64,
                        cache_dir=Path(tmpdir),
                        initial_pool_size=1,
                    )
