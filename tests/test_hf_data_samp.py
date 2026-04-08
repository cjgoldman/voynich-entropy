"""Tests for hf_data_samp module.

Tests cover the offline/unit-testable parts: dataclasses, text column
auto-detection, UTF-8 truncation, and the sampling flow via a mock dataset.
"""

import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, "src")

from hf_data_samp import (
    DatasetSpec,
    HFSample,
    DCLM,
    _resolve_text_column,
    _truncate_utf8,
    sample,
    sample_with_metadata,
)


# ---------------------------------------------------------------------------
# DatasetSpec / HFSample / DCLM
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_dataset_spec_defaults(self):
        spec = DatasetSpec(repo_id="org/dataset")
        assert spec.split == "train"
        assert spec.text_column is None
        assert spec.subset is None

    def test_dclm_preset(self):
        assert DCLM.repo_id == "mlfoundations/dclm-baseline-1.0"
        assert DCLM.split == "train"
        assert DCLM.text_column == "text"

    def test_hf_sample_fields(self):
        s = HFSample(
            text="hello",
            doc_index=0,
            dataset_id="x/y",
            byte_length=5,
            truncated=False,
        )
        assert s.byte_length == 5
        assert not s.truncated


# ---------------------------------------------------------------------------
# Text column auto-detection
# ---------------------------------------------------------------------------


class TestResolveTextColumn:
    def test_explicit_column(self):
        spec = DatasetSpec(repo_id="x", text_column="body")
        assert _resolve_text_column(spec, {"body": None, "text": None}) == "body"

    def test_auto_detect_text(self):
        spec = DatasetSpec(repo_id="x")
        assert _resolve_text_column(spec, {"text": None, "id": None}) == "text"

    def test_auto_detect_content_fallback(self):
        spec = DatasetSpec(repo_id="x")
        assert _resolve_text_column(spec, {"content": None, "id": None}) == "content"

    def test_auto_detect_prefers_text_over_content(self):
        spec = DatasetSpec(repo_id="x")
        assert _resolve_text_column(spec, {"content": None, "text": None}) == "text"

    def test_auto_detect_raises_on_missing(self):
        spec = DatasetSpec(repo_id="x")
        with pytest.raises(ValueError, match="Available columns"):
            _resolve_text_column(spec, {"id": None, "url": None})


# ---------------------------------------------------------------------------
# UTF-8 truncation
# ---------------------------------------------------------------------------


class TestTruncateUtf8:
    def test_no_truncation_needed(self):
        text, truncated = _truncate_utf8("hello", 100)
        assert text == "hello"
        assert not truncated

    def test_exact_fit(self):
        text, truncated = _truncate_utf8("hello", 5)
        assert text == "hello"
        assert not truncated

    def test_ascii_truncation(self):
        text, truncated = _truncate_utf8("hello world", 5)
        assert text == "hello"
        assert truncated

    def test_multibyte_character_boundary(self):
        # 'é' is 2 bytes in UTF-8 (0xC3 0xA9)
        text, truncated = _truncate_utf8("café", 4)
        # "caf" = 3 bytes, "é" would need 2 more = 5 total, so truncate
        assert text == "caf"
        assert truncated

    def test_emoji_boundary(self):
        # '🔥' is 4 bytes in UTF-8; "🔥abc" = 7 bytes total
        text, truncated = _truncate_utf8("🔥abc", 5)
        # 4 bytes for emoji + 1 byte for 'a' = 5; truncated from 7
        assert text == "🔥a"
        assert truncated

    def test_mid_codepoint_truncation(self):
        # Cut in the middle of a 3-byte character (e.g., '€' = 0xE2 0x82 0xAC)
        text, truncated = _truncate_utf8("€", 2)
        assert text == ""
        assert truncated


# ---------------------------------------------------------------------------
# Sampling (mocked dataset)
# ---------------------------------------------------------------------------


def _make_mock_dataset(docs, has_shuffle=True):
    """Create a mock IterableDataset that yields *docs* dicts."""
    ds = MagicMock()

    # Track transforms applied to determine final doc list
    state = {"docs": list(docs), "shuffled": False}

    def mock_shuffle(seed, buffer_size):
        # Reverse as a deterministic "shuffle" for testing
        state["docs"] = list(reversed(state["docs"]))
        state["shuffled"] = True
        return ds

    def mock_skip(n):
        state["docs"] = state["docs"][n:]
        return ds

    def mock_take(n):
        state["docs"] = state["docs"][:n]
        return ds

    ds.shuffle = mock_shuffle
    ds.skip = mock_skip
    ds.take = mock_take
    ds.__iter__ = lambda self: iter(state["docs"])

    return ds


class TestSampleWithMetadata:
    def test_basic_sampling(self):
        docs = [{"text": f"doc {i}"} for i in range(5)]
        mock_ds = _make_mock_dataset(docs)

        with patch("hf_data_samp.load_dataset", return_value=mock_ds):
            results = sample_with_metadata(DCLM, n=3)

        assert len(results) == 3
        assert results[0].text == "doc 0"
        assert results[0].doc_index == 0
        assert results[0].dataset_id == DCLM.repo_id
        assert not results[0].truncated

    def test_offset(self):
        docs = [{"text": f"doc {i}"} for i in range(10)]
        mock_ds = _make_mock_dataset(docs)

        with patch("hf_data_samp.load_dataset", return_value=mock_ds):
            results = sample_with_metadata(DCLM, n=2, offset=5)

        assert len(results) == 2
        assert results[0].text == "doc 5"
        assert results[0].doc_index == 5

    def test_seed_shuffles(self):
        docs = [{"text": f"doc {i}"} for i in range(5)]
        mock_ds = _make_mock_dataset(docs)

        with patch("hf_data_samp.load_dataset", return_value=mock_ds):
            results = sample_with_metadata(DCLM, n=3, seed=42)

        # Mock shuffle reverses, so first doc is "doc 4"
        assert results[0].text == "doc 4"

    def test_truncation_tracked(self):
        docs = [{"text": "a" * 10000}]
        mock_ds = _make_mock_dataset(docs)

        with patch("hf_data_samp.load_dataset", return_value=mock_ds):
            results = sample_with_metadata(DCLM, n=1, max_bytes=100)

        assert results[0].truncated
        assert results[0].byte_length == 100

    def test_fewer_than_n(self):
        docs = [{"text": "only one"}]
        mock_ds = _make_mock_dataset(docs)

        with patch("hf_data_samp.load_dataset", return_value=mock_ds):
            results = sample_with_metadata(DCLM, n=10)

        assert len(results) == 1


class TestSample:
    def test_returns_strings(self):
        docs = [{"text": f"doc {i}"} for i in range(3)]
        mock_ds = _make_mock_dataset(docs)

        with patch("hf_data_samp.load_dataset", return_value=mock_ds):
            results = sample(DCLM, n=3)

        assert results == ["doc 0", "doc 1", "doc 2"]


class TestAutoDetectIntegration:
    def test_content_column_auto_detected(self):
        docs = [{"content": "hello", "id": "1"}]
        mock_ds = _make_mock_dataset(docs)
        spec = DatasetSpec(repo_id="test/repo")

        with patch("hf_data_samp.load_dataset", return_value=mock_ds):
            results = sample(spec, n=1)

        assert results == ["hello"]
