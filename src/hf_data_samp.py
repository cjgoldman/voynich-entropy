"""HF Data Sampling Utility.

Lightweight streaming sampler for pulling small text samples from
Hugging Face datasets for comparative entropy analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Specifies which HF dataset to sample from."""

    repo_id: str
    split: str = "train"
    text_column: str | None = None
    subset: str | None = None


@dataclass
class HFSample:
    """A sampled document with provenance metadata."""

    text: str
    doc_index: int
    dataset_id: str
    byte_length: int
    truncated: bool


# Pre-configured spec for the DCLM baseline (the dataset used to train BLT).
DCLM = DatasetSpec(
    repo_id="mlfoundations/dclm-baseline-1.0",
    split="train",
    text_column="text",
)


# ---------------------------------------------------------------------------
# Text column auto-detection
# ---------------------------------------------------------------------------

_TEXT_COLUMN_PRIORITY = ("text", "content")


def _resolve_text_column(spec: DatasetSpec, features) -> str:
    """Return the text column name, auto-detecting if not set explicitly."""
    if spec.text_column is not None:
        return spec.text_column
    for candidate in _TEXT_COLUMN_PRIORITY:
        if candidate in features:
            return candidate
    available = list(features)
    raise ValueError(
        f"Could not auto-detect text column in {spec.repo_id}. "
        f"Available columns: {available}. "
        "Set text_column explicitly in DatasetSpec."
    )


# ---------------------------------------------------------------------------
# UTF-8 safe truncation
# ---------------------------------------------------------------------------

def _truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate *text* to fit within *max_bytes* of UTF-8.

    Returns (truncated_text, was_truncated).
    """
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text, False
    return raw[:max_bytes].decode("utf-8", errors="ignore"), True


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_with_metadata(
    spec: DatasetSpec,
    n: int = 10,
    *,
    offset: int = 0,
    seed: int | None = None,
    max_bytes: int = 8192,
) -> list[HFSample]:
    """Pull *n* text documents from a HF dataset via streaming.

    Args:
        spec: Dataset specification.
        n: Number of documents to retrieve.
        offset: Skip this many documents before sampling.
        seed: If set, shuffle the stream with this seed before sampling.
        max_bytes: Per-document UTF-8 byte budget.

    Returns:
        List of ``HFSample`` objects (may be shorter than *n* if the
        dataset has fewer documents).
    """
    ds = load_dataset(
        spec.repo_id,
        name=spec.subset,
        split=spec.split,
        streaming=True,
    )

    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=1000)

    if offset > 0:
        ds = ds.skip(offset)

    ds = ds.take(n)

    text_col: str | None = None
    results: list[HFSample] = []

    for idx, example in enumerate(ds):
        if text_col is None:
            text_col = _resolve_text_column(spec, example.keys())
        raw_text = example[text_col]
        text, truncated = _truncate_utf8(raw_text, max_bytes)
        results.append(
            HFSample(
                text=text,
                doc_index=offset + idx,
                dataset_id=spec.repo_id,
                byte_length=len(text.encode("utf-8")),
                truncated=truncated,
            )
        )

    return results


def sample(
    spec: DatasetSpec,
    n: int = 10,
    *,
    offset: int = 0,
    seed: int | None = None,
    max_bytes: int = 8192,
) -> list[str]:
    """Pull *n* text documents from a HF dataset via streaming.

    Convenience wrapper around :func:`sample_with_metadata` that returns
    plain strings only.
    """
    return [
        s.text
        for s in sample_with_metadata(
            spec,
            n,
            offset=offset,
            seed=seed,
            max_bytes=max_bytes,
        )
    ]
