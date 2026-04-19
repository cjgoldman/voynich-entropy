"""DCLM replay dataset and pool caching for replay fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

import hf_data_samp
import vms_uprep
from fine_tune.dataset import _encode_chunk

_REPLAY_SOURCES = {
    "DCLM": hf_data_samp.DCLM,
}


def _cache_path(cache_dir: Path, source: str, seed: int, pool_size: int) -> Path:
    """Path where the replay pool for (source, seed, size) is cached."""
    slug = source.lower()
    return cache_dir / f"{slug}-seed{seed}-n{pool_size}.jsonl"


def load_or_fetch_replay_pool(
    source: str,
    seed: int,
    pool_size: int,
    cache_dir: Path,
    max_bytes: int = 8192,
) -> list[str]:
    """Load a replay pool from cache, or fetch from HF and write the cache.

    The cache file is a JSONL of ``{"text": ..., "doc_index": ..., "truncated": ...}``
    records — one document per line. Subsequent runs with the same
    ``(source, seed, pool_size)`` read directly from disk without HF access.
    """
    if source not in _REPLAY_SOURCES:
        raise ValueError(
            f"Unknown replay source {source!r}. Known: {list(_REPLAY_SOURCES)}"
        )
    spec = _REPLAY_SOURCES[source]

    path = _cache_path(cache_dir, source, seed, pool_size)
    if path.exists():
        with open(path) as f:
            docs = [json.loads(line)["text"] for line in f if line.strip()]
        if len(docs) != pool_size:
            raise RuntimeError(
                f"Replay cache at {path} has {len(docs)} docs but pool_size "
                f"is {pool_size}. The file may have been truncated during a "
                "prior run. Delete it to force a re-fetch."
            )
        return docs

    samples = hf_data_samp.sample_with_metadata(
        spec, n=pool_size, seed=seed, max_bytes=max_bytes
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "text": s.text,
                        "doc_index": s.doc_index,
                        "truncated": s.truncated,
                    }
                )
                + "\n"
            )
    return [s.text for s in samples]


def load_replay_dataset_with_min_chunks(
    source: str,
    seed: int,
    min_chunks: int,
    max_seq_len: int,
    cache_dir: Path,
    initial_pool_size: int,
) -> tuple[list[str], "DCLMReplayDataset"]:
    """Fetch a replay pool large enough to yield at least ``min_chunks`` chunks.

    Since ``DCLMReplayDataset`` packs docs into at most ``len(docs)`` chunks
    (DCLM docs get truncated at ``max_seq_len``), we start with a pool of
    ``max(initial_pool_size, min_chunks)`` documents, build the dataset, and
    double the pool size if the resulting chunk count falls short. The HF
    cache is keyed on ``(source, seed, pool_size)`` and re-fetching with a
    larger size under the same seed yields a deterministic superset.
    """
    pool_size = max(initial_pool_size, min_chunks)
    max_iterations = 10
    for iteration in range(max_iterations):
        print(
            f"[replay] fetching pool_size={pool_size} "
            f"(need >= {min_chunks} chunks, iteration={iteration + 1})"
        )
        docs = load_or_fetch_replay_pool(
            source,
            seed=seed,
            pool_size=pool_size,
            cache_dir=cache_dir,
            max_bytes=max_seq_len,
        )
        ds = DCLMReplayDataset(docs, max_seq_len)
        print(f"[replay] pool_size={pool_size} yielded {len(ds)} chunks")
        if len(ds) >= min_chunks:
            return docs, ds
        pool_size *= 2
    raise RuntimeError(
        f"Unable to build replay dataset with >= {min_chunks} chunks "
        f"after {max_iterations} iterations (last pool_size={pool_size // 2}, "
        f"last chunks={len(ds)})"
    )


class DCLMReplayDataset(Dataset):
    """Replay dataset of pre-fetched DCLM documents packed into BLT chunks.

    Structurally symmetric to :class:`VoynichEntropyDataset`: each item is a
    ``torch.long`` tensor of shape ``(max_seq_len,)`` with BLT token IDs
    (``byte + 4``) right-padded with ``PAD_ID``.
    """

    def __init__(self, docs: list[str], max_seq_len: int) -> None:
        chunks = vms_uprep.stack_lines(docs, max_bytes=max_seq_len)
        self.tokens = [_encode_chunk(chunk, max_seq_len) for chunk in chunks]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]
