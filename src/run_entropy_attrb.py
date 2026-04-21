"""
Run entropy-reduction attribution over the full Voynich BMP corpus and
persist top-K context-token attributions per target glyph to SQLite.

Pipeline:
    prepare_annotated + stack_annotated_lines  (vms_uprep)
            ↓
    annotate_entropy                           (baseline per-byte entropies)
            ↓
    attribute_chunk                            (ablation / randomization)
            ↓
    AttributionStore.write_chunk               (top-K rows → SQLite)

The underlying entropy model is a BLT entropy model whose architecture
is pulled from the HF repo `facebook/blt-entropy`; the weights come
from a PyTorch Lightning checkpoint produced by `fine_tune.replay_train`.

Run from the src/ directory, e.g.:

    uv run python run_entropy_attrb.py \
        --checkpoint /workspace/data/experiments/rft-20260419-1848-r10.0/checkpoints/epoch=009.ckpt \
        --method ablation

Computational cost scales as roughly
    (#target glyphs) × (#context tokens in window) × (one forward pass),
which is large for the full manuscript.  Use --max-chunks to bound runs
during development.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as _load_safetensors

import random as _random

import entropy_attrb
import vms_uprep
from bytelatent.transformer import LMTransformer, LMTransformerArgs
from entropy_attrb_store import AttributionStore
from entropy_proc import annotate_entropy
from vms_annot import AnnotatedChunk, Attribution, SegmentKind

HF_REPO = "facebook/blt-entropy"
BLT_BYTE_OFFSET = 4
PAD_ID = 2


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _build_sliding_window_causal_mask(
    seq_len: int, window_size: int, device: torch.device
) -> torch.Tensor:
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = (rows >= cols) & (rows - cols < window_size)
    return torch.where(mask, 0.0, float("-inf"))


def load_entropy_model(
    checkpoint_path: Optional[Path], device: torch.device
) -> tuple[LMTransformer, LMTransformerArgs]:
    """Instantiate LMTransformer from the HF config, then load weights.

    If `checkpoint_path` is provided it must be either a Lightning `.ckpt`
    (keys prefixed with `model.`) or a bare state_dict.  If None, the
    pretrained HF safetensors weights are used.
    """
    config_path = hf_hub_download(HF_REPO, "config.json")
    with open(config_path) as f:
        hf_config = json.load(f)
    args_dict = hf_config.get("args", hf_config)

    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model_args = LMTransformerArgs(**args_dict)
        model = LMTransformer(model_args)
    finally:
        torch.set_default_dtype(prev_dtype)

    if checkpoint_path is None:
        weights_path = hf_hub_download(HF_REPO, "model.safetensors")
        state_dict = _load_safetensors(weights_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_state = ckpt.get("state_dict", ckpt)
        state_dict = {}
        for k, v in raw_state.items():
            # Lightning wrapper prefixed model params with "model.".
            if k.startswith("model."):
                state_dict[k[len("model.") :]] = v
            else:
                state_dict[k] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:3]}")

    model.attn_impl = "sdpa"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_args


# ---------------------------------------------------------------------------
# entropy_fn
# ---------------------------------------------------------------------------


def make_entropy_fn(
    model: LMTransformer,
    model_args: LMTransformerArgs,
    device: torch.device,
    batch_size: int,
):
    """Return a callable mapping list[str] -> list[list[float]].

    Each output inner list has length len(text.encode("utf-8")).  Inputs
    shorter than the batch max are padded with PAD_ID at the end; because
    attention is causal, end-padding does not affect the real tokens'
    entropies.  The per-byte entropy is Shannon entropy (nats) of the
    model's next-token distribution at that position.
    """
    window = model_args.sliding_window or 512

    def entropy_fn(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        token_lists = [
            [b + BLT_BYTE_OFFSET for b in t.encode("utf-8")] for t in texts
        ]
        out: list[list[float]] = [None] * len(texts)  # type: ignore
        # Group by identical lengths where possible to avoid padding waste,
        # but a simple fixed-batch pass is sufficient for correctness.
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch = token_lists[batch_start:batch_end]
            max_len = max(len(t) for t in batch)
            padded = torch.full(
                (len(batch), max_len), PAD_ID, dtype=torch.long, device=device
            )
            for i, t in enumerate(batch):
                if t:
                    padded[i, : len(t)] = torch.tensor(t, device=device)
            mask = _build_sliding_window_causal_mask(max_len, window, device).to(
                dtype=torch.bfloat16
            )
            with torch.no_grad():
                logits = model(padded, mask=mask)
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                ent = -(probs * log_probs).sum(dim=-1)  # (batch, seq)
            ent = ent.float().cpu()
            for i, t in enumerate(batch):
                out[batch_start + i] = ent[i, : len(t)].tolist()
        return out

    return entropy_fn


# ---------------------------------------------------------------------------
# Batched attribution (spec §4.3)
# ---------------------------------------------------------------------------


def attribute_chunk_batched(
    chunk: AnnotatedChunk,
    entropy_fn,
    method: str = "ablation",
    window_bytes: int = 512,
    target_filter=None,
    rng: Optional[_random.Random] = None,
) -> list[Attribution]:
    """Per-context-token batched attribution.

    Runs exactly one perturbed forward pass per candidate context token
    (not per (target, context) pair), then extracts each target's
    perturbed entropy from the appropriate slice.  Equivalent in output
    to entropy_attrb.attribute_chunk but ~(#targets) times faster.

    Requires chunk.annotations[i].byte_entropies to be populated.
    """
    if method not in ("ablation", "randomization"):
        raise ValueError(f"Unknown method: {method!r}")
    if target_filter is None:
        target_filter = lambda a: a.kind is SegmentKind.GLYPH  # noqa: E731

    offsets = entropy_attrb._byte_offsets(chunk.annotations)
    spans = entropy_attrb.find_token_spans(chunk)

    alphabet_by_width: dict[int, list[str]] = {}
    if method == "randomization":
        if rng is None:
            rng = _random.Random()
        alphabet_by_width = entropy_attrb._chunk_glyph_alphabet_by_byte_length(chunk)

    span_by_ann_index: dict[int, object] = {}
    for span in spans:
        for i in range(span.start, span.end):
            span_by_ann_index[i] = span

    # Build one perturbed text per span.  For ablation we also record
    # the bytes removed so we can remap each target's byte offset.
    pert_texts: list[str] = []
    bytes_removed: list[int] = []
    for span in spans:
        if method == "ablation":
            ablate_start, ablate_end = entropy_attrb._ablation_region(chunk, span)
            b0, b1 = offsets[ablate_start], offsets[ablate_end]
            text_bytes = chunk.text.encode("utf-8")
            pert_texts.append((text_bytes[:b0] + text_bytes[b1:]).decode("utf-8"))
            bytes_removed.append(b1 - b0)
        else:
            pert_texts.append(
                entropy_attrb._perturb_randomization(chunk, span, alphabet_by_width, rng)
            )
            bytes_removed.append(0)

    if not pert_texts:
        return []

    pert_entropies = entropy_fn(pert_texts)
    if len(pert_entropies) != len(pert_texts):
        raise ValueError(
            f"entropy_fn returned {len(pert_entropies)} results for "
            f"{len(pert_texts)} inputs"
        )
    for pert_text, pert_bytes in zip(pert_texts, pert_entropies):
        expected = len(pert_text.encode("utf-8"))
        if len(pert_bytes) != expected:
            raise ValueError(
                f"entropy_fn returned {len(pert_bytes)} bytes for "
                f"{expected}-byte string"
            )

    results: list[Attribution] = []
    for target_idx, ann in enumerate(chunk.annotations):
        if not target_filter(ann):
            continue
        if ann.byte_entropies is None:
            raise ValueError(
                f"Annotation {target_idx} has no byte_entropies; "
                "run entropy_proc.annotate_entropy first."
            )
        baseline = float(sum(ann.byte_entropies))
        target_byte_start = offsets[target_idx]
        target_n_bytes = len(ann.char.encode("utf-8"))
        window_start = target_byte_start - window_bytes
        target_span = span_by_ann_index.get(target_idx)

        for span, removed, pert_bytes in zip(spans, bytes_removed, pert_entropies):
            if span is target_span:
                continue
            if span.byte_end > target_byte_start:
                continue
            if span.byte_start < window_start:
                continue
            new_tbs = target_byte_start - removed
            # Target is strictly after the ablated region by construction
            # (span.byte_end <= target_byte_start), so new_tbs ≥ 0.
            pert_entropy = float(sum(pert_bytes[new_tbs : new_tbs + target_n_bytes]))
            results.append(
                Attribution(
                    target_ann_index=target_idx,
                    context_token=span,
                    method=method,
                    baseline_entropy=baseline,
                    perturbed_entropy=pert_entropy,
                    delta=pert_entropy - baseline,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_corpus_df(corpus_name: str):
    from voynpy import corpora

    ref = getattr(corpora, corpus_name)
    return ref.df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/workspace/data/experiments/rft-20260419-1848-r10.0/"
            "checkpoints/epoch=009.ckpt"
        ),
        help="Lightning .ckpt file.  Pass an empty string to use the "
        "pretrained HF weights instead.",
    )
    p.add_argument(
        "--corpus",
        default="vms_unicode_bmp",
        help="voynpy.corpora attribute name (default: vms_unicode_bmp).",
    )
    p.add_argument(
        "--method",
        choices=("ablation", "randomization", "both"),
        default="ablation",
    )
    p.add_argument("--window-bytes", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument(
        "--max-bytes", type=int, default=vms_uprep.DEFAULT_MAX_BYTES,
        help="Max UTF-8 bytes per chunk when stacking annotated lines.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SQLite path (default: data/attributions/<run_id>.sqlite).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda | cpu (auto by default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path: Optional[Path] = (
        args.checkpoint if args.checkpoint and str(args.checkpoint) else None
    )
    print(f"Loading model from {ckpt_path or 'HF ' + HF_REPO} onto {device} ...")
    model, model_args = load_entropy_model(ckpt_path, device)

    entropy_fn = make_entropy_fn(model, model_args, device, args.batch_size)

    print(f"Loading corpus '{args.corpus}' ...")
    df = load_corpus_df(args.corpus)
    annotated_lines = vms_uprep.prepare_annotated(
        df, max_bytes=args.max_bytes * len(df)
    )
    chunks = vms_uprep.stack_annotated_lines(annotated_lines, max_bytes=args.max_bytes)
    total_chunks = len(chunks)
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
    print(f"Prepared {len(chunks)}/{total_chunks} chunks.")

    methods = ("ablation", "randomization") if args.method == "both" else (args.method,)

    output = args.output or Path(
        f"/workspace/data/attributions/{args.corpus}_{ckpt_run_tag(ckpt_path)}.sqlite"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing attributions to {output}")

    with AttributionStore.open(str(output)) as store:
        for method in methods:
            run_id = store.start_run(
                corpus=args.corpus,
                method=method,
                window_bytes=args.window_bytes,
                top_k=args.top_k,
                model=str(ckpt_path) if ckpt_path else HF_REPO,
                seed=args.seed,
                chunk_range=f"0..{len(chunks) - 1}",
            )
            rng = _random.Random(args.seed) if method == "randomization" else None

            t0 = time.time()
            total_rows = 0
            for chunk_id, chunk in enumerate(chunks):
                # Baseline entropies: one forward pass over the unperturbed chunk.
                baseline = entropy_fn([chunk.text])[0]
                annotate_entropy(chunk, baseline)

                attrs = attribute_chunk_batched(
                    chunk,
                    entropy_fn,
                    method=method,
                    window_bytes=args.window_bytes,
                    rng=rng,
                )
                wrote = store.write_chunk(
                    run_id, chunk_id, chunk, attrs,
                    top_k=args.top_k, corpus=args.corpus,
                )
                total_rows += wrote
                dt = time.time() - t0
                print(
                    f"[{method}] chunk {chunk_id + 1}/{len(chunks)}  "
                    f"rows={wrote}  cum_rows={total_rows}  "
                    f"elapsed={dt:.1f}s",
                    flush=True,
                )
            store.finish_run(run_id)
            print(
                f"[{method}] run_id={run_id}  total_rows={total_rows}  "
                f"elapsed={time.time() - t0:.1f}s"
            )

    print("Done.")


def ckpt_run_tag(ckpt_path: Optional[Path]) -> str:
    if ckpt_path is None:
        return "hf-pretrained"
    # e.g. .../rft-20260419-1848-r10.0/checkpoints/epoch=009.ckpt → "rft-…_epoch=009"
    run_dir = ckpt_path.parent.parent.name
    stem = ckpt_path.stem
    return f"{run_dir}_{stem}"


if __name__ == "__main__":
    main()
