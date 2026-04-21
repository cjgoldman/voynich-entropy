"""
Entropy-reduction attribution for AnnotatedChunks.

For each target glyph g, we perturb each token t in g's preceding
context window, rerun the entropy model, and record how much t was
contributing to g's entropy reduction:

    A(g, t) = H_p(g; t) − H_0(g)

Positive values mean t was helping (removing or corrupting t made
the model less certain about g).  See specs/entropy_reduction_attrb.md
for the full design.

Pre-condition: chunk.annotations[i].byte_entropies must already be
populated by entropy_proc.annotate_entropy().

The entropy_fn callable is batched:
    entropy_fn(texts: list[str]) -> list[list[float]]
where len(result[i]) == len(texts[i].encode("utf-8")) for each i.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable, Literal, Optional

from vms_annot import (
    AnnotatedChunk,
    Attribution,
    GlyphAnnotation,
    SegmentKind,
    TokenSpan,
)

EntropyFn = Callable[[list[str]], list[list[float]]]


def _byte_offsets(annotations: list[GlyphAnnotation]) -> list[int]:
    """Return cumulative UTF-8 byte offsets, length = len(annotations) + 1.

    offsets[i] is the byte offset of annotation i in the reconstructed
    chunk text; offsets[-1] is the total byte length.
    """
    offsets = [0]
    cursor = 0
    for ann in annotations:
        cursor += len(ann.char.encode("utf-8"))
        offsets.append(cursor)
    return offsets


def find_token_spans(chunk: AnnotatedChunk) -> list[TokenSpan]:
    """Return one TokenSpan per maximal run of GLYPH annotations.

    Structural annotations (SPACE, LINE_SEP, PARA_SEP, PARA_START) split
    tokens but are not themselves tokens.  token_pos is numbered within
    the chunk, starting at 0.
    """
    offsets = _byte_offsets(chunk.annotations)
    anns = chunk.annotations
    n = len(anns)
    spans: list[TokenSpan] = []
    i = 0
    tok_pos = 0
    while i < n:
        if anns[i].kind is not SegmentKind.GLYPH:
            i += 1
            continue
        start = i
        while i < n and anns[i].kind is SegmentKind.GLYPH:
            i += 1
        end = i
        spans.append(
            TokenSpan(
                token_pos=tok_pos,
                start=start,
                end=end,
                byte_start=offsets[start],
                byte_end=offsets[end],
            )
        )
        tok_pos += 1
    return spans


def _chunk_glyph_alphabet_by_byte_length(
    chunk: AnnotatedChunk,
) -> dict[int, list[str]]:
    """Unique GLYPH codepoints in the chunk, bucketed by UTF-8 byte length.

    Randomization samples from the bucket matching the original glyph's
    byte width so the perturbed string has the same total byte length
    as the original — which is the invariant stated in the spec
    (section 6, "Multi-byte glyphs") and what `new_tbs = target_byte_start`
    in attribute_chunk relies on.
    """
    buckets: dict[int, set[str]] = {}
    for a in chunk.annotations:
        if a.kind is SegmentKind.GLYPH:
            buckets.setdefault(len(a.char.encode("utf-8")), set()).add(a.char)
    return {k: sorted(v) for k, v in buckets.items()}


def _ablation_region(chunk: AnnotatedChunk, span: TokenSpan) -> tuple[int, int]:
    """Return (ann_start, ann_end) of the annotation range to remove.

    Removes the token plus one separating SPACE.  Left-side space is
    preferred; falls back to right-side when the token is the first
    non-structural run (e.g. directly after PARA_START).
    """
    anns = chunk.annotations
    ablate_start = span.start
    ablate_end = span.end
    left_is_space = span.start > 0 and anns[span.start - 1].kind is SegmentKind.SPACE
    right_is_space = span.end < len(anns) and anns[span.end].kind is SegmentKind.SPACE
    if left_is_space:
        ablate_start = span.start - 1
    elif right_is_space:
        ablate_end = span.end + 1
    return ablate_start, ablate_end


def _perturb_ablation(
    chunk: AnnotatedChunk,
    span: TokenSpan,
    target_ann_index: int,
    offsets: list[int],
) -> tuple[str, int]:
    """Return (perturbed_text, new_target_byte_start)."""
    ablate_start, ablate_end = _ablation_region(chunk, span)
    b0, b1 = offsets[ablate_start], offsets[ablate_end]
    text_bytes = chunk.text.encode("utf-8")
    perturbed = (text_bytes[:b0] + text_bytes[b1:]).decode("utf-8")
    target_byte_start = offsets[target_ann_index]
    # target must be after the ablated region — caller guarantees this by
    # selecting only preceding-context tokens.
    assert target_byte_start >= b1, (
        f"target byte offset {target_byte_start} is inside ablated region "
        f"[{b0}, {b1})"
    )
    return perturbed, target_byte_start - (b1 - b0)


def _perturb_randomization(
    chunk: AnnotatedChunk,
    span: TokenSpan,
    alphabet_by_width: dict[int, list[str]],
    rng: random.Random,
) -> str:
    """Replace each GLYPH codepoint inside the span with a same-width sample.

    Each glyph is resampled from the subset of the chunk's alphabet that
    has the same UTF-8 byte length, preserving total byte length so
    downstream byte-offset arithmetic remains valid.
    """
    chars = list(chunk.text)
    for i in range(span.start, span.end):
        width = len(chars[i].encode("utf-8"))
        bucket = alphabet_by_width.get(width)
        if bucket:
            chars[i] = rng.choice(bucket)
        # else: no same-width alphabet entry; leave the glyph untouched
        # (happens only if this glyph is the sole codepoint of its width
        # in the chunk).
    return "".join(chars)


def _default_target_filter(ann: GlyphAnnotation) -> bool:
    return ann.kind is SegmentKind.GLYPH


def attribute_chunk(
    chunk: AnnotatedChunk,
    entropy_fn: EntropyFn,
    method: Literal["ablation", "randomization"] = "ablation",
    window_bytes: int = 512,
    target_filter: Optional[Callable[[GlyphAnnotation], bool]] = None,
    rng: Optional[random.Random] = None,
) -> list[Attribution]:
    """Return one Attribution per (target glyph, context token) pair.

    Args:
        chunk: AnnotatedChunk whose annotations already carry byte_entropies.
        entropy_fn: Batched callable. Given a list of strings, returns a
            parallel list of per-byte entropy lists.  Each inner list
            must have length len(text.encode("utf-8")).
        method: "ablation" removes the context token (plus a separating
            space); "randomization" replaces its GLYPH codepoints with
            samples from the chunk's glyph alphabet.
        window_bytes: UTF-8 bytes of strictly preceding context considered.
        target_filter: Predicate selecting target annotations.  Defaults
            to kind == GLYPH.
        rng: Used by "randomization" only.  Pass a seeded Random for
            reproducibility.

    Returns:
        List of Attribution records.  For each target glyph, at most one
        record per context token is produced.
    """
    if method not in ("ablation", "randomization"):
        raise ValueError(f"Unknown method: {method!r}")
    if target_filter is None:
        target_filter = _default_target_filter

    offsets = _byte_offsets(chunk.annotations)
    spans = find_token_spans(chunk)

    alphabet_by_width: dict[int, list[str]] = {}
    if method == "randomization":
        if rng is None:
            rng = random.Random()
        alphabet_by_width = _chunk_glyph_alphabet_by_byte_length(chunk)

    # Map a GLYPH annotation to its owning TokenSpan so we can skip
    # attributing a token to one of its own glyphs.
    span_by_ann_index: dict[int, TokenSpan] = {}
    for span in spans:
        for i in range(span.start, span.end):
            span_by_ann_index[i] = span

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

        context_tokens = [
            s
            for s in spans
            if s.byte_end <= target_byte_start
            and s.byte_start >= window_start
            and s is not target_span
        ]
        if not context_tokens:
            continue

        # Build perturbed variants in one batch per target.
        variants: list[tuple[TokenSpan, str, int]] = []
        for span in context_tokens:
            if method == "ablation":
                pert_text, new_tbs = _perturb_ablation(chunk, span, target_idx, offsets)
            else:
                pert_text = _perturb_randomization(chunk, span, alphabet_by_width, rng)
                new_tbs = target_byte_start  # length preserved by same-width sampling
            variants.append((span, pert_text, new_tbs))

        pert_texts = [v[1] for v in variants]
        pert_entropies = entropy_fn(pert_texts)
        if len(pert_entropies) != len(pert_texts):
            raise ValueError(
                f"entropy_fn returned {len(pert_entropies)} results for "
                f"{len(pert_texts)} inputs"
            )

        for (span, pert_text, new_tbs), pert_bytes in zip(variants, pert_entropies):
            expected_len = len(pert_text.encode("utf-8"))
            if len(pert_bytes) != expected_len:
                raise ValueError(
                    f"entropy_fn returned {len(pert_bytes)} bytes for "
                    f"{expected_len}-byte string"
                )
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


def attribute_and_attach(
    chunk: AnnotatedChunk,
    entropy_fn: EntropyFn,
    **kwargs,
) -> AnnotatedChunk:
    """Run attribute_chunk and store per-target results on each GlyphAnnotation.

    After this call, chunk.annotations[i].attributions is either a list
    of Attribution records (for targets with at least one context token)
    or None (when the target had no in-window context).
    """
    attributions = attribute_chunk(chunk, entropy_fn, **kwargs)
    by_target: dict[int, list[Attribution]] = {}
    for a in attributions:
        by_target.setdefault(a.target_ann_index, []).append(a)
    for idx, attrs in by_target.items():
        chunk.annotations[idx].attributions = attrs
    return chunk


def select_top_k(
    attributions: Iterable[Attribution],
    k: int = 10,
) -> list[tuple[int, Attribution]]:
    """Keep the top-K Attributions per (target, method) by |delta|.

    Returns a list of (rank, Attribution) pairs where rank 0 is the
    largest |delta| within each (target_ann_index, method) group.
    The returned list is not grouped; callers that need grouping can
    partition by attribution.target_ann_index and attribution.method.
    """
    groups: dict[tuple[int, str], list[Attribution]] = {}
    for a in attributions:
        groups.setdefault((a.target_ann_index, a.method), []).append(a)
    ranked: list[tuple[int, Attribution]] = []
    for attrs in groups.values():
        attrs.sort(key=lambda x: abs(x.delta), reverse=True)
        for rank, a in enumerate(attrs[:k]):
            ranked.append((rank, a))
    return ranked
