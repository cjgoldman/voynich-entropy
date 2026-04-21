"""
Render a per-glyph bar chart of top-1 attribution token distances for the
first paragraph of folio 1r.

For each target glyph g on folio 1r, par 1, we look up its top-1
(largest |delta|) entropy-reduction attribution in the SQLite store and
compute the distance, in whole tokens, between g's token and the
contributing context token.  The chart draws one bar per glyph with the
token distance as the bar height.

Run from /workspace/src/:
    uv run python plot_token_distance_1r.py \
        --db /workspace/data/attributions/vms_unicode_bmp_rft-20260419-1848-r10.0_epoch=009.sqlite \
        --out /workspace/data/attributions/token_distance_1r_par1.png
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from entropy_attrb import find_token_spans
from vms_annot import SegmentKind
from voy_entropy_display import display_token_distance_plot
import vms_uprep


DEFAULT_DB = Path(
    "/workspace/data/attributions/"
    "vms_unicode_bmp_rft-20260419-1848-r10.0_epoch=009.sqlite"
)
DEFAULT_OUT = Path(
    "/workspace/data/attributions/token_distance_1r_par1.png"
)


def build_chunk0(corpus_name: str = "vms_unicode_bmp"):
    """Rebuild chunk 0 of the corpus exactly as run_entropy_attrb does."""
    from voynpy import corpora

    df = getattr(corpora, corpus_name).df
    lines = vms_uprep.prepare_annotated(df, max_bytes=vms_uprep.DEFAULT_MAX_BYTES * len(df))
    chunks = vms_uprep.stack_annotated_lines(lines, max_bytes=vms_uprep.DEFAULT_MAX_BYTES)
    return chunks[0]


def par1_slice(chunk):
    """Return (start, end_exclusive) annotation indices covering 1r par 1.

    Includes trailing structural markers (LINE_SEP, PARA_SEP) that belong
    to the paragraph so the rendered glyph strip shows full line breaks.
    """
    anns = chunk.annotations
    first = None
    last = None
    for i, a in enumerate(anns):
        if a.kind is SegmentKind.GLYPH and a.folio == "1r" and a.par == 1:
            if first is None:
                first = i
            last = i
    if first is None:
        raise RuntimeError("No 1r par 1 glyphs in chunk 0")
    # Extend one index back to include the PARA_START marker if present.
    if first > 0 and anns[first - 1].kind is SegmentKind.PARA_START:
        first -= 1
    # Extend forward across any structural markers belonging to this para.
    end = last + 1
    while end < len(anns) and anns[end].kind in (
        SegmentKind.LINE_SEP, SegmentKind.PARA_SEP
    ):
        end += 1
    return first, end


def load_top1_distances(
    db_path: Path, chunk
) -> tuple[dict[int, int], dict[int, float]]:
    """Return (distances, deltas) maps keyed by target_ann_index for rank-0 rows.

    Distance is target's chunk-level token_pos minus the context token's
    token_pos (positive because context precedes the target).  Delta is
    the entropy reduction that the top context token contributed
    (perturbed_entropy − baseline_entropy; positive means the token was
    helping the model).
    """
    spans = find_token_spans(chunk)
    # Map annotation index -> chunk-level token_pos
    ann_to_chunk_token: dict[int, int] = {}
    for span in spans:
        for i in range(span.start, span.end):
            ann_to_chunk_token[i] = span.token_pos

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT target_ann_index, ctx_token_pos, delta FROM attributions "
        "WHERE folio='1r' AND par=1 AND method='ablation' AND rank=0"
    ).fetchall()
    conn.close()

    distances: dict[int, int] = {}
    deltas: dict[int, float] = {}
    for target_ann_index, ctx_token_pos, delta in rows:
        target_tok = ann_to_chunk_token.get(target_ann_index)
        if target_tok is None:
            continue
        distances[target_ann_index] = target_tok - ctx_token_pos
        deltas[target_ann_index] = float(delta)
    return distances, deltas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--corpus", default="vms_unicode_bmp")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1

    print(f"Rebuilding chunk 0 from corpus '{args.corpus}' ...")
    chunk = build_chunk0(args.corpus)

    start, end = par1_slice(chunk)
    par_anns = chunk.annotations[start:end]
    par_text = "".join(a.char for a in par_anns)
    print(f"1r par 1: {end - start} chars, {len(par_text.encode('utf-8'))} bytes")

    print("Loading top-1 attributions ...")
    distances_by_ann, deltas_by_ann = load_top1_distances(args.db, chunk)

    token_distances = []
    token_deltas = []
    for i, a in enumerate(par_anns):
        absolute = start + i
        token_distances.append(distances_by_ann.get(absolute))
        token_deltas.append(deltas_by_ann.get(absolute))

    n_bars = sum(1 for d in token_distances if d is not None)
    print(f"Glyphs with attributions: {n_bars}/{len(par_anns)}")

    from vms_annot import AnnotatedChunk
    par_chunk = AnnotatedChunk(text=par_text, annotations=par_anns)

    fig = display_token_distance_plot(
        token_distances,
        text=par_text,
        chunk=par_chunk,
        dpi=args.dpi,
        deltas=token_deltas,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=args.dpi)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
