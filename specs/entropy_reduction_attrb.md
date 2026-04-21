# Entropy Reduction Attribution

## 1. Introduction

To determine whether a text sample contains a meaningful language or is just
gibberish, we measure the **reduction of character-level entropy** that
long-range context provides. The assumption is that, while meaningful and
gibberish text may have comparable character-level entropy in isolation (e.g.
when looking at unigram or short n-gram statistics), meaningful text should
show a significant reduction in entropy once long-range context is taken into
account, because its characters become more predictable given the surrounding
text.

As a motivating example, for the snippet `"... the ca_ ..."` the next
character (denoted `_`) has high entropy under a local model — in English,
`t`, `r`, `n`, and `s` are all plausible continuations of `"ca"`. When we
extend the context to `"My friend has a number of pets. His dog likes to
chase the ca_"`, the next character becomes much more predictable: `t` is by
far the most likely continuation.

This module quantifies that effect on a per-glyph basis and **attributes**
the reduction to specific preceding tokens, producing a signal that can be
compared across corpora (Voynich vs. natural language vs. shuffled text).

## 2. Scope and Non-Goals

**In scope**
- A Python module, `entropy_attrb.py`, that runs on `AnnotatedChunk` objects
  that already carry baseline per-byte entropies (i.e. output of
  `entropy_proc.annotate_entropy`).
- Two perturbation strategies for the preceding context: token ablation and
  glyph randomization.
- Per-glyph attribution scores indicating how much each preceding token
  contributed to reducing that glyph's entropy.

**Out of scope (for the first iteration)**
- Training or fine-tuning the entropy model. The module treats BLT as a black
  box accessed through an `entropy_fn(text) -> list[float]` callable.
- Attribution across chunk boundaries. The sliding window is clipped to the
  current `AnnotatedChunk`.
- Interactive notebook rendering. Display belongs in `voy_entropy_display.py`
  and will consume the dataclasses defined here.

## 3. Definitions

- **Target glyph** — the `GlyphAnnotation` (kind `GLYPH`) whose entropy
  reduction we are attributing.
- **Context window** — the span of preceding bytes used as context for the
  target. Default width is **512 UTF-8 bytes**, clipped at the start of the
  chunk.
- **Baseline entropy** `H₀(g)` — the per-glyph entropy already stored in
  `GlyphAnnotation.byte_entropies` for glyph `g`, computed over the
  unperturbed chunk. We summarize it to a single scalar via
  `sum(byte_entropies)` (total information in bits for the glyph).
- **Perturbed entropy** `H_p(g; t)` — the scalar entropy of `g` recomputed
  after perturbing token `t` in the preceding context.
- **Attribution score** `A(g, t) = H_p(g; t) − H₀(g)`. Positive values mean
  token `t` was *helping* to reduce `g`'s entropy; values near zero mean `t`
  was irrelevant.

## 4. Methodology

Entropy is measured at the glyph/character level; perturbations are applied
at the **token/word level**, where a token is the run of `SegmentKind.GLYPH`
annotations between two `SegmentKind.SPACE` (or line/paragraph) markers in
the `AnnotatedChunk`.

### 4.1 Token ablation

For each target glyph `g` and each token `t` in `g`'s context window:

1. Remove every UTF-8 byte of `t` (and the separating space on one side, to
   avoid double spaces) from the chunk.
2. Re-invoke the entropy model on the perturbed chunk.
3. Record the perturbed entropy `H_p(g; t)` at the target glyph's new byte
   offset and compute `A(g, t)`.

Ablation is the simplest probe: it asks "how much worse does `g` get if this
token were never there?"

### 4.2 Glyph randomization

Ablation changes the *length* of the context, which can itself shift model
predictions. An alternative perturbation preserves positional structure:

1. Replace every `SegmentKind.GLYPH` codepoint inside token `t` with a glyph
   sampled uniformly from the corpus alphabet (same alphabet as `t`'s
   codepoints — e.g. ASCII for Latin text, the Voynich EVA set for VMS).
2. Leave spaces, line separators, and paragraph markers untouched.
3. Re-run the entropy model and compute `A(g, t)` as above.

This preserves positional information while degrading the semantic content
of token `t`, isolating the contribution of that token's *identity* rather
than its mere presence.

### 4.3 Batching and efficiency

A naïve implementation runs one forward pass per (target glyph, context
token) pair. In practice:

- Perturbations only need to be evaluated once per **context token**, not
  per target; a single perturbed forward pass yields updated entropies for
  **all** downstream glyphs simultaneously.
- Multiple perturbed variants of the same chunk can be batched into one
  model call where the underlying entropy function supports batching.

The module exposes a single `entropy_fn: Callable[[list[str]], list[list[float]]]`
so the caller controls batching and device placement.

## 5. Module Interface

File: `src/entropy_attrb.py`. Import style matches the rest of `src/` —
top-level modules, not a package.

### 5.1 Dataclasses

Added to `vms_annot.py` (to keep annotation types colocated):

```python
@dataclass
class TokenSpan:
    """A contiguous run of GLYPH annotations forming a single token."""
    token_pos: int                # 0-based index within the chunk
    start: int                    # inclusive annotation index
    end: int                      # exclusive annotation index
    byte_start: int               # UTF-8 byte offset in chunk.text
    byte_end: int

@dataclass
class Attribution:
    """Attribution of one context token's contribution to one target glyph."""
    target_ann_index: int         # index into chunk.annotations
    context_token: TokenSpan
    method: str                   # "ablation" | "randomization"
    baseline_entropy: float       # H_0(g)  (bits)
    perturbed_entropy: float      # H_p(g; t)
    delta: float                  # perturbed - baseline
```

### 5.2 Public functions

```python
def find_token_spans(chunk: AnnotatedChunk) -> list[TokenSpan]: ...

def attribute_chunk(
    chunk: AnnotatedChunk,
    entropy_fn: Callable[[list[str]], list[list[float]]],
    method: Literal["ablation", "randomization"] = "ablation",
    window_bytes: int = 512,
    target_filter: Callable[[GlyphAnnotation], bool] | None = None,
    rng: random.Random | None = None,
) -> list[Attribution]:
    """Return one Attribution per (target glyph, context token) pair.

    Pre-condition: chunk.annotations[i].byte_entropies must already be
    populated (i.e. entropy_proc.annotate_entropy has been run).

    target_filter defaults to "kind == GLYPH".
    rng is only used by the randomization method; pass a seeded Random
    instance for reproducibility.
    """
```

A convenience wrapper mirrors `entropy_proc.annotate_entropy`'s ergonomics:

```python
def attribute_and_attach(
    chunk: AnnotatedChunk,
    entropy_fn,
    **kwargs,
) -> AnnotatedChunk:
    """Same as attribute_chunk, but also stores per-target attributions on
    each GlyphAnnotation as ann.attributions (list[Attribution])."""
```

### 5.3 Pipeline position

```
vms_uprep.prepare_annotated / stack_annotated_lines
        ↓
entropy_proc.annotate_entropy           (baseline entropies)
        ↓
entropy_attrb.attribute_chunk           ← new module
        ↓
voy_entropy_display / notebook rendering
```

## 6. Edge Cases and Invariants

- **Chunk start.** If the window extends before byte 0 of the chunk, it is
  clipped. Targets near the start of a chunk therefore have fewer context
  tokens available; this is expected and should not be padded or stitched
  across chunks in v1.
- **Byte-offset tracking after ablation.** Removing a token shifts the byte
  offsets of every downstream glyph. The implementation must re-map target
  annotations to their new byte positions before slicing out the perturbed
  entropy. `GlyphAnnotation.char.encode("utf-8")` gives the per-glyph byte
  length needed for this remapping.
- **Non-glyph segments in the context.** `SPACE`, `LINE_SEP`, `PARA_SEP`,
  and `PARA_START` are never ablated or randomized; they are structural and
  removing them corrupts tokenization.
- **Multi-byte glyphs.** Randomization samples new codepoints from the same
  alphabet as the token being perturbed, not a fixed byte set, so UTF-8
  encoding is preserved.
- **Empty tokens.** A token consisting only of `$` (null token) should
  already have been filtered by `vms_uprep`; assert otherwise.

## 7. Configuration and Defaults

| Parameter         | Default       | Notes                                              |
|-------------------|---------------|----------------------------------------------------|
| `window_bytes`    | 512           | Preceding UTF-8 bytes considered as context        |
| `method`          | `"ablation"`  | `"ablation"` or `"randomization"`                  |
| `target_filter`   | GLYPH-only    | Callable over `GlyphAnnotation`                    |
| `rng` seed        | None          | Must be set for reproducible randomization         |

## 8. Persistence

Re-running attribution is expensive (one forward pass per perturbed context
token), so results need to be persisted and re-loaded for analysis and
display. This section covers what to store, how much it is, and the format
trade-offs.

### 8.1 What to store

Per (chunk, target glyph, method) we persist **only the top-K context
tokens by `|delta|`**, where `K = 10` by default. Keeping the top-K rather
than the full attribution matrix is the key size-control lever: the context
window holds roughly 100 tokens (512 bytes ÷ ~5 bytes/token), so top-10
drops storage by ~10×.

Each stored record carries enough provenance to join back to the original
chunk without re-deriving it:

| Field              | Type      | Notes                                         |
|--------------------|-----------|-----------------------------------------------|
| `run_id`           | TEXT      | UUID per attribution run; ties to config      |
| `corpus`           | TEXT      | e.g. `"vms"`, `"latin"`, `"german"`           |
| `chunk_id`         | INTEGER   | Position of chunk in its source document      |
| `folio`            | TEXT      | Target glyph's manuscript folio (nullable)    |
| `par`, `line`      | INTEGER   | Target glyph's paragraph / line (nullable)    |
| `token_pos`        | INTEGER   | Target glyph's token position on the line     |
| `target_ann_index` | INTEGER   | Index into `chunk.annotations`                |
| `target_char`      | TEXT      | The glyph itself (convenience for analysis)   |
| `baseline_entropy` | REAL      | `H_0(g)` in bits                              |
| `method`           | TEXT      | `"ablation"` or `"randomization"`             |
| `rank`             | INTEGER   | 0–9 within the target (0 = largest `|delta|`) |
| `ctx_token_pos`    | INTEGER   | Context token's `TokenSpan.token_pos`         |
| `ctx_token_text`   | TEXT      | Context token's glyphs (for human inspection) |
| `ctx_byte_offset`  | INTEGER   | Byte distance from target to context token    |
| `perturbed_entropy`| REAL      | `H_p(g; t)` in bits                           |
| `delta`            | REAL      | `perturbed − baseline`                        |

A sidecar `runs` table records the `run_id`, UTC timestamp, corpus, chunk
range, `method`, `window_bytes`, `top_k`, model identifier, and RNG seed so
a row's provenance is reproducible.

### 8.2 Scale estimate

Rough upper bound for the full Voynich manuscript:

- ~170 k target glyphs × 2 methods × 10 top-K rows ≈ **3.4 M rows**.
- Row payload is ~80 bytes (mostly numeric + a short `ctx_token_text`).
- Raw: ~270 MB. Compressed (Parquet/zstd or SQLite with small-string reuse):
  **50–100 MB per full-manuscript run**.

Per-chunk runs during development are vastly smaller (a few thousand rows)
and fit comfortably in memory.

### 8.3 Format options considered

**SQLite** (recommended for v1)
- Single file under `data/attributions/<run_id>.sqlite`, stdlib-only.
- Natural fit for ad-hoc queries from the notebook:
  `SELECT ctx_token_text, delta FROM attributions WHERE target_ann_index=? ORDER BY rank`.
- Index `(run_id, chunk_id, target_ann_index, method)` covers the hot path
  used by `voy_entropy_display`.
- Appends are cheap; multiple partial runs can be merged with
  `ATTACH DATABASE`.
- Downside: row storage, less compact than columnar for whole-table scans.

**Parquet** (recommended for analytics / export)
- One file per run, or partitioned by `corpus/method/chunk_id`.
- Excellent compression (zstd) and pandas/polars integration.
- Downside: not append-friendly — partial runs accumulate as many small
  files, and point queries require loading a row group.
- Good "freeze" format: dump a completed SQLite run to Parquet for
  long-term storage or cross-corpus analysis.

**HDF5**
- Fine for dense numeric arrays (e.g. a full `n_targets × n_context` matrix
  if we ever store un-truncated attributions), less natural for the
  row-shaped top-K output. Not adopted in v1.

**JSONL**
- Trivial to append and diff. Size grows ~3× vs. SQLite for this schema.
- Useful for tiny debug dumps; not the primary format.

**Pickle**
- Avoided — not portable across refactors, breaks on dataclass changes.

### 8.4 Proposed v1 design

- Module `entropy_attrb_store.py` with `AttributionStore` wrapping a SQLite
  connection.
- Two tables: `runs` and `attributions` (schema above).
- Writer API:
  ```python
  store = AttributionStore.open("data/attributions/vms_2026-04-19.sqlite")
  run_id = store.start_run(corpus="vms", method="ablation", window_bytes=512,
                           top_k=10, model="blt-entropy", seed=42)
  store.write_chunk(run_id, chunk_id, attributions)  # top-K already selected
  store.finish_run(run_id)
  ```
- Reader API returns a pandas DataFrame for notebook use:
  ```python
  df = store.load(run_id=..., folio="1r")
  ```
- Exporter: `store.to_parquet(run_id, path)` for sharing / archiving.
- Selecting top-K happens in the attribution module before handing rows to
  the store, so the store never sees the full matrix.

## 9. Testing

Under `tests/test_entropy_attrb.py`:

1. **Token-span extraction.** On a small hand-built `AnnotatedChunk`, verify
   `find_token_spans` returns correct indices and byte ranges, including
   tokens adjacent to `PARA_START` and `LINE_SEP`.
2. **Ablation round-trip.** With a stub `entropy_fn` that returns a
   deterministic function of input length, verify that ablating a
   single-token context produces the expected perturbed text and that
   baseline/perturbed slicing aligns to the right target glyph.
3. **Randomization determinism.** Same `rng` seed → same perturbed strings;
   alphabet of randomized glyphs is a subset of the alphabet of the source
   token.
4. **Window clipping.** Targets within the first `window_bytes` bytes of the
   chunk yield fewer attributions than later targets.
5. **Invariant check.** `len(chunk.text.encode("utf-8"))` matches the
   length expected by `entropy_fn` on every perturbed variant.

## 9. Open Questions

- **Summarizing per-byte entropy.** `sum(byte_entropies)` is simple but
  biases toward multi-byte glyphs. Alternatives: first-byte only, or mean.
  Decide once we see how noisy the attributions are.
- **Context tokens beyond 512 bytes.** Do we ever want a longer window, or
  a *decayed* weighting across position? Left for v2.
- **Comparability across corpora.** Voynich, Latin, and German each have
  different token-length distributions; raw `A(g, t)` may not be directly
  comparable. A per-corpus normalization (e.g. z-score against shuffled
  baseline) may be necessary for the eventual meaningful-vs-gibberish
  classifier.
