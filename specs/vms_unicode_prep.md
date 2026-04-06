# VMS Unicode Preparation Pipeline

## Purpose

This document specifies the `vms_uprep` module, which prepares Voynich manuscript Unicode data for ingestion into the BLT entropy model (a byte-level transformer). It covers both the plain-text preparation path and the annotated preparation path, which preserves glyph-level provenance through downstream processing.

## Pipeline Context

The `vms_uprep` module is the first stage in a three-stage processing pipeline:

```
voynpy.corpora.vms_unicode (DataFrame)
        ↓
vms_uprep.prepare() / prepare_annotated()
        ↓
entropy_proc.annotate_entropy()
        ↓
voy_entropy_display (Jupyter rendering)
```

**Why this pipeline exists:** The BLT entropy model operates on raw byte sequences, but researchers need to trace entropy values back to their manuscript coordinates (folio, paragraph, line, token). The annotation system was introduced so that provenance metadata survives the byte-encoding round-trip and can be visualized per-glyph in Jupyter notebooks.

## Initial Data Format

The Voynich Unicode data is sourced from `voynpy.corpora.vms_unicode.df`, a pandas DataFrame with the following columns:

- `folio`: The folio and side of the manuscript page (e.g., `"1r"` for folio 1 recto, `"1v"` for folio 1 verso).
- `par`: The paragraph number on the page.
- `line`: The line number within the paragraph.
- `t1` through `t26`: Token columns. Each cell contains a "word" — a sequence of glyphs separated by commas, or occasionally a single glyph. An empty cell is indicated by `$`. These are not tokens in the NLP sense, but glyph sequences treated as units for manuscript analysis.

Most lines have far fewer than 26 tokens; trailing columns contain `$`.

## Specified Capabilities

### Comma Removal

The module removes commas from token values in `t1`–`t26`. **Why:** Commas are delimiters in the original transcription format, not part of the glyph sequences. Keeping them would corrupt the byte representation fed to the BLT model.

### Space Separation

Tokens from `t1`–`t26` are concatenated into a single string per line, with spaces separating adjacent tokens. **Why:** This reconstitutes the visual word spacing of the manuscript line while remaining a simple, unambiguous encoding for the byte-level model.

### Empty Token Handling

Tokens equal to `$` (or pandas NA) are silently skipped during concatenation. **Why:** `$` is a placeholder for absent data, not a meaningful glyph — including it would introduce spurious bytes into the model input.

### Beginning of Paragraph Marker

A pilcrow `¶` (U+00B6) is prepended to the first line of each paragraph. **Why:** This gives the model an explicit signal for paragraph boundaries, which may carry structural information about the manuscript's composition.

### End of Line Marker

A Unicode Line Separator U+2028 is appended to every line. **Why:** The model needs an unambiguous end-of-line signal that won't collide with any Voynich glyph codepoint. U+2028 is a dedicated separator character in Unicode, distinct from ASCII newlines.

### End of Paragraph Marker

A Unicode Paragraph Separator U+2029 is appended after the line separator on the last line of each paragraph. **Why:** Distinguishes paragraph-final lines from paragraph-internal lines, allowing the model to learn paragraph-level structure.

### Byte Length Budget

- The module accepts a `max_bytes` parameter (default: **8192 bytes**).
- Byte length is computed over the UTF-8 encoding of the prepared strings.
- If the total byte length of all prepared lines exceeds `max_bytes`, the module should emit a `ByteLengthWarning` (a custom `UserWarning` subclass) to alert the caller. It should **not** truncate data mid-line.
- **Why:** The BLT model has a fixed input window. Exceeding it silently would produce corrupt or truncated input; warning allows the caller to decide how to react (e.g., use `stack_lines()` to chunk the data).

### Range Selection

The module accepts an optional `range_spec` dictionary specifying the start and end of the data to prepare:

```python
{
    "start": {"folio": "1r", "par": 1, "line": 1},
    "end":   {"folio": "10v", "par": 5, "line": 10}
}
```

Both endpoints are **inclusive**. The module validates that:
- The dict contains `start` and `end` keys, each with `folio`, `par`, and `line`.
- The specified coordinates exist in the dataframe.
- The start position is not after the end position.

If `range_spec` is `None`, the entire dataframe is processed.

**Why:** The manuscript is large enough that processing it in full may exceed the byte budget. Range selection lets researchers focus on specific sections or batch the manuscript into manageable pieces.

### Output Format (Plain Text)

`prepare()` returns a **list of strings**, one per manuscript line, each containing the concatenated tokens with the appropriate paragraph/line markers described above.

### Line Stacking

`stack_lines()` accepts a list of line strings (e.g., the output of `prepare()`) and groups them into larger chunks that each fit within a byte budget (`max_bytes`, default 8192). Lines are concatenated in order; when adding the next line would exceed the limit, a new chunk begins.

Returns a **list of strings**, where each string is a concatenation of consecutive lines.

**Why:** The BLT model input window is measured in bytes, not lines. Stacking maximizes the amount of manuscript context in each inference call without exceeding the window.

## Annotation System

### Motivation

Plain string output is sufficient for model inference, but researchers need to map entropy values back to specific manuscript coordinates. The annotation system maintains a parallel array of metadata — one `GlyphAnnotation` per Unicode codepoint — so that every character in the prepared text can be traced to its origin (or identified as a structural marker).

### Dependency: `vms_annot`

The annotation data structures are defined in the `vms_annot` module. The `vms_uprep` module imports and uses the following types:

#### `SegmentKind` (Enum)

Classifies each character in the prepared text:

| Value | Meaning |
|---|---|
| `GLYPH` | A Voynich glyph with manuscript provenance |
| `SPACE` | Word-boundary space between tokens |
| `PARA_START` | The `¶` marker (U+00B6) |
| `LINE_SEP` | Line separator (U+2028) |
| `PARA_SEP` | Paragraph separator (U+2029) |

#### `GlyphAnnotation` (Dataclass)

Per-character metadata:

| Field | Type | Description |
|---|---|---|
| `kind` | `SegmentKind` | What this character represents |
| `char` | `str` | The Unicode character itself |
| `folio` | `Optional[str]` | Manuscript folio (populated only for `GLYPH`) |
| `par` | `Optional[int]` | Paragraph number (populated only for `GLYPH`) |
| `line` | `Optional[int]` | Line number (populated only for `GLYPH`) |
| `token_pos` | `Optional[int]` | 0-based index among non-null tokens on the line (populated only for `GLYPH`) |
| `byte_entropies` | `Optional[list]` | Populated downstream by `entropy_proc.annotate_entropy()` |

**Invariant:** The annotations list is always the same length as the text string (measured in Unicode codepoints), maintaining a 1:1 mapping.

#### `AnnotatedLine` (Dataclass)

A single manuscript line with full metadata:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Identical to the corresponding string from `prepare()` |
| `annotations` | `list[GlyphAnnotation]` | One annotation per codepoint in `text` |
| `folio` | `str` | Manuscript folio |
| `par` | `int` | Paragraph number |
| `line` | `int` | Line number |

#### `AnnotatedChunk` (Dataclass)

A stacked group of annotated lines:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Identical to the corresponding string from `stack_lines()` |
| `annotations` | `list[GlyphAnnotation]` | One annotation per codepoint in `text` |

### Annotated API

#### `prepare_annotated(df, range_spec=None, max_bytes=8192)`

Identical behavior to `prepare()`, but returns a list of `AnnotatedLine` objects instead of plain strings. Each `AnnotatedLine.text` is byte-for-byte identical to the corresponding string that `prepare()` would return.

**Why:** This is the entry point for the provenance-preserving path. Downstream modules (`entropy_proc`) attach per-byte entropy values to the `GlyphAnnotation.byte_entropies` field, and display modules (`voy_entropy_display`) use `SegmentKind` to render glyphs differently from structural markers.

#### `stack_annotated_lines(lines, max_bytes=8192)`

Identical chunking logic to `stack_lines()`, but operates on `AnnotatedLine` objects and returns `AnnotatedChunk` objects. Text and annotation lists are concatenated in lockstep.

**Why:** Maintains the 1:1 text-to-annotation invariant through the stacking step, so that entropy values attached during inference can still be traced back to manuscript coordinates.

## Implementation Notes

- The `prepare()` and `prepare_annotated()` paths use mirrored internal helpers (`_row_to_line` / `_row_to_annotated_line`, `_build_output` / `_build_annotated_output`) to guarantee identical text output. Any change to the formatting logic must be applied to both paths.
- The dataframe is grouped by `(folio, par)` with `sort=False` to preserve the manuscript's original ordering when applying paragraph markers.
