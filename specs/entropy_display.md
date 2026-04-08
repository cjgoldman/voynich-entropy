# Entropy Display Pipeline

## Purpose

This document specifies the entropy display pipeline, which renders per-byte or per-glyph entropy values in Jupyter notebooks depending on the selected display mode (`"byte"` or `"glyph"`). The pipeline takes annotated manuscript chunks with attached entropy values and produces three visual outputs: an HTML per-byte entropy table with Voynich Unicode glyphs, a summary statistics panel, and a matplotlib line plot with optional manuscript metadata bands. It spans four modules: `entropy_proc`, `entropy_plot`, `voy_entropy_display`, and `voy_font`.

## Pipeline Context

The entropy display pipeline is the final stage in a four-stage processing pipeline:

```
voynpy.corpora.vms_unicode (DataFrame)
        ↓
vms_uprep.prepare_annotated()  →  AnnotatedLine[]
        ↓
vms_uprep.stack_annotated_lines()  →  AnnotatedChunk[]
        ↓
BLT entropy model inference  →  per-byte entropy floats
        ↓
entropy_proc.annotate_entropy()  →  AnnotatedChunk with byte_entropies
        ↓
voy_entropy_display  →  Jupyter rendering (table + summary + plot)
                        [mode="byte"|"glyph"]
```

**Why this pipeline exists:** The BLT entropy model produces raw per-byte entropy floats, but researchers need to visualize those values in the context of the manuscript's structure — seeing which glyphs, tokens, lines, and folios correspond to high or low entropy regions. The display pipeline bridges the gap between numerical model output and human-interpretable visualization.

## Dependencies

### `vms_annot`

Provides the annotation data structures that carry manuscript provenance through the pipeline. The display modules consume `AnnotatedChunk`, `GlyphAnnotation`, and `SegmentKind` to map byte positions back to manuscript coordinates and structural markers.

### `voy_font`

Provides Voynich Unicode font loading for Jupyter notebooks via base64-encoded CSS `@font-face` injection.

### `entropy_plot`

Provides the generic (non-Voynich-specific) plotting engine: dataclasses for band and font specifications, the core `plot_entropy()` matplotlib function, entropy summary display, and shared helper functions for color mapping, bar rendering, and glyph grouping.

## Data Structures

### `EntropyMode` (String Literal, `entropy_plot`)

Controls whether entropy is displayed at byte or glyph granularity. Accepted values are `"byte"` (default) and `"glyph"`.

- **`"byte"`**: One data point per UTF-8 byte. This is the existing behavior — the table shows one row per byte, the plot has one x-axis position per byte, and summary statistics are computed over byte-level entropy values.
- **`"glyph"`**: One data point per Unicode character. Multi-byte characters are collapsed into a single value by **summing the log-probabilities** (entropy values) of all bytes belonging to the glyph. Single-byte characters are unchanged. The table shows one row per glyph, the plot has one x-axis position per glyph, and summary statistics are computed over glyph-level entropy values.

**Why:** The BLT entropy model produces per-byte values, but researchers often want to reason about entropy at the character level — especially for Voynich glyphs that encode as 3-byte UTF-8 sequences. Byte-level display can fragment a single glyph's contribution across multiple rows or plot positions, making it harder to identify which *characters* are surprising. Glyph mode provides a natural unit of analysis that aligns with the manuscript's symbol inventory.

### `BandSpan` (Dataclass, `entropy_plot`)

A single contiguous region within a metadata band. In byte mode, positions are byte indices; in glyph mode, positions are glyph (character) indices:

| Field | Type | Description |
|---|---|---|
| `start` | `int` | Start position (bytes in byte mode, glyph index in glyph mode) |
| `width` | `int` | Width (bytes in byte mode, glyph count in glyph mode) |
| `text` | `str` | Short label rendered inside the span |

### `BandSpec` (Dataclass, `entropy_plot`)

A full metadata band (one horizontal row beneath the plot):

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Y-axis label (e.g., "Folio", "Line") |
| `color` | `str` | Hex color code (e.g., `"#0ea5e9"`) |
| `spans` | `list[BandSpan]` | Contiguous spans within the band (default empty) |

### `FontSpec` (Dataclass, `entropy_plot`)

Custom font specification for glyph labels on the x-axis:

| Field | Type | Description |
|---|---|---|
| `font_properties` | `Any` | Matplotlib `FontProperties` object |
| `char_predicate` | `Callable[[str], bool]` | Returns `True` for characters that should use this font |
| `font_size` | `float` | Font size in points (default `12.0`) |

### `GlyphShadingRule` (Dataclass, `entropy_plot`)

Defines background shading for specific character categories in the plot:

| Field | Type | Description |
|---|---|---|
| `chars` | `set[str]` | Set of characters this rule matches |
| `color` | `str` | Background color for matching characters |
| `alpha` | `float` | Opacity of the background shading |
| `legend_label` | `str` | Label shown in the plot legend |

## Specified Capabilities

### Voynich Font Loading

The `voy_font.load_voynich_font()` function loads the Voynich Unicode TrueType font (`BMPVoynichUnicode.ttf`) into Jupyter notebooks by encoding the font file as base64 and injecting a CSS `@font-face` rule via `IPython.display.HTML`. Loaded fonts are tracked in a module-level set to prevent duplicate injection.

**Why:** Voynich manuscript glyphs reside in the Unicode Private Use Area (U+E000–U+F8FF). Standard system fonts do not contain these codepoints, so the custom font must be embedded directly into the notebook's rendering context for glyphs to display correctly.

### Glyph Entropy Aggregation

`_aggregate_glyph_entropies()` in `entropy_plot` converts byte-level entropy values to glyph-level values using a text string and its corresponding byte entropies. It calls `_build_glyph_groups()` to decompose the text into `(char, byte_start, num_bytes)` tuples, then for each glyph, sums the entropy values for bytes `byte_start` through `byte_start + num_bytes`. The result is a list of floats with one entry per Unicode character.

**Why:** The BLT model produces per-byte cross-entropy values (non-negative, in nats). Summing cross-entropy over the bytes of a multi-byte character yields the total cross-entropy of the character — this is the mathematically correct aggregation because, under the autoregressive model, the total negative log-probability of a character equals the sum of the negative log-probabilities of its constituent bytes. The result is non-negative and increases for longer or more surprising glyphs, preserving the expected color-ramp direction (green = low, red = high).

### Entropy Color Ramp

Entropy values are mapped to colors using a five-stop linear interpolation ramp:

| Position | Color | Hex |
|---|---|---|
| 0.00 | Green | `#22c55e` |
| 0.25 | Lime | `#84cc16` |
| 0.50 | Yellow | `#eab308` |
| 0.75 | Orange | `#f97316` |
| 1.00 | Red | `#ef4444` |

The position is computed by normalizing the entropy value within the observed min/max range: `t = (value - min) / (max - min)`. RGB channels are linearly interpolated between adjacent stops. When all values are equal (min == max), a neutral slate gray (`#94a3b8`) is returned.

**Why:** Color-coding makes entropy patterns visible at a glance — low-entropy (predictable) bytes appear green, high-entropy (surprising) bytes appear red. The five-stop ramp provides smooth gradation without abrupt color transitions that could mislead interpretation.

### Per-Byte Entropy Table (HTML)

`display_entropy_table()` accepts a `mode` keyword argument (`"byte"` or `"glyph"`, default `"byte"`). In byte mode, it renders an HTML table in Jupyter with one row per byte of the input text. Each row contains:

- **Pos**: 0-based byte position.
- **Byte**: Raw byte value (0–255).
- **Glyph**: The Unicode character rendered in Voynich font, using `rowspan` to span all bytes belonging to the same multi-byte character.
- **Entropy**: Numeric entropy value, color-coded using the entropy color ramp.
- **Distribution**: A filled bar showing the entropy value relative to the maximum, color-coded to match.

Table rows are grouped by Unicode character, with alternating background bands (transparent / dark slate) to visually distinguish multi-byte glyph boundaries. A 1px border appears at the start of each new glyph group.

An optional header above the table renders the full input text in Voynich font on a dark background.

When `mode="glyph"`, the table changes to one row per Unicode character:

- **Pos**: 0-based glyph position (not byte position).
- **Glyph**: The Unicode character rendered in Voynich font (no rowspan needed since each row is one glyph).
- **Entropy**: The summed entropy for the glyph's bytes, color-coded using the entropy color ramp.
- **Distribution**: A filled bar showing the glyph entropy relative to the maximum glyph entropy.
- The **Byte** column is omitted in glyph mode.

Alternating background bands alternate per glyph. The glyph-group border styling is not needed since each row is already one glyph.

**Why:** The per-byte table is the primary analytical view. Researchers need to see exactly how the model's uncertainty distributes across the raw byte encoding of each glyph. The rowspan grouping prevents visual fragmentation of multi-byte characters, and the alternating bands make it easy to track which bytes belong to which glyph. Glyph mode provides a more compact view when byte-level detail is not needed — particularly useful for longer text spans where the byte table becomes unwieldy.

### Glyph Grouping

`_build_glyph_groups()` decomposes a Unicode string into a list of `(character, byte_start, num_bytes)` tuples. This mapping is used both by the HTML table (for rowspan and alternating bands) and by the matplotlib plot (for x-axis labels and background shading).

**Why:** Voynich Unicode glyphs encode to multi-byte UTF-8 sequences (typically 3 bytes each, since they reside in the PUA). The byte-level entropy model produces one value per byte, but the display must group these bytes back into their source characters for human readability.

### Entropy Bar

`_bar_html()` generates an inline HTML bar chart element: a dark container div with a colored fill div whose width is proportional to `value / max_val`. The fill color matches the entropy color ramp for that value.

**Why:** The bar provides a quick visual comparison of relative entropy magnitude across rows without requiring the reader to parse numeric values.

### Summary Statistics

`display_entropy_summary()` accepts `text` (required when `mode="glyph"`) and a `mode` keyword argument (`"byte"` or `"glyph"`, default `"byte"`). In glyph mode, entropy values are first aggregated to glyph level via `_aggregate_glyph_entropies()` before computing statistics. It renders an inline HTML panel showing four metrics:

- **Mean** entropy (color-coded).
- **Max** entropy (color-coded).
- **Min** entropy (color-coded).
- **Count**: total number of bytes (in byte mode) or glyphs (in glyph mode).

Each numeric value is color-mapped using the entropy color ramp relative to the observed min/max range.

When `mode="glyph"`, all statistics are computed over glyph-level entropy values (summed byte entropies per character). The count label reflects the number of glyphs rather than bytes.

**Why:** Summary statistics give researchers a quick overview of the entropy distribution before examining individual bytes or glyphs. The color coding provides immediate visual context for whether the mean is closer to the predictable or surprising end of the range.

### Matplotlib Entropy Line Plot

`plot_entropy()` accepts a `mode` keyword argument (`"byte"` or `"glyph"`, default `"byte"`). When `mode="glyph"` and `text` is provided, entropy values are aggregated to glyph level before plotting. It renders a matplotlib figure with:

1. **Colored line segments**: The entropy curve is drawn as consecutive line segments, each colored by the average entropy of its two endpoints using a green–yellow–red colormap. Point markers are overlaid at each byte position.

2. **Background shading**: The plot background is divided into vertical bands, one per Unicode character. If a `GlyphShadingRule` matches the character, that band uses the rule's color and alpha. Otherwise, alternating gray bands provide visual grouping.

3. **Glyph x-axis labels**: When `text` is provided, character labels are placed on the x-axis at positions determined by the mode — byte-aligned in byte mode, glyph-aligned in glyph mode. Characters matching the `FontSpec.char_predicate` are rendered in the custom font (Voynich); others use the default font.

4. **Auto-sized figure**: When `figsize` is not specified, the figure width is computed from the number of glyphs and a target density (`glyphs_per_inch`, default 4), with a minimum of 6 inches.

When `mode="glyph"`, the plot operates on glyph-level entropy values: one x-axis position per Unicode character, with the entropy value being the sum of byte entropies for that character. Background shading spans are one unit wide (one glyph) regardless of the character's byte count. The x-axis label for each position is the character itself (same as byte mode but without the multi-byte spanning). Auto-sizing uses the glyph count directly.

**Why:** The line plot reveals entropy trends and patterns across the text that are difficult to spot in a table. Colored segments make high-entropy regions pop visually, background shading ties the plot back to individual glyphs, and auto-sizing prevents labels from overlapping in short or long texts. In glyph mode, the plot provides a more natural view where each x-axis position corresponds to one character, avoiding the visual stretching that occurs when 3-byte PUA glyphs occupy three positions on the byte-level plot.

### Voynich Glyph Shading Rules

The `voy_entropy_display` module defines three `GlyphShadingRule` entries for the Voynich context:

| Characters | Color | Alpha | Legend Label |
|---|---|---|---|
| Space, Tab | `#22c55e` (green) | 0.12 | Space / Tab |
| `\n`, `\r`, U+2028, U+2029 | `#3b82f6` (blue) | 0.12 | Line Break |
| `¶` (U+00B6) | `#a855f7` (purple) | 0.12 | Paragraph (¶) |

**Why:** Structural separator characters (spaces, line breaks, paragraph markers) behave very differently from manuscript glyphs in the entropy model. Shading them distinctly lets researchers immediately see whether entropy spikes or dips correspond to actual glyph content or to formatting artifacts.

### Voynich Font Spec

`_voynich_font_spec()` wraps the Voynich matplotlib `FontProperties` as a `FontSpec` with a `char_predicate` that matches characters in the Private Use Area (U+E000–U+F8FF). This tells the plot renderer which characters need the custom font.

**Why:** The plot must mix two fonts — Voynich for PUA glyphs and the default font for structural markers like spaces and paragraph symbols. The predicate-based approach keeps font selection logic generic in `entropy_plot` while letting `voy_entropy_display` supply the Voynich-specific rule.

### Metadata Band Rendering

When an `AnnotatedChunk` is provided, the plot includes horizontal metadata bands below the main entropy curve. Four bands are rendered in order (top to bottom):

| Band | Label | Color | Span Key |
|---|---|---|---|
| Token | Token | `#334155` | `(folio, par, line, token_pos)` |
| Line | Line | `#475569` | `(folio, par, line)` |
| Paragraph | Par | `#059669` | `(folio, par)` |
| Folio | Folio | `#0ea5e9` | `folio` |

Each band is divided into contiguous spans using `_build_band_spans()`, which walks the annotation list and groups consecutive characters sharing the same segment key. Spans for separator characters (as determined by `SegmentKind`) are excluded — separators break the span at the appropriate structural level (e.g., `LINE_SEP` breaks line and token spans but not paragraph or folio spans).

Span labels are short identifiers: folio name for Folio, `P{n}` for Paragraph, `L{n}` for Line, `T{n}` for Token.

Bands are rendered using matplotlib `broken_barh()` with centered white text labels, no y-ticks, hidden spines, and a horizontal y-axis label. The layout uses `GridSpec` with height ratios of 12 (main plot) : 1.5 (glyph label row) : 1 per band, all sharing the x-axis.

When `mode="glyph"`, the `start` and `width` fields of `BandSpan` hold glyph-index positions, so each span covers a contiguous range of glyph positions. The span construction logic in `_build_band_spans()` counts characters rather than bytes.

**Why:** Manuscript structure is hierarchical — folios contain paragraphs, which contain lines, which contain tokens. Rendering these as stacked bands beneath the entropy curve lets researchers correlate entropy patterns with structural boundaries (e.g., "does entropy spike at paragraph transitions?") without cluttering the main plot.

### Band Span Construction

`_build_band_spans()` constructs contiguous spans from an annotation list for a given structural level. Span positions are in byte units when `mode="byte"` and glyph units when `mode="glyph"`. It walks the annotations sequentially, tracking a byte cursor and grouping consecutive characters that share the same segment key. The key varies by level:

- **folio**: `ann.folio`
- **par**: `(ann.folio, ann.par)`
- **line**: `(ann.folio, ann.par, ann.line)`
- **token**: `(ann.folio, ann.par, ann.line, ann.token_pos)`

For non-glyph characters (`SegmentKind` other than `GLYPH`), the function determines whether the separator breaks the current level's span based on its kind: `PARA_SEP` and `PARA_START` break paragraph, line, and token spans; `LINE_SEP` breaks line and token spans; word-boundary spaces break token spans only.

`_chunk_to_bands()` calls `_build_band_spans()` for each of the four levels and wraps the results as `BandSpec` objects, filtering out separator-only spans (where the key is `None`).

**Why:** The byte-level entropy model operates on flat byte sequences, but the manuscript has hierarchical structure. Band span construction recovers that structure from the annotation metadata so it can be visualized alongside the entropy curve.

### All-in-One Display

`display_entropy()` is the primary entry point, compositing all three visual components:

```python
display_entropy(text, token_ids, entropy_values,
                *, mode="byte", show_table=True, show_summary=True,
                show_plot=True, chunk=None, **kwargs)
```

- `mode`: `"byte"` (default) or `"glyph"`. Controls the granularity of all three views. The mode is forwarded to `display_entropy_table()`, `display_entropy_summary()`, and `display_entropy_plot()`.
- When `show_table=True`: calls `display_entropy_table()`, forwarding `text`, `mode`, and table-specific kwargs.
- When `show_summary=True`: calls `display_entropy_summary()`, forwarding `text`, `entropy_values`, and `mode`.
- When `show_plot=True`: calls `display_entropy_plot()`, forwarding `text`, `entropy_values`, `mode`, and the `chunk` for metadata bands if provided.

Extra keyword arguments are forwarded to `display_entropy_table()` (e.g., `font_size`, `bar_width`, `show_header`).

**Why:** Researchers typically want all three views together, but sometimes need only one (e.g., just the plot for a presentation, or just the table for detailed byte-level analysis). The compositor provides a single call for the common case while allowing selective rendering.

## Entropy Annotation

### `entropy_proc.annotate_entropy(chunk, entropy_values)`

Attaches per-byte entropy values to an `AnnotatedChunk`'s annotation list. The function:

1. Validates that the number of entropy values matches the total UTF-8 byte count of `chunk.text`.
2. Walks the annotation list and entropy values in parallel using a byte cursor.
3. For each `GlyphAnnotation`, computes the number of UTF-8 bytes for its character (`len(ann.char.encode("utf-8"))`).
4. Slices the corresponding entropy values and assigns them to `ann.byte_entropies`.
5. Raises `ValueError` if lengths do not match.

**Invariant:** After annotation, `len(ann.byte_entropies)` equals the number of UTF-8 bytes in `ann.char` for every annotation in the chunk.

**Why:** The BLT model produces entropy at byte granularity, but the annotation system operates at character granularity. This function bridges the two by attaching byte-level entropy slices to their parent characters, enabling both per-byte and per-character analysis in the display layer.

## Implementation Notes

- The `entropy_plot` module is intentionally generic — it knows nothing about Voynich manuscripts. All Voynich-specific logic (font loading, PUA character detection, shading rules, band colors, band construction from annotations) resides in `voy_entropy_display`. This separation allows `entropy_plot` to be reused for non-Voynich entropy visualization.
- The Voynich TTF font is loaded twice through different paths: `voy_font.load_voynich_font()` injects a base64 CSS `@font-face` rule for HTML table rendering, while `voy_entropy_display` registers the font with matplotlib's `fontManager` for plot axis labels. Both paths are required because HTML and matplotlib use independent font systems.
- The HTML table and matplotlib plot both use `_build_glyph_groups()` from `entropy_plot` to decompose text into byte-aligned character groups, ensuring consistent byte-to-character mapping across views.
- Band span construction handles the asymmetry between glyph and separator characters: glyphs carry full manuscript coordinates, while separators only carry a `SegmentKind` that determines which structural levels they break. This avoids requiring separators to store redundant coordinate data.
- The `mode` parameter propagates from `display_entropy()` down through all three rendering functions. The aggregation from byte to glyph entropy happens at the point of use (inside each rendering function or via the shared `_aggregate_glyph_entropies()` helper), not as a preprocessing step. This keeps the input contract consistent — all functions always receive byte-level entropy values and the source text, and perform aggregation internally when `mode="glyph"`. This avoids confusion about whether the caller should pre-aggregate.
- In glyph mode, band span positions are in glyph units. `_build_band_spans()` accepts a `mode` parameter and counts characters instead of bytes when `mode="glyph"`.
- The entropy color ramp uses linear RGB interpolation rather than perceptual colorspace interpolation. This is a deliberate simplicity trade-off — the five-stop ramp produces visually acceptable gradients without requiring additional color-science dependencies.
