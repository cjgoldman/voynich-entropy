"""
Voynich entropy display for Jupyter notebooks.

Renders per-byte entropy tables with Voynich Unicode font glyphs,
color-coded entropy bars, and summary statistics — the notebook
equivalent of the Rich terminal display in basic_run/blt_example.py.

Usage:
    from voy_entropy_display import display_entropy, display_entropy_table, display_entropy_plot

    # All-in-one display (table + summary + plot)
    display_entropy(text, token_ids, entropy_values)

    # Just the table (no plot)
    display_entropy(text, token_ids, entropy_values, show_plot=False)

    # Just the plot (no table)
    display_entropy(text, token_ids, entropy_values, show_table=False, show_summary=False)

    # Or call individual components directly
    display_entropy_table(text, token_ids, entropy_values)
    display_entropy_plot(entropy_values)
"""

import html as _html
from IPython.display import HTML, display
import matplotlib.font_manager as fm
from pathlib import Path

from voy_font import load_voynich_font
from entropy_plot import (
    BandSpan,
    BandSpec,
    FontSpec,
    GlyphShadingRule,
    plot_entropy,
    display_entropy_summary,
    _entropy_css_color,
    _bar_html,
    _build_glyph_groups,
    _aggregate_glyph_entropies,
)

# Load Voynich TTF for matplotlib rendering
_VOYNICH_TTF = Path(__file__).resolve().parent.parent / "voynich_fonts/Voynich/BMPVoynichUnicode.ttf"
_VOYNICH_FONT_PROP = None
if _VOYNICH_TTF.exists():
    fm.fontManager.addfont(str(_VOYNICH_TTF))
    _VOYNICH_FONT_PROP = fm.FontProperties(fname=str(_VOYNICH_TTF))


# ==============================================================================
# Voynich font / shading adapters
# ==============================================================================

def _voynich_font_spec():
    """Wrap the Voynich font as a generic FontSpec, or None if unavailable."""
    if _VOYNICH_FONT_PROP is None:
        return None
    return FontSpec(
        font_properties=_VOYNICH_FONT_PROP,
        char_predicate=lambda ch: len(ch) == 1 and 0xE000 <= ord(ch) <= 0xF8FF,
        font_size=12.0,
    )


def _is_cjk_char(ch: str) -> bool:
    """Return True for CJK Unified Ideographs and related ranges."""
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF        # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF     # CJK Extension A
            or 0x3000 <= cp <= 0x303F     # CJK Symbols and Punctuation
            or 0x3040 <= cp <= 0x30FF     # Hiragana + Katakana
            or 0xAC00 <= cp <= 0xD7AF     # Hangul Syllables
            or 0xFF00 <= cp <= 0xFFEF)    # Fullwidth Forms


def _cjk_font_spec():
    """Return a FontSpec for CJK characters if a suitable font is found."""
    for name in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
                  "Noto Sans SC", "WenQuanYi Micro Hei"]:
        try:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        except ValueError:
            continue
        return FontSpec(
            font_properties=fm.FontProperties(fname=path),
            char_predicate=_is_cjk_char,
            font_size=9.0,
        )
    return None


def _is_devanagari_char(ch: str) -> bool:
    """Return True for Devanagari characters and related marks."""
    cp = ord(ch)
    return (0x0900 <= cp <= 0x097F        # Devanagari
            or 0xA8E0 <= cp <= 0xA8FF     # Devanagari Extended
            or 0x1CD0 <= cp <= 0x1CFF)    # Vedic Extensions


def _devanagari_font_spec():
    """Return a FontSpec for Devanagari characters if a suitable font is found."""
    for name in ["Lohit Devanagari", "Lohit-Devanagari", "Noto Sans Devanagari",
                  "Noto Serif Devanagari", "Mangal"]:
        try:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        except ValueError:
            continue
        return FontSpec(
            font_properties=fm.FontProperties(fname=path),
            char_predicate=_is_devanagari_char,
            font_size=9.0,
        )
    return None


def _build_font_list():
    """Build a list of FontSpecs for all available custom fonts."""
    specs = []
    voy = _voynich_font_spec()
    if voy:
        specs.append(voy)
    cjk = _cjk_font_spec()
    if cjk:
        specs.append(cjk)
    deva = _devanagari_font_spec()
    if deva:
        specs.append(deva)
    return specs or None


_VOYNICH_SHADING = [
    GlyphShadingRule(chars={' ', '\t'}, color="#22c55e", alpha=0.12, legend_label="Space / Tab"),
    GlyphShadingRule(chars={'\n', '\r', '\u2028', '\u2029'}, color="#3b82f6", alpha=0.12, legend_label="Line Break"),
    GlyphShadingRule(chars={'\u00b6'}, color="#a855f7", alpha=0.12, legend_label="Paragraph (\u00b6)"),
]


# ==============================================================================
# Metadata band adapters (folio / par / line / token)
# ==============================================================================

_BAND_COLORS = {
    "folio":  "#0ea5e9",
    "par":    "#059669",
    "line":   "#475569",
    "token":  "#334155",
}

_BAND_LABELS = {
    "folio": "Folio",
    "par":   "Par",
    "line":  "Line",
    "token": "Token",
}


def _build_band_spans(annotations, level, *, mode="byte"):
    """Build contiguous spans for a metadata band level.

    Args:
        annotations: list of GlyphAnnotation (from AnnotatedChunk).
        level: One of "folio", "par", "line", "token".
        mode: "byte" or "glyph". In glyph mode, positions are glyph indices
              (each annotation contributes 1 unit regardless of byte count).

    Returns:
        List of (start, width, segment_key, label) tuples.
    """
    from vms_annot import SegmentKind

    spans = []
    cursor = 0
    _SENTINEL = object()
    current_key = _SENTINEL
    span_start = 0

    for ann in annotations:
        n_bytes = len(ann.char.encode("utf-8"))
        step = 1 if mode == "glyph" else n_bytes

        if ann.kind == SegmentKind.GLYPH:
            if level == "folio":
                key = ann.folio
            elif level == "par":
                key = (ann.folio, ann.par)
            elif level == "line":
                key = (ann.folio, ann.par, ann.line)
            else:  # token
                key = (ann.folio, ann.par, ann.line, ann.token_pos)
        else:
            breaks_token = True
            breaks_line = ann.kind in (SegmentKind.LINE_SEP, SegmentKind.PARA_SEP, SegmentKind.PARA_START)
            breaks_par = ann.kind in (SegmentKind.PARA_SEP, SegmentKind.PARA_START)

            if level == "token" and breaks_token:
                key = None
            elif level == "line" and breaks_line:
                key = None
            elif level == "par" and breaks_par:
                key = None
            else:
                key = current_key if current_key is not _SENTINEL else None

        if key != current_key:
            if cursor > span_start and current_key is not _SENTINEL:
                spans.append((span_start, cursor - span_start, current_key, _band_label(level, current_key)))
            span_start = cursor
            current_key = key

        cursor += step

    if cursor > span_start and current_key is not _SENTINEL:
        spans.append((span_start, cursor - span_start, current_key, _band_label(level, current_key)))

    return spans


def _band_label(level, key):
    """Return a short display label for a band span key."""
    if key is None:
        return ""
    if level == "folio":
        return str(key)
    elif level == "par":
        return f"P{key[1]}"
    elif level == "line":
        return f"L{key[2]}"
    else:  # token
        return f"T{key[3]}"


def _chunk_to_bands(chunk, *, mode="byte"):
    """Convert an AnnotatedChunk into a list of generic BandSpec objects.

    Args:
        chunk: An AnnotatedChunk with manuscript position annotations.
        mode: "byte" or "glyph". Forwarded to _build_band_spans.

    Returns:
        List of BandSpec in display order (token, line, par, folio).
    """
    bands = []
    for level in ("token", "line", "par", "folio"):
        raw_spans = _build_band_spans(chunk.annotations, level, mode=mode)
        specs = [
            BandSpan(start=s, width=w, text=lbl)
            for s, w, key, lbl in raw_spans
            if key is not None
        ]
        bands.append(BandSpec(
            label=_BAND_LABELS[level],
            color=_BAND_COLORS[level],
            spans=specs,
        ))
    return bands


# ==============================================================================
# Table rendering
# ==============================================================================

_TABLE_CSS = """\
<style>
.entropy-table {
    border-collapse: collapse;
    font-size: 13px;
    margin: 8px 0;
}
.entropy-table th {
    background: #1e293b;
    color: #e2e8f0;
    padding: 6px 10px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #334155;
}
.entropy-table td {
    padding: 4px 10px;
    border-bottom: 1px solid #334155;
    vertical-align: middle;
}
.entropy-table tr:hover td {
    background: #1e293b40;
}
.entropy-table .pos { color: #94a3b8; text-align: right; font-variant-numeric: tabular-nums; }
.entropy-table .byte { text-align: right; font-variant-numeric: tabular-nums; }
.entropy-table .glyph {
    font-family: 'VoynichUnicode', monospace;
    font-size: 20px;
    text-align: center;
    min-width: 32px;
}
.entropy-table .ent { text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }
.entropy-table tr.glyph-band-a td { background: transparent; }
.entropy-table tr.glyph-band-b td { background: rgba(30, 41, 59, 0.35); }
.entropy-table tr.glyph-band-a:hover td,
.entropy-table tr.glyph-band-b:hover td { background: #1e293b40; }
.entropy-table tr.glyph-group-first td { border-top: 1px solid #475569; }
.entropy-table .glyph-span {
    font-family: 'VoynichUnicode', monospace;
    font-size: 20px;
    text-align: center;
    min-width: 32px;
    vertical-align: middle;
    border-bottom: none;
}
.entropy-header {
    font-family: 'VoynichUnicode', sans-serif;
    font-size: 22px;
    line-height: 1.6;
    padding: 10px 16px;
    margin: 8px 0;
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 6px;
}
</style>
"""


def display_entropy_table(
    text,
    token_ids,
    entropy_values,
    *,
    show_header=True,
    font_size="20px",
    bar_width=120,
    mode="byte",
):
    """Display an entropy table in a Jupyter notebook.

    Args:
        text: The source Unicode string.
        token_ids: List of raw byte values (0-255) for each byte.
        entropy_values: List of float entropy values, one per byte.
        show_header: If True, show the input text in Voynich font above the table.
        font_size: CSS font size for glyphs in the table.
        bar_width: Width in pixels for the entropy distribution bar.
        mode: "byte" or "glyph". In glyph mode, one row per character with
              summed byte entropies; the Byte column is omitted.

    Returns:
        IPython HTML object.
    """
    load_voynich_font()

    if not entropy_values:
        result = HTML("<p style='color:#94a3b8;'>No entropy values to display.</p>")
        display(result)
        return result

    parts = [_TABLE_CSS]

    # Header: input text rendered in Voynich font
    if show_header:
        escaped = _html.escape(text)
        parts.append(
            f'<div class="entropy-header">'
            f'<span style="color:#94a3b8; font-family:sans-serif; '
            f'font-size:12px;">Input Text</span><br>'
            f'{escaped}</div>'
        )

    if mode == "glyph":
        # Glyph mode: one row per Unicode character, no Byte column
        glyph_entropies = _aggregate_glyph_entropies(text, entropy_values)
        min_e = min(glyph_entropies)
        max_e = max(glyph_entropies)

        parts.append('<table class="entropy-table">')
        parts.append(
            "<thead><tr>"
            "<th>Pos</th>"
            "<th>Glyph</th>"
            "<th>Entropy</th>"
            "<th>Distribution</th>"
            "</tr></thead><tbody>"
        )

        for glyph_idx, (ch, ent) in enumerate(zip(text, glyph_entropies)):
            color = _entropy_css_color(ent, min_e, max_e)
            band_class = "glyph-band-a" if glyph_idx % 2 == 0 else "glyph-band-b"
            bar = _bar_html(ent, max_e, width_px=bar_width, color=color)
            glyph_content = (
                f'<span style="font-family: VoynichUnicode, monospace; '
                f'font-size:{font_size};">{_html.escape(ch)}</span>'
            )

            parts.append(
                f'<tr class="{band_class}">'
                f'<td class="pos">{glyph_idx}</td>'
                f'<td class="glyph-span">{glyph_content}</td>'
                f'<td class="ent" style="color:{color};">{ent:.4f}</td>'
                f'<td style="color:{color};">{bar}</td>'
                f"</tr>"
            )
    else:
        # Byte mode: one row per byte with rowspan glyph grouping
        min_e = min(entropy_values)
        max_e = max(entropy_values)

        parts.append('<table class="entropy-table">')
        parts.append(
            "<thead><tr>"
            "<th>Pos</th>"
            "<th>Byte</th>"
            "<th>Glyph</th>"
            "<th>Entropy</th>"
            "<th>Distribution</th>"
            "</tr></thead><tbody>"
        )

        # Build glyph groups for rowspan and alternating bands
        glyph_groups = _build_glyph_groups(text)

        # Map each byte position to its group index for band toggling
        byte_to_group = {}
        for group_idx, (_ch, start, n_bytes) in enumerate(glyph_groups):
            for b in range(start, start + n_bytes):
                byte_to_group[b] = group_idx

        # Set of byte positions that start a new glyph group
        group_starts = {start for _, start, _ in glyph_groups}
        # Map start byte -> (char, num_bytes)
        group_info = {start: (ch, n_bytes) for ch, start, n_bytes in glyph_groups}

        for i, (tok, ent) in enumerate(zip(token_ids, entropy_values)):
            color = _entropy_css_color(ent, min_e, max_e)

            # Alternating band class
            g_idx = byte_to_group.get(i, 0)
            band_class = "glyph-band-a" if g_idx % 2 == 0 else "glyph-band-b"
            first_class = " glyph-group-first" if i in group_starts else ""

            bar = _bar_html(ent, max_e, width_px=bar_width, color=color)

            # Build glyph cell only on first byte of each group (rowspan)
            glyph_td = ""
            if i in group_info:
                ch, n_bytes = group_info[i]
                glyph_content = (
                    f'<span style="font-family: VoynichUnicode, monospace; '
                    f'font-size:{font_size};">{_html.escape(ch)}</span>'
                )
                rowspan_attr = f' rowspan="{n_bytes}"' if n_bytes > 1 else ""
                glyph_td = f'<td class="glyph-span"{rowspan_attr}>{glyph_content}</td>'

            parts.append(
                f'<tr class="{band_class}{first_class}">'
                f'<td class="pos">{i}</td>'
                f'<td class="byte">{tok}</td>'
                f'{glyph_td}'
                f'<td class="ent" style="color:{color};">{ent:.4f}</td>'
                f'<td style="color:{color};">{bar}</td>'
                f"</tr>"
            )

    parts.append("</tbody></table>")

    html_str = "\n".join(parts)
    result = HTML(html_str)
    display(result)
    return result


# ==============================================================================
# Plot (thin wrapper around generic plot_entropy)
# ==============================================================================

def display_entropy_plot(
    entropy_values,
    *,
    text=None,
    chunk=None,
    figsize=None,
    dpi=200,
    glyphs_per_inch=4,
    mode="byte",
):
    """Display a matplotlib line plot of entropy values.

    Args:
        entropy_values: List of float entropy values.
        text: Optional source text — if provided, character labels are placed
              on the x-axis.
        chunk: Optional AnnotatedChunk — if provided, horizontal metadata
               bands (folio, paragraph, line, token) are rendered below
               the main plot.
        figsize: Matplotlib figure size tuple.
        dpi: Figure DPI.
        glyphs_per_inch: Target glyph density when figsize is auto-computed.
        mode: "byte" or "glyph". Forwarded to band construction and plot_entropy.

    Returns:
        The matplotlib Figure.
    """
    bands = _chunk_to_bands(chunk, mode=mode) if chunk is not None else None
    return plot_entropy(
        entropy_values,
        text=text,
        bands=bands,
        fonts=_build_font_list(),
        shading_rules=_VOYNICH_SHADING,
        figsize=figsize,
        dpi=dpi,
        glyphs_per_inch=glyphs_per_inch,
        mode=mode,
    )


# ==============================================================================
# All-in-one display
# ==============================================================================

def display_entropy(text, token_ids, entropy_values, *, mode="byte", show_table=True, show_summary=True, show_plot=True, chunk=None, **kwargs):
    """Display the full entropy analysis: table, summary, and optional plot.

    Args:
        text: Source Unicode string.
        token_ids: List of raw byte values (0-255).
        entropy_values: List of float entropy values, one per byte.
        mode: "byte" or "glyph". Controls granularity of all three views.
        show_table: If True, show the entropy table.
        show_summary: If True, show summary statistics.
        show_plot: If True, show a matplotlib line plot.
        chunk: Optional AnnotatedChunk for manuscript position bands.
        **kwargs: Passed through to display_entropy_table.
    """
    if show_table:
        display_entropy_table(text, token_ids, entropy_values, mode=mode, **kwargs)
    if show_summary:
        display_entropy_summary(entropy_values, text=text, mode=mode)
    if show_plot:
        display_entropy_plot(entropy_values, text=text, chunk=chunk, mode=mode)
