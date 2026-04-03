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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

from voy_font import load_voynich_font

# Load Voynich TTF for matplotlib rendering
_VOYNICH_TTF = Path(__file__).resolve().parent.parent / "voynich_fonts/Voynich/CustomVoynichUnicode.ttf"
_VOYNICH_FONT_PROP = None
if _VOYNICH_TTF.exists():
    fm.fontManager.addfont(str(_VOYNICH_TTF))
    _VOYNICH_FONT_PROP = fm.FontProperties(fname=str(_VOYNICH_TTF))

# ==============================================================================
# Color helpers
# ==============================================================================

# Entropy color ramp: green (low) -> yellow -> orange -> red (high)
_ENTROPY_COLORS = [
    (0.00, "#22c55e"),  # green
    (0.25, "#84cc16"),  # lime
    (0.50, "#eab308"),  # yellow
    (0.75, "#f97316"),  # orange
    (1.00, "#ef4444"),  # red
]


def _entropy_css_color(value, min_val, max_val):
    """Map an entropy value to a CSS hex color on the green-to-red ramp."""
    if max_val == min_val:
        return "#94a3b8"  # slate gray
    t = (value - min_val) / (max_val - min_val)
    t = max(0.0, min(1.0, t))
    # Find the two stops that bracket t
    for i in range(len(_ENTROPY_COLORS) - 1):
        t0, c0 = _ENTROPY_COLORS[i]
        t1, c1 = _ENTROPY_COLORS[i + 1]
        if t0 <= t <= t1:
            local_t = (t - t0) / (t1 - t0)
            r0, g0, b0 = int(c0[1:3], 16), int(c0[3:5], 16), int(c0[5:7], 16)
            r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
            r = int(r0 + (r1 - r0) * local_t)
            g = int(g0 + (g1 - g0) * local_t)
            b = int(b0 + (b1 - b0) * local_t)
            return f"#{r:02x}{g:02x}{b:02x}"
    return _ENTROPY_COLORS[-1][1]


def _bar_html(value, max_val, width_px=120, color="#94a3b8"):
    """Render an inline entropy bar as a colored HTML div."""
    if max_val == 0:
        return ""
    pct = min(value / max_val, 1.0)
    filled_px = int(round(pct * width_px))
    return (
        f'<div style="display:inline-block; width:{width_px}px; height:12px; '
        f'background:#1e293b; border-radius:2px; overflow:hidden;">'
        f'<div style="width:{filled_px}px; height:100%; '
        f'background:{color}; border-radius:2px;"></div></div>'
    )


# ==============================================================================
# Byte-to-character mapping
# ==============================================================================

def _build_byte_char_map(text):
    """Map byte positions to the source Unicode character.

    For multi-byte characters, only the first byte gets the character label;
    continuation bytes are mapped to None.
    """
    byte_map = {}
    byte_offset = 0
    for ch in text:
        encoded = ch.encode("utf-8")
        byte_map[byte_offset] = ch
        for j in range(1, len(encoded)):
            byte_map[byte_offset + j] = None
        byte_offset += len(encoded)
    return byte_map


def _build_glyph_groups(text):
    """Build a list of glyph groups from the source text.

    Each group is a tuple (char, start_byte, num_bytes) indicating which
    byte positions belong to the same Unicode character.
    """
    groups = []
    byte_offset = 0
    for ch in text:
        num_bytes = len(ch.encode("utf-8"))
        groups.append((ch, byte_offset, num_bytes))
        byte_offset += num_bytes
    return groups


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
.entropy-summary {
    display: inline-block;
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 12px 20px;
    margin: 8px 0;
    font-size: 14px;
}
.entropy-summary .label { color: #94a3b8; margin-right: 8px; }
.entropy-summary .value { font-weight: 700; font-variant-numeric: tabular-nums; margin-right: 16px; }
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
):
    """Display a per-byte entropy table in a Jupyter notebook.

    Args:
        text: The source Unicode string.
        token_ids: List of raw byte values (0-255) for each byte.
        entropy_values: List of float entropy values, one per byte.
        show_header: If True, show the input text in Voynich font above the table.
        font_size: CSS font size for glyphs in the table.
        bar_width: Width in pixels for the entropy distribution bar.

    Returns:
        IPython HTML object.
    """
    load_voynich_font()

    if not entropy_values:
        result = HTML("<p style='color:#94a3b8;'>No entropy values to display.</p>")
        display(result)
        return result

    min_e = min(entropy_values)
    max_e = max(entropy_values)

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

    # Table
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


def display_entropy_summary(entropy_values):
    """Display summary statistics for entropy values.

    Args:
        entropy_values: List of float entropy values.

    Returns:
        IPython HTML object.
    """
    if not entropy_values:
        result = HTML("<p style='color:#94a3b8;'>No entropy values to display.</p>")
        display(result)
        return result

    min_e = min(entropy_values)
    max_e = max(entropy_values)
    mean_e = sum(entropy_values) / len(entropy_values)

    color_min = _entropy_css_color(min_e, min_e, max_e)
    color_max = _entropy_css_color(max_e, min_e, max_e)
    color_mean = _entropy_css_color(mean_e, min_e, max_e)

    html_str = (
        _TABLE_CSS
        + f'<div class="entropy-summary">'
        f'<span class="label">Mean entropy:</span>'
        f'<span class="value" style="color:{color_mean};">{mean_e:.4f}</span>'
        f'<span class="label">Max:</span>'
        f'<span class="value" style="color:{color_max};">{max_e:.4f}</span>'
        f'<span class="label">Min:</span>'
        f'<span class="value" style="color:{color_min};">{min_e:.4f}</span>'
        f'<span class="label">Bytes:</span>'
        f'<span class="value">{len(entropy_values)}</span>'
        f"</div>"
    )
    result = HTML(html_str)
    display(result)
    return result


# ==============================================================================
# Plot
# ==============================================================================

def display_entropy_plot(
    entropy_values,
    *,
    text=None,
    figsize=None,
    dpi=200,
    glyphs_per_inch=4,
):
    """Display a matplotlib line plot of per-byte entropy.

    Args:
        entropy_values: List of float entropy values.
        text: Optional source text — if provided, byte-aligned ASCII character
              labels are placed on the x-axis.  Non-ASCII characters (including
              Voynich PUA glyphs) are shown as "·" since matplotlib cannot use
              the CSS-injected Voynich font.
        figsize: Matplotlib figure size tuple.  If None, the width is computed
                 automatically so that there are roughly *glyphs_per_inch*
                 glyphs per horizontal inch.
        dpi: Figure DPI.
        glyphs_per_inch: Target glyph density when figsize is auto-computed.

    Returns:
        The matplotlib Figure.
    """
    if figsize is None:
        n_glyphs = len(text) if text else len(entropy_values)
        width = max(4.0, n_glyphs / glyphs_per_inch)
        figsize = (width, 3)

    if not entropy_values:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig

    positions = np.arange(len(entropy_values))
    values = np.array(entropy_values)
    min_e, max_e = values.min(), values.max()
    if min_e == max_e:
        max_e = min_e + 1.0  # avoid degenerate normalizer

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Alternating glyph-group bands (mirrors the table's band styling)
    _SPACE_CHARS = {' ', '\t'}
    _LINE_SEP_CHARS = {'\n', '\r', '\u2028', '\u2029'}
    _PILCROW_CHARS = {'\u00b6'}
    if text:
        glyph_groups = _build_glyph_groups(text)
        for group_idx, (ch, start, n_bytes) in enumerate(glyph_groups):
            x0 = start - 0.5
            x1 = start + n_bytes - 0.5
            if ch in _SPACE_CHARS:
                ax.axvspan(x0, x1, color="#22c55e", alpha=0.12, zorder=0)
            elif ch in _LINE_SEP_CHARS:
                ax.axvspan(x0, x1, color="#3b82f6", alpha=0.12, zorder=0)
            elif ch in _PILCROW_CHARS:
                ax.axvspan(x0, x1, color="#a855f7", alpha=0.12, zorder=0)
            elif group_idx % 2 == 0:
                ax.axvspan(x0, x1, color="#94a3b8", alpha=0.25, zorder=0)
            else:
                ax.axvspan(x0, x1, color="#1e293b", alpha=0.25, zorder=0)

    # Color each segment by entropy
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "entropy", ["#22c55e", "#eab308", "#ef4444"]
    )
    norm = plt.Normalize(min_e, max_e)

    # Plot line segments with color gradient
    for i in range(len(positions) - 1):
        avg_e = (values[i] + values[i + 1]) / 2
        ax.plot(
            positions[i : i + 2],
            values[i : i + 2],
            color=cmap(norm(avg_e)),
            linewidth=1.5,
        )

    # Scatter points
    ax.scatter(positions, values, c=values, cmap=cmap, norm=norm, s=12, zorder=5)

    # Legend for glyph band colors
    legend_handles = []
    if text:
        import matplotlib.patches as mpatches
        # Only add entries for band types that actually appear
        chars_in_text = set(text)
        if chars_in_text & _SPACE_CHARS:
            legend_handles.append(mpatches.Patch(color="#22c55e", alpha=0.3, label="Space / Tab"))
        if chars_in_text & _LINE_SEP_CHARS:
            legend_handles.append(mpatches.Patch(color="#3b82f6", alpha=0.3, label="Line Break"))
        if chars_in_text & _PILCROW_CHARS:
            legend_handles.append(mpatches.Patch(color="#a855f7", alpha=0.3, label="Paragraph (¶)"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    # Character labels on x-axis using the Voynich TTF font
    if text and _VOYNICH_FONT_PROP is not None:
        glyph_groups_for_ticks = _build_glyph_groups(text)
        tick_positions = []
        tick_labels = []
        for ch, start, n_bytes in glyph_groups_for_ticks:
            center = start + (n_bytes - 1) / 2.0
            tick_positions.append(center)
            tick_labels.append(ch)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=7)
        for label in ax.get_xticklabels():
            ch = label.get_text()
            if ch and len(ch) == 1 and 0xE000 <= ord(ch) <= 0xF8FF:
                label.set_fontproperties(_VOYNICH_FONT_PROP)
                label.set_fontsize(12)
    else:
        ax.set_xlabel("Byte Position")

    ax.set_ylabel("Entropy")
    ax.set_title("Per-Byte Entropy", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()
    return fig


# ==============================================================================
# All-in-one display
# ==============================================================================

def display_entropy(text, token_ids, entropy_values, *, show_table=True, show_summary=True, show_plot=True, **kwargs):
    """Display the full entropy analysis: table, summary, and optional plot.

    Args:
        text: Source Unicode string.
        token_ids: List of raw byte values (0-255).
        entropy_values: List of float entropy values, one per byte.
        show_table: If True, show the per-byte entropy table.
        show_summary: If True, show summary statistics.
        show_plot: If True, show a matplotlib line plot.
        **kwargs: Passed through to display_entropy_table.
    """
    if show_table:
        display_entropy_table(text, token_ids, entropy_values, **kwargs)
    if show_summary:
        display_entropy_summary(entropy_values)
    if show_plot:
        display_entropy_plot(entropy_values, text=text)
