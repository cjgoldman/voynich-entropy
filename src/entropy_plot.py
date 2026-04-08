"""
Generic per-byte entropy visualization for Jupyter notebooks.

Provides a reusable entropy line plot with optional hierarchical metadata
bands (e.g., chapters/sections, folios/paragraphs) and customizable font
rendering.  Domain-specific adapters (such as the Voynich display layer)
convert their own annotation types into the generic BandSpec / FontSpec
abstractions defined here.

Usage:
    from entropy_plot import plot_entropy, BandSpec, BandSpan, FontSpec, GlyphShadingRule

    # Minimal — just the entropy line chart
    plot_entropy(entropy_values)

    # With character labels on x-axis
    plot_entropy(entropy_values, text=source_text)

    # With metadata bands below the plot
    bands = [
        BandSpec(label="Chapter", color="#0ea5e9", spans=[BandSpan(0, 120, "Ch1"), ...]),
        BandSpec(label="Section", color="#059669", spans=[BandSpan(0, 40, "S1"), ...]),
    ]
    plot_entropy(entropy_values, text=source_text, bands=bands)
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

EntropyMode = Literal["byte", "glyph"]

from IPython.display import HTML, display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


# ==============================================================================
# Dataclasses
# ==============================================================================

@dataclass
class BandSpan:
    """A single contiguous region within a metadata band."""
    start: int
    width: int
    text: str  # short label rendered inside the span


@dataclass
class BandSpec:
    """One horizontal metadata band to render below the entropy line plot."""
    label: str          # y-axis label, e.g. "Chapter", "Folio"
    color: str          # hex color, e.g. "#0ea5e9"
    spans: list[BandSpan] = field(default_factory=list)


@dataclass
class FontSpec:
    """Optional custom font for glyph labels on the entropy plot."""
    font_properties: Any                    # matplotlib FontProperties object
    char_predicate: Callable[[str], bool]   # returns True for chars that use this font
    font_size: float = 12.0


@dataclass
class GlyphShadingRule:
    """A rule mapping characters to a background color band in the plot."""
    chars: set[str]
    color: str
    alpha: float
    legend_label: str


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


def _aggregate_glyph_entropies(text, entropy_values):
    """Convert byte-level entropy values to glyph-level by summing per character.

    Args:
        text: Source Unicode string.
        entropy_values: List of float entropy values, one per UTF-8 byte.

    Returns:
        List of floats with one entry per Unicode character (sum of byte entropies).
    """
    groups = _build_glyph_groups(text)
    glyph_entropies = []
    for _ch, byte_start, num_bytes in groups:
        glyph_entropies.append(sum(entropy_values[byte_start:byte_start + num_bytes]))
    return glyph_entropies


# ==============================================================================
# Summary display
# ==============================================================================

_SUMMARY_CSS = """\
<style>
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
</style>
"""


def display_entropy_summary(entropy_values, *, text=None, mode="byte"):
    """Display summary statistics for entropy values.

    Args:
        entropy_values: List of float entropy values.
        text: Source text (required when mode="glyph").
        mode: "byte" or "glyph". In glyph mode, entropy values are aggregated
              per character before computing statistics.

    Returns:
        IPython HTML object.
    """
    if not entropy_values:
        result = HTML("<p style='color:#94a3b8;'>No entropy values to display.</p>")
        display(result)
        return result

    if mode == "glyph":
        if text is None:
            raise ValueError("text is required when mode='glyph'")
        values = _aggregate_glyph_entropies(text, entropy_values)
    else:
        values = entropy_values

    min_e = min(values)
    max_e = max(values)
    mean_e = sum(values) / len(values)

    color_min = _entropy_css_color(min_e, min_e, max_e)
    color_max = _entropy_css_color(max_e, min_e, max_e)
    color_mean = _entropy_css_color(mean_e, min_e, max_e)

    count_label = "Glyphs:" if mode == "glyph" else "Bytes:"

    html_str = (
        _SUMMARY_CSS
        + f'<div class="entropy-summary">'
        f'<span class="label">Mean entropy:</span>'
        f'<span class="value" style="color:{color_mean};">{mean_e:.4f}</span>'
        f'<span class="label">Max:</span>'
        f'<span class="value" style="color:{color_max};">{max_e:.4f}</span>'
        f'<span class="label">Min:</span>'
        f'<span class="value" style="color:{color_min};">{min_e:.4f}</span>'
        f'<span class="label">{count_label}</span>'
        f'<span class="value">{len(values)}</span>'
        f"</div>"
    )
    result = HTML(html_str)
    display(result)
    return result


# ==============================================================================
# Band rendering
# ==============================================================================

def _render_metadata_band(ax, band, alpha=0.75):
    """Render a single horizontal metadata band on the given axes.

    Args:
        ax: Matplotlib axes for this band row.
        band: A BandSpec instance.
        alpha: Opacity for the band rectangles.
    """
    for span in band.spans:
        ax.broken_barh(
            [(span.start - 0.5, span.width)],
            (0, 1),
            facecolors=band.color,
            alpha=alpha,
            edgecolors="none",
        )
        cx = span.start + span.width / 2.0 - 0.5
        ax.text(
            cx, 0.5, span.text,
            ha="center", va="center",
            fontsize=5, color="white", fontweight="bold",
            clip_on=True,
        )

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel(band.label, rotation=0, labelpad=40, va="center", fontsize=7, color="#94a3b8")


# ==============================================================================
# Core entropy plot
# ==============================================================================

def plot_entropy(
    entropy_values,
    *,
    text=None,
    bands=None,
    font=None,
    shading_rules=None,
    figsize=None,
    dpi=200,
    glyphs_per_inch=4,
    mode="byte",
):
    """Display a matplotlib line plot of entropy values.

    Args:
        entropy_values: List of float entropy values, one per byte.
        text: Optional source text — if provided, character labels are placed
              on the x-axis.
        bands: Optional list of BandSpec — if provided, horizontal metadata
               bands are rendered below the main plot.
        font: Optional FontSpec — custom font for matching glyph labels.
        shading_rules: Optional list of GlyphShadingRule — character-category
                       background colors.  Unmatched characters get alternating
                       gray bands.
        figsize: Matplotlib figure size tuple.  If None, the width is computed
                 automatically so that there are roughly *glyphs_per_inch*
                 glyphs per horizontal inch.
        dpi: Figure DPI.
        glyphs_per_inch: Target glyph density when figsize is auto-computed.
        mode: "byte" or "glyph". In glyph mode, entropy values are aggregated
              per character before plotting.

    Returns:
        The matplotlib Figure.
    """
    # In glyph mode, aggregate byte entropies to glyph level
    if mode == "glyph" and text is not None:
        plot_values = _aggregate_glyph_entropies(text, entropy_values)
    else:
        plot_values = list(entropy_values)

    has_bands = bands is not None and len(bands) > 0

    if figsize is None:
        if mode == "glyph":
            n_glyphs = len(text) if text else len(plot_values)
        else:
            n_glyphs = len(plot_values) if plot_values else (len(text) if text else 0)
        width = max(4.0, n_glyphs / glyphs_per_inch)
        height = 4.0 if has_bands else 3.0
        figsize = (width, height)

    if not plot_values:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig

    # Create figure layout: main plot + glyph label row + N band rows
    if has_bands:
        from matplotlib.gridspec import GridSpec
        n_bands = len(bands)
        height_ratios = [12, 1.5] + [1] * n_bands
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(2 + n_bands, 1, height_ratios=height_ratios, hspace=0.08, figure=fig)
        ax = fig.add_subplot(gs[0])
        glyph_ax = fig.add_subplot(gs[1], sharex=ax)
        band_axes = []
        for i in range(n_bands):
            band_axes.append(fig.add_subplot(gs[2 + i], sharex=ax))
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    positions = np.arange(len(plot_values))
    values = np.array(plot_values)
    min_e, max_e = values.min(), values.max()
    if min_e == max_e:
        max_e = min_e + 1.0

    # Alternating glyph-group bands with optional shading rules
    if text:
        glyph_groups = _build_glyph_groups(text)
        if mode == "glyph":
            # One x-unit per glyph
            for glyph_idx, (ch, _start, _n_bytes) in enumerate(glyph_groups):
                x0 = glyph_idx - 0.5
                x1 = glyph_idx + 0.5
                matched = False
                if shading_rules:
                    for rule in shading_rules:
                        if ch in rule.chars:
                            ax.axvspan(x0, x1, color=rule.color, alpha=rule.alpha, zorder=0)
                            matched = True
                            break
                if not matched:
                    if glyph_idx % 2 == 0:
                        ax.axvspan(x0, x1, color="#94a3b8", alpha=0.25, zorder=0)
                    else:
                        ax.axvspan(x0, x1, color="#1e293b", alpha=0.25, zorder=0)
        else:
            # Byte mode: spans cover byte ranges
            for group_idx, (ch, start, n_bytes) in enumerate(glyph_groups):
                x0 = start - 0.5
                x1 = start + n_bytes - 0.5
                matched = False
                if shading_rules:
                    for rule in shading_rules:
                        if ch in rule.chars:
                            ax.axvspan(x0, x1, color=rule.color, alpha=rule.alpha, zorder=0)
                            matched = True
                            break
                if not matched:
                    if group_idx % 2 == 0:
                        ax.axvspan(x0, x1, color="#94a3b8", alpha=0.25, zorder=0)
                    else:
                        ax.axvspan(x0, x1, color="#1e293b", alpha=0.25, zorder=0)

    # Color each segment by entropy
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "entropy", ["#22c55e", "#eab308", "#ef4444"]
    )
    norm = plt.Normalize(min_e, max_e)

    for i in range(len(positions) - 1):
        avg_e = (values[i] + values[i + 1]) / 2
        ax.plot(
            positions[i : i + 2],
            values[i : i + 2],
            color=cmap(norm(avg_e)),
            linewidth=1.5,
        )

    ax.scatter(positions, values, c=values, cmap=cmap, norm=norm, s=12, zorder=5)

    # Legend for shading rules
    legend_handles = []
    if text and shading_rules:
        chars_in_text = set(text)
        for rule in shading_rules:
            if chars_in_text & rule.chars:
                legend_handles.append(
                    mpatches.Patch(color=rule.color, alpha=rule.alpha + 0.18, label=rule.legend_label)
                )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    # Character labels on x-axis
    if has_bands:
        ax.tick_params(axis="x", labelbottom=False)
    elif text:
        glyph_groups_for_ticks = _build_glyph_groups(text)
        tick_positions = []
        tick_labels = []
        if mode == "glyph":
            for glyph_idx, (ch, _start, _n_bytes) in enumerate(glyph_groups_for_ticks):
                tick_positions.append(glyph_idx)
                tick_labels.append(ch)
        else:
            for ch, start, n_bytes in glyph_groups_for_ticks:
                center = start + (n_bytes - 1) / 2.0
                tick_positions.append(center)
                tick_labels.append(ch)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=7)
        if font is not None:
            for label in ax.get_xticklabels():
                ch = label.get_text()
                if ch and font.char_predicate(ch):
                    label.set_fontproperties(font.font_properties)
                    label.set_fontsize(font.font_size)
    else:
        ax.set_xlabel("Byte Position" if mode == "byte" else "Glyph Position")

    ax.set_ylabel("Entropy")
    title = "Per-Byte Entropy" if mode == "byte" else "Per-Glyph Entropy"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Render metadata bands
    if has_bands:
        for band, band_ax in zip(bands, band_axes):
            _render_metadata_band(band_ax, band)
        for band_ax in band_axes:
            band_ax.tick_params(axis="x", labelbottom=False)

        # Glyph label row
        glyph_ax.set_yticks([])
        glyph_ax.set_ylim(0, 1)
        glyph_ax.patch.set_visible(False)
        glyph_ax.tick_params(axis="x", labelbottom=False, labeltop=False, bottom=False, top=False)
        for spine in glyph_ax.spines.values():
            spine.set_visible(False)
        glyph_ax.set_ylabel("Glyph", rotation=0, labelpad=40, va="center", fontsize=7, color="#94a3b8")
        if text:
            glyph_groups_for_ticks = _build_glyph_groups(text)
            if mode == "glyph":
                for glyph_idx, (ch, _start, _n_bytes) in enumerate(glyph_groups_for_ticks):
                    fp = font.font_properties if (font and font.char_predicate(ch)) else None
                    fs = font.font_size if fp else 7
                    glyph_ax.text(
                        glyph_idx, 0.5, ch,
                        ha="center", va="center",
                        fontsize=fs,
                        fontproperties=fp,
                        clip_on=True,
                        transform=glyph_ax.get_xaxis_transform(),
                    )
            else:
                for ch, start, n_bytes in glyph_groups_for_ticks:
                    center = start + (n_bytes - 1) / 2.0
                    fp = font.font_properties if (font and font.char_predicate(ch)) else None
                    fs = font.font_size if fp else 7
                    glyph_ax.text(
                        center, 0.5, ch,
                        ha="center", va="center",
                        fontsize=fs,
                        fontproperties=fp,
                        clip_on=True,
                        transform=glyph_ax.get_xaxis_transform(),
                    )

    if has_bands:
        fig.subplots_adjust(hspace=0.08)
    else:
        fig.tight_layout()
    plt.show()
    return fig
