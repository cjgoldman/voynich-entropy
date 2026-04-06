"""
VMS Unicode data preparation for the BLT entropy model.

Prepares Voynich manuscript Unicode dataframe data into byte-level
sequences suitable for ingestion by a byte-level transformer model.

Usage:
    from voynpy.corpora import vms_unicode
    from vms_uprep import prepare

    # Prepare a specific range
    lines = prepare(vms_unicode.df, {
        "start": {"folio": "1r", "par": 1, "line": 1},
        "end":   {"folio": "1r", "par": 2, "line": 3}
    })

    # Prepare the entire manuscript
    lines = prepare(vms_unicode.df)
"""

import warnings
import pandas as pd

from vms_annot import (
    AnnotatedChunk,
    AnnotatedLine,
    GlyphAnnotation,
    SegmentKind,
)

# ==============================================================================
# Constants
# ==============================================================================

NULL_TOKEN = "$"
LINE_SEP = "\u2028"
PARA_SEP = "\u2029"
PARA_START = "\u00b6"  # ¶
DEFAULT_MAX_BYTES = 8192


class ByteLengthWarning(UserWarning):
    """Raised when prepared output exceeds the specified byte limit."""
    pass


# ==============================================================================
# Private helpers
# ==============================================================================

def _get_token_cols(df):
    """Return token column names (t1, t2, ...) sorted by numeric suffix."""
    cols = [c for c in df.columns if c.startswith("t") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def _row_to_line(row, token_cols):
    """Convert a dataframe row into a space-separated string with commas removed."""
    parts = []
    for col in token_cols:
        cell = row[col]
        if pd.isna(cell) or cell == NULL_TOKEN:
            continue
        parts.append(cell.replace(",", ""))
    return " ".join(parts)


def _find_row_index(df, folio, par, line):
    """Return the positional iloc index for a given folio/par/line triple."""
    mask = (
        (df["folio"] == folio)
        & (df["par"].astype(int) == int(par))
        & (df["line"].astype(int) == int(line))
    )
    matches = df.index[mask]
    if len(matches) == 0:
        available = df["folio"].unique().tolist()
        raise ValueError(
            f"No row found for folio={folio!r}, par={par}, line={line}. "
            f"Available folios: {available}"
        )
    return df.index.get_loc(matches[0])


def _validate_range_spec(spec):
    """Validate the structure of a range_spec dict."""
    if not isinstance(spec, dict):
        raise ValueError("range_spec must be a dict")
    for key in ("start", "end"):
        if key not in spec:
            raise ValueError(f"range_spec missing required key: {key!r}")
        sub = spec[key]
        if not isinstance(sub, dict):
            raise ValueError(f"range_spec[{key!r}] must be a dict")
        for field in ("folio", "par", "line"):
            if field not in sub:
                raise ValueError(
                    f"range_spec[{key!r}] missing required key: {field!r}"
                )


def _slice_df(df, range_spec):
    """Validate range_spec and return the inclusive slice of the dataframe."""
    _validate_range_spec(range_spec)
    s = range_spec["start"]
    e = range_spec["end"]
    start_pos = _find_row_index(df, s["folio"], s["par"], s["line"])
    end_pos = _find_row_index(df, e["folio"], e["par"], e["line"])
    if start_pos > end_pos:
        raise ValueError(
            f"Start position (row {start_pos}) is after end position (row {end_pos})"
        )
    return df.iloc[start_pos: end_pos + 1]


def _build_output(sliced_df, token_cols):
    """Build the list of marked-up line strings from a dataframe slice."""
    output = []
    for (_folio, _par), group in sliced_df.groupby(["folio", "par"], sort=False):
        rows = list(group.iterrows())
        last_idx = len(rows) - 1
        for i, (_row_idx, row) in enumerate(rows):
            content = _row_to_line(row, token_cols)
            is_first = i == 0
            is_last = i == last_idx
            if is_first:
                content = PARA_START + content
            content += LINE_SEP
            if is_last:
                content += PARA_SEP
            output.append(content)
    return output


def _check_byte_length(lines, max_bytes):
    """Check total UTF-8 byte length and raise if over budget."""
    total = sum(len(line.encode("utf-8")) for line in lines)
    if total > max_bytes:
        msg = (
            f"Prepared data is {total} bytes, exceeding the limit of "
            f"{max_bytes} bytes."
        )
        warnings.warn(msg, ByteLengthWarning)
        raise ValueError(msg)


# ==============================================================================
# Annotated helpers
# ==============================================================================

def _make_ann(kind, char, **kwargs):
    """Shorthand for creating a GlyphAnnotation."""
    return GlyphAnnotation(kind=kind, char=char, **kwargs)


def _row_to_annotated_line(row, token_cols, folio, par, line):
    """Convert a dataframe row into text + parallel GlyphAnnotation list.

    Mirrors _row_to_line() exactly but builds annotations per character.
    """
    text_parts = []
    annotations = []
    token_idx = 0
    for col in token_cols:
        cell = row[col]
        if pd.isna(cell) or cell == NULL_TOKEN:
            continue
        word = cell.replace(",", "")
        if text_parts:
            text_parts.append(" ")
            annotations.append(_make_ann(SegmentKind.SPACE, " "))
        for ch in word:
            text_parts.append(ch)
            annotations.append(_make_ann(
                SegmentKind.GLYPH, ch,
                folio=folio, par=par, line=line, token_pos=token_idx,
            ))
        token_idx += 1
    return "".join(text_parts), annotations


def _build_annotated_output(sliced_df, token_cols):
    """Build AnnotatedLine objects from a dataframe slice.

    Mirrors _build_output() but returns AnnotatedLine instead of plain strings.
    """
    output = []
    for (_folio, _par), group in sliced_df.groupby(["folio", "par"], sort=False):
        rows = list(group.iterrows())
        last_idx = len(rows) - 1
        for i, (_row_idx, row) in enumerate(rows):
            folio = row["folio"]
            par = int(row["par"])
            ln = int(row["line"])
            content, anns = _row_to_annotated_line(row, token_cols, folio, par, ln)
            is_first = i == 0
            is_last = i == last_idx

            text = ""
            annotations = []

            if is_first:
                text += PARA_START
                annotations.append(_make_ann(SegmentKind.PARA_START, PARA_START))

            text += content
            annotations.extend(anns)

            text += LINE_SEP
            annotations.append(_make_ann(SegmentKind.LINE_SEP, LINE_SEP))

            if is_last:
                text += PARA_SEP
                annotations.append(_make_ann(SegmentKind.PARA_SEP, PARA_SEP))

            output.append(AnnotatedLine(
                text=text,
                annotations=annotations,
                folio=folio,
                par=par,
                line=ln,
            ))
    return output


# ==============================================================================
# Public API
# ==============================================================================

def prepare(df, range_spec=None, max_bytes=DEFAULT_MAX_BYTES):
    """Prepare VMS Unicode dataframe data for the BLT entropy model.

    Args:
        df: pandas DataFrame with columns folio, par, line, t1–t26.
        range_spec: Optional dict specifying start/end folio/par/line.
            If None, the entire dataframe is processed.
        max_bytes: Maximum total UTF-8 byte length (default 8192).

    Returns:
        List of strings, one per manuscript line, with paragraph/line markers.

    Raises:
        ValueError: If range_spec is invalid or byte limit is exceeded.
    """
    if range_spec is not None:
        sliced = _slice_df(df, range_spec)
    else:
        sliced = df

    token_cols = _get_token_cols(sliced)
    lines = _build_output(sliced, token_cols)
    _check_byte_length(lines, max_bytes)
    return lines

def stack_lines(lines, max_bytes=DEFAULT_MAX_BYTES):
    """Stack a list of line strings into a single string with separators.

    Args:
        lines: List of line strings (e.g. output of prepare()).
        max_bytes: Maximum total UTF-8 byte length (default 8192).

    Returns:
        A list of strings where each sting is a stack of lines up to the byte limit.
    Raises:
        ValueError: If the byte limit is exceeded.
    """
    stacked = []
    current_stack = ""
    for line in lines:
        candidate_stack = current_stack + line
        if len(candidate_stack.encode("utf-8")) > max_bytes:
            if current_stack:
                stacked.append(current_stack)
            current_stack = line
        else:
            current_stack = candidate_stack
    if current_stack:
        stacked.append(current_stack)
    return stacked


def prepare_annotated(df, range_spec=None, max_bytes=DEFAULT_MAX_BYTES):
    """Prepare VMS Unicode dataframe data with full glyph provenance annotations.

    Args:
        df: pandas DataFrame with columns folio, par, line, t1-t26.
        range_spec: Optional range dict, same format as prepare().
        max_bytes: Maximum total UTF-8 byte length.

    Returns:
        List of AnnotatedLine, one per manuscript line.
        Each AnnotatedLine.text is identical to the corresponding string
        that prepare() would have returned.
    """
    if range_spec is not None:
        sliced = _slice_df(df, range_spec)
    else:
        sliced = df

    token_cols = _get_token_cols(sliced)
    annotated_lines = _build_annotated_output(sliced, token_cols)
    _check_byte_length([al.text for al in annotated_lines], max_bytes)
    return annotated_lines


def stack_annotated_lines(lines, max_bytes=DEFAULT_MAX_BYTES):
    """Stack AnnotatedLine objects into AnnotatedChunk objects.

    Concatenates text and annotation lists simultaneously. Each resulting
    AnnotatedChunk.text is identical to the corresponding string that
    stack_lines() would have produced from the same lines.

    Args:
        lines: List of AnnotatedLine (output of prepare_annotated()).
        max_bytes: Maximum UTF-8 byte length per chunk.

    Returns:
        List of AnnotatedChunk.
    """
    stacked = []
    current_text = ""
    current_anns = []
    for line in lines:
        candidate = current_text + line.text
        if len(candidate.encode("utf-8")) > max_bytes:
            if current_text:
                stacked.append(AnnotatedChunk(text=current_text, annotations=current_anns))
            current_text = line.text
            current_anns = list(line.annotations)
        else:
            current_text = candidate
            current_anns.extend(line.annotations)
    if current_text:
        stacked.append(AnnotatedChunk(text=current_text, annotations=current_anns))
    return stacked
