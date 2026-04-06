"""
Voynich manuscript annotation data structures.

Dataclasses for preserving folio/paragraph/line/token provenance
as text flows through the VMS preparation and entropy pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class SegmentKind(Enum):
    """Type of segment in the annotated text."""
    GLYPH = auto()       # A real Voynich glyph with manuscript provenance
    SPACE = auto()       # Word-boundary space between tokens on the same line
    PARA_START = auto()  # ¶ marker (U+00B6)
    LINE_SEP = auto()    # U+2028 line separator
    PARA_SEP = auto()    # U+2029 paragraph separator


@dataclass
class GlyphAnnotation:
    """Annotation for a single Unicode character in the prepared text.

    For GLYPH kind, manuscript coordinates (folio, par, line, token_pos)
    are populated.  For structural markers (SPACE, PARA_START, etc.),
    those fields are None.

    The char field always stores the Unicode character so that byte
    length can be computed via len(char.encode("utf-8")).
    """
    kind: SegmentKind
    char: str

    # Manuscript coordinates — populated only when kind == GLYPH
    folio: Optional[str] = None
    par: Optional[int] = None
    line: Optional[int] = None
    token_pos: Optional[int] = None  # 0-based index among non-null tokens on the line

    # Populated by entropy_proc.annotate_entropy(); None until then
    byte_entropies: Optional[list] = field(default=None, repr=False)


@dataclass
class AnnotatedLine:
    """One manuscript line with its structural markers, before chunking.

    Invariant: len(annotations) == len(text) in Unicode codepoints.
    """
    text: str
    annotations: list[GlyphAnnotation]
    folio: str
    par: int
    line: int


@dataclass
class AnnotatedChunk:
    """One chunk produced by stack_annotated_lines(), ready for BLT inference.

    Invariant: len(annotations) == len(text) in Unicode codepoints.
    """
    text: str
    annotations: list[GlyphAnnotation]
