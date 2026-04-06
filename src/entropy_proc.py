"""
Attach BLT per-byte entropy values to AnnotatedChunk metadata.

Usage:
    from entropy_proc import annotate_entropy

    chunk = annotate_entropy(chunk, entropy_values)
    # Now each chunk.annotations[i].byte_entropies is a list of floats
"""

from vms_annot import AnnotatedChunk


def annotate_entropy(chunk, entropy_values):
    """Attach per-byte entropy values to each GlyphAnnotation in the chunk.

    Walks the annotation list and entropy values in parallel using a byte
    cursor.  Each annotation receives a slice of entropy_values whose
    length equals the UTF-8 byte count of that character.

    Args:
        chunk: An AnnotatedChunk (output of stack_annotated_lines()).
        entropy_values: List of floats, one per UTF-8 byte of chunk.text.

    Returns:
        The same chunk with annotation.byte_entropies populated.

    Raises:
        ValueError: If len(entropy_values) != total UTF-8 byte count of chunk.text.
    """
    expected_bytes = len(chunk.text.encode("utf-8"))
    if len(entropy_values) != expected_bytes:
        raise ValueError(
            f"entropy_values length ({len(entropy_values)}) does not match "
            f"text byte length ({expected_bytes})"
        )

    byte_cursor = 0
    for ann in chunk.annotations:
        n_bytes = len(ann.char.encode("utf-8"))
        ann.byte_entropies = entropy_values[byte_cursor : byte_cursor + n_bytes]
        byte_cursor += n_bytes

    assert byte_cursor == len(entropy_values), (
        f"Byte cursor ({byte_cursor}) != entropy length ({len(entropy_values)})"
    )

    return chunk
