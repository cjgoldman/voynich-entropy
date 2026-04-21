"""Tests for the entropy_attrb module and its SQLite store."""

import os
import random
import sys
import tempfile

import pytest

sys.path.insert(0, "src")

from entropy_attrb import (
    attribute_and_attach,
    attribute_chunk,
    find_token_spans,
    select_top_k,
)
from entropy_attrb_store import AttributionStore
from entropy_proc import annotate_entropy
from vms_annot import (
    AnnotatedChunk,
    Attribution,
    GlyphAnnotation,
    SegmentKind,
    TokenSpan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _glyph(ch, folio="1r", par=1, line=1, token_pos=0):
    return GlyphAnnotation(
        kind=SegmentKind.GLYPH,
        char=ch,
        folio=folio,
        par=par,
        line=line,
        token_pos=token_pos,
    )


def _space():
    return GlyphAnnotation(kind=SegmentKind.SPACE, char=" ")


def _line_sep():
    return GlyphAnnotation(kind=SegmentKind.LINE_SEP, char="\u2028")


def _para_start():
    return GlyphAnnotation(kind=SegmentKind.PARA_START, char="\u00b6")


def _para_sep():
    return GlyphAnnotation(kind=SegmentKind.PARA_SEP, char="\u2029")


def _make_chunk(tokens, *, with_para_start=False, with_line_sep=False):
    """Build an AnnotatedChunk from a list of word strings (space-separated)."""
    anns = []
    text_parts = []
    if with_para_start:
        anns.append(_para_start())
        text_parts.append("\u00b6")
    for t_idx, word in enumerate(tokens):
        if t_idx > 0:
            anns.append(_space())
            text_parts.append(" ")
        for ch in word:
            anns.append(_glyph(ch, token_pos=t_idx))
            text_parts.append(ch)
    if with_line_sep:
        anns.append(_line_sep())
        text_parts.append("\u2028")
    return AnnotatedChunk(text="".join(text_parts), annotations=anns)


def _populate_baseline_entropy(chunk, constant=1.0):
    """Give each annotation a known baseline entropy value per byte."""
    text_bytes = chunk.text.encode("utf-8")
    annotate_entropy(chunk, [constant] * len(text_bytes))


def length_entropy_fn(texts):
    """Deterministic stub: each byte gets entropy = (1 / len_bytes).

    Shorter strings → larger per-byte entropy; useful for asserting
    ablation-induced shifts without worrying about model details.
    """
    results = []
    for t in texts:
        n = len(t.encode("utf-8"))
        value = 0.0 if n == 0 else 1.0 / n
        results.append([value] * n)
    return results


def byte_identity_fn(texts):
    """Deterministic stub: per-byte entropy = (byte_value / 256)."""
    results = []
    for t in texts:
        bs = t.encode("utf-8")
        results.append([b / 256.0 for b in bs])
    return results


def context_sensitive_fn(texts):
    """Per-byte entropy at position i depends on the preceding 3 bytes too.

    Useful for randomization tests: perturbing context bytes actually
    changes the reported entropy at the target position.
    """
    results = []
    for t in texts:
        bs = t.encode("utf-8")
        r = []
        for i, b in enumerate(bs):
            window = bs[max(0, i - 3) : i + 1]
            r.append(sum(window) / (256.0 * len(window)))
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# find_token_spans
# ---------------------------------------------------------------------------


class TestFindTokenSpans:
    def test_simple_two_tokens(self):
        chunk = _make_chunk(["ab", "cd"])
        spans = find_token_spans(chunk)
        assert len(spans) == 2
        assert spans[0] == TokenSpan(
            token_pos=0, start=0, end=2, byte_start=0, byte_end=2
        )
        assert spans[1] == TokenSpan(
            token_pos=1, start=3, end=5, byte_start=3, byte_end=5
        )

    def test_with_para_start_and_line_sep(self):
        chunk = _make_chunk(["ab", "cd"], with_para_start=True, with_line_sep=True)
        spans = find_token_spans(chunk)
        # PARA_START is 2 bytes (U+00B6), LINE_SEP is 3 bytes (U+2028).
        # Annotations: [PARA_START, a, b, SPACE, c, d, LINE_SEP]
        assert [s.start for s in spans] == [1, 4]
        assert [s.end for s in spans] == [3, 6]
        # Byte offsets: PARA_START=2, then "ab"=2+2=4 bytes...
        assert spans[0].byte_start == 2
        assert spans[0].byte_end == 4
        assert spans[1].byte_start == 5  # 4 + space
        assert spans[1].byte_end == 7

    def test_multibyte_glyph_bytes(self):
        # ¶ (2 bytes) inside a token isn't realistic, but multi-byte glyphs are.
        # Use 'é' (2 bytes) to check byte accounting.
        anns = [
            _glyph("é", token_pos=0),
            _glyph("a", token_pos=0),
            _space(),
            _glyph("b", token_pos=1),
        ]
        chunk = AnnotatedChunk(text="éa b", annotations=anns)
        spans = find_token_spans(chunk)
        assert spans[0].byte_start == 0
        assert spans[0].byte_end == 3  # é (2) + a (1)
        assert spans[1].byte_start == 4
        assert spans[1].byte_end == 5

    def test_empty_chunk(self):
        chunk = AnnotatedChunk(text="", annotations=[])
        assert find_token_spans(chunk) == []


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


class TestAblation:
    def test_removes_token_and_left_space(self):
        # text: "ab cd ef"; target is in "ef"; ablate "cd" → "ab ef"
        chunk = _make_chunk(["ab", "cd", "ef"])
        _populate_baseline_entropy(chunk)

        # Target is the 'e' glyph (index 6 in annotations).
        # Annotations: [a, b, SPACE, c, d, SPACE, e, f]
        def only_e(ann):
            return ann.kind is SegmentKind.GLYPH and ann.char == "e"

        captured = {}

        def capturing_fn(texts):
            captured["texts"] = list(texts)
            return length_entropy_fn(texts)

        attrs = attribute_chunk(
            chunk,
            capturing_fn,
            method="ablation",
            window_bytes=512,
            target_filter=only_e,
        )
        # Two preceding tokens in window: "ab" and "cd".
        assert len(attrs) == 2
        # When we ablate "cd" we expect "ab ef" (left space "cd " removed).
        # When we ablate "ab" we expect "cd ef" (right space " cd" removed;
        # left side has no space because "ab" is at chunk start).
        assert "ab ef" in captured["texts"]
        assert "cd ef" in captured["texts"]

    def test_delta_sign_for_length_sensitive_fn(self):
        # length_entropy_fn returns higher per-byte entropy for shorter
        # strings, so ablating a token (shrinking the input) raises
        # perturbed entropy → positive delta.
        chunk = _make_chunk(["ab", "cd", "ef"])
        _populate_baseline_entropy(chunk, constant=0.01)

        def only_e(ann):
            return ann.kind is SegmentKind.GLYPH and ann.char == "e"

        attrs = attribute_chunk(
            chunk,
            length_entropy_fn,
            method="ablation",
            target_filter=only_e,
        )
        for a in attrs:
            assert a.delta > 0
            assert a.perturbed_entropy > a.baseline_entropy

    def test_skips_target_own_token(self):
        # Target 'c' should not attribute to its own token "cd".
        chunk = _make_chunk(["ab", "cd", "ef"])
        _populate_baseline_entropy(chunk)

        def only_c(ann):
            return ann.kind is SegmentKind.GLYPH and ann.char == "c"

        attrs = attribute_chunk(
            chunk,
            length_entropy_fn,
            method="ablation",
            target_filter=only_c,
        )
        # Only preceding "ab" should appear; not "cd" (own) or "ef" (following).
        token_chars = set()
        for a in attrs:
            span = a.context_token
            token_chars.add(
                "".join(x.char for x in chunk.annotations[span.start : span.end])
            )
        assert token_chars == {"ab"}

    def test_requires_byte_entropies(self):
        chunk = _make_chunk(["ab", "cd"])
        # Do not populate byte_entropies.
        with pytest.raises(ValueError, match="byte_entropies"):
            attribute_chunk(chunk, length_entropy_fn, method="ablation")


# ---------------------------------------------------------------------------
# Randomization
# ---------------------------------------------------------------------------


class TestRandomization:
    def test_same_seed_same_results(self):
        chunk = _make_chunk(["ab", "cd", "ef"])
        _populate_baseline_entropy(chunk)

        rng1 = random.Random(42)
        rng2 = random.Random(42)
        a1 = attribute_chunk(chunk, byte_identity_fn, method="randomization", rng=rng1)
        a2 = attribute_chunk(chunk, byte_identity_fn, method="randomization", rng=rng2)

        assert len(a1) == len(a2)
        for x, y in zip(a1, a2):
            assert x.delta == pytest.approx(y.delta)

    def test_different_seeds_differ(self):
        # Use a larger alphabet so the chance of identical draws is tiny.
        # context_sensitive_fn looks at preceding bytes, so randomizing
        # context tokens changes the target's reported entropy.
        chunk = _make_chunk(["abcdef", "ghijkl", "mnopqr"])
        _populate_baseline_entropy(chunk)

        a1 = attribute_chunk(
            chunk,
            context_sensitive_fn,
            method="randomization",
            rng=random.Random(1),
        )
        a2 = attribute_chunk(
            chunk,
            context_sensitive_fn,
            method="randomization",
            rng=random.Random(999),
        )
        assert any(x.delta != pytest.approx(y.delta) for x, y in zip(a1, a2))

    def test_alphabet_restricted_to_chunk_glyphs(self):
        # Check that randomization only samples from codepoints that actually
        # appear as GLYPHs in the chunk.  We capture the perturbed strings.
        chunk = _make_chunk(["abc", "xyz"])
        _populate_baseline_entropy(chunk)
        allowed = {a.char for a in chunk.annotations if a.kind is SegmentKind.GLYPH}

        captured = []

        def capture(texts):
            captured.extend(texts)
            return byte_identity_fn(texts)

        def only_z(ann):
            return ann.kind is SegmentKind.GLYPH and ann.char == "z"

        attribute_chunk(
            chunk,
            capture,
            method="randomization",
            rng=random.Random(0),
            target_filter=only_z,
        )
        # Each captured string must contain only allowed glyphs or the single
        # space we inserted.
        for pert in captured:
            for ch in pert:
                assert ch in allowed or ch == " "

    def test_preserves_byte_length(self):
        chunk = _make_chunk(["abc", "def"])
        _populate_baseline_entropy(chunk)
        original_bytes = len(chunk.text.encode("utf-8"))

        captured = []

        def capture(texts):
            captured.extend(texts)
            return byte_identity_fn(texts)

        attribute_chunk(
            chunk,
            capture,
            method="randomization",
            rng=random.Random(7),
        )
        for pert in captured:
            assert len(pert.encode("utf-8")) == original_bytes

    def test_preserves_byte_length_with_mixed_widths(self):
        # Chunk with both single-byte (ASCII) and 2-byte (é) glyphs.
        # Same-width bucketed sampling must preserve byte length so that
        # new_tbs = target_byte_start stays valid.
        anns = [
            _glyph("é", token_pos=0),
            _glyph("a", token_pos=0),
            _space(),
            _glyph("b", token_pos=1),
            _glyph("c", token_pos=1),
            _space(),
            _glyph("d", token_pos=2),
            _glyph("é", token_pos=2),
        ]
        chunk = AnnotatedChunk(text="éa bc dé", annotations=anns)
        _populate_baseline_entropy(chunk)
        original_bytes = len(chunk.text.encode("utf-8"))

        captured = []

        def capture(texts):
            captured.extend(texts)
            return byte_identity_fn(texts)

        # Many rounds to stress different random draws.
        attribute_chunk(
            chunk,
            capture,
            method="randomization",
            rng=random.Random(0),
        )
        assert captured  # must have produced at least one perturbed variant
        for pert in captured:
            assert len(pert.encode("utf-8")) == original_bytes


# ---------------------------------------------------------------------------
# Window clipping
# ---------------------------------------------------------------------------


class TestWindowClipping:
    def test_early_target_has_fewer_context_tokens(self):
        # All glyphs unique so target_filter matches exactly one annotation.
        chunk = _make_chunk(["aa", "bb", "cc", "dd", "xy"])
        _populate_baseline_entropy(chunk)

        # Byte layout (one byte per ASCII glyph, one per SPACE):
        #   aa 0..2 | bb 3..5 | cc 6..8 | dd 9..11 | xy 12..14

        def only(char):
            return lambda a: a.kind is SegmentKind.GLYPH and a.char == char

        # 'y' at byte 13.  window=7 → [6, 13).  cc (6..8) and dd (9..11)
        # both fall fully inside; xy is the target's own token (excluded).
        attrs_small = attribute_chunk(
            chunk,
            length_entropy_fn,
            method="ablation",
            window_bytes=7,
            target_filter=only("y"),
        )
        # window 100 → all four preceding tokens (aa, bb, cc, dd).
        attrs_large = attribute_chunk(
            chunk,
            length_entropy_fn,
            method="ablation",
            window_bytes=100,
            target_filter=only("y"),
        )
        # 'x' is the first glyph of its own token; no preceding token
        # can be attributed since "xy" is excluded as the target's own token.
        # But with window 100, aa/bb/cc/dd still precede 'x' — its window is
        # [−88, 12) and all four preceding tokens fit.
        attrs_x_small = attribute_chunk(
            chunk,
            length_entropy_fn,
            method="ablation",
            window_bytes=2,
            target_filter=only("x"),
        )
        assert len(attrs_small) == 2
        assert len(attrs_large) == 4
        # window=2 for 'x' (byte 12) → [10, 12); no token fits fully.
        assert len(attrs_x_small) == 0


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_perturbed_length_matches_entropy_fn_input(self):
        chunk = _make_chunk(["ab", "cd", "ef"])
        _populate_baseline_entropy(chunk)

        def strict_fn(texts):
            out = []
            for t in texts:
                n = len(t.encode("utf-8"))
                out.append([0.5] * n)
            return out

        # Should not raise: lengths line up.
        attribute_chunk(chunk, strict_fn, method="ablation")
        attribute_chunk(chunk, strict_fn, method="randomization", rng=random.Random(0))

    def test_entropy_fn_wrong_length_raises(self):
        chunk = _make_chunk(["ab", "cd", "ef"])
        _populate_baseline_entropy(chunk)

        def wrong_fn(texts):
            # Always return a single float per text.
            return [[0.5] for _ in texts]

        with pytest.raises(ValueError, match="bytes for"):
            attribute_chunk(chunk, wrong_fn, method="ablation")


# ---------------------------------------------------------------------------
# attribute_and_attach
# ---------------------------------------------------------------------------


class TestAttachMode:
    def test_attaches_attributions_to_targets(self):
        chunk = _make_chunk(["ab", "cd", "ef"])
        _populate_baseline_entropy(chunk)
        attribute_and_attach(chunk, length_entropy_fn, method="ablation")

        # Glyphs in the first token have no preceding context → None.
        first_token_glyph = chunk.annotations[0]
        assert first_token_glyph.attributions is None

        # Glyphs in later tokens should have attributions.
        e_idx = chunk.text.index("e")
        # text index aligns with annotation index (1 char per ann).
        e_ann = chunk.annotations[e_idx]
        assert e_ann.attributions is not None
        assert all(isinstance(a, Attribution) for a in e_ann.attributions)


# ---------------------------------------------------------------------------
# select_top_k
# ---------------------------------------------------------------------------


class TestSelectTopK:
    def test_ranks_by_abs_delta(self):
        span = TokenSpan(token_pos=0, start=0, end=1, byte_start=0, byte_end=1)
        attrs = [
            Attribution(
                target_ann_index=5,
                context_token=span,
                method="ablation",
                baseline_entropy=1.0,
                perturbed_entropy=1.5,
                delta=0.5,
            ),
            Attribution(
                target_ann_index=5,
                context_token=span,
                method="ablation",
                baseline_entropy=1.0,
                perturbed_entropy=3.0,
                delta=2.0,
            ),
            Attribution(
                target_ann_index=5,
                context_token=span,
                method="ablation",
                baseline_entropy=1.0,
                perturbed_entropy=0.9,
                delta=-0.1,
            ),
        ]
        ranked = select_top_k(attrs, k=2)
        assert len(ranked) == 2
        assert ranked[0][0] == 0
        assert ranked[0][1].delta == pytest.approx(2.0)
        assert ranked[1][0] == 1
        assert ranked[1][1].delta == pytest.approx(0.5)

    def test_groups_by_target_and_method(self):
        span = TokenSpan(token_pos=0, start=0, end=1, byte_start=0, byte_end=1)

        def mk(idx, method, delta):
            return Attribution(
                target_ann_index=idx,
                context_token=span,
                method=method,
                baseline_entropy=0.0,
                perturbed_entropy=delta,
                delta=delta,
            )

        attrs = [
            mk(1, "ablation", 0.1),
            mk(1, "randomization", 0.2),
            mk(2, "ablation", 0.3),
        ]
        ranked = select_top_k(attrs, k=10)
        assert len(ranked) == 3
        # Each group gets its own rank-0.
        by_group = {}
        for rank, a in ranked:
            by_group[(a.target_ann_index, a.method)] = rank
        assert set(by_group.values()) == {0}


# ---------------------------------------------------------------------------
# AttributionStore
# ---------------------------------------------------------------------------


class TestAttributionStore:
    def test_end_to_end_write_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "attr.sqlite")
            chunk = _make_chunk(["ab", "cd", "ef"])
            _populate_baseline_entropy(chunk)
            attrs = attribute_chunk(chunk, length_entropy_fn, method="ablation")

            with AttributionStore.open(path) as store:
                run_id = store.start_run(
                    corpus="test",
                    method="ablation",
                    window_bytes=512,
                    top_k=10,
                    model="stub",
                    seed=42,
                )
                written = store.write_chunk(
                    run_id, 0, chunk, attrs, top_k=10, corpus="test"
                )
                store.finish_run(run_id)

            assert written > 0

            # Re-open and check round-trip via raw SQL (no pandas dependency).
            import sqlite3

            conn = sqlite3.connect(path)
            rows = conn.execute(
                "SELECT run_id, chunk_id, target_ann_index, method, rank, delta "
                "FROM attributions ORDER BY target_ann_index, rank"
            ).fetchall()
            run_rows = conn.execute(
                "SELECT run_id, method, window_bytes, top_k FROM runs"
            ).fetchall()
            conn.close()

            assert len(run_rows) == 1
            assert run_rows[0][1] == "ablation"
            assert run_rows[0][2] == 512
            # Every attribution row should reference the same run and be ranked 0..k.
            for row in rows:
                assert row[0] == run_id
                assert row[3] == "ablation"
                assert row[4] >= 0

    def test_primary_key_blocks_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "attr.sqlite")
            chunk = _make_chunk(["ab", "cd", "ef"])
            _populate_baseline_entropy(chunk)
            attrs = attribute_chunk(chunk, length_entropy_fn, method="ablation")

            with AttributionStore.open(path) as store:
                run_id = store.start_run(
                    corpus="test",
                    method="ablation",
                    window_bytes=512,
                    top_k=10,
                )
                store.write_chunk(run_id, 0, chunk, attrs)
                import sqlite3

                with pytest.raises(sqlite3.IntegrityError):
                    store.write_chunk(run_id, 0, chunk, attrs)

    def test_ctx_byte_offset_is_negative(self):
        """Preceding context tokens should have negative ctx_byte_offset."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "attr.sqlite")
            chunk = _make_chunk(["ab", "cd", "ef"])
            _populate_baseline_entropy(chunk)
            attrs = attribute_chunk(chunk, length_entropy_fn, method="ablation")

            with AttributionStore.open(path) as store:
                run_id = store.start_run(
                    corpus="test",
                    method="ablation",
                    window_bytes=512,
                    top_k=10,
                )
                store.write_chunk(run_id, 0, chunk, attrs)
                import sqlite3

                conn = sqlite3.connect(path)
                offsets = [
                    r[0]
                    for r in conn.execute(
                        "SELECT ctx_byte_offset FROM attributions"
                    ).fetchall()
                ]
                conn.close()

            assert offsets
            assert all(o < 0 for o in offsets)
