"""
SQLite-backed persistence for entropy-reduction attributions.

A run records one attribute_chunk() sweep over a corpus.  Each row in
the attributions table is the top-K contribution of a single context
token to a single target glyph under a single method, with enough
provenance to join back to the source chunk.

Schema is stable across partial runs: different SQLite files can be
merged with ATTACH DATABASE; the (run_id, chunk_id, target_ann_index,
method, rank) primary key prevents duplicate rows.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
import uuid
from typing import Iterable, Optional

from entropy_attrb import select_top_k
from vms_annot import AnnotatedChunk, Attribution

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    corpus        TEXT,
    chunk_range   TEXT,
    method        TEXT NOT NULL,
    window_bytes  INTEGER NOT NULL,
    top_k         INTEGER NOT NULL,
    model         TEXT,
    seed          INTEGER,
    finished_utc  TEXT
)
"""

_ATTR_DDL = """
CREATE TABLE IF NOT EXISTS attributions (
    run_id            TEXT NOT NULL,
    corpus            TEXT,
    chunk_id          INTEGER NOT NULL,
    folio             TEXT,
    par               INTEGER,
    line              INTEGER,
    token_pos         INTEGER,
    target_ann_index  INTEGER NOT NULL,
    target_char       TEXT NOT NULL,
    baseline_entropy  REAL NOT NULL,
    method            TEXT NOT NULL,
    rank              INTEGER NOT NULL,
    ctx_token_pos     INTEGER NOT NULL,
    ctx_token_text    TEXT,
    ctx_byte_offset   INTEGER NOT NULL,
    perturbed_entropy REAL NOT NULL,
    delta             REAL NOT NULL,
    PRIMARY KEY (run_id, chunk_id, target_ann_index, method, rank),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
)
"""

_ATTR_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_attributions_lookup
    ON attributions (run_id, chunk_id, target_ann_index, method)
"""


def _ctx_token_text(chunk: AnnotatedChunk, span) -> str:
    return "".join(a.char for a in chunk.annotations[span.start : span.end])


class AttributionStore:
    """Thin SQLite wrapper for attribution runs."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()

    # ------------------------------------------------------------------ open

    @classmethod
    def open(cls, path: str) -> "AttributionStore":
        """Open (creating if needed) a SQLite database at path."""
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = ON")
        return cls(conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "AttributionStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(_RUNS_DDL)
            self.conn.execute(_ATTR_DDL)
            self.conn.execute(_ATTR_INDEX_DDL)

    # ------------------------------------------------------------------ runs

    def start_run(
        self,
        *,
        corpus: str,
        method: str,
        window_bytes: int,
        top_k: int,
        model: Optional[str] = None,
        seed: Optional[int] = None,
        chunk_range: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Insert a row into runs and return its run_id (UUID4 by default)."""
        run_id = run_id or str(uuid.uuid4())
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs (run_id, timestamp_utc, corpus, chunk_range, "
                "method, window_bytes, top_k, model, seed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    ts,
                    corpus,
                    chunk_range,
                    method,
                    window_bytes,
                    top_k,
                    model,
                    seed,
                ),
            )
        return run_id

    def finish_run(self, run_id: str) -> None:
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET finished_utc = ? WHERE run_id = ?",
                (ts, run_id),
            )

    # ------------------------------------------------------------- write

    def write_chunk(
        self,
        run_id: str,
        chunk_id: int,
        chunk: AnnotatedChunk,
        attributions: Iterable[Attribution],
        top_k: int = 10,
        corpus: Optional[str] = None,
    ) -> int:
        """Insert top-K attributions for one chunk.  Returns rows written.

        Attributions are grouped by (target_ann_index, method) and ranked
        by |delta| descending; only the first top_k rows per group are
        written.
        """
        rows = []
        for rank, a in select_top_k(attributions, k=top_k):
            target_ann = chunk.annotations[a.target_ann_index]
            span = a.context_token
            # Signed byte offset: ctx_token_start − target_start.  Negative
            # for preceding context (the expected case).
            target_byte_start = sum(
                len(x.char.encode("utf-8"))
                for x in chunk.annotations[: a.target_ann_index]
            )
            ctx_byte_offset = span.byte_start - target_byte_start
            rows.append(
                (
                    run_id,
                    corpus,
                    chunk_id,
                    target_ann.folio,
                    target_ann.par,
                    target_ann.line,
                    target_ann.token_pos,
                    a.target_ann_index,
                    target_ann.char,
                    a.baseline_entropy,
                    a.method,
                    rank,
                    span.token_pos,
                    _ctx_token_text(chunk, span),
                    ctx_byte_offset,
                    a.perturbed_entropy,
                    a.delta,
                )
            )
        if not rows:
            return 0
        with self.conn:
            self.conn.executemany(
                "INSERT INTO attributions ("
                "run_id, corpus, chunk_id, folio, par, line, token_pos, "
                "target_ann_index, target_char, baseline_entropy, method, "
                "rank, ctx_token_pos, ctx_token_text, ctx_byte_offset, "
                "perturbed_entropy, delta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------- read

    def load(
        self,
        run_id: Optional[str] = None,
        folio: Optional[str] = None,
        chunk_id: Optional[int] = None,
        method: Optional[str] = None,
        target_ann_index: Optional[int] = None,
    ):
        """Return a pandas DataFrame of matching attribution rows.

        Filters default to None (no filter).  pandas is imported lazily
        so the store module works in environments without pandas
        (callers that avoid .load() will not trigger the import).
        """
        import pandas as pd

        clauses = []
        params: list = []
        for col, val in (
            ("run_id", run_id),
            ("folio", folio),
            ("chunk_id", chunk_id),
            ("method", method),
            ("target_ann_index", target_ann_index),
        ):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM attributions {where} ORDER BY chunk_id, target_ann_index, method, rank"
        return pd.read_sql_query(query, self.conn, params=params)

    def to_parquet(self, run_id: str, path: str) -> None:
        """Export one run's attributions to a Parquet file."""
        df = self.load(run_id=run_id)
        df.to_parquet(path, index=False)
