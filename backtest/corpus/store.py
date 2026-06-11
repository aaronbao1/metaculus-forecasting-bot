"""Corpus store: SQLite metadata + JSON blobs for bulky series.

Layout: root/corpus.db + root/blobs/. Tables: questions, series,
contamination, crosswalk. Big payloads (CP/price history, raw snapshots)
live as JSON files under blobs/ and are referenced by relative path from
the series table (spec §3.1).

Provenance convention: "withheld" (the API tier hid a populated field —
Metaculus post-lockdown reality) is recorded in provenance["withheld"],
DISTINCT from "missing" (the source never had the field, stored as ''/None
with no withheld marker).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backtest.core import BacktestQuestion, QType

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    qid TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    native_id TEXT NOT NULL,
    qtype TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    resolution_criteria TEXT NOT NULL DEFAULT '',
    fine_print TEXT NOT NULL DEFAULT '',
    open_time TEXT,
    scheduled_close_time TEXT,
    spot_scoring_time TEXT,
    actual_resolve_time TEXT,
    resolution TEXT,            -- JSON-encoded; SQL NULL means unresolved
    options TEXT NOT NULL DEFAULT '[]',
    scaling TEXT,
    question_weight REAL NOT NULL DEFAULT 1.0,
    provenance TEXT NOT NULL DEFAULT '{}',
    schema_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS series (
    qid TEXT NOT NULL,
    kind TEXT NOT NULL,         -- "cp" | "market_price"
    blob_path TEXT NOT NULL,    -- relative to store root
    n_points INTEGER NOT NULL,
    PRIMARY KEY (qid, kind)
);
CREATE TABLE IF NOT EXISTS contamination (
    qid TEXT NOT NULL,
    model_set_hash TEXT NOT NULL,
    stratum TEXT NOT NULL,      -- clean | contaminated | contaminated-confirmed
    PRIMARY KEY (qid, model_set_hash)
);
CREATE TABLE IF NOT EXISTS crosswalk (
    qid TEXT NOT NULL,
    market_qid TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    verified INTEGER NOT NULL DEFAULT 0,
    manual_override TEXT,       -- non-null always wins in market_for()
    PRIMARY KEY (qid, market_qid)
);
"""


@dataclass
class FetchReport:
    fetched: int = 0
    withheld: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _from_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _safe(qid: str) -> str:
    """qid -> filesystem-safe blob filename fragment."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in qid)


class CorpusStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.root / "corpus.db")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def upsert_question(self, q: BacktestQuestion) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO questions
               (qid, platform, native_id, qtype, title, description,
                resolution_criteria, fine_print, open_time,
                scheduled_close_time, spot_scoring_time,
                actual_resolve_time, resolution, options, scaling,
                question_weight, provenance, schema_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (q.qid, q.platform, q.native_id, q.qtype.value, q.title,
             q.description, q.resolution_criteria, q.fine_print,
             _iso(q.open_time), _iso(q.scheduled_close_time),
             _iso(q.spot_scoring_time), _iso(q.actual_resolve_time),
             None if q.resolution is None else json.dumps(q.resolution),
             json.dumps(list(q.options)),
             None if q.scaling is None else json.dumps(q.scaling),
             q.question_weight, json.dumps(q.provenance), SCHEMA_VERSION))
        self.conn.commit()

    def get_question(self, qid: str) -> BacktestQuestion | None:
        row = self.conn.execute(
            "SELECT * FROM questions WHERE qid = ?", (qid,)).fetchone()
        return self._row_to_question(row) if row else None

    @staticmethod
    def _row_to_question(row: sqlite3.Row) -> BacktestQuestion:
        return BacktestQuestion(
            qid=row["qid"], platform=row["platform"],
            native_id=row["native_id"], qtype=QType(row["qtype"]),
            title=row["title"], description=row["description"],
            resolution_criteria=row["resolution_criteria"],
            fine_print=row["fine_print"],
            open_time=_from_iso(row["open_time"]),
            scheduled_close_time=_from_iso(row["scheduled_close_time"]),
            spot_scoring_time=_from_iso(row["spot_scoring_time"]),
            actual_resolve_time=_from_iso(row["actual_resolve_time"]),
            resolution=(None if row["resolution"] is None
                        else json.loads(row["resolution"])),
            options=tuple(json.loads(row["options"])),
            scaling=(None if row["scaling"] is None
                     else json.loads(row["scaling"])),
            question_weight=row["question_weight"],
            provenance=json.loads(row["provenance"]))
