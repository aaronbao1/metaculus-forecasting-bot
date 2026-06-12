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

import hashlib
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
    """SQLite + blob store for backtest corpus data.

    Threading contract: single-process use only. WAL mode is enabled so
    async fetchers can safely call the store from coroutines in the same
    process. Concurrent multi-thread WRITES are not safe.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            self.root / "corpus.db", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.conn.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return None

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

    # ------------------------------------------------------------------
    # Series blobs
    # ------------------------------------------------------------------

    def put_series(self, qid: str, kind: str,
                   points: list[tuple[str, object]]) -> None:
        """Replace the (qid, kind) series. points: (iso_timestamp, value).

        All timestamps are normalised to UTC ISO-8601 on write and sorted by
        their true UTC datetime, so mixed-offset inputs ("+02:00", "Z",
        "-05:00") are stored and retrieved in correct chronological order.
        Naive timestamps are treated as UTC.

        Blob filenames include a SHA-1 disambiguator so that qids whose safe
        representations collide (e.g. "p:0xAB/CD" vs "p:0xAB_CD") never
        overwrite each other.  Layout:
            blobs/<safe(qid)>__<sha1(qid)[:8]>__<kind>.json
        """
        def _to_utc_iso(ts: str) -> tuple[datetime, str]:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc, dt_utc.isoformat()

        normalized: list[tuple[str, object]] = []
        sort_keys: list[datetime] = []
        for ts, value in points:
            dt_utc, iso_utc = _to_utc_iso(ts)
            normalized.append((iso_utc, value))
            sort_keys.append(dt_utc)
        pts = [normalized[i]
               for i in sorted(range(len(normalized)),
                                key=lambda i: sort_keys[i])]

        qid_hash = hashlib.sha1(qid.encode()).hexdigest()[:8]
        path = self.blobs / f"{_safe(qid)}__{qid_hash}__{_safe(kind)}.json"
        path.write_text(json.dumps(pts))
        self.conn.execute(
            "INSERT OR REPLACE INTO series (qid, kind, blob_path, n_points)"
            " VALUES (?,?,?,?)",
            (qid, kind, str(path.relative_to(self.root)), len(pts)))
        self.conn.commit()

    def value_at(self, qid: str, kind: str, t: datetime) -> object | None:
        """Step-function lookup: last point with ts <= t, else None."""
        row = self.conn.execute(
            "SELECT blob_path FROM series WHERE qid = ? AND kind = ?",
            (qid, kind)).fetchone()
        if row is None:
            return None
        # missing blob = corrupt store, fail loud
        pts = json.loads((self.root / row["blob_path"]).read_text())
        # Normalise the probe time once so every comparison is tz-aware.
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        best = None
        for ts, value in pts:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt <= t:
                best = value
            else:
                break
        return best

    # ------------------------------------------------------------------
    # Contamination strata
    # ------------------------------------------------------------------

    def set_contamination(self, qid: str, model_set_hash: str,
                          stratum: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO contamination"
            " (qid, model_set_hash, stratum) VALUES (?,?,?)",
            (qid, model_set_hash, stratum))
        self.conn.commit()

    def get_contamination(self, qid: str,
                          model_set_hash: str) -> str | None:
        row = self.conn.execute(
            "SELECT stratum FROM contamination"
            " WHERE qid = ? AND model_set_hash = ?",
            (qid, model_set_hash)).fetchone()
        return row["stratum"] if row else None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, *, platforms=None, qtypes=None, resolved_only=True,
              open_after=None, close_before=None, stratum=None,
              model_set_hash=None, limit=None) -> list[BacktestQuestion]:
        """Date filters compare ISO-8601 strings lexicographically — fine at
        date granularity for the formats the fetchers write."""
        sql = "SELECT q.* FROM questions q"
        where: list[str] = []
        args: list = []
        if stratum is not None:
            if model_set_hash is None:
                raise ValueError("stratum filter requires model_set_hash")
            sql += (" JOIN contamination c ON c.qid = q.qid"
                    " AND c.model_set_hash = ? AND c.stratum = ?")
            args += [model_set_hash, stratum]
        if platforms:
            where.append(
                f"q.platform IN ({','.join('?' * len(platforms))})")
            args += list(platforms)
        if qtypes:
            vals = [t.value if isinstance(t, QType) else t for t in qtypes]
            where.append(f"q.qtype IN ({','.join('?' * len(vals))})")
            args += vals
        if resolved_only:
            where.append("q.resolution IS NOT NULL")
        if open_after is not None:
            where.append("q.open_time > ?")
            args.append(open_after.isoformat())
        if close_before is not None:
            where.append("q.scheduled_close_time < ?")
            args.append(close_before.isoformat())
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY q.qid"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        return [self._row_to_question(r)
                for r in self.conn.execute(sql, args).fetchall()]
