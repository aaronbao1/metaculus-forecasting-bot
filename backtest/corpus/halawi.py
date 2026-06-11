"""Halawi et al. 2024 (arXiv 2402.18563) dataset importer.

Reads LOCAL files only — never the network. Download first (manual step):

    pip install -U "huggingface_hub[cli]"
    huggingface-cli download YuehHanChen/forecasting \
        --repo-type dataset --local-dir data/backtest/raw/halawi

VERIFY THE RECORD SHAPE against the downloaded files before a real import:
this importer targets the curated binary split and tolerates the common
field aliases via _first(); unknown layouts surface as per-record errors,
never crashes. Dataset text is stored VERBATIM with provenance (text may
drift from current platform copies — spec §3.2).

Resolution-date validation runs on every record (resolve < open =>
quarantined): >=3.8% of even the curated set had resolved-before-forecast
defects (spec §3.4).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport


def _first(rec: dict, *names, default=None):
    for n in names:
        if rec.get(n) not in (None, ""):
            return rec[n]
    return default


def _parse_dt(v) -> datetime | None:
    if v in (None, ""):
        return None
    return datetime.fromisoformat(str(v))


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        # Only exact binary endpoints are unambiguous; anything else
        # (e.g. 0.5 for annulled/ambiguous) is not a confident outcome.
        if v == 1 or v == 1.0:
            return True
        if v == 0 or v == 0.0:
            return False
        return None
    if isinstance(v, str) and v.strip().lower() in ("yes", "no"):
        return v.strip().lower() == "yes"
    return None


def _iter_records(dataset_dir: Path):
    for path in sorted(Path(dataset_dir).rglob("*.json*")):
        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
            yield from (data if isinstance(data, list) else [data])


def import_halawi(store: CorpusStore, dataset_dir) -> FetchReport:
    report = FetchReport()
    for rec in _iter_records(dataset_dir):
        try:
            title = str(_first(rec, "question", "title", default=""))
            rid = str(_first(rec, "id", "question_id")
                      or hashlib.sha256(title.encode()).hexdigest()[:12])
            resolution = _to_bool(_first(rec, "resolution", "answer"))
            if resolution is None:
                report.errors.append(f"{rid}: no usable resolution")
                continue
            open_time = _parse_dt(_first(
                rec, "date_begin", "begin_date", "open_time"))
            close_time = _parse_dt(_first(
                rec, "date_close", "close_date", "close_time"))
            resolve_time = _parse_dt(_first(
                rec, "date_resolve_at", "resolve_date", "resolution_date"))
            if (open_time and resolve_time
                    and _aware(resolve_time) < _aware(open_time)):
                report.errors.append(
                    f"{rid}: resolved before open — quarantined")
                continue
            store.upsert_question(BacktestQuestion(
                qid=f"halawi:{rid}", platform="halawi", native_id=rid,
                qtype=QType.BINARY, title=title,
                description=str(_first(
                    rec, "background", "description", default="")),
                resolution_criteria=str(_first(
                    rec, "resolution_criteria", default="")),
                open_time=open_time, scheduled_close_time=close_time,
                actual_resolve_time=resolve_time, resolution=resolution,
                provenance={
                    "source": "halawi",
                    "origin_platform": str(_first(
                        rec, "source", "platform", default="unknown")),
                    "license_note": ("research dataset (arXiv 2402.18563);"
                                     " text stored verbatim"),
                    "imported_at":
                        datetime.now(timezone.utc).isoformat()}))
            report.fetched += 1
        except Exception as e:      # per-record isolation
            report.errors.append(f"record skipped: {e}")
    return report
