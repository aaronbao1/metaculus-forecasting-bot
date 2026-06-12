"""ForecastBench datasets importer (arXiv 2409.19839, CC BY-SA 4.0).

Reads a LOCAL clone only — never the network. Download first (manual):

    git clone https://github.com/forecastingresearch/forecastbench-datasets \
        data/backtest/raw/forecastbench-datasets

Targets the nightly JSON layout: question_sets/*.json holding
{"questions": [...]} and resolution_sets/*.json holding
{"resolutions": [...]} anywhere under the given directory (rglob — the
repo nests them under datasets/). Questions and resolutions join on
(source, id). Combination questions (combination_of != "N/A") are derived
and skipped. freeze_datetime_value is the platform crowd value at freeze —
stored as a one-point "cp" series so CP-echo opponents work on
ForecastBench questions too.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport


def _parse_dt(v) -> datetime | None:
    # Dataset-source questions (acled/fred/wikipedia/...) carry the literal
    # string "N/A" in market date fields — verified against the real repo
    # 2026-06-11 (8,596/11,412 records would otherwise error out).
    if v in (None, "", "N/A"):
        return None
    return datetime.fromisoformat(str(v))


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def import_forecastbench(store: CorpusStore, datasets_dir) -> FetchReport:
    report = FetchReport()
    datasets_dir = Path(datasets_dir)

    resolutions: dict[tuple[str, str], object] = {}
    for path in sorted(datasets_dir.rglob("resolution_sets/*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            report.errors.append(f"{path.name}: {e}")
            continue
        for r in data.get("resolutions") or []:
            resolutions[(str(r.get("source")), str(r.get("id")))] = \
                r.get("resolved_to")

    seen: set[str] = set()
    for path in sorted(datasets_dir.rglob("question_sets/*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            report.errors.append(f"{path.name}: {e}")
            continue
        for rec in data.get("questions") or []:
            try:
                if rec.get("combination_of") not in (None, "N/A", [], ""):
                    continue            # derived combo question
                source = str(rec.get("source"))
                rid = str(rec.get("id"))
                qid = f"forecastbench:{source}:{rid}"
                resolved_to = resolutions.get((source, rid))
                resolution = None
                if resolved_to in (0, 1, 0.0, 1.0):
                    resolution = bool(round(float(resolved_to)))
                open_time = _parse_dt(
                    rec.get("market_info_open_datetime"))
                close_time = _parse_dt(
                    rec.get("market_info_close_datetime"))
                resolve_time = _parse_dt(rec.get("resolution_date"))
                if (open_time and resolve_time
                        and _aware(resolve_time) < _aware(open_time)):
                    report.errors.append(
                        f"{qid}: resolved before open — quarantined")
                    continue
                store.upsert_question(BacktestQuestion(
                    qid=qid, platform="forecastbench",
                    native_id=f"{source}:{rid}", qtype=QType.BINARY,
                    title=str(rec.get("question") or ""),
                    description=str(rec.get("background") or ""),
                    resolution_criteria=str(
                        rec.get("resolution_criteria") or ""),
                    open_time=open_time,
                    scheduled_close_time=close_time,
                    actual_resolve_time=resolve_time,
                    resolution=resolution,
                    provenance={
                        "source": "forecastbench",
                        "origin_platform": source,
                        "license": "CC BY-SA 4.0",
                        "question_set": path.name,
                        "imported_at":
                            datetime.now(timezone.utc).isoformat()}))
                freeze_t = rec.get("freeze_datetime")
                freeze_v = rec.get("freeze_datetime_value")
                if freeze_t and freeze_v not in (None, ""):
                    try:
                        store.put_series(
                            qid, "cp", [(str(freeze_t), float(freeze_v))])
                    except (TypeError, ValueError):
                        pass            # non-numeric freeze values
                if qid not in seen:
                    seen.add(qid)
                    report.fetched += 1
            except Exception as e:      # per-record isolation
                report.errors.append(
                    f"{path.name}:{rec.get('id')}: {e}")
    return report
