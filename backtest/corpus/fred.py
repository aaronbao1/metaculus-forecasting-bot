"""FRED target-series fetcher: stores observations as kind="target" series
under qid "fred:<series_id>" plus a minimal unresolved question row so the
series has an addressable, titled qid (synthetic.generate_questions and
TimeSeriesSystem key on qids).

FRED data is public domain with attribution; an API key is free
(FRED_API_KEY). Vintage-correctness note: this pulls the CURRENT revision
series — for revision-sensitive experiments use ALFRED vintages (not
implemented)."""
from __future__ import annotations

import logging
import os

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport

logger = logging.getLogger(__name__)

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"


async def _http_get_json(url: str, params: dict):
    """Monkeypatch seam (same convention as timetravel/providers.py)."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0,
                                 follow_redirects=True) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_fred(store: CorpusStore, series_ids: list[str],
                     api_key: str | None = None) -> FetchReport:
    report = FetchReport()
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        report.errors.append("FRED_API_KEY not configured")
        return report
    for sid in series_ids:
        try:
            meta = await _http_get_json(FRED_SERIES_URL, {
                "series_id": sid, "api_key": key, "file_type": "json"})
            info = ((meta or {}).get("seriess") or [{}])[0]
            data = await _http_get_json(FRED_OBS_URL, {
                "series_id": sid, "api_key": key, "file_type": "json"})
        except Exception as e:
            report.errors.append(f"{sid}: {type(e).__name__}: {e}")
            continue
        points = []
        for obs in (data or {}).get("observations") or []:
            value = obs.get("value")
            if value in (None, "", "."):        # "." = FRED missing marker
                continue
            try:
                points.append((f"{obs['date']}T00:00:00+00:00",
                               float(value)))
            except (ValueError, KeyError) as e:
                report.errors.append(f"{sid}: bad observation {obs}: {e}")
        if not points:
            report.errors.append(f"{sid}: no usable observations")
            continue
        qid = f"fred:{sid}"
        store.upsert_question(BacktestQuestion(
            qid=qid, platform="fred", native_id=sid, qtype=QType.NUMERIC,
            title=info.get("title") or sid,
            description=(f"FRED series {sid} "
                         f"({info.get('units') or 'unknown units'}); "
                         "container row for the target series — generate "
                         "questions with `backtest synth-questions`."),
            scaling=None,
            provenance={"source": "fred", "units": info.get("units"),
                        "license": "public domain w/ attribution",
                        "revision_note": "current vintage, not ALFRED"}))
        store.put_series(qid, "target", points)
        report.fetched += 1
    return report
