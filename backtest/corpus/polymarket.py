"""Polymarket fetcher: Gamma /markets enumeration + CLOB prices-history.

No auth needed. Gamma quirk (verified live in bot/markets.py): outcomes /
outcomePrices / clobTokenIds are JSON-encoded STRINGS. Resolved markets
return EMPTY price history below fidelity~720 (12h) — 12h-1d granularity
is the design assumption for price-at-T (spec §3.2).

Snapshot-on-sight: market history is decaying (Polymarket pruning is
undocumented; /orderbook-history already died) — everything fetched is
stored permanently via put_series.

Resolution inference: a settled Yes/No market's outcomePrices collapse to
0/1; >=0.99 on the Yes outcome -> True, <=0.01 -> False, anything between
stays None (closed-but-not-settled; unscoreable).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport

logger = logging.getLogger(__name__)

GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES = "https://clob.polymarket.com/prices-history"
FIDELITY_MIN = 720          # minutes; resolved markets empty below ~12h
PAGE_SIZE = 100
HTTP_TIMEOUT_S = 30.0


async def _get_json(url: str, params: dict | None = None,
                    headers: dict | None = None, client=None):
    import httpx
    if client is not None:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as one_shot:
        resp = await one_shot.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _parse_dt(v) -> datetime | None:
    if v in (None, ""):
        return None
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


async def _ingest_market(store: CorpusStore, m: dict,
                         report: FetchReport) -> bool:
    outcomes = [str(o).lower() for o in json.loads(m["outcomes"])]
    if sorted(outcomes) != ["no", "yes"]:
        return False                       # only Yes/No binaries
    yes_idx = outcomes.index("yes")        # map index via outcomes field
    prices = [float(p) for p in json.loads(m["outcomePrices"])]
    clob_ids = json.loads(m.get("clobTokenIds") or "[]")

    yes_price = prices[yes_idx]
    if yes_price >= 0.99:
        resolution: bool | None = True
    elif yes_price <= 0.01:
        resolution = False
    else:
        resolution = None                  # closed but not settled 0/1

    qid = f"polymarket:{m['id']}"
    open_time = _parse_dt(m.get("startDate"))
    resolve_time = _parse_dt(m.get("closedTime"))   # Gamma settle timestamp
    # resolution-date validation at import (spec §3.4)
    if open_time and resolve_time and resolve_time < open_time:
        report.errors.append(
            f"quarantined {qid}: resolve_time < open_time")
        return False
    store.upsert_question(BacktestQuestion(
        qid=qid, platform="polymarket", native_id=str(m["id"]),
        qtype=QType.BINARY, title=m.get("question") or "",
        description=m.get("description") or "",
        open_time=open_time,
        scheduled_close_time=_parse_dt(m.get("endDate")),
        resolution=resolution,
        provenance={
            "source": "polymarket",
            "condition_id": m.get("conditionId"),
            "slug": m.get("slug"),
            "note": "resolution inferred from settled outcomePrices",
            "fetched_at": datetime.now(timezone.utc).isoformat()}))

    if len(clob_ids) > yes_idx:            # snapshot-on-sight
        hist = await _get_json(CLOB_PRICES, params={
            "market": clob_ids[yes_idx], "fidelity": FIDELITY_MIN})
        pts = [(datetime.fromtimestamp(h["t"], tz=timezone.utc).isoformat(),
                float(h["p"]))
               for h in (hist.get("history") or [])]
        if pts:
            store.put_series(qid, "market_price", pts)
    report.fetched += 1
    return True


async def fetch_polymarket(store: CorpusStore, *, closed=True,
                           start_date_min=None,
                           limit=None) -> FetchReport:
    report = FetchReport()
    offset = 0
    while True:
        params: dict = {"closed": str(bool(closed)).lower(),
                        "limit": PAGE_SIZE, "offset": offset}
        if start_date_min is not None:
            params["start_date_min"] = start_date_min
        try:
            page = await _get_json(GAMMA_MARKETS, params=params)
        except Exception as e:
            report.errors.append(f"gamma page offset={offset}: {e}")
            break
        for m in page or []:
            if limit is not None and report.fetched >= limit:
                return report
            try:
                await _ingest_market(store, m, report)
            except Exception as e:         # per-market isolation
                logger.warning("polymarket %s failed: %s", m.get("id"), e)
                report.errors.append(f"market {m.get('id')}: {e}")
        if not page or len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return report
