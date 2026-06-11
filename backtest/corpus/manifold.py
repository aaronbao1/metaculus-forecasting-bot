"""Manifold fetcher: /v0/markets enumeration + /v0/bets chain.

Price-at-T is reconstructed EXACTLY from the bet chain: each
non-redemption bet's probAfter becomes a step point at its createdTime,
seeded with the earliest bet's probBefore at market creation. Redemption
bets are EXCLUDED — they move no price (spec §3.2). Times are epoch ms.

Play-money caveat recorded in provenance (provenance["play_money"]) so
reports can stratify market evidence quality. MC markets need
answerId-filtered bets — out of scope here (binary only), explicit seam.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport

logger = logging.getLogger(__name__)

MANIFOLD_BASE = "https://api.manifold.markets/v0"
PAGE_SIZE = 500
BETS_LIMIT = 1000     # single page; >1000-bet markets are a seam (beforeTime)
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


def _ms(v) -> datetime | None:
    if v is None:
        return None
    return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)


async def _ingest_market(store: CorpusStore, m: dict,
                         report: FetchReport) -> None:
    if m.get("outcomeType") != "BINARY" or not m.get("isResolved"):
        return
    resolution = {"YES": True, "NO": False}.get(
        str(m.get("resolution", "")).upper())
    if resolution is None:
        return                              # MKT / CANCEL: not scoreable
    qid = f"manifold:{m['id']}"
    open_time = _ms(m.get("createdTime"))
    resolve_time = _ms(m.get("resolutionTime"))
    # resolution-date validation at import (spec §3.4)
    if open_time and resolve_time and resolve_time < open_time:
        report.errors.append(
            f"quarantined {qid}: resolve_time < open_time")
        return
    store.upsert_question(BacktestQuestion(
        qid=qid, platform="manifold", native_id=str(m["id"]),
        qtype=QType.BINARY, title=m.get("question") or "",
        open_time=open_time,
        scheduled_close_time=_ms(m.get("closeTime")),
        resolution=resolution,
        provenance={
            "source": "manifold", "play_money": True,
            "url": m.get("url"),
            "fetched_at": datetime.now(timezone.utc).isoformat()}))
    report.fetched += 1

    try:
        bets = await _get_json(f"{MANIFOLD_BASE}/bets", params={
            "contractId": m["id"], "limit": BETS_LIMIT})
    except Exception as e:    # question already stored; series failed
        logger.warning("manifold bets %s failed: %s", m.get("id"), e)
        report.errors.append(f"bets {m.get('id')}: {e}")
        return
    chain = sorted(
        (b for b in (bets or [])
         if not b.get("isRedemption")
         and b.get("probAfter") is not None),
        key=lambda b: b["createdTime"])
    if not chain:
        return
    pts = [(_ms(m.get("createdTime")).isoformat(),
            float(chain[0]["probBefore"]))]
    pts += [(_ms(b["createdTime"]).isoformat(), float(b["probAfter"]))
            for b in chain]
    store.put_series(qid, "market_price", pts)   # snapshot-on-sight


async def fetch_manifold(store: CorpusStore, *, limit=None) -> FetchReport:
    report = FetchReport()
    before = None
    while True:
        params: dict = {"limit": PAGE_SIZE}
        if before:
            params["before"] = before
        try:
            page = await _get_json(f"{MANIFOLD_BASE}/markets",
                                   params=params)
        except Exception as e:
            report.errors.append(f"markets page: {e}")
            break
        if not page:
            break
        for m in page:
            if limit is not None and report.fetched >= limit:
                return report
            try:
                await _ingest_market(store, m, report)
            except Exception as e:     # per-market isolation
                logger.warning("manifold %s failed: %s", m.get("id"), e)
                report.errors.append(f"market {m.get('id')}: {e}")
        if len(page) < PAGE_SIZE:
            break
        before = page[-1]["id"]
    return report
