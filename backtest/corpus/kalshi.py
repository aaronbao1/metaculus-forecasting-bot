"""Kalshi fetcher: settled markets + daily candlesticks.

GET /markets?status=settled (cursor pagination), then per market
GET /series/{series}/markets/{ticker}/candlesticks with
period_interval=1440 (daily) and include_latest_before_start=true.

Candlestick price encoding changed on 2026-03-12:
  - Live endpoint (post-2026-03-12): fixed-point dollar STRINGS in
    price.close_dollars (e.g. "0.5600"), already in [0, 1].
  - /historical archive path (pre-2026-03-12): integer cents in
    price.close (e.g. 56), divide by 100 to get probability.
price.close_dollars may be absent on no-trade days -> fall back to
yes_bid/yes_ask mid (close_dollars first, then close cents).

ROLLING WINDOW WARNING: the live candlesticks API keeps only ~3 months of
history (GET /historical/cutoff reports the boundary); older series must
come from the /historical/* archive endpoints — an explicit post-MVP
fallback seam. Snapshot-on-sight (spec §3.2) makes every successful fetch
permanent locally, which is the primary mitigation.

The markets payload does not carry series_ticker; it is derived from the
event ticker prefix (event "FED-26JUN" -> series "FED") — verify against
live payloads when the archive fallback is built.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
PAGE_SIZE = 200
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


def _epoch(dt: datetime | None) -> int | None:
    return int(dt.timestamp()) if dt else None


def _candle_prob(c: dict) -> float | None:
    price = c.get("price") or {}
    if price.get("close_dollars") is not None:
        return float(price["close_dollars"])          # already dollars 0-1
    if price.get("close") is not None:                # legacy /historical archive path
        return float(price["close"]) / 100.0
    bid = c.get("yes_bid") or {}
    ask = c.get("yes_ask") or {}
    if bid.get("close_dollars") is not None and ask.get("close_dollars") is not None:
        return (float(bid["close_dollars"]) + float(ask["close_dollars"])) / 2.0
    if bid.get("close") is not None and ask.get("close") is not None:
        return (float(bid["close"]) + float(ask["close"])) / 2.0 / 100.0
    return None


async def _ingest_market(store: CorpusStore, m: dict,
                         report: FetchReport) -> None:
    ticker = m["ticker"]
    qid = f"kalshi:{ticker}"
    result = (m.get("result") or "").lower()
    if result == "scalar":
        report.errors.append(
            f"skipped {qid}: scalar market (non-binary)")
        return
    resolution = {"yes": True, "no": False}.get(result)
    open_time = _parse_dt(m.get("open_time"))
    close_time = _parse_dt(m.get("close_time"))
    settled_time = _parse_dt(m.get("settled_time"))
    # resolution-date validation at import (spec §3.4)
    if open_time and settled_time and settled_time < open_time:
        report.errors.append(
            f"quarantined {qid}: resolve_time < open_time")
        return
    store.upsert_question(BacktestQuestion(
        qid=qid, platform="kalshi", native_id=ticker,
        qtype=QType.BINARY,
        title=m.get("title") or "",
        resolution_criteria=m.get("rules_primary") or "",
        open_time=open_time, scheduled_close_time=close_time,
        resolution=resolution,
        provenance={
            "source": "kalshi",
            "event_ticker": m.get("event_ticker"),
            "fetched_at": datetime.now(timezone.utc).isoformat()}))
    report.fetched += 1

    series_ticker = (m.get("series_ticker")
                     or str(m.get("event_ticker") or ticker).split("-")[0])
    try:
        candles = await _get_json(
            f"{KALSHI_BASE}/series/{series_ticker}/markets/{ticker}"
            "/candlesticks",
            params={"start_ts": _epoch(open_time) or 0,
                    "end_ts": _epoch(close_time) or 0,
                    "period_interval": 1440,
                    "include_latest_before_start": "true"})
    except Exception as e:    # series fetch fails; question already stored
        logger.warning("kalshi candlesticks %s failed: %s", ticker, e)
        report.errors.append(f"candlesticks {ticker}: {e}")
        return
    pts = []
    for c in candles.get("candlesticks") or []:
        prob = _candle_prob(c)
        ts = c.get("end_period_ts")
        if prob is None or ts is None:
            continue
        pts.append((datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    prob))
    if pts:
        store.put_series(qid, "market_price", pts)   # snapshot-on-sight


async def fetch_kalshi(store: CorpusStore, *, status="settled",
                       min_settled_ts=None, limit=None) -> FetchReport:
    report = FetchReport()
    cursor = ""
    while True:
        params: dict = {"status": status, "limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        if min_settled_ts is not None:
            params["min_settled_ts"] = min_settled_ts
        try:
            page = await _get_json(f"{KALSHI_BASE}/markets", params=params)
        except Exception as e:
            report.errors.append(f"markets page: {e}")
            break
        for m in page.get("markets") or []:
            if limit is not None and report.fetched >= limit:
                return report
            try:
                await _ingest_market(store, m, report)
            except Exception as e:     # per-market isolation
                logger.warning("kalshi %s failed: %s", m.get("ticker"), e)
                report.errors.append(f"market {m.get('ticker')}: {e}")
        cursor = page.get("cursor") or ""
        if not cursor:
            break
    return report
