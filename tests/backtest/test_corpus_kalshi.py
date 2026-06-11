from datetime import datetime, timezone

import backtest.corpus.kalshi as kalshi_mod
from backtest.corpus.kalshi import fetch_kalshi
from backtest.corpus.store import CorpusStore

MARKETS_PAGE_1 = {
    "cursor": "abc",
    "markets": [{
        "ticker": "FED-26JUN-T3.00",
        "event_ticker": "FED-26JUN",
        "title": "Fed funds above 3.00 after June 2026 meeting?",
        "rules_primary":
            "Resolves Yes if the target range floor exceeds 3.00.",
        "open_time": "2026-01-02T00:00:00Z",
        "close_time": "2026-06-18T00:00:00Z",
        "result": "no",
        "status": "settled",
    }, {
        # settled BEFORE open -> quarantined (resolution-date validation,
        # spec §3.4)
        "ticker": "BAD-26JAN-T1",
        "event_ticker": "BAD-26JAN",
        "title": "Settled before it opened?",
        "open_time": "2026-02-01T00:00:00Z",
        "close_time": "2026-03-01T00:00:00Z",
        "settled_time": "2026-01-15T00:00:00Z",    # before open_time
        "result": "yes",
        "status": "settled",
    }],
}
MARKETS_PAGE_2 = {"cursor": "", "markets": []}

CANDLES = {"candlesticks": [
    # prices in CENTS; price.close may be null on no-trade days ->
    # fall back to yes_bid/yes_ask mid
    {"end_period_ts": 1767312000,                  # 2026-01-02
     "price": {"close": 42},
     "yes_bid": {"close": 41}, "yes_ask": {"close": 43}},
    {"end_period_ts": 1769904000,                  # 2026-02-01
     "price": {"close": None},
     "yes_bid": {"close": 30}, "yes_ask": {"close": 34}},
]}


async def test_fetch_kalshi_ingests_settled_market(tmp_path, monkeypatch):
    seen = {"urls": [], "params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        seen["urls"].append(url)
        seen["params"].append(params)
        if url.endswith("/markets"):
            return (MARKETS_PAGE_1 if not params.get("cursor")
                    else MARKETS_PAGE_2)
        if "/candlesticks" in url:
            return CANDLES
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(kalshi_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_kalshi(store)
    assert report.fetched == 1                 # BAD-26JAN-T1 quarantined
    assert any("quarantined kalshi:BAD-26JAN-T1" in e
               for e in report.errors)
    assert seen["params"][0]["status"] == "settled"

    q = store.get_question("kalshi:FED-26JUN-T3.00")
    assert q.resolution is False               # result: "no"
    assert q.qtype.value == "binary"
    assert q.resolution_criteria.startswith("Resolves Yes")
    assert q.scheduled_close_time == datetime(2026, 6, 18,
                                              tzinfo=timezone.utc)

    candle_url = next(u for u in seen["urls"] if "/candlesticks" in u)
    # series ticker derived from event ticker prefix
    assert ("/series/FED/markets/FED-26JUN-T3.00/candlesticks"
            in candle_url)
    cp = seen["params"][seen["urls"].index(candle_url)]
    assert cp["period_interval"] == 1440
    assert cp["include_latest_before_start"] == "true"

    t = datetime(2026, 1, 15, tzinfo=timezone.utc)
    assert store.value_at("kalshi:FED-26JUN-T3.00",
                          "market_price", t) == 0.42
    t2 = datetime(2026, 2, 15, tzinfo=timezone.utc)
    assert store.value_at("kalshi:FED-26JUN-T3.00",
                          "market_price", t2) == 0.32   # bid/ask mid


async def test_fetch_kalshi_passes_min_settled_ts_and_limit(
        tmp_path, monkeypatch):
    seen = {"params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/markets"):
            seen["params"].append(params)
            return MARKETS_PAGE_1
        return {"candlesticks": []}

    monkeypatch.setattr(kalshi_mod, "_get_json", fake_get)
    report = await fetch_kalshi(CorpusStore(tmp_path),
                                min_settled_ts=1767225600, limit=1)
    assert report.fetched == 1
    assert seen["params"][0]["min_settled_ts"] == 1767225600


async def test_fetch_kalshi_isolates_candle_failure(tmp_path, monkeypatch):
    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/markets"):
            return (MARKETS_PAGE_1 if not params.get("cursor")
                    else MARKETS_PAGE_2)
        raise RuntimeError("candles 404 — outside rolling window")

    monkeypatch.setattr(kalshi_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_kalshi(store)
    # question still ingested; only the series failed
    assert report.fetched == 1
    assert store.get_question("kalshi:FED-26JUN-T3.00") is not None
    assert any("candlesticks" in e for e in report.errors)
