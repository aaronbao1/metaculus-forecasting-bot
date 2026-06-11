from datetime import datetime, timezone

import backtest.corpus.polymarket as polymarket_mod
from backtest.corpus.polymarket import fetch_polymarket
from backtest.corpus.store import CorpusStore

# Gamma /markets?closed=true returns a bare array; outcomes/outcomePrices/
# clobTokenIds are JSON-encoded STRINGS (same quirk bot/markets.py handles)
GAMMA_PAGE = [
    {"id": "516710", "question": "Will Z win the 2026 election?",
     "conditionId": "0xabc", "slug": "will-z-win",
     "description": "Resolves to Yes if Z wins.",
     "outcomes": '["No", "Yes"]',              # reversed on purpose:
     "outcomePrices": '["0", "1"]',            # Yes settled at index 1
     "clobTokenIds": '["1111", "2222"]',
     "startDate": "2026-01-01T00:00:00Z",
     "endDate": "2026-03-01T00:00:00Z",
     "closed": True, "volumeNum": 250000.0},
    # not a Yes/No binary -> skipped
    {"id": "516711", "question": "Up or down market?",
     "outcomes": '["Up", "Down"]', "outcomePrices": '["0.6", "0.4"]',
     "clobTokenIds": '["3333", "4444"]',
     "startDate": "2026-01-01T00:00:00Z",
     "endDate": "2026-03-01T00:00:00Z",
     "closed": True, "volumeNum": 10.0},
    # settled BEFORE open -> quarantined (resolution-date validation,
    # spec §3.4)
    {"id": "516712", "question": "Settled before it opened?",
     "outcomes": '["No", "Yes"]', "outcomePrices": '["0", "1"]',
     "clobTokenIds": '["5555", "6666"]',
     "startDate": "2026-02-01T00:00:00Z",
     "endDate": "2026-03-01T00:00:00Z",
     "closedTime": "2026-01-15T00:00:00Z",     # before startDate
     "closed": True, "volumeNum": 50.0},
]

PRICES_HISTORY = {"history": [
    {"t": 1767225600, "p": 0.30},      # 2026-01-01T00:00:00Z
    {"t": 1769904000, "p": 0.65},      # 2026-02-01T00:00:00Z
]}


async def test_fetch_polymarket_ingests_and_snapshots_prices(
        tmp_path, monkeypatch):
    seen = {"gamma_params": [], "clob_params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        if "gamma-api" in url:
            seen["gamma_params"].append(params)
            return GAMMA_PAGE if params.get("offset", 0) == 0 else []
        if "prices-history" in url:
            seen["clob_params"].append(params)
            return PRICES_HISTORY
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(polymarket_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_polymarket(store)
    assert report.fetched == 1        # Up/Down skipped, 516712 quarantined
    assert any("quarantined polymarket:516712" in e for e in report.errors)
    assert store.get_question("polymarket:516712") is None
    assert seen["gamma_params"][0]["closed"] == "true"

    q = store.get_question("polymarket:516710")
    assert q.resolution is True                # Yes price "1" at index 1
    assert q.qtype.value == "binary"
    assert q.title == "Will Z win the 2026 election?"
    assert q.scheduled_close_time == datetime(2026, 3, 1,
                                              tzinfo=timezone.utc)
    assert q.provenance["condition_id"] == "0xabc"

    # price history requested for the YES token, chosen via the outcomes
    # field (index 1 here), at fidelity >= 720 (resolved markets return
    # empty below ~12h granularity)
    assert seen["clob_params"][0]["market"] == "2222"
    assert seen["clob_params"][0]["fidelity"] == 720
    t = datetime(2026, 1, 15, tzinfo=timezone.utc)
    assert store.value_at("polymarket:516710", "market_price", t) == 0.30
    t2 = datetime(2026, 2, 15, tzinfo=timezone.utc)
    assert store.value_at("polymarket:516710", "market_price", t2) == 0.65


async def test_fetch_polymarket_passes_filters_and_limit(
        tmp_path, monkeypatch):
    seen = {"params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        if "gamma-api" in url:
            seen["params"].append(params)
            return GAMMA_PAGE
        return {"history": []}

    monkeypatch.setattr(polymarket_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_polymarket(store, start_date_min="2026-01-01",
                                    limit=1)
    assert report.fetched == 1                 # limit stops after first
    assert seen["params"][0]["start_date_min"] == "2026-01-01"


async def test_fetch_polymarket_tolerates_array_typed_gamma_fields(
        tmp_path, monkeypatch):
    """If Gamma ever returns real arrays instead of JSON-encoded strings,
    _ingest_market must still succeed (no TypeError from json.loads)."""
    array_market = {
        "id": "999001",
        "question": "Will array fields work?",
        "conditionId": "0xdef",
        "slug": "array-fields",
        "description": "Tests array-typed Gamma fields.",
        # ARRAYS instead of JSON-encoded strings:
        "outcomes": ["No", "Yes"],
        "outcomePrices": ["0", "1"],
        "clobTokenIds": ["7777", "8888"],
        "startDate": "2026-01-01T00:00:00Z",
        "endDate": "2026-06-01T00:00:00Z",
        "closed": True, "volumeNum": 100000.0,
    }

    async def fake_get(url, params=None, headers=None, client=None):
        if "gamma-api" in url:
            return [array_market] if params.get("offset", 0) == 0 else []
        return PRICES_HISTORY

    monkeypatch.setattr(polymarket_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_polymarket(store)

    # must NOT land in errors — should ingest cleanly
    assert report.errors == [], f"Unexpected errors: {report.errors}"
    assert report.fetched == 1

    q = store.get_question("polymarket:999001")
    assert q is not None
    assert q.resolution is True   # Yes price "1" at index 1


async def test_fetch_polymarket_isolates_bad_market(tmp_path, monkeypatch):
    bad = [{"id": "X", "question": "broken", "outcomes": "not json",
            "outcomePrices": "[]", "clobTokenIds": "[]"}] + GAMMA_PAGE[:1]

    async def fake_get(url, params=None, headers=None, client=None):
        if "gamma-api" in url:
            return bad if params.get("offset", 0) == 0 else []
        return PRICES_HISTORY

    monkeypatch.setattr(polymarket_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_polymarket(store)
    assert report.fetched == 1
    assert any("X" in e for e in report.errors)
