from datetime import datetime, timezone

import backtest.corpus.manifold as manifold_mod
from backtest.corpus.manifold import fetch_manifold
from backtest.corpus.store import CorpusStore

# /v0/markets returns a bare array of LiteMarkets; times are epoch MS
MARKETS_PAGE = [
    {"id": "abc123", "question": "Will W resign in 2026?",
     "outcomeType": "BINARY", "isResolved": True, "resolution": "YES",
     "createdTime": 1767225600000,             # 2026-01-01
     "closeTime": 1772323200000,               # 2026-03-01
     "url": "https://manifold.markets/u/will-w-resign",
     "volume": 1234.5},
    # non-binary -> skipped
    {"id": "mc999", "question": "Which team wins?",
     "outcomeType": "MULTIPLE_CHOICE", "isResolved": True,
     "resolution": "a1", "createdTime": 1767225600000,
     "closeTime": 1772323200000},
    # resolved MKT/CANCEL -> not scoreable as binary -> skipped
    {"id": "cancel1", "question": "Cancelled?", "outcomeType": "BINARY",
     "isResolved": True, "resolution": "CANCEL",
     "createdTime": 1767225600000, "closeTime": 1772323200000},
    # resolved BEFORE creation -> quarantined (resolution-date validation,
    # spec §3.4)
    {"id": "bad1", "question": "Resolved before created?",
     "outcomeType": "BINARY", "isResolved": True, "resolution": "YES",
     "createdTime": 1769904000000,             # 2026-02-01
     "resolutionTime": 1767225600000,          # 2026-01-01, before created
     "closeTime": 1772323200000},
]

# /v0/bets returns newest-first; redemption bets carry no price signal
BETS = [
    {"createdTime": 1769904000000,             # 2026-02-01
     "probBefore": 0.35, "probAfter": 0.62, "isRedemption": False},
    {"createdTime": 1768608000000,             # 2026-01-17
     "probBefore": 0.50, "probAfter": 0.50, "isRedemption": True},
    {"createdTime": 1767916800000,             # 2026-01-09
     "probBefore": 0.40, "probAfter": 0.35, "isRedemption": False},
]


async def test_fetch_manifold_bet_chain_step_series(tmp_path, monkeypatch):
    seen = {"bet_params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/v0/markets"):
            return MARKETS_PAGE if not params.get("before") else []
        if url.endswith("/v0/bets"):
            seen["bet_params"].append(params)
            return BETS
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(manifold_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_manifold(store)
    assert report.fetched == 1     # MC + CANCEL skipped, bad1 quarantined
    assert any("quarantined manifold:bad1" in e for e in report.errors)
    assert seen["bet_params"][0]["contractId"] == "abc123"

    q = store.get_question("manifold:abc123")
    assert q.resolution is True
    assert q.qtype.value == "binary"
    assert q.scheduled_close_time == datetime(2026, 3, 1,
                                              tzinfo=timezone.utc)
    assert q.provenance["play_money"] is True  # caveat recorded

    # step function: market creation seeds earliest probBefore, then each
    # non-redemption bet contributes (createdTime, probAfter)
    at = lambda *a: store.value_at("manifold:abc123", "market_price",
                                   datetime(*a, tzinfo=timezone.utc))
    assert at(2026, 1, 5) == 0.40              # before first bet
    assert at(2026, 1, 10) == 0.35             # after first bet
    assert at(2026, 1, 20) == 0.35             # redemption bet EXCLUDED
    assert at(2026, 2, 2) == 0.62              # after last bet


async def test_fetch_manifold_limit_and_bets_failure_isolated(
        tmp_path, monkeypatch):
    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/v0/markets"):
            return MARKETS_PAGE if not params.get("before") else []
        raise RuntimeError("bets endpoint down")

    monkeypatch.setattr(manifold_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_manifold(store, limit=1)
    # question stored even though its bet series failed
    assert report.fetched == 1
    assert store.get_question("manifold:abc123") is not None
    assert any("bets" in e for e in report.errors)
