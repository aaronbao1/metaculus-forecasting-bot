import pytest

import bot.markets as markets_mod
from bot.markets import (MarketQuote, fetch_manifold, fetch_polymarket,
                         format_market_section, market_research)

MANIFOLD_PAYLOAD = [
    {"question": "Will the Fed hike rates in 2026?", "probability": 0.4,
     "outcomeType": "BINARY", "volume": 7083.88, "isResolved": False,
     "url": "https://manifold.markets/x/fed-hike"},
    {"question": "Resolved one", "probability": 0.9, "outcomeType": "BINARY",
     "volume": 99999.0, "isResolved": True,
     "url": "https://manifold.markets/x/resolved"},
    {"question": "MC market without probability", "outcomeType": "MULTIPLE_CHOICE",
     "volume": 5000.0, "isResolved": False,
     "url": "https://manifold.markets/x/mc"},
]

POLYMARKET_PAYLOAD = {
    "events": [{
        "title": "Fed Decision in June?",
        "slug": "fed-decision-in-june",
        "active": True, "closed": False,
        "markets": [
            {"question": "25 bps decrease after June 2026 meeting?",
             "groupItemTitle": "25 bps decrease",
             "outcomes": '["Yes", "No"]',
             "outcomePrices": '["0.0025", "0.9975"]',
             "volumeNum": 14244992.29, "active": True, "closed": False},
            {"question": "Closed sub-market",
             "groupItemTitle": "stale",
             "outcomes": '["Yes", "No"]',
             "outcomePrices": '["0.5", "0.5"]',
             "volumeNum": 1.0, "active": True, "closed": True},
            {"question": "Malformed prices",
             "outcomes": '["Yes", "No"]',
             "outcomePrices": "not json",
             "volumeNum": 2.0, "active": True, "closed": False},
        ],
    }],
    "pagination": {"hasMore": False, "totalResults": 1},
}


async def test_fetch_manifold_maps_fields_and_filters(monkeypatch):
    async def fake_get(url, params, client=None):
        assert "search-markets" in url
        assert params["term"] == "Fed?"
        return MANIFOLD_PAYLOAD

    monkeypatch.setattr(markets_mod, "_get_json", fake_get)
    quotes = await fetch_manifold("Fed?")
    assert len(quotes) == 1            # resolved + no-probability MC dropped
    q = quotes[0]
    assert q.source == "Manifold"
    assert q.probability == pytest.approx(0.4)
    assert q.volume == pytest.approx(7083.88)
    assert q.url.endswith("/fed-hike")


async def test_fetch_polymarket_decodes_nested_json_strings(monkeypatch):
    async def fake_get(url, params, client=None):
        assert "public-search" in url
        assert params["events_status"] == "active"
        return POLYMARKET_PAYLOAD

    monkeypatch.setattr(markets_mod, "_get_json", fake_get)
    quotes = await fetch_polymarket("Fed?")
    assert len(quotes) == 1            # closed + malformed sub-markets dropped
    q = quotes[0]
    assert q.source == "Polymarket"
    assert q.probability == pytest.approx(0.0025)
    assert "25 bps decrease" in q.title
    assert q.url == "https://polymarket.com/event/fed-decision-in-june"


async def test_market_research_composes_sorted_section(monkeypatch):
    async def fake_manifold(query, limit=5, client=None):
        return [MarketQuote("Manifold", "Small market", 0.40, 7000.0, "u1")]

    async def fake_polymarket(query, limit=5, client=None):
        return [MarketQuote("Polymarket", "Big market", 0.0025, 14000000.0, "u2")]

    monkeypatch.setattr(markets_mod, "fetch_manifold", fake_manifold)
    monkeypatch.setattr(markets_mod, "fetch_polymarket", fake_polymarket)
    text = await market_research("Fed?")
    assert "Big market" in text and "Small market" in text
    # sorted by volume: Polymarket's larger market first
    assert text.index("Big market") < text.index("Small market")
    assert "0.2%" in text              # probability rendered as percent
    assert "40%" in text
    assert "anchor" in text.lower()    # anchor-not-target guidance present


async def test_market_research_survives_one_source_failing(monkeypatch):
    async def boom(query, limit=5, client=None):
        raise RuntimeError("api down")

    async def fake_polymarket(query, limit=5, client=None):
        return [MarketQuote("Polymarket", "Only market", 0.5, 10.0, "u")]

    monkeypatch.setattr(markets_mod, "fetch_manifold", boom)
    monkeypatch.setattr(markets_mod, "fetch_polymarket", fake_polymarket)
    text = await market_research("Q?")
    assert "Only market" in text


async def test_market_research_empty_when_nothing_found(monkeypatch):
    async def nothing(query, limit=5, client=None):
        return []

    monkeypatch.setattr(markets_mod, "fetch_manifold", nothing)
    monkeypatch.setattr(markets_mod, "fetch_polymarket", nothing)
    assert await market_research("Q?") == ""


def test_format_market_section_handles_no_probability():
    text = format_market_section(
        [MarketQuote("Manifold", "T", None, 100.0, "u")])
    assert "T" in text


async def test_polymarket_yes_index_lookup_not_positional(monkeypatch):
    payload = {"events": [{
        "title": "E", "slug": "e", "active": True, "closed": False,
        "markets": [
            {"question": "reversed order", "groupItemTitle": "rev",
             "outcomes": '["No", "Yes"]', "outcomePrices": '["0.9", "0.1"]',
             "volumeNum": 10.0, "active": True, "closed": False},
            {"question": "no yes outcome", "groupItemTitle": "updown",
             "outcomes": '["Up", "Down"]', "outcomePrices": '["0.6", "0.4"]',
             "volumeNum": 20.0, "active": True, "closed": False},
        ]}]}

    async def fake_get(url, params, client=None):
        return payload

    monkeypatch.setattr(markets_mod, "_get_json", fake_get)
    quotes = await fetch_polymarket("Q?")
    assert len(quotes) == 1                      # Up/Down market skipped
    assert quotes[0].probability == pytest.approx(0.1)   # Yes at index 1


async def test_polymarket_null_markets_and_events_are_safe(monkeypatch):
    async def fake_get(url, params, client=None):
        return {"events": [
            {"title": "no markets", "slug": "x", "active": True,
             "closed": False, "markets": None},
            None if False else {"title": "ok", "slug": "ok", "active": True,
                                "closed": False,
                                "markets": [{"question": "q",
                                             "outcomes": '["Yes", "No"]',
                                             "outcomePrices": '["0.5", "0.5"]',
                                             "volumeNum": 1.0,
                                             "active": True, "closed": False}]},
        ]}

    monkeypatch.setattr(markets_mod, "_get_json", fake_get)
    quotes = await fetch_polymarket("Q?")
    assert len(quotes) == 1

    async def fake_get_null(url, params, client=None):
        return {"events": None}

    monkeypatch.setattr(markets_mod, "_get_json", fake_get_null)
    assert await fetch_polymarket("Q?") == []


async def test_market_research_per_source_cap_and_global_top8(monkeypatch):
    async def fake_manifold(query, limit=5, client=None):
        return [MarketQuote("Manifold", f"M{i}", 0.5, float(i), f"mu{i}")
                for i in range(6)]                       # volumes 0..5

    async def fake_polymarket(query, limit=5, client=None):
        return [MarketQuote("Polymarket", f"P{i}", 0.5, 1000.0 + i, f"pu{i}")
                for i in range(6)]                       # all dwarf Manifold

    monkeypatch.setattr(markets_mod, "fetch_manifold", fake_manifold)
    monkeypatch.setattr(markets_mod, "fetch_polymarket", fake_polymarket)
    text = await market_research("Q?")
    lines = [l for l in text.splitlines() if l.startswith("- [")]
    assert len(lines) == 8
    # top-4 per source survive: Manifold present despite tiny volumes
    assert sum("[Manifold]" in l for l in lines) == 4
    assert sum("[Polymarket]" in l for l in lines) == 4
    assert "M5" in text and "M1" not in text             # per-source top-4
    assert "P5" in text and "P1" not in text


def test_fmt_prob_never_renders_false_certainty():
    from bot.markets import _fmt_prob
    assert _fmt_prob(0.0004) == "<0.1%"
    assert _fmt_prob(0.9996) == ">99.9%"
    assert _fmt_prob(0.0) == "<0.1%"
    assert _fmt_prob(1.0) == ">99.9%"
    assert _fmt_prob(0.0025) == "0.2%"
    assert _fmt_prob(0.40) == "40%"
