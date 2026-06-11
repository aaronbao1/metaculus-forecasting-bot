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
    async def fake_get(url, params):
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
    async def fake_get(url, params):
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
    async def fake_manifold(query, limit=5):
        return [MarketQuote("Manifold", "Small market", 0.40, 7000.0, "u1")]

    async def fake_polymarket(query, limit=5):
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
    async def boom(query, limit=5):
        raise RuntimeError("api down")

    async def fake_polymarket(query, limit=5):
        return [MarketQuote("Polymarket", "Only market", 0.5, 10.0, "u")]

    monkeypatch.setattr(markets_mod, "fetch_manifold", boom)
    monkeypatch.setattr(markets_mod, "fetch_polymarket", fake_polymarket)
    text = await market_research("Q?")
    assert "Only market" in text


async def test_market_research_empty_when_nothing_found(monkeypatch):
    async def nothing(query, limit=5):
        return []

    monkeypatch.setattr(markets_mod, "fetch_manifold", nothing)
    monkeypatch.setattr(markets_mod, "fetch_polymarket", nothing)
    assert await market_research("Q?") == ""


def test_format_market_section_handles_no_probability():
    text = format_market_section(
        [MarketQuote("Manifold", "T", None, 100.0, "u")])
    assert "T" in text
