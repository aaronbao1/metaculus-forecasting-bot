"""Prediction-market odds as a research input (Polymarket + Manifold, public
APIs, no keys). Injecting market odds into research is a cheap, measured win
(adjacentwire placed #11 on this change alone) — but market parroting is a
documented failure mode (0.994 correlation with shown prices), so the section
explicitly frames prices as one anchor among several, never the target.

API contracts verified live 2026-06-11:
- Manifold: GET /v0/search-markets returns a bare array; binary markets carry
  `probability`; `closeTime` is epoch ms; MC answers need a second call (we
  restrict to BINARY).
- Polymarket: GET /public-search returns {events: [...]}; prices live on
  nested markets; `outcomes`/`outcomePrices` are JSON-encoded STRINGS; use
  the *Num numeric variants; check both `active` and `closed`.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MANIFOLD_SEARCH = "https://api.manifold.markets/v0/search-markets"
POLYMARKET_SEARCH = "https://gamma-api.polymarket.com/public-search"
HTTP_TIMEOUT_S = 15.0
MAX_QUOTES = 8


@dataclass
class MarketQuote:
    source: str               # "Manifold" | "Polymarket"
    title: str
    probability: float | None  # P(Yes) in [0,1]; None if unavailable
    volume: float
    url: str


async def _get_json(url: str, params: dict, client=None):
    import httpx
    if client is not None:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as one_shot:
        resp = await one_shot.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_manifold(query: str, limit: int = 5,
                         client=None) -> list[MarketQuote]:
    data = await _get_json(MANIFOLD_SEARCH, params={
        "term": query, "limit": limit, "filter": "open",
        "sort": "liquidity", "contractType": "BINARY"}, client=client)
    quotes = []
    for m in (data or []):
        try:
            if m.get("isResolved") or m.get("probability") is None:
                continue
            quotes.append(MarketQuote(
                source="Manifold",
                title=str(m["question"]),
                probability=float(m["probability"]),
                volume=float(m.get("volume") or 0.0),
                url=str(m.get("url") or "")))
        except Exception as e:
            logger.warning("skipping malformed Manifold market: %s", e)
    return quotes


async def fetch_polymarket(query: str, limit: int = 5,
                           client=None) -> list[MarketQuote]:
    data = await _get_json(POLYMARKET_SEARCH, params={
        "q": query, "limit_per_type": limit, "events_status": "active"},
        client=client)
    quotes = []
    for event in ((data or {}).get("events") or []):
        try:
            if not event.get("active", True) or event.get("closed", False):
                continue
            event_url = f"https://polymarket.com/event/{event.get('slug', '')}"
            markets = event.get("markets") or []
        except Exception as e:
            logger.warning("skipping malformed Polymarket event: %s", e)
            continue
        for m in markets:
            try:
                if not m.get("active", True) or m.get("closed", False):
                    continue
                outcomes = json.loads(m["outcomes"])
                prices = [float(p) for p in json.loads(m["outcomePrices"])]
                yes_idx = next(i for i, o in enumerate(outcomes)
                               if str(o).lower() == "yes")
                label = m.get("groupItemTitle") or m.get("question", "")
                title = (f"{event.get('title', '')} — {label}"
                         if label and label != event.get("title") else
                         event.get("title", label))
                quotes.append(MarketQuote(
                    source="Polymarket",
                    title=title,
                    probability=prices[yes_idx],
                    volume=float(m.get("volumeNum") or 0.0),
                    url=event_url))
            except Exception as e:
                logger.warning("skipping malformed Polymarket market: %s", e)
    return quotes


def _fmt_volume(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def _fmt_prob(p: float | None) -> str:
    """Never render a non-degenerate price as a literal 0%/100% — a false
    certainty anchor is exactly the failure mode market data can cause."""
    if p is None:
        return "n/a"
    pct = p * 100.0
    if 0.0 <= pct < 0.1:
        return "<0.1%"
    if 99.9 < pct <= 100.0:
        return ">99.9%"
    return f"{pct:.1f}%" if pct < 1 or pct > 99 else f"{pct:.0f}%"


def format_market_section(quotes: list[MarketQuote]) -> str:
    lines = [
        "Possibly-related prediction markets (matched by keyword — judge "
        "relevance yourself; ignore markets that are not actually about this "
        "question). Treat prices as one anchor among several, never the "
        "answer: have explicit reasons to concur or deviate.",
        "",
    ]
    for q in quotes:
        lines.append(f"- [{q.source}] {q.title} — P(Yes)={_fmt_prob(q.probability)} "
                     f"(volume {_fmt_volume(q.volume)}) {q.url}")
    return "\n".join(lines)


async def market_research(question_text: str) -> str:
    """Top markets across both venues, by volume. Empty string if none —
    the research compositor then simply omits this provider."""
    import httpx
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as client:
        results = await asyncio.gather(
            fetch_manifold(question_text, client=client),
            fetch_polymarket(question_text, client=client),
            return_exceptions=True)
    quotes: list[MarketQuote] = []
    for src, r in zip(("Manifold", "Polymarket"), results):
        if isinstance(r, BaseException):
            logger.warning("market source %s failed: %s", src, r)
        else:
            # per-source cap so Polymarket's real-money volumes don't crowd
            # play-money venues out of the section entirely
            quotes.extend(sorted(r, key=lambda q: q.volume,
                                 reverse=True)[:MAX_QUOTES // 2])
    if not quotes:
        return ""
    quotes.sort(key=lambda q: q.volume, reverse=True)
    return format_market_section(quotes[:MAX_QUOTES])
