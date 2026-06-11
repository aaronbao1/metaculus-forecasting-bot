"""Historical research providers. Same signature as the live providers in
bot/research.py — async fn(question: dict, cfg) -> str — so the existing
globals()-dispatch resolves them by config name once
install_historical_providers() registers them into bot.research's namespace.
Each provider reads ACTIVE_CONTEXT for as_of/ledger/store/llm, writes an
evidence-ledger entry for everything it surfaces, and raises RuntimeError
when no context is set or the cutoff mode forbids it (bot/research.py's
never-raise contract converts that into a skipped provider)."""
from __future__ import annotations

import logging
import os
from datetime import datetime

from backtest.core import LedgerEntry
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext

logger = logging.getLogger(__name__)

ASKNEWS_N_ARTICLES = 10


def _require_context(provider: str) -> AsOfContext:
    ctx = ACTIVE_CONTEXT.get()
    if ctx is None:
        raise RuntimeError(
            f"{provider}: no AsOfContext set (ACTIVE_CONTEXT is None); "
            "historical providers only run inside the backtest harness")
    return ctx


def _fmt_price(p: float) -> str:
    """Never render a non-degenerate price as a literal 0%/100% (same rule
    as bot/markets.py — a false certainty anchor is the failure mode)."""
    pct = p * 100.0
    if 0.0 <= pct < 0.1:
        return "<0.1%"
    if 99.9 < pct <= 100.0:
        return ">99.9%"
    return f"{pct:.1f}%" if pct < 1 or pct > 99 else f"{pct:.0f}%"


async def _markets_historical_research(question: dict, cfg) -> str:
    """Crosswalk-matched market price at T. Leak-free by construction: the
    price series was snapshotted on sight; value_at is a step-function lookup
    of the last point <= T."""
    ctx = _require_context("markets_historical")
    if ctx.store is None:
        raise RuntimeError("markets_historical: AsOfContext.store is not set")
    qid = question.get("qid")
    if not qid:
        raise RuntimeError("markets_historical: question dict has no 'qid' "
                           "(the backtest adapter must pass it)")
    from backtest.corpus.crosswalk import market_for
    market_qid = market_for(ctx.store, qid)
    if market_qid is None:
        return ""               # no equivalent market; compositor skips section
    price = ctx.store.value_at(market_qid, "market_price", ctx.as_of)
    if price is None:
        return ""               # no price history at/before T
    ctx.record(LedgerEntry(
        provider="markets_historical", query=qid, url=None,
        doc_date=ctx.as_of.isoformat(), date_trust="snapshot",
        note=f"price-at-T from {market_qid} (leak-free by construction)"))
    return (f"Crosswalk-matched prediction market {market_qid} priced "
            f"P(Yes)={_fmt_price(float(price))} as of "
            f"{ctx.as_of.date().isoformat()}. Treat the price as one anchor "
            "among several, never the answer: have explicit reasons to "
            "concur or deviate.")


def _get_asknews_sdk():
    """Lazy import seam: the SDK may not be installed in test/CI
    environments; tests monkeypatch this module attribute."""
    from asknews_sdk import AsyncAskNewsSDK
    return AsyncAskNewsSDK


def _field(item, name, default=""):
    """AskNews SDK returns dicts or attribute objects depending on
    return_type/version — accept both."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


async def _asknews_historical_research(question: dict, cfg) -> str:
    """AskNews archive search with historical=True and end_timestamp=T.
    Ingestion-time snapshots (leak-low by architecture, unaudited) — allowed
    in strict mode; maximizes parity with the live asknews provider."""
    ctx = _require_context("asknews_historical")
    # same credential contract as bot/research._asknews_research
    has_oauth = bool(os.getenv("ASKNEWS_CLIENT_ID")
                     and os.getenv("ASKNEWS_SECRET"))
    if not (has_oauth or os.getenv("ASKNEWS_API_KEY")):
        raise RuntimeError("AskNews credentials not configured")
    sdk_cls = _get_asknews_sdk()
    if has_oauth:
        sdk = sdk_cls(client_id=os.getenv("ASKNEWS_CLIENT_ID"),
                      client_secret=os.getenv("ASKNEWS_SECRET"),
                      scopes={"news"})
    else:
        sdk = sdk_cls(api_key=os.getenv("ASKNEWS_API_KEY"), scopes={"news"})
    resp = await sdk.news.search_news(
        query=question["question_text"], n_articles=ASKNEWS_N_ARTICLES,
        return_type="dicts", historical=True,
        end_timestamp=int(ctx.as_of.timestamp()))
    articles = getattr(resp, "as_dicts", None) or []
    parts = []
    for a in articles:
        title = _field(a, "eng_title") or _field(a, "title")
        summary = _field(a, "summary")
        url = str(_field(a, "article_url") or "")
        pub = _field(a, "pub_date", None)
        pub_iso = (pub.isoformat() if isinstance(pub, datetime)
                   else (str(pub) if pub else None))
        ctx.record(LedgerEntry(
            provider="asknews_historical", query=question["question_text"],
            url=url or None, doc_date=pub_iso, date_trust="snapshot",
            note="AskNews ingestion-time snapshot (historical archive)"))
        parts.append(f"[{pub_iso or 'undated'}] {title} — {summary} ({url})")
    if not parts:
        return ""
    return ("Historical news snapshots (AskNews archive, ingestion-time, "
            f"end_timestamp={ctx.as_of.isoformat()}):\n"
            + "\n".join(f"- {p}" for p in parts))
