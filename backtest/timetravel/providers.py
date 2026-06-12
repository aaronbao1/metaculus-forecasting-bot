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
import re
from datetime import datetime, timedelta

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
    # historical=True silently returns ZERO articles unless BOTH
    # start_timestamp and end_timestamp are set (live-verified 2026-06-11).
    # Window start: question open time (Halawi protocol) when the caller
    # provides it, else 90 days before T.
    end_ts = int(ctx.as_of.timestamp())
    open_iso = question.get("open_time_iso")
    if open_iso:
        start_dt = datetime.fromisoformat(
            str(open_iso).replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            from datetime import timezone as _tz
            start_dt = start_dt.replace(tzinfo=_tz.utc)
        start_ts = min(int(start_dt.timestamp()), end_ts - 3600)
    else:
        start_ts = end_ts - 90 * 86400
    resp = await sdk.news.search_news(
        query=question["question_text"], n_articles=ASKNEWS_N_ARTICLES,
        return_type="dicts", historical=True,
        start_timestamp=start_ts, end_timestamp=end_ts)
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


# --- shared HTTP seams (monkeypatch targets in tests; never hit in CI) -----

HTTP_TIMEOUT_S = 30.0


async def _http_get_json(url: str, params: dict):
    import httpx
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def _http_get_text(url: str) -> str:
    import httpx
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def _http_post_json(url: str, payload: dict, headers: dict):
    import httpx
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


# --- GDELT DOC 2.0 + Wayback (DIY RetroSearch; strongest snapshot trust) ----

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
WAYBACK_CDX_URL = "http://web.archive.org/cdx/search/cdx"
GDELT_MAX_DOCS = 6
GDELT_WINDOWS = 2            # 3-month ArtList windows stepping back from T
GDELT_DOC_CHARS = 1500


def _gdelt_ts(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


async def _wayback_snapshot_before(url: str, t: datetime):
    """Latest Wayback capture at or before t -> (raw_id_url, iso_ts) or None.
    CDX JSON output: first row is the header; limit=-1 returns the last
    matching capture."""
    rows = await _http_get_json(WAYBACK_CDX_URL, params={
        "url": url, "output": "json", "to": _gdelt_ts(t),
        "filter": "statuscode:200", "limit": "-1"})
    if not rows or len(rows) < 2:
        return None
    ts, original = rows[-1][1], rows[-1][2]
    # Defense-in-depth: never trust the 'to' filter alone — re-check the
    # capture timestamp client-side so a CDX quirk cannot sneak in a post-T row.
    if ts > _gdelt_ts(t):
        return None
    raw = f"https://web.archive.org/web/{ts}id_/{original}"
    return raw, datetime.strptime(ts, "%Y%m%d%H%M%S").isoformat()


async def _gdelt_wayback_research(question: dict, cfg) -> str:
    """GDELT ArtList URL enumeration (3-month windows ending at T — ArtList
    only searches a trailing window, so we page windows backwards), then for
    each URL the latest Wayback capture <= T fetched in raw id_ mode. An
    article with no pre-T capture is dropped: not provably pre-T."""
    ctx = _require_context("gdelt_wayback")
    articles = []
    # GDELT DOC API serves historical STARTDATETIME/ENDDATETIME windows
    # (live-verified 2026-06-11: 2024 windows return in-window articles);
    # each query span is capped ~3 months, hence the windowing.
    for w in range(GDELT_WINDOWS):
        end = ctx.as_of - timedelta(days=90 * w)
        start = end - timedelta(days=90)
        try:
            data = await _http_get_json(GDELT_DOC_URL, params={
                "query": question["question_text"], "mode": "ArtList",
                "format": "json", "maxrecords": GDELT_MAX_DOCS,
                "sort": "hybridrel",
                "startdatetime": _gdelt_ts(start),
                "enddatetime": _gdelt_ts(end)})
        except Exception as e:
            logger.warning("GDELT window %d failed: %s", w, e)
            continue
        articles.extend((data or {}).get("articles") or [])
        if len(articles) >= GDELT_MAX_DOCS:
            break
    parts = []
    for art in articles[:GDELT_MAX_DOCS]:
        url = art.get("url")
        if not url:
            continue
        try:
            snap = await _wayback_snapshot_before(url, ctx.as_of)
        except Exception as e:
            logger.warning("Wayback CDX lookup failed for %s: %s", url, e)
            continue
        if snap is None:
            continue
        raw_url, snap_iso = snap
        try:
            text = _strip_html(await _http_get_text(raw_url))[:GDELT_DOC_CHARS]
        except Exception as e:
            logger.warning("Wayback fetch failed for %s: %s", raw_url, e)
            continue
        ctx.record(LedgerEntry(
            provider="gdelt_wayback", query=question["question_text"],
            url=raw_url, doc_date=snap_iso, date_trust="snapshot",
            note=f"Wayback capture <= {ctx.as_of.isoformat()} of {url}"))
        parts.append(f"[snapshot {snap_iso}] {art.get('title', url)}\n{text}")
    if not parts:
        return ""
    return ("Archived web snapshots (GDELT URL enumeration + Wayback "
            "captures taken on or before T):\n\n" + "\n\n".join(parts))


# --- Exa endCrawlDate (strict-allowed, semi-trusted) ------------------------

EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_NUM_RESULTS = 6
EXA_DOC_CHARS = 1000


async def _exa_historical_research(question: dict, cfg) -> str:
    """Exa search filtered to pages crawled on or before T. Allowed in strict
    mode but semi-trusted: crawl date >> publication metadata, unaudited —
    every ledger entry carries the flag."""
    ctx = _require_context("exa_historical")
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY not configured")
    data = await _http_post_json(
        EXA_SEARCH_URL,
        payload={"query": question["question_text"],
                 "endCrawlDate": ctx.as_of.isoformat().replace("+00:00", "Z"),
                 "numResults": EXA_NUM_RESULTS,
                 "contents": {"text": {"maxCharacters": EXA_DOC_CHARS}}},
        headers={"x-api-key": api_key, "Content-Type": "application/json"})
    parts = []
    for r in (data or {}).get("results") or []:
        url = r.get("url")
        published = r.get("publishedDate")
        # Determine trust label honestly: publisher trust only when publishedDate
        # is present and <= T; otherwise low trust (the leak-screen decides
        # inclusion in strict mode — we never exclude here in pragmatic mode).
        try:
            pub_dt = datetime.fromisoformat(
                published.replace("Z", "+00:00")) if published else None
        except (ValueError, AttributeError):
            pub_dt = None
        if pub_dt is not None and pub_dt <= ctx.as_of:
            date_trust = "publisher"
            note = ("semi-trusted: endCrawlDate filter; crawl date >> "
                    "publication metadata, unaudited")
        else:
            date_trust = "low"
            note = ("semi-trusted: endCrawlDate filter; crawl date >> "
                    "publication metadata, unaudited"
                    "; published date missing or post-T")
        ctx.record(LedgerEntry(
            provider="exa_historical", query=question["question_text"],
            url=url, doc_date=published, date_trust=date_trust, note=note))
        parts.append(f"[{published or 'undated'}] {r.get('title', url)} — "
                     f"{(r.get('text') or '')[:EXA_DOC_CHARS]} ({url})")
    if not parts:
        return ""
    return (f"Web results crawled on or before {ctx.as_of.date().isoformat()} "
            "(Exa endCrawlDate):\n" + "\n".join(f"- {p}" for p in parts))


# --- serper_dated (PRAGMATIC ONLY) + provider installation ------------------

SERPER_URL = "https://google.serper.dev/search"
SERPER_NUM = 10


async def _serper_dated_research(question: dict, cfg) -> str:
    """Date-filtered live Google search (Serper, tbs cdr params). Live search
    date filters measurably leak (71-81% of questions affected), so this
    provider is pragmatic-only, every result is leak-screened per document,
    and survivors are stamped date_trust=low."""
    ctx = _require_context("serper_dated")
    if ctx.cutoff_mode != "pragmatic":
        raise RuntimeError(
            "serper_dated is pragmatic-only: live search date filters leak "
            f"(cutoff_mode={ctx.cutoff_mode!r})")
    if ctx.llm is None:
        raise RuntimeError(
            "serper_dated: AsOfContext.llm is required for the leak screen")
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not configured")
    t = ctx.as_of
    tbs = f"cdr:1,cd_min:1/1/1990,cd_max:{t.month}/{t.day}/{t.year}"
    data = await _http_post_json(
        SERPER_URL,
        payload={"q": question["question_text"], "num": SERPER_NUM,
                 "tbs": tbs},
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"})
    docs = [{"title": r.get("title", ""), "url": r.get("link"),
             "date": r.get("date"), "snippet": r.get("snippet", "")}
            for r in (data or {}).get("organic") or []]
    if not docs:
        return ""
    from backtest.timetravel.leakage import screen_docs
    survivors = await screen_docs(docs, ctx.as_of, ctx.llm)
    parts = []
    for d in survivors:
        ctx.record(LedgerEntry(
            provider="serper_dated", query=question["question_text"],
            url=d["url"], doc_date=d.get("date"), date_trust="low",
            note="pragmatic-mode dated search result (survived leak screen)"))
        parts.append(f"[{d.get('date') or 'undated'}] {d['title']} — "
                     f"{d['snippet']} ({d['url']})")
    if not parts:
        return ""
    return ("Date-filtered web search results (PRAGMATIC mode; "
            "date_trust=low; leak-screened):\n"
            + "\n".join(f"- {p}" for p in parts))


HISTORICAL_PROVIDERS = (
    _asknews_historical_research,
    _gdelt_wayback_research,
    _exa_historical_research,
    _markets_historical_research,
    _serper_dated_research,
)

_HISTORICAL_SECTION_TITLES = {
    "markets_historical": "Prediction-market signals (price at T)",
    "asknews_historical": "News search (AskNews archive at T)",
    "gdelt_wayback": "News archive (GDELT+Wayback ≤ T)",
    "exa_historical": "Web search (Exa crawl ≤ T)",
    "serper_dated": "Web search (date-filtered, leak-screened)",
}


def install_historical_providers() -> None:
    """Register the historical providers into bot.research's module namespace
    so its existing globals()-dispatch (bot/research.py run_research) resolves
    them by config name, e.g. research_providers=("asknews_historical", ...).
    Also registers friendly section titles into bot.research._SECTION_TITLES
    so composed research reports render polished headers.
    Additive runtime registration — no bot/ file edits, no live-path change."""
    import bot.research
    for fn in HISTORICAL_PROVIDERS:
        setattr(bot.research, fn.__name__, fn)
    bot.research._SECTION_TITLES.update(_HISTORICAL_SECTION_TITLES)
