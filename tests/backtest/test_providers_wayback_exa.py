"""gdelt_wayback: GDELT DOC 2.0 ArtList URL enumeration in 3-month windows
ending at T, then Wayback CDX lookup for the latest capture <= T fetched in
raw id_ mode — the strongest snapshot guarantee. exa_historical: POST
api.exa.ai/search with endCrawlDate=T; publisher date trust, semi-trusted.
All HTTP goes through providers._http_get_json/_http_get_text/_http_post_json
module seams, monkeypatched here — never the network."""
from datetime import datetime, timedelta, timezone

import pytest

import backtest.timetravel.providers as providers
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot.config import BotConfig

T = datetime(2024, 6, 15, tzinfo=timezone.utc)
QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F")
CFG = BotConfig()
RAW_URL = ("https://web.archive.org/web/20240601120000id_/"
           "https://news.example/story")


@pytest.fixture
def ctx():
    c = AsOfContext(as_of=T)
    token = ACTIVE_CONTEXT.set(c)
    yield c
    ACTIVE_CONTEXT.reset(token)


async def test_gdelt_requires_context():
    with pytest.raises(RuntimeError, match="no AsOfContext"):
        await providers._gdelt_wayback_research(QUESTION, CFG)


async def test_gdelt_wayback_snapshot_flow(ctx, monkeypatch):
    calls = []

    async def fake_get_json(url, params):
        calls.append((url, dict(params)))
        if url == providers.GDELT_DOC_URL:
            if params["enddatetime"] == providers._gdelt_ts(T):
                return {"articles": [{"url": "https://news.example/story",
                                      "title": "Story"}]}
            return {"articles": []}            # older window: nothing
        if url == providers.WAYBACK_CDX_URL:
            return [["urlkey", "timestamp", "original", "mimetype",
                     "statuscode", "digest", "length"],
                    ["x", "20240601120000", "https://news.example/story",
                     "text/html", "200", "D", "123"]]
        raise AssertionError(f"unexpected URL {url}")

    async def fake_get_text(url):
        assert url == RAW_URL
        return "<html><script>x()</script><body><p>Pre-T content.</p></body></html>"

    monkeypatch.setattr(providers, "_http_get_json", fake_get_json)
    monkeypatch.setattr(providers, "_http_get_text", fake_get_text)
    out = await providers._gdelt_wayback_research(QUESTION, CFG)
    assert "Pre-T content." in out
    assert "x()" not in out                    # script stripped
    gdelt = [c for c in calls if c[0] == providers.GDELT_DOC_URL]
    assert len(gdelt) == providers.GDELT_WINDOWS
    assert gdelt[0][1]["mode"] == "ArtList"
    assert gdelt[0][1]["enddatetime"] == providers._gdelt_ts(T)
    assert gdelt[0][1]["startdatetime"] == \
        providers._gdelt_ts(T - timedelta(days=90))
    assert gdelt[1][1]["enddatetime"] == \
        providers._gdelt_ts(T - timedelta(days=90))
    cdx = [c for c in calls if c[0] == providers.WAYBACK_CDX_URL]
    assert len(cdx) == 1
    assert cdx[0][1]["to"] == providers._gdelt_ts(T)
    assert cdx[0][1]["url"] == "https://news.example/story"
    assert len(ctx.ledger) == 1
    e = ctx.ledger[0]
    assert e.provider == "gdelt_wayback"
    assert e.date_trust == "snapshot"
    assert e.url == RAW_URL
    assert e.doc_date == "2024-06-01T12:00:00"


async def test_gdelt_drops_articles_without_pre_T_snapshot(ctx, monkeypatch):
    fetched = []

    async def fake_get_json(url, params):
        if url == providers.GDELT_DOC_URL:
            if params["enddatetime"] == providers._gdelt_ts(T):
                return {"articles": [{"url": "https://news.example/story",
                                      "title": "Story"}]}
            return {"articles": []}
        if url == providers.WAYBACK_CDX_URL:
            return [["urlkey", "timestamp", "original", "mimetype",
                     "statuscode", "digest", "length"]]   # header only
        raise AssertionError(url)

    async def fake_get_text(url):
        fetched.append(url)
        return "should never be fetched"

    monkeypatch.setattr(providers, "_http_get_json", fake_get_json)
    monkeypatch.setattr(providers, "_http_get_text", fake_get_text)
    assert await providers._gdelt_wayback_research(QUESTION, CFG) == ""
    assert fetched == []                       # not provably pre-T: no fetch
    assert ctx.ledger == []


async def test_exa_requires_context_and_key(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "k")
    with pytest.raises(RuntimeError, match="no AsOfContext"):
        await providers._exa_historical_research(QUESTION, CFG)


async def test_exa_missing_key_raises(ctx, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EXA_API_KEY"):
        await providers._exa_historical_research(QUESTION, CFG)


async def test_exa_end_crawl_date_z_suffix(ctx, monkeypatch):
    """endCrawlDate must be Z-suffixed ISO (Exa documents Z-suffixed ISO)."""
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    captured = {}

    async def fake_post(url, payload, headers):
        captured.update(payload=payload)
        return {"results": []}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    await providers._exa_historical_research(QUESTION, CFG)
    assert captured["payload"]["endCrawlDate"] == "2024-06-15T00:00:00Z"


async def test_exa_trust_label_honest(ctx, monkeypatch):
    """pre-T publishedDate -> date_trust='publisher';
    missing/post-T publishedDate -> date_trust='low' with note."""
    monkeypatch.setenv("EXA_API_KEY", "exa-key")

    async def fake_post(url, payload, headers):
        return {"results": [
            # pre-T: full trust
            {"url": "https://doc.example/a", "title": "Doc A",
             "publishedDate": "2024-05-01T00:00:00Z", "text": "alpha"},
            # post-T: low trust
            {"url": "https://doc.example/b", "title": "Doc B",
             "publishedDate": "2024-07-01T00:00:00Z", "text": "beta"},
            # missing: low trust
            {"url": "https://doc.example/c", "title": "Doc C",
             "publishedDate": None, "text": "gamma"},
        ]}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    out = await providers._exa_historical_research(QUESTION, CFG)
    assert len(ctx.ledger) == 3
    # pre-T: publisher trust
    assert ctx.ledger[0].date_trust == "publisher"
    assert "missing or post-T" not in ctx.ledger[0].note
    # post-T: low trust with note
    assert ctx.ledger[1].date_trust == "low"
    assert "missing or post-T" in ctx.ledger[1].note
    # missing: low trust with note
    assert ctx.ledger[2].date_trust == "low"
    assert "missing or post-T" in ctx.ledger[2].note
    # all docs still included in output
    assert "Doc A" in out and "Doc B" in out and "Doc C" in out


async def test_wayback_cdx_ts_after_t_drops_doc(ctx, monkeypatch):
    """If CDX returns a row whose timestamp > T, the doc must be dropped
    (never trust the 'to' filter alone — client-side recheck)."""
    fetched = []

    async def fake_get_json(url, params):
        if url == providers.GDELT_DOC_URL:
            return {"articles": [{"url": "https://news.example/future",
                                  "title": "Future Story"}]}
        if url == providers.WAYBACK_CDX_URL:
            # timestamp is AFTER T (20240616 > 20240615)
            return [["urlkey", "timestamp", "original", "mimetype",
                     "statuscode", "digest", "length"],
                    ["x", "20240616000000", "https://news.example/future",
                     "text/html", "200", "D", "1"]]
        raise AssertionError(url)

    async def fake_get_text(url):
        fetched.append(url)
        return "post-T content"

    monkeypatch.setattr(providers, "_http_get_json", fake_get_json)
    monkeypatch.setattr(providers, "_http_get_text", fake_get_text)
    result = await providers._gdelt_wayback_research(QUESTION, CFG)
    assert result == ""
    assert fetched == []   # post-T capture must not be fetched
    assert ctx.ledger == []


async def test_exa_end_crawl_date_and_publisher_ledger(ctx, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    captured = {}

    async def fake_post(url, payload, headers):
        captured.update(url=url, payload=payload, headers=headers)
        return {"results": [
            {"url": "https://doc.example/a", "title": "Doc A",
             "publishedDate": "2024-05-01T00:00:00Z", "text": "alpha text"},
            {"url": "https://doc.example/b", "title": "Doc B",
             "publishedDate": None, "text": "beta text"},
        ]}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    out = await providers._exa_historical_research(QUESTION, CFG)
    assert captured["url"] == providers.EXA_SEARCH_URL
    assert captured["payload"]["endCrawlDate"] == "2024-06-15T00:00:00Z"
    assert captured["payload"]["query"] == "Q?"
    assert captured["headers"]["x-api-key"] == "exa-key"
    assert "Doc A" in out and "alpha text" in out and "Doc B" in out
    assert len(ctx.ledger) == 2
    assert all(e.provider == "exa_historical" for e in ctx.ledger)
    # Doc A: pre-T published -> publisher trust; Doc B: missing -> low trust
    assert ctx.ledger[0].date_trust == "publisher"
    assert ctx.ledger[1].date_trust == "low"
    assert all("semi-trusted" in e.note for e in ctx.ledger)
    assert ctx.ledger[0].doc_date == "2024-05-01T00:00:00Z"
    assert ctx.ledger[1].doc_date is None
