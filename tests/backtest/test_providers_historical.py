"""markets_historical (crosswalk + store.value_at price-at-T; leak-free by
construction) and asknews_historical (SDK historical=True end_timestamp=T).
Both read ACTIVE_CONTEXT and refuse to run without it. The AskNews SDK may
not be installed: the provider imports it lazily through the
_get_asknews_sdk module attribute, which tests monkeypatch."""
from datetime import datetime, timezone

import pytest

import backtest.timetravel.providers as providers
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot.config import BotConfig

T = datetime(2024, 6, 15, tzinfo=timezone.utc)
PUB = datetime(2024, 6, 1, tzinfo=timezone.utc)
QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F",
                qid="metaculus:31678")
CFG = BotConfig()


@pytest.fixture
def ctx():
    c = AsOfContext(as_of=T)
    token = ACTIVE_CONTEXT.set(c)
    yield c
    ACTIVE_CONTEXT.reset(token)


class FakeStore:
    def __init__(self, value=0.42):
        self.value = value
        self.calls = []

    def value_at(self, qid, kind, t):
        self.calls.append((qid, kind, t))
        return self.value


async def test_markets_requires_context():
    assert ACTIVE_CONTEXT.get() is None
    with pytest.raises(RuntimeError, match="no AsOfContext"):
        await providers._markets_historical_research(QUESTION, CFG)


async def test_markets_requires_store_and_qid(ctx):
    with pytest.raises(RuntimeError, match="store"):
        await providers._markets_historical_research(QUESTION, CFG)
    ctx.store = FakeStore()
    with pytest.raises(RuntimeError, match="qid"):
        await providers._markets_historical_research(
            dict(question_text="Q?"), CFG)


async def test_markets_formats_price_at_T_and_ledgers_snapshot(
        ctx, monkeypatch):
    ctx.store = FakeStore(value=0.42)
    import backtest.corpus.crosswalk as crosswalk
    monkeypatch.setattr(crosswalk, "market_for",
                        lambda store, qid: "polymarket:abc")
    out = await providers._markets_historical_research(QUESTION, CFG)
    assert "42%" in out
    assert "polymarket:abc" in out
    assert "2024-06-15" in out
    assert ctx.store.calls == [("polymarket:abc", "market_price", T)]
    assert len(ctx.ledger) == 1
    e = ctx.ledger[0]
    assert e.provider == "markets_historical"
    assert e.query == "metaculus:31678"
    assert e.date_trust == "snapshot"


async def test_markets_empty_when_no_crosswalk_match(ctx, monkeypatch):
    ctx.store = FakeStore()
    import backtest.corpus.crosswalk as crosswalk
    monkeypatch.setattr(crosswalk, "market_for", lambda store, qid: None)
    assert await providers._markets_historical_research(QUESTION, CFG) == ""
    assert ctx.ledger == []


async def test_markets_empty_when_no_price_at_T(ctx, monkeypatch):
    ctx.store = FakeStore(value=None)
    import backtest.corpus.crosswalk as crosswalk
    monkeypatch.setattr(crosswalk, "market_for",
                        lambda store, qid: "polymarket:abc")
    assert await providers._markets_historical_research(QUESTION, CFG) == ""
    assert ctx.ledger == []


class FakeAskNewsSDK:
    last = None

    def __init__(self, **kwargs):
        FakeAskNewsSDK.last = self
        self.kwargs = kwargs
        self.news = self
        self.search_kwargs = None

    async def search_news(self, **kwargs):
        self.search_kwargs = kwargs

        class Resp:
            as_dicts = [
                {"eng_title": "Title A", "summary": "Sum A",
                 "article_url": "https://a.example/1", "pub_date": PUB},
                {"eng_title": "Title B", "summary": "Sum B",
                 "article_url": "https://b.example/2", "pub_date": PUB},
            ]
        return Resp()


async def test_asknews_requires_context(monkeypatch):
    monkeypatch.setenv("ASKNEWS_API_KEY", "k")
    with pytest.raises(RuntimeError, match="no AsOfContext"):
        await providers._asknews_historical_research(QUESTION, CFG)


async def test_asknews_requires_credentials(ctx, monkeypatch):
    for var in ("ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        await providers._asknews_historical_research(QUESTION, CFG)


async def test_asknews_historical_end_timestamp_and_snapshot_ledger(
        ctx, monkeypatch):
    monkeypatch.setenv("ASKNEWS_API_KEY", "k")
    monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
    monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
    monkeypatch.setattr(providers, "_get_asknews_sdk",
                        lambda: FakeAskNewsSDK)
    out = await providers._asknews_historical_research(QUESTION, CFG)
    sk = FakeAskNewsSDK.last.search_kwargs
    assert sk["historical"] is True
    assert sk["end_timestamp"] == int(T.timestamp())
    assert sk["query"] == "Q?"
    assert "Title A" in out and "Sum A" in out and "Title B" in out
    assert len(ctx.ledger) == 2
    assert {e.provider for e in ctx.ledger} == {"asknews_historical"}
    assert all(e.date_trust == "snapshot" for e in ctx.ledger)
    assert ctx.ledger[0].url == "https://a.example/1"
    assert ctx.ledger[0].doc_date == PUB.isoformat()


async def test_asknews_oauth_credentials_pass_through(ctx, monkeypatch):
    monkeypatch.delenv("ASKNEWS_API_KEY", raising=False)
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "cid")
    monkeypatch.setenv("ASKNEWS_SECRET", "sec")
    monkeypatch.setattr(providers, "_get_asknews_sdk",
                        lambda: FakeAskNewsSDK)
    await providers._asknews_historical_research(QUESTION, CFG)
    assert FakeAskNewsSDK.last.kwargs["client_id"] == "cid"
    assert FakeAskNewsSDK.last.kwargs["client_secret"] == "sec"
