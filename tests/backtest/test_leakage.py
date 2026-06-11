"""screen_docs: an LLM judge flags documents revealing post-as_of
information; flagged docs are dropped and every discard is stamped into the
evidence ledger. serper_dated: date-filtered live search (tbs cdr params),
pragmatic-only — raises in strict mode, and bot/research.py's never-raise
contract converts that into a skipped provider. install_historical_providers
registers all five providers into bot.research's namespace."""
from datetime import datetime, timezone

import pytest

import backtest.timetravel.providers as providers
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from backtest.timetravel.leakage import LEAK_JUDGE_MODEL, screen_docs
from bot.config import BotConfig

T = datetime(2024, 6, 15, tzinfo=timezone.utc)
QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F")
DOCS = [
    {"title": "Pre-T report", "url": "https://ok.example/1",
     "date": "2024-05-20", "text": "Surveys in May 2024 show ..."},
    {"title": "Post-T result", "url": "https://leak.example/2",
     "date": "2024-07-01", "text": "The outcome, announced July 1 ..."},
]


class FakeJudge:
    def __init__(self, completion):
        self.completion = completion
        self.calls = []

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append(dict(model=model, prompt=prompt,
                               temperature=temperature, run_index=run_index))
        return self.completion


@pytest.fixture
def ctx():
    c = AsOfContext(as_of=T, cutoff_mode="pragmatic")
    token = ACTIVE_CONTEXT.set(c)
    yield c
    ACTIVE_CONTEXT.reset(token)


@pytest.fixture
def installed():
    providers.install_historical_providers()
    yield
    import bot.research
    for fn in providers.HISTORICAL_PROVIDERS:
        if hasattr(bot.research, fn.__name__):
            delattr(bot.research, fn.__name__)


async def test_screen_docs_drops_flagged_and_ledgers_discards(ctx):
    judge = FakeJudge("1: CLEAN\n2: LEAK")
    survivors = await screen_docs(DOCS, T, judge)
    assert survivors == [DOCS[0]]
    call = judge.calls[0]
    assert call["model"] == LEAK_JUDGE_MODEL
    assert call["temperature"] == 0.0 and call["run_index"] == 0
    assert "Pre-T report" in call["prompt"]
    assert len(ctx.ledger) == 1
    e = ctx.ledger[0]
    assert e.provider == "leak_screen"
    assert e.url == "https://leak.example/2"
    assert e.doc_date == "2024-07-01"
    assert e.date_trust == "low"
    assert "discarded" in e.note


async def test_screen_docs_unparseable_verdict_drops_conservatively(ctx):
    survivors = await screen_docs(DOCS, T, FakeJudge("no idea, sorry"))
    assert survivors == []
    assert len(ctx.ledger) == 2


async def test_screen_docs_empty_input_skips_the_llm():
    judge = FakeJudge("unused")
    assert await screen_docs([], T, judge) == []
    assert judge.calls == []


async def test_screen_docs_without_context_still_screens():
    assert ACTIVE_CONTEXT.get() is None
    survivors = await screen_docs(DOCS, T, FakeJudge("1: CLEAN\n2: LEAK"))
    assert survivors == [DOCS[0]]


async def test_serper_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    ctx = AsOfContext(as_of=T, cutoff_mode="strict")
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        with pytest.raises(RuntimeError, match="pragmatic-only"):
            await providers._serper_dated_research(QUESTION, BotConfig())
    finally:
        ACTIVE_CONTEXT.reset(token)


async def test_serper_pragmatic_screens_and_ledgers(ctx, monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    captured = {}

    async def fake_post(url, payload, headers):
        captured.update(url=url, payload=payload, headers=headers)
        return {"organic": [
            {"title": "Pre-T report", "link": "https://ok.example/1",
             "date": "2024-05-20", "snippet": "Surveys in May ..."},
            {"title": "Post-T result", "link": "https://leak.example/2",
             "date": "2024-07-01", "snippet": "The outcome, announced ..."},
        ]}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    ctx.llm = FakeJudge("1: CLEAN\n2: LEAK")
    out = await providers._serper_dated_research(QUESTION, BotConfig())
    assert captured["url"] == providers.SERPER_URL
    assert captured["headers"]["X-API-KEY"] == "serper-key"
    assert "cd_max:6/15/2024" in captured["payload"]["tbs"]
    assert captured["payload"]["tbs"].startswith("cdr:1,")
    assert "Pre-T report" in out
    assert "leak.example" not in out
    survivors = [e for e in ctx.ledger if e.provider == "serper_dated"]
    discards = [e for e in ctx.ledger if e.provider == "leak_screen"]
    assert len(survivors) == 1 and survivors[0].date_trust == "low"
    assert len(discards) == 1 and discards[0].url == "https://leak.example/2"


async def test_serper_requires_llm_for_the_screen(ctx, monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    ctx.llm = None
    with pytest.raises(RuntimeError, match="llm"):
        await providers._serper_dated_research(QUESTION, BotConfig())


def test_install_registers_all_five_into_bot_research(installed):
    import bot.research
    for fn in providers.HISTORICAL_PROVIDERS:
        assert getattr(bot.research, fn.__name__) is fn
    assert bot.research._serper_dated_research is \
        providers._serper_dated_research
    # Addendum: section titles must also be registered for polished headers
    expected_titles = {
        "markets_historical": "Prediction-market signals (price at T)",
        "asknews_historical": "News search (AskNews archive at T)",
        "gdelt_wayback": "News archive (GDELT+Wayback ≤ T)",
        "exa_historical": "Web search (Exa crawl ≤ T)",
        "serper_dated": "Web search (date-filtered, leak-screened)",
    }
    for name, title in expected_titles.items():
        assert bot.research._SECTION_TITLES.get(name) == title, (
            f"_SECTION_TITLES missing or wrong for {name!r}")


async def test_strict_mode_serper_skipped_by_run_research_never_raise(
        installed, monkeypatch):
    """Cutoff-mode enforcement end to end: under a strict context the serper
    provider raises; run_research's never-raise contract logs and skips it,
    keeping the surviving providers' sections."""
    import bot.research as research_mod
    from bot.research import run_research

    async def stub(q, cfg):
        return "stub section"

    monkeypatch.setattr(research_mod, "_asknews_research", stub)
    monkeypatch.setenv("SERPER_API_KEY", "k")
    ctx = AsOfContext(as_of=T, cutoff_mode="strict")
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        out = await run_research(QUESTION, BotConfig(
            research_providers=("serper_dated", "asknews")))
    finally:
        ACTIVE_CONTEXT.reset(token)
    assert out.providers == ("asknews",)
    assert "stub section" in out.text
    assert "serper" not in out.provider
