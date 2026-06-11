import pytest

import bot.research as research_mod
from bot.config import BotConfig
from bot.research import ResearchResult, run_research

QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F")


def _cfg(providers, **kw):
    return BotConfig(research_providers=tuple(providers), **kw)


async def test_composes_all_successful_providers(monkeypatch):
    async def fake_asknews(q, cfg):
        return "asknews report"

    async def fake_search(q, cfg):
        return "search-llm report"

    monkeypatch.setattr(research_mod, "_asknews_research", fake_asknews)
    monkeypatch.setattr(research_mod, "_search_llm_research", fake_search)
    out = await run_research(QUESTION, _cfg(["asknews", "search_llm"]))
    assert isinstance(out, ResearchResult)
    assert out.providers == ("asknews", "search_llm")
    assert "asknews report" in out.text
    assert "search-llm report" in out.text
    # sections appear in configured order
    assert out.text.index("asknews report") < out.text.index("search-llm report")


async def test_runs_providers_concurrently(monkeypatch):
    import asyncio

    started = []

    async def slow_a(q, cfg):
        started.append("a")
        await asyncio.sleep(0.05)
        assert "b" in started          # b started before a finished
        return "A"

    async def slow_b(q, cfg):
        started.append("b")
        await asyncio.sleep(0.05)
        assert "a" in started
        return "B"

    monkeypatch.setattr(research_mod, "_asknews_research", slow_a)
    monkeypatch.setattr(research_mod, "_search_llm_research", slow_b)
    out = await run_research(QUESTION, _cfg(["asknews", "search_llm"]))
    assert out.providers == ("asknews", "search_llm")


async def test_partial_failure_keeps_survivors(monkeypatch):
    async def boom(q, cfg):
        raise RuntimeError("asknews down")

    async def fake_search(q, cfg):
        return "search-llm report"

    monkeypatch.setattr(research_mod, "_asknews_research", boom)
    monkeypatch.setattr(research_mod, "_search_llm_research", fake_search)
    out = await run_research(QUESTION, _cfg(["asknews", "search_llm"]))
    assert out.providers == ("search_llm",)
    assert "search-llm report" in out.text


async def test_empty_string_success_counts_as_failure(monkeypatch):
    async def empty(q, cfg):
        return "   "

    async def fake_search(q, cfg):
        return "search-llm report"

    monkeypatch.setattr(research_mod, "_asknews_research", empty)
    monkeypatch.setattr(research_mod, "_search_llm_research", fake_search)
    out = await run_research(QUESTION, _cfg(["asknews", "search_llm"]))
    assert out.providers == ("search_llm",)


async def test_returns_empty_flagged_result_when_all_fail(monkeypatch):
    async def boom(q, cfg):
        raise RuntimeError("down")

    monkeypatch.setattr(research_mod, "_asknews_research", boom)
    monkeypatch.setattr(research_mod, "_search_llm_research", boom)
    out = await run_research(QUESTION, _cfg(["asknews", "search_llm"]))
    assert out.providers == ()
    assert out.provider == "none"
    assert out.text == ""


async def test_provider_str_field_joins_names(monkeypatch):
    async def fake_asknews(q, cfg):
        return "asknews report"

    async def fake_search(q, cfg):
        return "search-llm report"

    monkeypatch.setattr(research_mod, "_asknews_research", fake_asknews)
    monkeypatch.setattr(research_mod, "_search_llm_research", fake_search)
    out = await run_research(QUESTION, _cfg(["asknews", "search_llm"]))
    assert out.provider == "asknews,search_llm"


async def test_hung_provider_times_out_without_killing_run(monkeypatch):
    import asyncio

    async def hang(q, cfg):
        await asyncio.sleep(60)

    async def fake_search(q, cfg):
        return "search-llm report"

    monkeypatch.setattr(research_mod, "_asknews_research", hang)
    monkeypatch.setattr(research_mod, "_search_llm_research", fake_search)
    out = await run_research(
        QUESTION, _cfg(["asknews", "search_llm"], research_timeout_s=0.05))
    assert out.providers == ("search_llm",)


async def test_markets_provider_flows_through_run_research(monkeypatch):
    import bot.markets as markets_mod

    async def fake_market_research(question_text):
        assert question_text == "Q?"
        return "MARKET_SECTION_SENTINEL"

    async def boom(q, cfg):
        raise RuntimeError("down")

    monkeypatch.setattr(markets_mod, "market_research", fake_market_research)
    monkeypatch.setattr(research_mod, "_asknews_research", boom)
    monkeypatch.setattr(research_mod, "_search_llm_research", boom)
    out = await run_research(
        QUESTION, _cfg(["asknews", "search_llm", "markets"]))
    assert out.providers == ("markets",)
    assert "MARKET_SECTION_SENTINEL" in out.text
    assert "Prediction-market signals" in out.text


async def test_unknown_provider_name_is_skipped_not_fatal(monkeypatch):
    async def fake_search(q, cfg):
        return "search-llm report"

    monkeypatch.setattr(research_mod, "_search_llm_research", fake_search)
    out = await run_research(QUESTION, _cfg(["bogus_typo", "search_llm"]))
    assert out.providers == ("search_llm",)


async def test_timeout_actually_bounds_wall_clock(monkeypatch):
    import asyncio
    import time

    async def hang(q, cfg):
        await asyncio.sleep(60)

    monkeypatch.setattr(research_mod, "_asknews_research", hang)
    monkeypatch.setattr(research_mod, "_search_llm_research", hang)
    start = time.monotonic()
    out = await run_research(
        QUESTION, _cfg(["asknews", "search_llm"], research_timeout_s=0.1))
    assert time.monotonic() - start < 5.0        # not the 60s hang
    assert out.provider == "none"
