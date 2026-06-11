import bot.research as research_mod
from bot.config import BotConfig
from bot.research import ResearchResult, run_research

QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F")


async def test_uses_first_working_provider(monkeypatch):
    async def fake_asknews(q, cfg):
        return "asknews report"
    monkeypatch.setattr(research_mod, "_asknews_research", fake_asknews)
    out = await run_research(QUESTION, BotConfig())
    assert isinstance(out, ResearchResult)
    assert out.provider == "asknews"
    assert "asknews report" in out.text


async def test_falls_back_when_provider_raises(monkeypatch):
    async def boom(q, cfg):
        raise RuntimeError("asknews down")
    async def fake_search(q, cfg):
        return "search-llm report"
    monkeypatch.setattr(research_mod, "_asknews_research", boom)
    monkeypatch.setattr(research_mod, "_search_llm_research", fake_search)
    out = await run_research(QUESTION, BotConfig())
    assert out.provider == "search_llm"


async def test_returns_empty_flagged_result_when_all_fail(monkeypatch):
    async def boom(q, cfg):
        raise RuntimeError("down")
    monkeypatch.setattr(research_mod, "_asknews_research", boom)
    monkeypatch.setattr(research_mod, "_search_llm_research", boom)
    out = await run_research(QUESTION, BotConfig())
    assert out.provider == "none"
    assert out.text == ""
