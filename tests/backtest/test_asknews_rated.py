"""Task 11: asknews_rated provider — relevance-filtered dossiers."""
import asyncio
from datetime import datetime, timezone

from backtest.timetravel import providers as prov_mod
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot.config import BotConfig

DOSSIER = (
    "Historical news snapshots (AskNews archive):\n"
    "- [2026-01-01] OPEC production cut agreed (http://a)\n"
    "- [2026-01-02] Celebrity gossip (http://b)\n")


class FakeLLM:
    def __init__(self):
        self.calls = []

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append({"model": model, "run_index": run_index})
        score = 90 if "OPEC" in prompt else 10
        return f'{{"scores": {{"relevance": {score}}}}}'


def _run(question):
    llm = FakeLLM()
    ctx = AsOfContext(as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
                      llm=llm)
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        text = asyncio.run(prov_mod._asknews_rated_research(
            question, BotConfig()))
    finally:
        ACTIVE_CONTEXT.reset(token)
    return text, llm


def test_filters_low_relevance_items(monkeypatch):
    async def fake_asknews(question, cfg):
        return DOSSIER
    monkeypatch.setattr(prov_mod, "_asknews_historical_research",
                        fake_asknews)
    text, llm = _run({"qid": "t:1", "question_text": "Oil?"})
    assert "OPEC" in text and "gossip" not in text
    assert "relevance-filtered" in text
    assert all(c["model"] == BotConfig().parser_model for c in llm.calls)


def test_empty_dossier_passes_through(monkeypatch):
    async def fake_asknews(question, cfg):
        return ""
    monkeypatch.setattr(prov_mod, "_asknews_historical_research",
                        fake_asknews)
    text, llm = _run({"qid": "t:1", "question_text": "Oil?"})
    assert text == "" and not llm.calls
