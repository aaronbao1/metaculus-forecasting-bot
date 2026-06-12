"""Canary leakage test: plant a document whose content exists only after T
through the pragmatic dated-search provider; the leak screen must drop it
from the research text and the evidence ledger must record the discard."""
from datetime import datetime, timezone

import backtest.timetravel.providers as providers
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot.config import BotConfig

T = datetime(2024, 6, 15, tzinfo=timezone.utc)


class ScriptedJudge:
    def __init__(self, completion):
        self.completion = completion

    async def invoke(self, model, prompt, temperature, run_index=0):
        return self.completion


async def test_planted_post_T_doc_is_dropped_and_ledgered(monkeypatch):
    canary = {"title": "Final results announced",
              "link": "https://canary.example/results",
              "date": "2024-07-01",
              "snippet": "The final results, announced July 1 2024, were ..."}
    clean = {"title": "Polling ahead of the vote",
             "link": "https://ok.example/polls",
             "date": "2024-05-20",
             "snippet": "Surveys conducted in May 2024 show ..."}

    async def fake_post(url, payload, headers):
        return {"organic": [clean, canary]}

    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    ctx = AsOfContext(as_of=T, cutoff_mode="pragmatic",
                      llm=ScriptedJudge("1: CLEAN\n2: LEAK"))
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        out = await providers._serper_dated_research(
            dict(question_text="Who won the vote?"), BotConfig())
    finally:
        ACTIVE_CONTEXT.reset(token)
    assert "canary.example" not in out       # planted doc never reaches research
    assert "ok.example" in out
    discards = [e for e in ctx.ledger if e.provider == "leak_screen"]
    assert len(discards) == 1
    assert discards[0].url == "https://canary.example/results"
    assert discards[0].doc_date == "2024-07-01"
    assert discards[0].date_trust == "low"
