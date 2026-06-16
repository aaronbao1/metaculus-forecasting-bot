import pytest

from bot.config import BotConfig, EnsembleMember
from bot.ensemble_bot import CalibratedEnsembleBot

CANNED_BINARY = "reasoning...\nstatus quo...\npremortem...\nProbability: 64%"
CANNED_MC = "reasoning...\nAlpha: 20%\nBeta: 30%\nGamma: 50%"
CANNED_NUMERIC = """reasoning...
Percentile 2.5: 5
Percentile 10: 10
Percentile 25: 20
Percentile 50: 40
Percentile 75: 60
Percentile 90: 80
Percentile 97.5: 95"""

COMMON = dict(question_text="Q?", background_info="", resolution_criteria="R",
              fine_print="", research="news", close_iso="2026-07-01T00:00:00Z")


def make_bot(monkeypatch, canned):
    cfg = BotConfig(ensemble=[EnsembleMember("fake/model-a", runs=3),
                              EnsembleMember("fake/model-b", runs=3)])
    bot = CalibratedEnsembleBot(bot_config=cfg)

    async def fake_invoke(self, member_and_prompt):
        return canned
    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", fake_invoke,
                        raising=True)
    return bot


async def test_binary_pipeline_aggregates_and_caps(monkeypatch):
    bot = make_bot(monkeypatch, CANNED_BINARY)
    result = await bot.run_binary_ensemble(**COMMON)
    # six identical members at 0.64 -> trimmed GMO = 0.64, inside caps
    assert result.final == pytest.approx(0.64, abs=1e-6)
    assert len(result.member_probs) == 6
    assert result.spread == pytest.approx(0.0)


async def test_binary_extreme_consensus_gets_capped(monkeypatch):
    bot = make_bot(monkeypatch, "Probability: 99.9%")
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.final == 0.98          # hard cap


async def test_binary_spread_is_recorded_for_divergent_members(monkeypatch):
    cfg = BotConfig(ensemble=[EnsembleMember("fake/model-a", runs=3),
                              EnsembleMember("fake/model-b", runs=3)])
    bot = CalibratedEnsembleBot(bot_config=cfg)

    async def fake_invoke(self, member_and_prompt):
        member, prompt = member_and_prompt
        return {"fake/model-a": "Probability: 20%",
                "fake/model-b": "Probability: 70%"}[member.model]

    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", fake_invoke,
                        raising=True)
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.spread == pytest.approx(0.5)


async def test_skips_when_too_few_members_survive(monkeypatch):
    bot = make_bot(monkeypatch, "I refuse.")
    with pytest.raises(RuntimeError, match="only 0 ensemble members"):
        await bot.run_binary_ensemble(**COMMON)


async def test_mc_pipeline(monkeypatch):
    bot = make_bot(monkeypatch, CANNED_MC)
    result = await bot.run_mc_ensemble(**COMMON, options=["Alpha", "Beta", "Gamma"])
    assert result.final["Gamma"] == pytest.approx(0.5, abs=0.02)
    assert sum(result.final.values()) == pytest.approx(1.0)


async def test_numeric_pipeline_medians_percentiles(monkeypatch):
    bot = make_bot(monkeypatch, CANNED_NUMERIC)
    result = await bot.run_numeric_ensemble(
        **COMMON, units="things",
        lower_bound_message="not below 0", upper_bound_message="not above 100")
    assert result.median_percentiles[50.0] == 40
    assert 2.5 in result.median_percentiles
    assert 97.5 in result.median_percentiles


CANNED_DATE = """reasoning...
Percentile 2.5: 2026-02-01
Percentile 10: 2026-03-01
Percentile 25: 2026-04-01
Percentile 50: 2026-06-01
Percentile 75: 2026-08-01
Percentile 90: 2026-10-01
Percentile 97.5: 2026-12-01"""


async def test_date_pipeline_medians_timestamps(monkeypatch):
    from datetime import datetime, timezone
    bot = make_bot(monkeypatch, CANNED_DATE)
    result = await bot.run_date_ensemble(
        **COMMON, lower_bound_message="not before 2026-01-01",
        upper_bound_message="not after 2027-01-01")
    assert result.median_percentiles[50.0] == \
        datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()
    assert 2.5 in result.median_percentiles
    assert 97.5 in result.median_percentiles
    assert len(result.member_models) == 6
