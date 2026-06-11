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


async def test_binary_extreme_consensus_gets_capped(monkeypatch):
    bot = make_bot(monkeypatch, "Probability: 99.9%")
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.final == 0.98          # hard cap


async def test_skips_when_too_few_members_survive(monkeypatch):
    bot = make_bot(monkeypatch, "I refuse.")
    with pytest.raises(RuntimeError, match="only 0 ensemble members"):
        await bot.run_binary_ensemble(**COMMON)


def make_divergent_bot(monkeypatch, supervisor_reply="Probability: 60%",
                       **cfg_overrides):
    """3 runs at 20% (model-a) + 3 at 70% (model-b) -> spread 0.5."""
    cfg = BotConfig(ensemble=[EnsembleMember("fake/model-a", runs=3),
                              EnsembleMember("fake/model-b", runs=3)],
                    supervisor_model="fake/supervisor", **cfg_overrides)
    bot = CalibratedEnsembleBot(bot_config=cfg)
    calls = []

    async def fake_invoke(self, member_and_prompt):
        member, prompt = member_and_prompt
        calls.append(member.model)
        return {"fake/model-a": "Probability: 20%",
                "fake/model-b": "Probability: 70%",
                "fake/supervisor": supervisor_reply}[member.model]

    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", fake_invoke,
                        raising=True)
    return bot, calls


async def test_supervisor_triggers_on_high_spread_and_blends(monkeypatch):
    from bot.aggregation import from_log_odds, to_log_odds
    bot, calls = make_divergent_bot(monkeypatch)
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.spread == pytest.approx(0.5)
    assert result.supervisor_prob == pytest.approx(0.60)
    assert calls.count("fake/supervisor") == 1
    expected = from_log_odds(
        0.5 * to_log_odds([0.60])[0]
        + 0.5 * to_log_odds([result.aggregate_raw])[0])
    assert result.final == pytest.approx(float(expected), abs=1e-6)


async def test_supervisor_not_called_below_trigger(monkeypatch):
    bot = make_bot(monkeypatch, CANNED_BINARY)        # all members identical
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.spread == pytest.approx(0.0)
    assert result.supervisor_prob is None


async def test_supervisor_garbage_falls_back_to_raw(monkeypatch):
    bot, calls = make_divergent_bot(monkeypatch, supervisor_reply="I refuse.")
    result = await bot.run_binary_ensemble(**COMMON)
    assert calls.count("fake/supervisor") == 1
    assert result.supervisor_prob is None
    assert result.final == pytest.approx(result.aggregate_raw, abs=1e-6)


async def test_supervisor_disabled_never_calls(monkeypatch):
    bot, calls = make_divergent_bot(monkeypatch, supervisor_enabled=False)
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.supervisor_prob is None
    assert "fake/supervisor" not in calls


async def test_supervisor_prompt_receives_member_reasoning(monkeypatch):
    captured = {}
    cfg = BotConfig(ensemble=[EnsembleMember("fake/model-a", runs=3),
                              EnsembleMember("fake/model-b", runs=3)],
                    supervisor_model="fake/supervisor")
    bot = CalibratedEnsembleBot(bot_config=cfg)

    async def fake_invoke(self, member_and_prompt):
        member, prompt = member_and_prompt
        if member.model == "fake/supervisor":
            captured["prompt"] = prompt
            return "Probability: 50%"
        return {"fake/model-a": "because of reason RARE_TOKEN_A\nProbability: 20%",
                "fake/model-b": "because of reason RARE_TOKEN_B\nProbability: 70%",
                }[member.model]

    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", fake_invoke,
                        raising=True)
    await bot.run_binary_ensemble(**COMMON)
    assert "RARE_TOKEN_A" in captured["prompt"]
    assert "RARE_TOKEN_B" in captured["prompt"]
    assert "20%" in captured["prompt"]      # member numbers shown
    assert "news" in captured["prompt"]     # research included


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


def make_triage_bot(monkeypatch, tier_reply="Tier: NEAR_CERTAIN",
                    **cfg_overrides):
    cfg = BotConfig(ensemble=[EnsembleMember("fake/model-a", runs=3),
                              EnsembleMember("fake/model-b", runs=3)],
                    parser_model="fake/triage", **cfg_overrides)
    bot = CalibratedEnsembleBot(bot_config=cfg)
    calls = []

    async def fake_invoke(self, member_and_prompt):
        member, prompt = member_and_prompt
        calls.append(member.model)
        if member.model == "fake/triage":
            return tier_reply
        return CANNED_BINARY
    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", fake_invoke,
                        raising=True)
    return bot, calls


async def test_tier_is_classified_and_attached_even_when_routing_off(monkeypatch):
    bot, calls = make_triage_bot(monkeypatch)       # triage_enabled defaults False
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.tier == "near_certain"
    assert calls.count("fake/triage") == 1
    assert len(result.member_probs) == 6            # full ensemble: no routing


async def test_triage_routing_reduces_runs_for_near_certain(monkeypatch):
    bot, calls = make_triage_bot(monkeypatch, triage_enabled=True,
                                 triage_fast_runs=2, min_members=2)
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.tier == "near_certain"
    assert len(result.member_probs) == 4            # 2 models x 2 fast runs


async def test_triage_routing_keeps_full_ensemble_for_goldilocks(monkeypatch):
    bot, calls = make_triage_bot(monkeypatch, tier_reply="Tier: GOLDILOCKS",
                                 triage_enabled=True)
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.tier == "goldilocks"
    assert len(result.member_probs) == 6


async def test_triage_failure_is_unknown_and_full_ensemble(monkeypatch):
    bot, calls = make_triage_bot(monkeypatch, tier_reply="cannot classify",
                                 triage_enabled=True)
    result = await bot.run_binary_ensemble(**COMMON)
    assert result.tier == "unknown"
    assert len(result.member_probs) == 6
