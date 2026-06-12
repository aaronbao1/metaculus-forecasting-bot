"""Task 1: new BotConfig fields (n_trim, extremize_a, prompt_overrides,
playbook_path) — single deliberate config-hash perturbation (roadmap #8).
Defaults must reproduce current behavior exactly."""
import asyncio

import pytest

from bot.aggregation import aggregate_binary, extremize, geometric_mean_of_odds
from bot.calibration import cap
from bot.config import BotConfig, EnsembleMember, config_hash


def test_new_fields_defaults():
    cfg = BotConfig()
    assert cfg.n_trim == 1
    assert cfg.extremize_a == 1.0
    assert cfg.prompt_overrides == {}
    assert cfg.playbook_path is None


@pytest.mark.parametrize("patch", [
    {"n_trim": 2},
    {"extremize_a": 1.41},
    {"prompt_overrides": {"binary": "X"}},
    {"playbook_path": "skills/forecasting-playbook.md"},
])
def test_non_default_fields_perturb_hash(patch):
    from dataclasses import replace
    assert config_hash(replace(BotConfig(), **patch)) != config_hash(BotConfig())


def test_aggregate_binary_threads_n_trim():
    probs = [0.05, 0.4, 0.5, 0.6, 0.95]
    assert aggregate_binary(probs, "trimmed_gmo", n_trim=0) == \
        pytest.approx(geometric_mean_of_odds(probs))
    assert aggregate_binary(probs, "trimmed_gmo", n_trim=2) == \
        pytest.approx(geometric_mean_of_odds([0.5]))
    # default unchanged: 1 per side
    assert aggregate_binary(probs, "trimmed_gmo") == \
        pytest.approx(geometric_mean_of_odds([0.4, 0.5, 0.6]))


def _bot(cfg):
    from bot.ensemble_bot import CalibratedEnsembleBot
    return CalibratedEnsembleBot(cfg)


def _run_binary(bot, prob_text="Probability: 70%"):
    async def fake(member_and_prompt):
        return prob_text

    bot._invoke_model = fake
    return asyncio.run(bot.run_binary_ensemble(
        question_text="Q", background_info="B", resolution_criteria="R",
        fine_print="F", research="RES", close_iso="2026-02-01T00:00:00+00:00"))


def test_extremize_a_applied_between_aggregation_and_caps():
    cfg = BotConfig(ensemble=[EnsembleMember("m", runs=1)], min_members=1,
                    extremize_a=2.0)
    r = _run_binary(_bot(cfg))
    assert r.aggregate_raw == pytest.approx(0.7, abs=1e-9)
    assert r.final == pytest.approx(cap(extremize(0.7, 2.0)), abs=1e-9)


def test_extremize_identity_by_default():
    cfg = BotConfig(ensemble=[EnsembleMember("m", runs=1)], min_members=1)
    r = _run_binary(_bot(cfg))
    assert r.final == pytest.approx(0.7, abs=1e-9)


def test_from_member_forecasts_honors_n_trim_and_extremize(tmp_path):
    from datetime import datetime, timezone

    from backtest.core import BacktestQuestion, QType
    from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
    from backtest.timetravel.asof import AsOfContext

    q = BacktestQuestion(qid="t:1", platform="t", native_id="1",
                         qtype=QType.BINARY, title="Q")
    ctx = AsOfContext(as_of=datetime(2026, 1, 1, tzinfo=timezone.utc))
    probs = [0.05, 0.4, 0.5, 0.6, 0.95]
    adapter = CalibratedEnsembleAdapter.variant(
        base=BotConfig(calibration_dir=str(tmp_path / "nope")),
        n_trim=0, extremize_a=2.0)
    art = asyncio.run(adapter.from_member_forecasts(q, ctx, probs))
    assert art.error is None
    expected = cap(extremize(geometric_mean_of_odds(probs), 2.0))
    assert art.forecast.binary_prob == pytest.approx(expected, abs=1e-9)
