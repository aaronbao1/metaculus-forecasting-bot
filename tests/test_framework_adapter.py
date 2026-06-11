"""Framework-adapter tests. forecasting-tools is installed in the venv; no
network or API keys are touched — LLM and research seams are stubbed."""
import json

import pytest

from bot.config import BotConfig, EnsembleMember
from bot.ensemble_bot import (CalibratedEnsembleBot, build_framework_bot,
                              build_pchip_distribution_class)
from bot.logging_store import read_forecasts
from bot.research import ResearchResult

DECLARED = [(2.5, 10.0), (10, 20.0), (25, 30.0), (50, 40.0),
            (75, 50.0), (90, 60.0), (97.5, 70.0)]


def _percentiles():
    from forecasting_tools.data_models.numeric_report import Percentile
    return [Percentile(percentile=p / 100.0, value=v) for p, v in DECLARED]


def _dist(cls, **overrides):
    kwargs = dict(declared_percentiles=_percentiles(),
                  open_lower_bound=False, open_upper_bound=False,
                  lower_bound=0.0, upper_bound=100.0,
                  zero_point=None, cdf_size=201)
    kwargs.update(overrides)
    return cls(**kwargs)


def test_get_cdf_uses_pchip_construction_not_framework_default():
    """Regression: the framework publishes via get_cdf(), not the deprecated
    .cdf property — overriding only .cdf silently ships the default CDF."""
    from forecasting_tools import NumericDistribution
    cls = build_pchip_distribution_class()
    ours = _dist(cls).get_cdf()
    base = _dist(NumericDistribution).get_cdf()
    assert len(ours) == 201
    assert ours[0].percentile == pytest.approx(0.0, abs=1e-9)
    assert ours[-1].percentile == pytest.approx(1.0, abs=1e-9)
    # PCHIP-smoothed interior must differ from the framework construction —
    # if get_cdf() ever stops routing through our machinery, this fails
    diffs = [abs(a.percentile - b.percentile) for a, b in zip(ours, base)]
    assert max(diffs) > 1e-4


def test_get_cdf_delegates_for_log_scaled_and_discrete():
    from forecasting_tools import NumericDistribution
    cls = build_pchip_distribution_class()
    # log-scaled (zero_point set): framework construction, not ours
    ours = _dist(cls, zero_point=-1.0).get_cdf()
    base = _dist(NumericDistribution, zero_point=-1.0).get_cdf()
    assert [p.percentile for p in ours] == [p.percentile for p in base]
    # discrete (cdf_size != 201): framework construction with correct length
    ours_d = _dist(cls, cdf_size=101).get_cdf()
    assert len(ours_d) == 101


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    cfg = BotConfig(ensemble=[EnsembleMember("fake/model-a", runs=3),
                              EnsembleMember("fake/model-b", runs=3)],
                    log_path=str(tmp_path / "forecasts.jsonl"))

    async def fake_invoke(self, member_and_prompt):
        return "reasoning...\nProbability: 64%"

    async def fake_run_research(question, cfg_):
        return ResearchResult(text="stub research",
                              provider="asknews,markets",
                              providers=("asknews", "markets"))

    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", fake_invoke,
                        raising=True)
    import bot.ensemble_bot as eb
    monkeypatch.setattr(eb, "run_research", fake_run_research)
    return build_framework_bot(cfg), cfg


async def test_adapter_logs_full_provenance(adapter):
    from forecasting_tools import BinaryQuestion
    bot, cfg = adapter
    q = BinaryQuestion(question_text="Q?", id_of_post=123, id_of_question=456)
    research = await bot.run_research(q)
    assert research == "stub research"
    pred = await bot._run_forecast_on_binary(q, research)
    assert pred.prediction_value == pytest.approx(0.64, abs=1e-6)
    rows = read_forecasts(cfg.log_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["question_id"] == 123
    assert row["id_of_question"] == 456
    assert row["research_providers"] == "asknews,markets"
    assert row["raw"] == pytest.approx(0.64, abs=1e-6)
    assert row["spread"] == pytest.approx(0.0)
    assert row["config_hash"]


async def test_unwritable_log_does_not_fail_the_question(adapter, monkeypatch):
    from forecasting_tools import BinaryQuestion
    bot, cfg = adapter
    import bot.ensemble_bot as eb

    def explode(record, path):
        raise OSError("disk full")

    monkeypatch.setattr(eb, "append_forecast", explode)
    q = BinaryQuestion(question_text="Q?", id_of_post=1)
    await bot.run_research(q)
    pred = await bot._run_forecast_on_binary(q, "research")
    assert pred.prediction_value == pytest.approx(0.64, abs=1e-6)


async def test_cost_brake_slices_question_list(adapter, monkeypatch):
    from forecasting_tools import BinaryQuestion, ForecastBot
    bot, cfg = adapter
    cfg.max_questions_per_run = 2
    captured = {}

    async def fake_super(self, questions, return_exceptions=False):
        captured["n"] = len(questions)
        return []

    monkeypatch.setattr(ForecastBot, "forecast_questions", fake_super)
    qs = [BinaryQuestion(question_text=f"Q{i}?") for i in range(5)]
    await bot.forecast_questions(qs)
    assert captured["n"] == 2
