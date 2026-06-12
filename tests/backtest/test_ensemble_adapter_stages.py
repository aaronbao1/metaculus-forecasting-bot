"""Stage entry points: from_research_text (skip research) and
from_member_forecasts (aggregation+calibration only, no LLM)."""
from datetime import datetime, timezone

import pytest

import bot.research as research_mod
from backtest.core import BacktestQuestion, QType
from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
from backtest.timetravel.asof import AsOfContext
from bot.config import BotConfig, EnsembleMember

AS_OF = datetime(2024, 5, 15, 12, 0, tzinfo=timezone.utc)


class StubLLM:
    def __init__(self, reply="Reasoning...\nProbability: 64%"):
        self.reply = reply
        self.calls = []

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append(dict(model=model, prompt=prompt,
                               temperature=temperature, run_index=run_index))
        return self.reply


def _cfg(tmp_path, **kw):
    return BotConfig(
        ensemble=[EnsembleMember("fake/model-a", runs=2, temperature=0.3),
                  EnsembleMember("fake/model-b", runs=2, temperature=0.7)],
        calibration_dir=str(tmp_path / "calib"),
        log_path=str(tmp_path / "forecasts.jsonl"),
        **kw)


def _binary_q():
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="Will X happen?",
        description="Background.", resolution_criteria="Resolves YES if X.",
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc))


def _mc_q():
    return BacktestQuestion(
        qid="metaculus:2", platform="metaculus", native_id="2",
        qtype=QType.MULTIPLE_CHOICE, title="Which colour?",
        options=("Red", "Blue"),
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc))


async def test_from_research_text_skips_run_research(tmp_path, monkeypatch):
    async def explode(question, cfg_):
        raise AssertionError("run_research must not be called")

    monkeypatch.setattr(research_mod, "run_research", explode)
    llm = StubLLM()
    ctx = AsOfContext(as_of=AS_OF, llm=llm)
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.from_research_text(_binary_q(), ctx,
                                           "FROZEN RESEARCH")
    assert art.error is None
    assert art.forecast.binary_prob == pytest.approx(0.64, abs=1e-6)
    research = next(s for s in art.stages if s.stage == "research")
    assert research.outputs["provider"] == "injected"
    assert research.outputs["text"] == "FROZEN RESEARCH"
    # the injected text actually reaches the member prompts
    assert "FROZEN RESEARCH" in llm.calls[0]["prompt"]


async def test_from_member_forecasts_aggregation_and_calibration_only(
        tmp_path):
    adapter = CalibratedEnsembleAdapter(
        _cfg(tmp_path, aggregation_method="mean"))
    ctx = AsOfContext(as_of=AS_OF)            # no llm: must not be needed
    art = await adapter.from_member_forecasts(_binary_q(), ctx,
                                              [0.6, 0.7, 0.8])
    assert art.error is None
    assert art.forecast.binary_prob == pytest.approx(0.7, abs=1e-9)
    assert [s.stage for s in art.stages] == ["aggregation", "calibration"]
    agg = art.stages[0]
    assert agg.inputs["member_probs"] == [0.6, 0.7, 0.8]
    assert agg.inputs["method"] == "mean"


async def test_from_member_forecasts_applies_caps(tmp_path):
    adapter = CalibratedEnsembleAdapter(
        _cfg(tmp_path, aggregation_method="mean"))
    ctx = AsOfContext(as_of=AS_OF)
    art = await adapter.from_member_forecasts(_binary_q(), ctx,
                                              [0.999, 0.999, 0.999])
    assert art.forecast.binary_prob == pytest.approx(0.98)   # cap_hi


async def test_from_member_forecasts_non_binary_raises(tmp_path):
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    ctx = AsOfContext(as_of=AS_OF)
    with pytest.raises(NotImplementedError):
        await adapter.from_member_forecasts(_mc_q(), ctx, [{"Red": 1.0}])


async def test_from_member_forecasts_runtime_error_is_isolated(tmp_path):
    adapter = CalibratedEnsembleAdapter(
        _cfg(tmp_path, aggregation_method="mean"))
    ctx = AsOfContext(as_of=AS_OF)
    art = await adapter.from_member_forecasts(_binary_q(), ctx, [])
    assert art.forecast is None
    assert "no forecasts" in art.error
