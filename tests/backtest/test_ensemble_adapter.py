"""CalibratedEnsembleAdapter tests — no network, no forecasting-tools LLMs:
bot.research.run_research is monkeypatched and every model call goes through
a stub ctx.llm."""
from datetime import datetime, timezone

import pytest

import bot.research as research_mod
from backtest.core import BacktestQuestion, QType
from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot.config import BotConfig, EnsembleMember
from bot.research import ResearchResult

AS_OF = datetime(2024, 5, 15, 12, 0, tzinfo=timezone.utc)

BINARY_REPLY = "Reasoning...\nProbability: 64%"


class StubLLM:
    def __init__(self, reply=BINARY_REPLY, fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = []

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append(dict(model=model, prompt=prompt,
                               temperature=temperature, run_index=run_index))
        if self.fail:
            raise RuntimeError("boom")
        return self.reply


class FakeStore:
    def __init__(self, result="clean"):
        self.result = result
        self.calls = []

    def get_contamination(self, qid, model_set_hash):
        self.calls.append((qid, model_set_hash))
        return self.result


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
        fine_print="None.",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc))


@pytest.fixture
def stub_research(monkeypatch):
    seen = {}

    async def fake_run_research(question, cfg_):
        seen["ctx"] = ACTIVE_CONTEXT.get()
        seen["as_of"] = cfg_.as_of
        seen["question"] = question
        return ResearchResult(text="stub research", provider="asknews",
                              providers=("asknews",))

    monkeypatch.setattr(research_mod, "run_research", fake_run_research)
    return seen


async def test_binary_forecast_routes_through_ctx_llm(tmp_path, stub_research):
    llm = StubLLM()
    ctx = AsOfContext(as_of=AS_OF, llm=llm)
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_binary_q(), ctx)

    assert art.error is None
    assert art.qid == "metaculus:1"
    assert art.as_of == AS_OF
    assert art.cutoff_mode == "strict"
    assert art.system_name == "calibrated_ensemble"
    assert art.config_fingerprint == adapter.config_fingerprint()
    assert art.forecast.qtype is QType.BINARY
    # identity Platt (empty calibration dir) + caps [0.02, 0.98] -> 0.64
    assert art.forecast.binary_prob == pytest.approx(0.64, abs=1e-6)
    # 2 members x 2 runs routed through ctx.llm with per-member run_index
    assert len(llm.calls) == 4
    by_model = {}
    for c in llm.calls:
        by_model.setdefault(c["model"], []).append(c["run_index"])
    assert sorted(by_model["fake/model-a"]) == [0, 1]
    assert sorted(by_model["fake/model-b"]) == [0, 1]
    temps = {c["model"]: c["temperature"] for c in llm.calls}
    assert temps == {"fake/model-a": 0.3, "fake/model-b": 0.7}
    # as_of threaded into the prompt as `today`, research text injected
    assert "2024-05-15" in llm.calls[0]["prompt"]
    assert "stub research" in llm.calls[0]["prompt"]


async def test_active_context_set_during_research_and_reset_after(
        tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM())
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    await adapter.forecast(_binary_q(), ctx)
    assert stub_research["ctx"] is ctx
    assert stub_research["as_of"] == AS_OF
    assert ACTIVE_CONTEXT.get() is None


async def test_stage_records_cover_pipeline(tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM())
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_binary_q(), ctx)
    names = [s.stage for s in art.stages]
    assert names.count("research") == 1
    assert names.count("member_llm") == 4
    assert names.count("aggregation") == 1
    assert names.count("calibration") == 1
    research = next(s for s in art.stages if s.stage == "research")
    assert research.outputs["text"] == "stub research"
    assert research.outputs["provider"] == "asknews"
    member = next(s for s in art.stages if s.stage == "member_llm")
    assert member.model in ("fake/model-a", "fake/model-b")
    assert "prompt" in member.inputs
    assert member.outputs["completion"] == BINARY_REPLY
    agg = next(s for s in art.stages if s.stage == "aggregation")
    assert agg.inputs["method"] == "trimmed_gmo"
    assert agg.outputs["aggregate_raw"] == pytest.approx(0.64, abs=1e-6)
    cal = next(s for s in art.stages if s.stage == "calibration")
    assert cal.outputs["final"] == pytest.approx(0.64, abs=1e-6)


async def test_failure_returns_error_artifact_never_raises(
        tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM(fail=True))
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_binary_q(), ctx)
    assert art.forecast is None
    assert "ensemble members survived" in art.error
    failed = [s for s in art.stages if s.stage == "member_llm"]
    assert len(failed) == 4
    assert all("boom" in s.outputs["error"] for s in failed)
    assert ACTIVE_CONTEXT.get() is None


async def test_contamination_from_store_else_contaminated(
        tmp_path, stub_research):
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    # no store -> conservative default
    art = await adapter.forecast(
        _binary_q(), AsOfContext(as_of=AS_OF, llm=StubLLM()))
    assert art.contamination == "contaminated"
    # store present -> looked up by (qid, model_set_hash)
    fake = FakeStore("clean")
    art2 = await adapter.forecast(
        _binary_q(), AsOfContext(as_of=AS_OF, llm=StubLLM(), store=fake))
    assert art2.contamination == "clean"
    qid, mhash = fake.calls[0]
    assert qid == "metaculus:1"
    assert len(mhash) == 12


def test_variant_patches_config():
    v = CalibratedEnsembleAdapter.variant(aggregation_method="mean",
                                          cap_hi=0.95)
    assert v.cfg.aggregation_method == "mean"
    assert v.cfg.cap_hi == 0.95
    assert (v.config_fingerprint()
            != CalibratedEnsembleAdapter().config_fingerprint())


MC_REPLY = "Analysis...\nRed: 70%\nBlue: 30%"
NUMERIC_REPLY = (
    "Percentile 2.5: 10\nPercentile 10: 20\nPercentile 25: 30\n"
    "Percentile 50: 40\nPercentile 75: 50\nPercentile 90: 60\n"
    "Percentile 97.5: 70")


def _mc_q():
    return BacktestQuestion(
        qid="metaculus:2", platform="metaculus", native_id="2",
        qtype=QType.MULTIPLE_CHOICE, title="Which colour?",
        options=("Red", "Blue"),
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc))


def _numeric_q(**scaling_overrides):
    scaling = {"range_min": 0.0, "range_max": 100.0, "zero_point": None,
               "open_lower_bound": False, "open_upper_bound": True,
               "cdf_size": 201, "unit": "widgets"}
    scaling.update(scaling_overrides)
    return BacktestQuestion(
        qid="metaculus:3", platform="metaculus", native_id="3",
        qtype=QType.NUMERIC, title="How many widgets?",
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc),
        scaling=scaling)


async def test_mc_dispatch(tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM(reply=MC_REPLY))
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_mc_q(), ctx)
    assert art.error is None
    assert art.forecast.qtype is QType.MULTIPLE_CHOICE
    assert art.forecast.mc_probs["Red"] == pytest.approx(0.7, abs=1e-6)
    assert sum(art.forecast.mc_probs.values()) == pytest.approx(1.0)
    names = [s.stage for s in art.stages]
    assert names.count("aggregation") == 1
    assert names.count("calibration") == 1


async def test_numeric_dispatch_builds_201_point_cdf(tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM(reply=NUMERIC_REPLY))
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_numeric_q(), ctx)
    assert art.error is None
    cdf = art.forecast.cdf
    assert isinstance(cdf, tuple)
    assert len(cdf) == 201
    assert cdf[0] == pytest.approx(0.0, abs=1e-9)      # closed lower bound
    assert cdf[-1] == pytest.approx(0.999, abs=1e-9)   # open upper bound
    assert all(b - a >= 5e-05 - 1e-9 for a, b in zip(cdf, cdf[1:]))


async def test_numeric_nonstandard_cdf_size_is_error_artifact(
        tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM(reply=NUMERIC_REPLY))
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_numeric_q(cdf_size=101), ctx)
    assert art.forecast is None
    assert "NotImplementedError" in art.error
