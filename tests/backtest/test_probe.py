"""memorization_probe (Halawi App. A.2 analogue): zero-research direct
outcome prompt per model, temperature=0 run_index=0 through CachedLLM (so
re-probing is free); confident-correct => contaminated-confirmed for that
model."""
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.timetravel.llm_cache import CachedLLM
from backtest.timetravel.probe import memorization_probe


def _q(qtype=QType.BINARY, resolution=True, options=()):
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1", qtype=qtype,
        title="Did X happen by 2024-06-01?", resolution_criteria="Resolves "
        "Yes if X happened before 2024-06-01.", resolution=resolution,
        options=options,
        actual_resolve_time=datetime(2024, 6, 1, tzinfo=timezone.utc))


class ScriptedLLM:
    def __init__(self, completions):
        self.completions = completions       # model -> completion text
        self.calls = []

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append(dict(model=model, prompt=prompt,
                               temperature=temperature, run_index=run_index))
        return self.completions[model]


async def test_binary_confident_correct_flags_contamination():
    llm = ScriptedLLM({"model-a": "I recall this.\nProbability: 97%",
                       "model-b": "Unsure.\nProbability: 60%"})
    out = await memorization_probe(_q(resolution=True),
                                   ["model-a", "model-b"], llm)
    assert out == {"model-a": True, "model-b": False}
    assert all(c["temperature"] == 0.0 and c["run_index"] == 0
               for c in llm.calls)
    assert {c["model"] for c in llm.calls} == {"model-a", "model-b"}


async def test_binary_confident_correct_on_no_resolution():
    llm = ScriptedLLM({"m": "Probability: 3%"})
    assert await memorization_probe(_q(resolution=False), ["m"], llm) == \
        {"m": True}
    # confident but WRONG is not contamination evidence
    llm2 = ScriptedLLM({"m": "Probability: 97%"})
    assert await memorization_probe(_q(resolution=False), ["m"], llm2) == \
        {"m": False}


async def test_unparseable_completion_is_not_confident():
    llm = ScriptedLLM({"m": "I have no idea."})
    assert await memorization_probe(_q(), ["m"], llm) == {"m": False}


async def test_mc_probe_matches_winning_option():
    q = _q(qtype=QType.MULTIPLE_CHOICE, resolution="Option B",
           options=("Option A", "Option B"))
    llm = ScriptedLLM({"hit": "From memory:\nOption B",
                       "miss": "From memory:\nOption A"})
    out = await memorization_probe(q, ["hit", "miss"], llm)
    assert out == {"hit": True, "miss": False}


async def test_unresolved_or_numeric_questions_never_flag():
    llm = ScriptedLLM({})
    assert await memorization_probe(_q(resolution=None), ["m"], llm) == \
        {"m": False}
    assert await memorization_probe(
        _q(qtype=QType.NUMERIC, resolution=42.0), ["m"], llm) == {"m": False}
    assert llm.calls == []                   # no spend on unprobeable types


async def test_probe_is_cached_second_run_makes_no_live_calls(
        tmp_path, monkeypatch):
    live_calls = []

    async def fake_call_live(self, model, prompt, temperature):
        live_calls.append(model)
        return "Probability: 97%", 0, 0, 0.0

    monkeypatch.setattr(CachedLLM, "_call_live", fake_call_live)
    llm = CachedLLM(tmp_path / "cache")
    q = _q(resolution=True)
    first = await memorization_probe(q, ["m1"], llm)
    second = await memorization_probe(q, ["m1"], llm)
    assert first == second == {"m1": True}
    assert live_calls == ["m1"]              # second probe fully cached
