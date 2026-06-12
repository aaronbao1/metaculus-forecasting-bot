"""Task 14: PromptFitness — cached-research prompt evaluation with a hard
budget cap. Uses a real CachedLLM with a stubbed live seam so the cache-hit
economics are the production ones."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord, artifact_to_dict)
from backtest.corpus.store import CorpusStore
from backtest.optimize.fitness import (BudgetExceeded, Candidate,
                                       PromptFitness)
from backtest.timetravel.llm_cache import CachedLLM
from bot.config import BotConfig, EnsembleMember
from bot.prompts import BINARY_PROMPT

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _fixture(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = CorpusStore(tmp_path / "corpus")
    arts = []
    for i, res in enumerate([True, False, True]):
        q = BacktestQuestion(
            qid=f"t:{i}", platform="t", native_id=str(i),
            qtype=QType.BINARY, title=f"Q{i}?", open_time=T0,
            scheduled_close_time=T0 + timedelta(days=10 + i),
            resolution=res)
        store.upsert_question(q)
        arts.append(ForecastArtifact(
            qid=q.qid, horizon="close-7d", as_of=T0 + timedelta(days=3),
            system_name="current", config_fingerprint="x",
            forecast=Forecast(qtype=QType.BINARY, binary_prob=0.5),
            stages=[StageRecord(stage="research",
                                outputs={"text": f"research for Q{i}"})]))
    (run_dir / "artifacts.jsonl").write_text(
        "\n".join(json.dumps(artifact_to_dict(a)) for a in arts))
    return run_dir, store


def _llm(tmp_path, monkeypatch, cost=0.01):
    llm = CachedLLM(tmp_path / "llm_cache", live=True)

    calls = {"live": 0}

    async def fake_live(model, prompt, temperature):
        calls["live"] += 1
        return "Probability: 70%", 0, 0, cost

    monkeypatch.setattr(llm, "_call_live", fake_live)
    return llm, calls


def _base_cfg(tmp_path):
    return BotConfig(ensemble=[EnsembleMember("m", runs=2)], min_members=1,
                     calibration_dir=str(tmp_path / "no-cal"))


def test_prompt_fitness_evaluates_and_costs_then_caches(tmp_path, monkeypatch):
    run_dir, store = _fixture(tmp_path)
    llm, calls = _llm(tmp_path, monkeypatch)
    fit = PromptFitness(run_dir, store, "current", llm, budget_usd=5.0,
                        base_config=_base_cfg(tmp_path))
    cand = Candidate("prompt:binary", {"template": BINARY_PROMPT})
    r1 = asyncio.run(fit.evaluate(cand))
    assert r1.n == 3
    assert calls["live"] == 6                  # 3 questions x 2 runs, fresh
    assert fit.counting.fresh_calls == 6
    assert r1.cost_usd == pytest.approx(0.06)  # actual cost from cache records

    r2 = asyncio.run(fit.evaluate(cand))       # identical template -> cached
    assert calls["live"] == 6                  # no new live calls
    assert fit.counting.cached_calls == 6
    assert r2.cost_usd == 0.0
    assert r2.score == pytest.approx(r1.score)


def test_mutated_template_spends_again(tmp_path, monkeypatch):
    run_dir, store = _fixture(tmp_path)
    llm, calls = _llm(tmp_path, monkeypatch)
    fit = PromptFitness(run_dir, store, "current", llm, budget_usd=5.0,
                        base_config=_base_cfg(tmp_path))
    asyncio.run(fit.evaluate(
        Candidate("prompt:binary", {"template": BINARY_PROMPT})))
    asyncio.run(fit.evaluate(
        Candidate("prompt:binary",
                  {"template": BINARY_PROMPT + "\nBe bold."})))
    assert calls["live"] == 12


def test_budget_exceeded_raises_before_overspend(tmp_path, monkeypatch):
    run_dir, store = _fixture(tmp_path)
    llm, calls = _llm(tmp_path, monkeypatch, cost=0.5)
    fit = PromptFitness(run_dir, store, "current", llm, budget_usd=1.0,
                        base_config=_base_cfg(tmp_path),
                        est_cost_per_fresh=0.5)
    with pytest.raises(BudgetExceeded):
        asyncio.run(fit.evaluate(
            Candidate("prompt:binary", {"template": BINARY_PROMPT})))
    assert fit.counting.cost_usd <= 1.5        # bounded overshoot (one item)


def test_fitted_platt_in_base_config_rejected(tmp_path, monkeypatch):
    run_dir, store = _fixture(tmp_path)
    llm, _ = _llm(tmp_path, monkeypatch)
    cal = tmp_path / "fitted"
    cal.mkdir()
    (cal / "binary.json").write_text('{"a": 1.2, "b": -0.3, "n_fit": 200}')
    with pytest.raises(ValueError, match="leakage"):
        PromptFitness(run_dir, store, "current", llm, budget_usd=1.0,
                      base_config=BotConfig(calibration_dir=str(cal)))
