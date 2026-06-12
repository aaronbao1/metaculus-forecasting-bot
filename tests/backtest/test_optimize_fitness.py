"""Task 13: candidate/fitness framework + $0 ReplayFitness."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord, artifact_to_dict)
from backtest.corpus.store import CorpusStore
from backtest.optimize.fitness import (Candidate, ReplayFitness,
                                       temporal_split)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _q(i, resolution):
    return BacktestQuestion(
        qid=f"t:{i}", platform="t", native_id=str(i), qtype=QType.BINARY,
        title=f"Q{i}?", open_time=T0,
        scheduled_close_time=T0 + timedelta(days=10 + i),
        resolution=resolution)


def fixture_run(tmp_path, member_probs_by_q=None):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    store = CorpusStore(tmp_path / "corpus")
    probs_map = member_probs_by_q or {
        0: ([0.7, 0.8, 0.9], True), 1: ([0.6, 0.7, 0.8], True),
        2: ([0.2, 0.3, 0.4], False), 3: ([0.1, 0.2, 0.3], False)}
    arts = []
    for i, (probs, res) in probs_map.items():
        q = _q(i, res)
        store.upsert_question(q)
        arts.append(ForecastArtifact(
            qid=q.qid, horizon="close-7d", as_of=T0 + timedelta(days=3 + i),
            system_name="current", config_fingerprint="x",
            forecast=Forecast(qtype=QType.BINARY, binary_prob=0.5),
            stages=[StageRecord(stage="aggregation",
                                inputs={"member_probs": probs})]))
    (run_dir / "artifacts.jsonl").write_text(
        "\n".join(json.dumps(artifact_to_dict(a)) for a in arts))
    return run_dir, store


def test_temporal_split_orders_by_close_time():
    qs = [_q(i, True) for i in (3, 0, 2, 1)]
    train, val = temporal_split(qs, train_frac=0.5)
    assert [q.qid for q in train] == ["t:0", "t:1"]
    assert [q.qid for q in val] == ["t:2", "t:3"]


def test_replay_fitness_scores_params_candidate(tmp_path):
    run_dir, store = fixture_run(tmp_path)
    fit = ReplayFitness(run_dir, store, "current")
    res = asyncio.run(fit.evaluate(Candidate("params", {"n_trim": 0})))
    assert res.n == 4
    assert res.score > 0                       # well-aligned probs beat chance
    fb = res.per_item[0]["feedback"]
    assert "forecast=" in fb and "resolved=" in fb and "baseline_score=" in fb


def test_replay_fitness_caps_to_half_scores_zero(tmp_path):
    run_dir, store = fixture_run(tmp_path)
    fit = ReplayFitness(run_dir, store, "current")
    res = asyncio.run(fit.evaluate(
        Candidate("params", {"cap_lo": 0.5, "cap_hi": 0.5})))
    assert res.score == pytest.approx(0.0, abs=1e-9)


def test_replay_fitness_qids_subset(tmp_path):
    run_dir, store = fixture_run(tmp_path)
    fit = ReplayFitness(run_dir, store, "current")
    res = asyncio.run(fit.evaluate(Candidate("params", {}),
                                   qids={"t:0", "t:2"}))
    assert res.n == 2


def test_replay_fitness_rejects_prompt_kind(tmp_path):
    run_dir, store = fixture_run(tmp_path)
    fit = ReplayFitness(run_dir, store, "current")
    with pytest.raises(ValueError, match="PromptFitness"):
        asyncio.run(fit.evaluate(Candidate("prompt:binary", {"t": "x"})))
