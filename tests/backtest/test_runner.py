"""Runner tests: full offline runs over a real CorpusStore in tmp_path with
a stub system. No network; CachedLLM is constructed but never invoked."""
import json
from datetime import datetime, timezone

import pytest

import backtest.experiments.runner as runner_mod
from backtest.core import BacktestQuestion, Forecast, ForecastArtifact, QType
from backtest.corpus.store import CorpusStore
from backtest.experiments.profiles import PROFILES, Profile
from backtest.experiments.runner import RunResult, run_experiment
from backtest.experiments.spec import ExperimentSpec


class StubSystem:
    name = "stub"

    def __init__(self):
        self.calls = 0

    def config_fingerprint(self) -> str:
        return "stub-v1"

    async def forecast(self, q, ctx):
        self.calls += 1
        return ForecastArtifact(
            qid=q.qid, horizon="", as_of=ctx.as_of, system_name=self.name,
            config_fingerprint="stub-v1",
            forecast=Forecast(qtype=QType.BINARY, binary_prob=0.8),
            cutoff_mode=ctx.cutoff_mode)


def _q(i: int, resolution: bool) -> BacktestQuestion:
    return BacktestQuestion(
        qid=f"metaculus:{i}", platform="metaculus", native_id=str(i),
        qtype=QType.BINARY, title=f"Question {i}?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
        resolution=resolution)


@pytest.fixture
def store(tmp_path):
    s = CorpusStore(tmp_path / "corpus")
    s.upsert_question(_q(1, True))
    s.upsert_question(_q(2, False))
    s.put_series("metaculus:1", "cp",
                 [("2024-05-01T00:00:00+00:00", 0.66)])
    return s


def _spec(**kw):
    base = dict(name="t", corpus={"qtypes": ["binary"]},
                horizons=["close-7d"], cutoff_mode="strict",
                profile="smoke",
                variants=[{"name": "stub", "patch": {}}],
                baselines=[], seed=0)
    base.update(kw)
    return ExperimentSpec(**base)


@pytest.fixture
def stub_systems(monkeypatch):
    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)
    return stub


async def test_full_run_writes_manifest_artifacts_scores(
        tmp_path, store, stub_systems):
    result = await run_experiment(_spec(), store, tmp_path / "runs",
                                  live_llm=False, yes=True)
    assert isinstance(result, RunResult)
    run_dir = result.run_dir

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert len(manifest["statuses"]) == 2     # 2 q × 1 horizon × 1 system
    assert set(manifest["statuses"].values()) == {"done"}

    artifacts = [json.loads(l) for l in
                 (run_dir / "artifacts.jsonl").read_text().splitlines()]
    assert len(artifacts) == 2
    assert all(a["horizon"] == "close-7d" for a in artifacts)

    rows = [json.loads(l) for l in
            (run_dir / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert set(rows[0]) == {"qid", "horizon", "variant", "log_score",
                            "baseline_score", "brier", "h2h_cp",
                            "h2h_market", "stratum", "resolution"}
    by_qid = {r["qid"]: r for r in rows}
    assert by_qid["metaculus:1"]["h2h_cp"] is not None   # CP series present
    assert by_qid["metaculus:2"]["h2h_cp"] is None
    assert isinstance(by_qid["metaculus:1"]["baseline_score"], float)
    assert (run_dir / "stats.json").exists()


async def test_run_id_is_deterministic(tmp_path, store, stub_systems):
    r1 = await run_experiment(_spec(), store, tmp_path / "a",
                              live_llm=False, yes=True)
    r2 = await run_experiment(_spec(), store, tmp_path / "b",
                              live_llm=False, yes=True)
    assert r1.run_id == r2.run_id


async def test_resume_skips_completed(tmp_path, store, stub_systems,
                                      monkeypatch):
    await run_experiment(_spec(), store, tmp_path / "runs",
                         live_llm=False, yes=True)
    stub2 = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub2})
    result = await run_experiment(_spec(), store, tmp_path / "runs",
                                  live_llm=False, yes=True)
    assert stub2.calls == 0
    assert set(result.statuses.values()) == {"done"}


async def test_unreachable_horizon_is_skipped(tmp_path, store, stub_systems):
    # close − 365d is before open_time -> horizon_to_asof returns None
    result = await run_experiment(_spec(horizons=["close-365d"]), store,
                                  tmp_path / "runs", live_llm=False, yes=True)
    assert set(result.statuses.values()) == {"skipped"}
    assert stub_systems.calls == 0


async def test_system_crash_is_isolated(tmp_path, store, monkeypatch):
    class Crashy(StubSystem):
        async def forecast(self, q, ctx):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": Crashy()})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)
    result = await run_experiment(_spec(), store, tmp_path / "runs",
                                  live_llm=False, yes=True)
    assert set(result.statuses.values()) == {"error"}
    artifacts = [json.loads(l) for l in
                 (result.run_dir / "artifacts.jsonl").read_text().splitlines()]
    assert all("kaboom" in a["error"] for a in artifacts)
    assert not (result.run_dir / "scores.jsonl").read_text().strip()


async def test_cost_brake_aborts(tmp_path, store, stub_systems, monkeypatch):
    monkeypatch.setitem(PROFILES, "smoke",
                        Profile(max_questions=2, cost_cap_usd=0.01,
                                est_cost_per_question_usd=1.0))
    with pytest.raises(RuntimeError, match="cost brake"):
        await run_experiment(_spec(), store, tmp_path / "runs",
                             live_llm=False, yes=True)


async def test_confirmation_gate_aborts_on_no(tmp_path, store, stub_systems,
                                              monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    result = await run_experiment(_spec(), store, tmp_path / "runs",
                                  live_llm=False, yes=False)
    assert result.statuses == {}
    assert not (result.run_dir / "artifacts.jsonl").exists()


# ---------------------------------------------------------------------------
# Addendum: probe wiring
# ---------------------------------------------------------------------------

class ScriptedLLM:
    """Stub LLM that returns pre-scripted completions by model name."""

    def __init__(self, completions: dict):
        self.completions = completions

    async def invoke(self, model: str, prompt: str, temperature: float,
                     run_index: int = 0) -> str:
        return self.completions[model]


async def test_probe_contaminated_confirmed_in_score_row(
        tmp_path, monkeypatch):
    """A question whose probe completion is confidently-correct gets
    stratum 'contaminated-confirmed' stamped into its score row."""
    from backtest.corpus.model_releases import model_set_hash
    from bot.config import BotConfig

    s = CorpusStore(tmp_path / "corpus")
    # question that resolves True — we'll make probe confidently correct
    q = _q(1, True)
    s.upsert_question(q)
    s.put_series("metaculus:1", "cp", [("2024-05-01T00:00:00+00:00", 0.66)])

    # Stub system returns a valid forecast
    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)

    # Scripted LLM: confident "Probability: 97%" -> contaminated-confirmed
    models = [m.model for m in BotConfig().ensemble]
    completions = {m: "I recall it happened.\nProbability: 97%" for m in models}
    scripted_llm = ScriptedLLM(completions)

    # Monkeypatch CachedLLM constructor so the runner uses our scripted LLM
    monkeypatch.setattr(runner_mod, "CachedLLM",
                        lambda cache_dir, live: scripted_llm)

    result = await run_experiment(
        _spec(), s, tmp_path / "runs",
        live_llm=True, yes=True, probe=True)

    rows = [json.loads(l) for l in
            (result.run_dir / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["stratum"] == "contaminated-confirmed"
