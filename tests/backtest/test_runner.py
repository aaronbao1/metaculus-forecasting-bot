"""Runner tests: full offline runs over a real CorpusStore in tmp_path with
a stub system. No network; CachedLLM is constructed but never invoked."""
import json
from datetime import datetime, timezone
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Addendum: regression locks from adversarial review
# ---------------------------------------------------------------------------

async def test_asymmetric_error_pairing(tmp_path, store, monkeypatch):
    """stats.json n should equal the common-cell count (the variant that errors
    on question 2 should not be counted for that cell in the paired comparison).
    """
    from backtest.core import StageRecord

    class ErrorOnQ2(StubSystem):
        name = "erroring"

        async def forecast(self, q, ctx):
            if q.qid == "metaculus:2":
                raise RuntimeError("deliberate error on q2")
            return await super().forecast(q, ctx)

    stub_a = StubSystem()
    stub_a.name = "variant_a"
    stub_b = ErrorOnQ2()
    stub_b.name = "variant_b"

    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"variant_a": stub_a,
                                             "variant_b": stub_b})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)

    # 2 variants, 2 questions, 1 horizon → variant_b errors on q2 → only q1 common
    spec = _spec(variants=[{"name": "variant_a", "patch": {}},
                            {"name": "variant_b", "patch": {}}],
                 baselines=[])
    result = await run_experiment(spec, store, tmp_path / "runs",
                                  live_llm=False, yes=True)

    stats = json.loads((result.run_dir / "stats.json").read_text())
    # Only one pairing key exists
    assert len(stats) == 1
    pair_key = next(iter(stats))
    # common-cell count must be 1 (only q1 succeeded for both variants)
    assert stats[pair_key]["n"] == 1


async def test_resume_no_duplication(tmp_path, store, monkeypatch):
    """An interrupted-then-resumed run must not produce duplicate score rows."""
    call_count = 0
    original_forecast = StubSystem.forecast

    async def interruptable_forecast(self, q, ctx):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            # Simulate crash after first artifact is written
            raise RuntimeError("simulated mid-run crash")
        return await original_forecast(self, q, ctx)

    stub1 = StubSystem()
    stub1.forecast = interruptable_forecast.__get__(stub1, StubSystem)
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub1})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)

    # First run — crashes mid-way (second call raises)
    result1 = await run_experiment(_spec(), store, tmp_path / "runs",
                                   live_llm=False, yes=True)
    # Exactly one question is "done", one is "error"
    statuses1 = result1.statuses
    assert "done" in statuses1.values()
    assert "error" in statuses1.values()

    # Resume — fresh stub, no crash
    stub2 = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub2})
    result2 = await run_experiment(_spec(), store, tmp_path / "runs",
                                   live_llm=False, yes=True)
    assert set(result2.statuses.values()) == {"done"}

    # scores.jsonl must have exactly 2 rows (no duplicates)
    rows = [json.loads(l) for l in
            (result2.run_dir / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    qids = [r["qid"] for r in rows]
    assert len(qids) == len(set(qids)), "duplicate score rows detected"


async def test_mid_run_actual_spend_brake(tmp_path, store, monkeypatch):
    """After each forecast the runner accumulates actual cost; it must raise
    RuntimeError when actual spend crosses cost_cap_usd, and the manifest
    must remain loadable and resumable after the raise.
    """
    from backtest.core import StageRecord

    EXPENSIVE = 10.0  # cost per forecast cell

    class ExpensiveSystem(StubSystem):
        async def forecast(self, q, ctx):
            art = await super().forecast(q, ctx)
            art.stages.append(StageRecord(stage="llm", cost_usd=EXPENSIVE))
            return art

    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": ExpensiveSystem()})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)
    # cap lower than cost of 2 cells so the brake fires on the second cell
    monkeypatch.setitem(PROFILES, "smoke",
                        Profile(max_questions=2, cost_cap_usd=15.0,
                                est_cost_per_question_usd=0.02))

    run_dir_parent = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="cost brake.*actual"):
        await run_experiment(_spec(), store, run_dir_parent,
                             live_llm=False, yes=True)

    # Manifest must still be valid JSON and contain at least one "done" cell
    import glob
    manifests = list(Path(run_dir_parent).glob("*/manifest.json"))
    assert manifests, "no manifest written before brake fired"
    manifest = json.loads(manifests[0].read_text())
    assert "statuses" in manifest
    assert "done" in manifest["statuses"].values()
    # actual_cost_usd must be present in the manifest
    assert "actual_cost_usd" in manifest


def test_corpus_limit_caps_below_profile(tmp_path):
    """spec corpus 'limit' tightens the profile's max_questions cap."""
    from backtest.experiments.runner import _select_questions
    from backtest.experiments.profiles import PROFILES
    from backtest.experiments.spec import ExperimentSpec

    store = CorpusStore(tmp_path / "store")
    for i in range(5):
        store.upsert_question(_q(i, resolution=True))
    spec = ExperimentSpec(name="cap", corpus={"limit": 2},
                          horizons=["close-7d"], cutoff_mode="strict",
                          profile="standard", variants=[], baselines=[],
                          seed=0)
    qs = _select_questions(spec, store, PROFILES["standard"])
    assert len(qs) == 2
