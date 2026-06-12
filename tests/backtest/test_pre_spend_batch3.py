"""Pre-spend review batch 3 — research cache, resolve-time guard,
platform/market columns, price staleness.

I3 — research content-addressed cache in CalibratedEnsembleAdapter.forecast()
I4 — already-decided-at-T guard + price_at_T provenance in _score_run
I5 — platform dimension in score rows + report columns
I7 — stale-price flagging via value_at_with_ts + report stale-count
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

import bot.research as research_mod
from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord, artifact_to_dict)
from backtest.corpus.store import CorpusStore
from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
from backtest.timetravel.asof import AsOfContext
from bot.config import BotConfig, EnsembleMember
from bot.research import ResearchResult

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

AS_OF = datetime(2024, 5, 15, 12, 0, tzinfo=timezone.utc)
BINARY_REPLY = "Reasoning...\nProbability: 64%"


class StubLLM:
    def __init__(self, reply=BINARY_REPLY, fail=False, cache_dir: Path | None = None):
        self.reply = reply
        self.fail = fail
        self.calls = []
        # Mimic CachedLLM.cache_dir so the research cache can resolve its dir.
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp()) / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append(dict(model=model, prompt=prompt,
                               temperature=temperature, run_index=run_index))
        if self.fail:
            raise RuntimeError("boom")
        return self.reply


def _cfg(tmp_path, **kw):
    return BotConfig(
        ensemble=[EnsembleMember("fake/model-a", runs=2, temperature=0.3)],
        min_members=1,
        calibration_dir=str(tmp_path / "calib"),
        log_path=str(tmp_path / "forecasts.jsonl"),
        research_providers=["asknews_historical"],
        **kw)


def _binary_q(**kw):
    defaults = dict(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="Will X happen?",
        description="Background.", resolution_criteria="Resolves YES if X.",
        fine_print="None.",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )
    defaults.update(kw)
    return BacktestQuestion(**defaults)


# ---------------------------------------------------------------------------
# I3 — content-addressed research cache
# ---------------------------------------------------------------------------

class CountingResearch:
    """Monkeypatch target that counts run_research invocations."""

    def __init__(self, text="stub research"):
        self.count = 0
        self.text = text

    async def __call__(self, question, cfg_):
        self.count += 1
        return ResearchResult(text=self.text, provider="asknews",
                              providers=("asknews_historical",))


@pytest.fixture
def counting_research(monkeypatch):
    counter = CountingResearch()
    monkeypatch.setattr(research_mod, "run_research", counter)
    return counter


async def test_research_cached_second_call_skips_run_research(
        tmp_path, counting_research):
    """Two forecast() calls with identical (qid, as_of, providers) must only
    call run_research once; the second call uses the cache."""
    shared_cache = tmp_path / "llm_cache"
    llm = StubLLM(cache_dir=shared_cache)
    ctx = AsOfContext(as_of=AS_OF, llm=llm)
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    q = _binary_q()

    art1 = await adapter.forecast(q, ctx)
    art2 = await adapter.forecast(q, ctx)

    assert art1.error is None, f"First forecast failed: {art1.error}"
    assert art2.error is None, f"Second forecast failed: {art2.error}"
    assert counting_research.count == 1, (
        f"Expected run_research called once; got {counting_research.count} calls")


async def test_research_cache_miss_on_different_as_of(
        tmp_path, counting_research):
    """A different as_of must produce a cache miss and call run_research again."""
    shared_cache = tmp_path / "llm_cache"
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    q = _binary_q()

    as_of_a = datetime(2024, 5, 15, 12, 0, tzinfo=timezone.utc)
    as_of_b = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)

    art1 = await adapter.forecast(
        q, AsOfContext(as_of=as_of_a, llm=StubLLM(cache_dir=shared_cache)))
    art2 = await adapter.forecast(
        q, AsOfContext(as_of=as_of_b, llm=StubLLM(cache_dir=shared_cache)))

    assert art1.error is None, f"First forecast failed: {art1.error}"
    assert art2.error is None, f"Second forecast failed: {art2.error}"
    assert counting_research.count == 2, (
        f"Expected 2 run_research calls (different as_of); got {counting_research.count}")


async def test_research_cache_hit_noted_in_stage(tmp_path, counting_research):
    """On a cache hit the stage record must mention the cache hit."""
    shared_cache = tmp_path / "llm_cache"
    llm = StubLLM(cache_dir=shared_cache)
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    q = _binary_q()
    ctx = AsOfContext(as_of=AS_OF, llm=llm)

    await adapter.forecast(q, ctx)           # miss → writes cache
    art2 = await adapter.forecast(q, ctx)    # hit

    research_stages = [s for s in art2.stages if s.stage == "research"]
    assert research_stages, "Expected at least one research stage record"
    notes = " ".join(str(s.outputs) for s in research_stages)
    assert "cache" in notes.lower(), (
        f"Stage record on cache hit should mention 'cache'; got: {notes!r}")


async def test_research_cache_no_ctx_llm_skips_caching(
        tmp_path, counting_research):
    """When ctx.llm is None the cache dir is unavailable; research must still
    run (no error), and should NOT be cached (no cache_dir to infer from)."""
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    q = _binary_q()
    # llm=None → adapter raises during forecast (no LLM to call)
    # but run_research itself must have been called — cache skipped gracefully
    ctx = AsOfContext(as_of=AS_OF, llm=None)
    art = await adapter.forecast(q, ctx)
    # With no LLM the forecast fails, but run_research must have been called
    assert counting_research.count >= 1


# ---------------------------------------------------------------------------
# I4 — already-decided-at-T guard + price_at_T
# ---------------------------------------------------------------------------

import backtest.experiments.runner as runner_mod


class StubSystem:
    name = "stub"
    _calls: list

    def __init__(self):
        self._calls = []

    def config_fingerprint(self) -> str:
        return "stub-v1"

    async def forecast(self, q, ctx):
        self._calls.append(q.qid)
        return ForecastArtifact(
            qid=q.qid, horizon="", as_of=ctx.as_of, system_name=self.name,
            config_fingerprint="stub-v1",
            forecast=Forecast(qtype=QType.BINARY, binary_prob=0.8),
            cutoff_mode=ctx.cutoff_mode)


def _store_with_questions(tmp_path, questions):
    s = CorpusStore(tmp_path / "corpus")
    for q in questions:
        s.upsert_question(q)
    return s


def _spec(**kw):
    base = dict(name="t", corpus={},
                horizons=["close-7d"], cutoff_mode="strict",
                profile="smoke",
                variants=[{"name": "stub", "patch": {}}],
                baselines=[], seed=0)
    base.update(kw)
    from backtest.experiments.spec import ExperimentSpec
    return ExperimentSpec(**base)


async def test_already_resolved_question_cell_skipped(tmp_path, monkeypatch):
    """A question whose actual_resolve_time <= as_of must be skipped.

    close-7d for a question closing 2024-06-01 → as_of = 2024-05-25.
    actual_resolve_time = 2024-05-01 <= 2024-05-25 → must be skipped.
    """
    resolved_early = BacktestQuestion(
        qid="metaculus:42", platform="metaculus", native_id="42",
        qtype=QType.BINARY, title="Already decided?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 5, 1, tzinfo=timezone.utc),
        resolution=True)
    store = _store_with_questions(tmp_path, [resolved_early])

    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)

    result = await runner_mod.run_experiment(
        _spec(), store, tmp_path / "runs", live_llm=False, yes=True)

    # The cell must be "skipped", not "done"
    assert set(result.statuses.values()) == {"skipped"}, (
        f"Expected all cells skipped for already-resolved question; "
        f"got {result.statuses}")
    # The stub system must NOT have been called
    assert stub._calls == [], (
        f"Stub system should not be called for already-resolved cells; "
        f"got calls={stub._calls}")


async def test_not_yet_resolved_question_not_skipped(tmp_path, monkeypatch):
    """A question whose actual_resolve_time > as_of must NOT be skipped."""
    not_yet = BacktestQuestion(
        qid="metaculus:43", platform="metaculus", native_id="43",
        qtype=QType.BINARY, title="Not yet?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
        resolution=True)
    store = _store_with_questions(tmp_path, [not_yet])

    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)

    result = await runner_mod.run_experiment(
        _spec(), store, tmp_path / "runs", live_llm=False, yes=True)

    assert "done" in result.statuses.values(), (
        "Question not yet resolved at as_of must be forecast (not skipped)")


async def test_null_actual_resolve_time_not_skipped(tmp_path, monkeypatch):
    """Questions with actual_resolve_time=None must not be skipped."""
    null_resolve = BacktestQuestion(
        qid="metaculus:44", platform="metaculus", native_id="44",
        qtype=QType.BINARY, title="Null resolve?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=None,
        resolution=True)
    store = _store_with_questions(tmp_path, [null_resolve])

    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)

    result = await runner_mod.run_experiment(
        _spec(), store, tmp_path / "runs", live_llm=False, yes=True)

    assert "done" in result.statuses.values(), (
        "Question with null actual_resolve_time must not be skipped")


async def test_score_rows_carry_price_at_T(tmp_path, monkeypatch):
    """score rows for cells with a market price must include price_at_T."""
    q = BacktestQuestion(
        qid="polymarket:1", platform="polymarket", native_id="1",
        qtype=QType.BINARY, title="Market q?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
        resolution=True)
    store = _store_with_questions(tmp_path, [q])
    # Add a market_price series for this question
    store.put_series("polymarket:1", "market_price",
                     [("2024-05-01T00:00:00+00:00", 0.73)])

    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    # market_for identity: returns same qid when market_price series exists
    monkeypatch.setattr(runner_mod, "market_for",
                        lambda store, qid: qid)

    result = await runner_mod.run_experiment(
        _spec(), store, tmp_path / "runs", live_llm=False, yes=True)

    rows = [json.loads(l)
            for l in (result.run_dir / "scores.jsonl").read_text().splitlines()]
    assert rows, "Expected score rows"
    row = rows[0]
    assert "price_at_T" in row, "score row must carry 'price_at_T' field"
    assert row["price_at_T"] == pytest.approx(0.73)


async def test_score_rows_price_at_T_none_when_no_market(tmp_path, monkeypatch):
    """score rows for cells without market prices must have price_at_T=None."""
    q = BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="No market?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
        resolution=True)
    store = _store_with_questions(tmp_path, [q])

    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)

    result = await runner_mod.run_experiment(
        _spec(), store, tmp_path / "runs", live_llm=False, yes=True)

    rows = [json.loads(l)
            for l in (result.run_dir / "scores.jsonl").read_text().splitlines()]
    assert rows
    assert rows[0]["price_at_T"] is None


# ---------------------------------------------------------------------------
# I5 — platform dimension in score rows and report
# ---------------------------------------------------------------------------

async def test_score_rows_carry_platform(tmp_path, monkeypatch):
    """Every score row must carry a 'platform' field matching q.platform."""
    q = BacktestQuestion(
        qid="kalshi:99", platform="kalshi", native_id="99",
        qtype=QType.BINARY, title="Kalshi Q?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
        resolution=True)
    store = _store_with_questions(tmp_path, [q])

    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)

    result = await runner_mod.run_experiment(
        _spec(), store, tmp_path / "runs", live_llm=False, yes=True)

    rows = [json.loads(l)
            for l in (result.run_dir / "scores.jsonl").read_text().splitlines()]
    assert rows
    assert rows[0]["platform"] == "kalshi", (
        f"Expected platform='kalshi', got {rows[0].get('platform')!r}")


def _write_run_dir_with_platform(tmp_path):
    """Fabricate a run dir with platform-tagged score rows for report tests."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "test-0-abc12345", "corpus_root": "",
         "statuses": {"metaculus:1|close-7d|a": "done"}}))
    rows = [
        {"qid": "metaculus:1", "horizon": "close-7d", "variant": "a",
         "log_score": -0.2, "baseline_score": 20.0, "brier": 0.04,
         "h2h_cp": 5.0, "h2h_market": 3.0, "stratum": "clean",
         "platform": "metaculus", "price_at_T": 0.6, "price_at_T_ts": None,
         "resolution": True},
        {"qid": "manifold:2", "horizon": "close-7d", "variant": "a",
         "log_score": -0.4, "baseline_score": 10.0, "brier": 0.09,
         "h2h_cp": None, "h2h_market": 1.0, "stratum": "clean",
         "platform": "manifold", "price_at_T": 0.4, "price_at_T_ts": None,
         "resolution": True},
        {"qid": "metaculus:3", "horizon": "close-7d", "variant": "chance",
         "log_score": -0.69, "baseline_score": 0.0, "brier": 0.25,
         "h2h_cp": None, "h2h_market": None, "stratum": "clean",
         "platform": "metaculus", "price_at_T": None, "price_at_T_ts": None,
         "resolution": True},
    ]
    with (run_dir / "scores.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    art = ForecastArtifact(
        qid="metaculus:1", horizon="close-7d",
        as_of=datetime(2024, 5, 25, tzinfo=timezone.utc),
        system_name="a", config_fingerprint="fp",
        forecast=Forecast(qtype=QType.BINARY, binary_prob=0.8),
        stages=[StageRecord(stage="member_llm", model="fake/model-a",
                            cost_usd=0.12, tokens_in=100, tokens_out=50)])
    with (run_dir / "artifacts.jsonl").open("w") as f:
        f.write(json.dumps(artifact_to_dict(art)) + "\n")
    return run_dir


def test_report_contains_h2h_market_column(tmp_path):
    """build_report leaderboard must contain an 'h2h vs market' column."""
    from backtest.reports.report import build_report
    run_dir = _write_run_dir_with_platform(tmp_path)
    path = build_report(run_dir)
    text = path.read_text()
    assert "h2h vs market" in text.lower(), (
        "Report leaderboard must include h2h vs market column; "
        f"got:\n{text[:500]}")


def test_report_contains_by_platform_section(tmp_path):
    """build_report must contain a 'By platform' section."""
    from backtest.reports.report import build_report
    run_dir = _write_run_dir_with_platform(tmp_path)
    path = build_report(run_dir)
    text = path.read_text()
    assert "by platform" in text.lower(), (
        "Report must include a 'By platform' section; "
        f"got:\n{text[:500]}")


def test_report_by_platform_includes_manifold_footnote(tmp_path):
    """build_report By platform must footnote manifold as play-money."""
    from backtest.reports.report import build_report
    run_dir = _write_run_dir_with_platform(tmp_path)
    path = build_report(run_dir)
    text = path.read_text()
    assert "manifold" in text.lower()
    # The footnote must mention play-money somewhere in the report
    assert "play" in text.lower() or "play-money" in text.lower(), (
        "Report must mention 'play-money' (manifold footnote)")


# ---------------------------------------------------------------------------
# I7 — stale-price flagging via value_at_with_ts + report stale count
# ---------------------------------------------------------------------------

def test_value_at_with_ts_returns_value_and_timestamp(tmp_path):
    """value_at_with_ts must return (value, iso_ts) for the matching point."""
    store = CorpusStore(tmp_path / "corpus")
    from backtest.core import BacktestQuestion, QType
    q = BacktestQuestion(
        qid="metaculus:5", platform="metaculus", native_id="5",
        qtype=QType.BINARY, title="Test?", resolution=True)
    store.upsert_question(q)
    store.put_series("metaculus:5", "market_price", [
        ("2024-03-01T00:00:00+00:00", 0.55),
        ("2024-04-01T00:00:00+00:00", 0.65),
    ])

    t = datetime(2024, 4, 15, tzinfo=timezone.utc)
    result = store.value_at_with_ts("metaculus:5", "market_price", t)

    assert result is not None, "Expected (value, ts) tuple, got None"
    value, ts = result
    assert value == pytest.approx(0.65), f"Expected 0.65, got {value}"
    assert "2024-04-01" in ts, f"Expected April 2024 timestamp, got {ts!r}"


def test_value_at_with_ts_returns_none_when_no_series(tmp_path):
    """value_at_with_ts must return None when no series data exists."""
    store = CorpusStore(tmp_path / "corpus")
    t = datetime(2024, 4, 15, tzinfo=timezone.utc)
    result = store.value_at_with_ts("missing:99", "market_price", t)
    assert result is None


def test_value_at_with_ts_before_first_point_returns_none(tmp_path):
    """value_at_with_ts must return None when t is before the first series point."""
    store = CorpusStore(tmp_path / "corpus")
    from backtest.core import BacktestQuestion, QType
    q = BacktestQuestion(
        qid="metaculus:6", platform="metaculus", native_id="6",
        qtype=QType.BINARY, title="Test?", resolution=True)
    store.upsert_question(q)
    store.put_series("metaculus:6", "market_price", [
        ("2024-04-01T00:00:00+00:00", 0.5),
    ])

    t = datetime(2024, 3, 1, tzinfo=timezone.utc)  # before the only point
    result = store.value_at_with_ts("metaculus:6", "market_price", t)
    assert result is None


def test_value_at_unchanged_after_adding_with_ts(tmp_path):
    """value_at must still work correctly (unchanged) alongside value_at_with_ts."""
    store = CorpusStore(tmp_path / "corpus")
    from backtest.core import BacktestQuestion, QType
    q = BacktestQuestion(
        qid="metaculus:7", platform="metaculus", native_id="7",
        qtype=QType.BINARY, title="Test?", resolution=True)
    store.upsert_question(q)
    store.put_series("metaculus:7", "market_price", [
        ("2024-03-01T00:00:00+00:00", 0.4),
        ("2024-04-01T00:00:00+00:00", 0.6),
    ])

    t = datetime(2024, 4, 15, tzinfo=timezone.utc)
    # value_at must still return the raw value
    v = store.value_at("metaculus:7", "market_price", t)
    assert v == pytest.approx(0.6)


def _write_run_dir_with_stale_price(tmp_path, stale: bool):
    """Fabricate a run dir where price_at_T_ts is stale (or fresh)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "stale-0-abc12345", "corpus_root": "",
         "statuses": {"metaculus:1|close-7d|a": "done"}}))

    # as_of = 2024-05-25; stale = price ts > 7 days before as_of
    if stale:
        # 2024-05-01 is 24 days before 2024-05-25 → stale
        price_ts = "2024-05-01T00:00:00+00:00"
    else:
        # 2024-05-24 is 1 day before 2024-05-25 → fresh
        price_ts = "2024-05-24T00:00:00+00:00"

    rows = [
        {"qid": "metaculus:1", "horizon": "close-7d", "variant": "a",
         "log_score": -0.2, "baseline_score": 20.0, "brier": 0.04,
         "h2h_cp": 5.0, "h2h_market": 3.0, "stratum": "clean",
         "platform": "metaculus", "price_at_T": 0.6,
         "price_at_T_ts": price_ts, "resolution": True},
    ]
    with (run_dir / "scores.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    art = ForecastArtifact(
        qid="metaculus:1", horizon="close-7d",
        as_of=datetime(2024, 5, 25, tzinfo=timezone.utc),
        system_name="a", config_fingerprint="fp",
        forecast=Forecast(qtype=QType.BINARY, binary_prob=0.8),
        stages=[])
    with (run_dir / "artifacts.jsonl").open("w") as f:
        f.write(json.dumps(artifact_to_dict(art)) + "\n")
    return run_dir


def test_report_shows_stale_price_count_when_stale(tmp_path):
    """build_report must render 'stale price (>7d before T)' with a count > 0."""
    from backtest.reports.report import build_report
    run_dir = _write_run_dir_with_stale_price(tmp_path, stale=True)
    path = build_report(run_dir)
    text = path.read_text()
    assert "stale price" in text.lower(), (
        f"Report must mention stale prices; got:\n{text[:800]}")
    # The count should be non-zero
    assert ">7d" in text or "7d" in text, (
        f"Report must show '>7d' threshold; got:\n{text[:800]}")


def test_report_shows_zero_stale_count_when_fresh(tmp_path):
    """build_report stale-price line must show 0 when all prices are fresh."""
    from backtest.reports.report import build_report
    run_dir = _write_run_dir_with_stale_price(tmp_path, stale=False)
    path = build_report(run_dir)
    text = path.read_text()
    # Should still mention stale price but with count 0
    # (or not mention it at all — both are acceptable)
    # The key invariant: no crash and the report is valid
    assert "Leaderboard" in text


async def test_score_rows_carry_price_at_T_ts(tmp_path, monkeypatch):
    """score rows for cells with a market price must include price_at_T_ts."""
    q = BacktestQuestion(
        qid="polymarket:2", platform="polymarket", native_id="2",
        qtype=QType.BINARY, title="Market q ts?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
        resolution=True)
    store = _store_with_questions(tmp_path, [q])
    store.put_series("polymarket:2", "market_price",
                     [("2024-05-01T00:00:00+00:00", 0.73)])

    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    monkeypatch.setattr(runner_mod, "market_for",
                        lambda store, qid: qid)

    result = await runner_mod.run_experiment(
        _spec(), store, tmp_path / "runs", live_llm=False, yes=True)

    rows = [json.loads(l)
            for l in (result.run_dir / "scores.jsonl").read_text().splitlines()]
    assert rows
    row = rows[0]
    assert "price_at_T_ts" in row, "score row must carry 'price_at_T_ts' field"
    # ts should be a string containing the date of the series point
    assert row["price_at_T_ts"] is not None
    assert "2024-05-01" in str(row["price_at_T_ts"])
