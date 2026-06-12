"""Pre-spend review batch 2 — seeded feasible sampling, market_for identity
fallback, research sees descriptions.

B2 — _select_questions: seeded shuffle draws uniformly from all platforms,
     horizon-infeasible questions are filtered before the cap is applied, and
     the same seed always produces the same selection.

B3 — market_for: identity fallback returns qid when it is a market-platform
     question with a market_price series; explicit crosswalk rows still win.

I6 — _question_dict: resolution_criteria falls back to description for
     questions that carry substance only in description (polymarket style).
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from backtest.core import BacktestQuestion, QType
from backtest.corpus.crosswalk import market_for
from backtest.corpus.store import CorpusStore
from backtest.experiments.profiles import PROFILES, Profile
from backtest.experiments.runner import _select_questions
from backtest.experiments.spec import ExperimentSpec
from backtest.systems.ensemble_adapter import _question_dict


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _metaculus_q(i: int, *, feasible: bool = True) -> BacktestQuestion:
    """A metaculus question. feasible=True → horizon close-7d is valid."""
    if feasible:
        # open_time leaves room for close-7d (open + 30d, close open+60d)
        open_t = datetime(2024, 1, 1, tzinfo=timezone.utc)
        close_t = datetime(2024, 3, 1, tzinfo=timezone.utc)
    else:
        # open_time == scheduled_close_time → close-7d lands before open → skip
        open_t = datetime(2024, 3, 1, tzinfo=timezone.utc)
        close_t = datetime(2024, 3, 1, tzinfo=timezone.utc)
    return BacktestQuestion(
        qid=f"metaculus:{i}", platform="metaculus", native_id=str(i),
        qtype=QType.BINARY, title=f"Metaculus Q{i}",
        open_time=open_t, scheduled_close_time=close_t,
        resolution=True)


def _market_q(i: int, platform: str = "polymarket") -> BacktestQuestion:
    """A market-platform question (sorts AFTER metaculus alphabetically)."""
    open_t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    close_t = datetime(2024, 3, 1, tzinfo=timezone.utc)
    return BacktestQuestion(
        qid=f"{platform}:{i:04d}", platform=platform, native_id=str(i),
        qtype=QType.BINARY, title=f"{platform.capitalize()} Q{i}",
        open_time=open_t, scheduled_close_time=close_t,
        resolution=True)


def _spec(seed: int = 0, limit: int | None = None,
          horizons: list | None = None) -> ExperimentSpec:
    corpus: dict = {}
    if limit is not None:
        corpus["limit"] = limit
    return ExperimentSpec(
        name="t", corpus=corpus,
        horizons=horizons or ["close-7d"],
        cutoff_mode="strict",
        profile="smoke",
        variants=[], baselines=[],
        seed=seed)


# Profile with a high max_questions so it doesn't constrain our small fixtures.
_UNLIMITED_PROFILE = Profile(
    max_questions=1000,
    cost_cap_usd=500.0,
    est_cost_per_question_usd=0.02)


# ---------------------------------------------------------------------------
# B2 — _select_questions
# ---------------------------------------------------------------------------

def test_seeded_selection_contains_market_platform_questions(tmp_path):
    """Corpus where polymarket qids sort last alphabetically (p > m) but the
    cap is smaller than total: the seeded shuffle must draw across all platforms
    so at least one polymarket question appears in the selection.

    We use 10 metaculus + 10 polymarket questions with a cap of 10.  With a
    pure alphabetical LIMIT-10 SQL query every selected question would be
    metaculus — the shuffle fixes that.  We verify with a seed that provably
    mixes both platforms (empirically seed=0 always does; the assertion is
    deterministic for the same seed).
    """
    store = CorpusStore(tmp_path / "store")
    for i in range(10):
        store.upsert_question(_metaculus_q(i))
    for i in range(10):
        store.upsert_question(_market_q(i))

    spec = _spec(seed=0, limit=10)
    qs = _select_questions(spec, store, _UNLIMITED_PROFILE)

    assert len(qs) == 10
    platforms = {q.platform for q in qs}
    assert "polymarket" in platforms, (
        "seeded shuffle should draw across platforms; polymarket absent")


def test_only_horizon_feasible_questions_selected(tmp_path):
    """Questions where all spec horizons map to None must be excluded before
    the cap is applied — they would otherwise waste a cap slot."""
    store = CorpusStore(tmp_path / "store")
    # 3 feasible, 2 infeasible (open_time == close_time → close-7d → None)
    for i in range(3):
        store.upsert_question(_metaculus_q(i, feasible=True))
    for i in range(3, 5):
        store.upsert_question(_metaculus_q(i, feasible=False))

    spec = _spec(seed=0, limit=10, horizons=["close-7d"])
    qs = _select_questions(spec, store, _UNLIMITED_PROFILE)

    assert len(qs) == 3, (
        "only the 3 horizon-feasible questions should be selected")
    for q in qs:
        # confirm they are the feasible ones
        assert q.qid in {"metaculus:0", "metaculus:1", "metaculus:2"}


def test_same_seed_same_selection(tmp_path):
    """Two calls with the same seed on the same corpus → identical qid lists."""
    store = CorpusStore(tmp_path / "store")
    for i in range(20):
        store.upsert_question(_metaculus_q(i))
    for i in range(20):
        store.upsert_question(_market_q(i))

    spec = _spec(seed=42, limit=15)
    qs1 = _select_questions(spec, store, _UNLIMITED_PROFILE)
    qs2 = _select_questions(spec, store, _UNLIMITED_PROFILE)

    assert [q.qid for q in qs1] == [q.qid for q in qs2], (
        "same seed must produce identical selection (resume safety)")


def test_different_seeds_different_order(tmp_path):
    """Two different seeds on the same corpus → different qid orders / sets.

    With 20 questions and a cap of 10, the probability that two independent
    uniform-random shuffles produce the same top-10 is astronomically small,
    so this is deterministically reliable.
    """
    store = CorpusStore(tmp_path / "store")
    for i in range(20):
        store.upsert_question(_metaculus_q(i))

    qs_a = _select_questions(spec := _spec(seed=0, limit=10), store,
                             _UNLIMITED_PROFILE)
    qs_b = _select_questions(_spec(seed=99, limit=10), store, _UNLIMITED_PROFILE)

    # Either the qids themselves or their order must differ.
    assert [q.qid for q in qs_a] != [q.qid for q in qs_b], (
        "different seeds must produce different selection order/composition")


def test_result_sorted_by_qid_for_stable_iteration(tmp_path):
    """The returned list must be sorted by qid regardless of shuffle order."""
    store = CorpusStore(tmp_path / "store")
    for i in range(10):
        store.upsert_question(_metaculus_q(i))

    spec = _spec(seed=7)
    qs = _select_questions(spec, store, _UNLIMITED_PROFILE)
    qids = [q.qid for q in qs]
    assert qids == sorted(qids), "returned list must be sorted by qid"


# ---------------------------------------------------------------------------
# B3 — market_for identity fallback
# ---------------------------------------------------------------------------

def test_market_for_identity_fallback_with_series(tmp_path):
    """A polymarket question with a market_price series → market_for returns
    the qid itself (identity fallback)."""
    store = CorpusStore(tmp_path / "store")
    q = _market_q(1, "polymarket")
    store.upsert_question(q)
    store.put_series(q.qid, "market_price",
                     [("2024-02-01T00:00:00+00:00", 0.65)])

    result = market_for(store, q.qid)
    assert result == q.qid, (
        f"Expected identity fallback {q.qid!r}, got {result!r}")


def test_market_for_no_series_returns_none(tmp_path):
    """A polymarket question WITHOUT a market_price series → None (no fallback
    without data)."""
    store = CorpusStore(tmp_path / "store")
    q = _market_q(2, "polymarket")
    store.upsert_question(q)
    # No series stored.

    result = market_for(store, q.qid)
    assert result is None, (
        "market_for must return None when no market_price series exists")


def test_market_for_crosswalk_row_wins_over_identity(tmp_path):
    """An explicit verified crosswalk row for a market-platform qid takes
    precedence over the identity fallback."""
    store = CorpusStore(tmp_path / "store")
    q = _market_q(3, "kalshi")
    store.upsert_question(q)
    store.put_series(q.qid, "market_price",
                     [("2024-02-01T00:00:00+00:00", 0.5)])

    # Insert a crosswalk row pointing to a different market qid.
    other_qid = "polymarket:9999"
    store.conn.execute(
        "INSERT INTO crosswalk (qid, market_qid, score, verified, manual_override)"
        " VALUES (?,?,?,?,NULL)",
        (q.qid, other_qid, 0.95, 1))
    store.conn.commit()

    result = market_for(store, q.qid)
    assert result == other_qid, (
        "verified crosswalk row must win over identity fallback")


def test_market_for_manual_override_wins_over_identity(tmp_path):
    """manual_override on a market-platform qid still beats identity."""
    store = CorpusStore(tmp_path / "store")
    q = _market_q(4, "manifold")
    store.upsert_question(q)
    store.put_series(q.qid, "market_price",
                     [("2024-02-01T00:00:00+00:00", 0.4)])

    override_qid = "kalshi:OVERRIDE"
    store.conn.execute(
        "INSERT INTO crosswalk (qid, market_qid, score, verified, manual_override)"
        " VALUES (?,?,?,?,?)",
        (q.qid, "polymarket:0", 0.5, 0, override_qid))
    store.conn.commit()

    result = market_for(store, q.qid)
    assert result == override_qid, (
        "manual_override must win over identity fallback")


# ---------------------------------------------------------------------------
# I6 — _question_dict description fallback
# ---------------------------------------------------------------------------

def _polymarket_question(*, resolution_criteria: str = "",
                          description: str = "") -> BacktestQuestion:
    return BacktestQuestion(
        qid="polymarket:9001", platform="polymarket", native_id="9001",
        qtype=QType.BINARY,
        title="Will the thing happen?",
        description=description,
        resolution_criteria=resolution_criteria,
        fine_print="",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        resolution=True)


def test_question_dict_resolution_criteria_fallback_to_description():
    """When resolution_criteria is empty but description is set, _question_dict
    must expose description as resolution_criteria so the research stage sees
    the question substance."""
    q = _polymarket_question(
        resolution_criteria="",
        description="This resolves YES if the event happens per official data.")

    d = _question_dict(q)

    assert d["resolution_criteria"] == q.description, (
        "resolution_criteria must fall back to description when empty")


def test_question_dict_resolution_criteria_used_when_set():
    """When resolution_criteria is non-empty it must be used as-is."""
    q = _polymarket_question(
        resolution_criteria="Resolves YES if X.",
        description="Background context.")

    d = _question_dict(q)

    assert d["resolution_criteria"] == "Resolves YES if X.", (
        "non-empty resolution_criteria must not be replaced by description")


def test_question_dict_includes_description_field():
    """description must always be present in _question_dict for completeness."""
    q = _polymarket_question(
        resolution_criteria="Resolves YES.",
        description="Extra background.")

    d = _question_dict(q)

    assert "description" in d
    assert d["description"] == "Extra background."
