"""Pre-spend review batch 1 — tz normalise, dotenv, cost capture, no-research
spend guard.

B1 — covered in test_corpus_store.py (store round-trip) and
     test_asof_threading.py (horizon_to_asof).  This file covers the
     remaining three items.

B4 — scripts/backtest.py must call load_dotenv at module top.
I1 — CachedLLM._call_live must capture real cost via MonetaryCostManager.
I2 — ensemble_adapter.forecast() raises RuntimeError (→ error artifact) when
     run_research returns provider="none", BEFORE any LLM call.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# B4 — scripts/backtest.py has load_dotenv wired to the repo root
# ---------------------------------------------------------------------------

def test_backtest_script_has_load_dotenv_call():
    """Assert that scripts/backtest.py contains the load_dotenv call wired to
    the repo root — mirrors the pattern in scripts/run_tournament.py."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "scripts" / "backtest.py"
           ).read_text()
    # Must import load_dotenv
    assert "from dotenv import load_dotenv" in src, (
        "scripts/backtest.py must import load_dotenv from dotenv")
    # Must call load_dotenv with an argument referencing the .env file.
    # We accept any of the natural forms:
    #   load_dotenv(".env")
    #   load_dotenv(Path(...) / ".env")
    #   load_dotenv(_Path(...) / ".env")
    # The argument may span multiple tokens, so search the whole source for
    # the combination of the call + the .env reference near it.
    assert "load_dotenv(" in src, (
        "scripts/backtest.py must call load_dotenv()")
    # ".env" must appear somewhere in the file as a string argument to load_dotenv
    # (covers Path-joined forms like '/ ".env"' as well as bare '".env"').
    after_call = src[src.index("load_dotenv("):]
    assert '".env"' in after_call or "'.env'" in after_call, (
        "load_dotenv() call must reference the .env path")


def test_backtest_script_importable(monkeypatch):
    """Importing scripts.backtest (with load_dotenv monkeypatched) must not
    raise — it must survive a cold import with no .env present."""
    import importlib
    import sys

    # Stub load_dotenv so we don't need a real .env
    dotenv_stub = type(sys)("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)

    # Force re-import of the module
    sys.modules.pop("scripts.backtest", None)
    sys.modules.pop("scripts", None)
    # Use importlib to load the file directly
    from pathlib import Path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scripts.backtest",
        Path(__file__).resolve().parents[2] / "scripts" / "backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # must not raise


# ---------------------------------------------------------------------------
# I1 — CachedLLM._call_live captures real cost via MonetaryCostManager
# ---------------------------------------------------------------------------

async def test_live_call_captures_cost_via_monetary_cost_manager(
        tmp_path, monkeypatch):
    """When GeneralLlm.invoke returns a plain str (as it does in practice)
    the cost must still be captured non-zero via MonetaryCostManager.current_usage.
    The cache record must carry that cost_usd, not the old 0.0 dead-path value.
    """
    from backtest.timetravel.llm_cache import CachedLLM, cache_key

    FAKE_COST = 0.00314

    # Stub GeneralLlm to return a plain str (real behaviour post-fix)
    class FakeLlm:
        def __init__(self, model, temperature, timeout=None):
            pass
        async def invoke(self, prompt):
            return "plain string completion"

    monkeypatch.setattr("forecasting_tools.GeneralLlm", FakeLlm)

    # Stub MonetaryCostManager so that current_usage returns FAKE_COST after
    # the invoke, simulating what litellm's callback would record.
    # _call_live does `from forecasting_tools import ..., MonetaryCostManager`
    # at call time, so patching forecasting_tools.MonetaryCostManager is
    # sufficient (the module-level import will re-resolve at next call).
    class FakeMonetaryCostManager:
        def __init__(self, *a, **kw):
            self._usage = 0.0
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        @property
        def current_usage(self):
            return FAKE_COST

    monkeypatch.setattr("forecasting_tools.MonetaryCostManager",
                        FakeMonetaryCostManager)

    llm = CachedLLM(tmp_path / "cache")
    result = await llm.invoke("model/x", "hello", 0.0)
    assert result == "plain string completion"

    key = cache_key("model/x", "hello", 0.0, 0)
    rec = json.loads((tmp_path / "cache" / f"{key}.json").read_text())
    assert rec["cost_usd"] == pytest.approx(FAKE_COST), (
        f"Expected cost_usd={FAKE_COST} from MonetaryCostManager, "
        f"got {rec['cost_usd']}")


# ---------------------------------------------------------------------------
# I2 — ensemble_adapter refuses LLM spend when research provider == "none"
# ---------------------------------------------------------------------------

async def test_no_research_returns_error_artifact_no_llm_calls(
        tmp_path, monkeypatch):
    """When run_research returns provider='none' the adapter must return an
    error artifact mentioning research (and the error must name the refusal
    reason) WITHOUT making any LLM calls."""
    from datetime import datetime, timezone

    import bot.research as research_mod
    from backtest.core import BacktestQuestion, QType
    from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
    from backtest.timetravel.asof import AsOfContext
    from bot.config import BotConfig, EnsembleMember
    from bot.research import ResearchResult

    # Stub run_research → provider="none"
    async def fake_no_research(question, cfg_):
        return ResearchResult(text="", provider="none", providers=())

    monkeypatch.setattr(research_mod, "run_research", fake_no_research)

    # Counting stub LLM — must never be called
    class CountingLLM:
        call_count = 0
        async def invoke(self, model, prompt, temperature, run_index=0):
            CountingLLM.call_count += 1
            return "Probability: 50%"

    CountingLLM.call_count = 0

    cfg = BotConfig(
        ensemble=[EnsembleMember("fake/model", runs=1, temperature=0.3)],
        min_members=1,
        calibration_dir=str(tmp_path / "calib"),
        log_path=str(tmp_path / "forecasts.jsonl"),
    )
    q = BacktestQuestion(
        qid="test:42", platform="test", native_id="42",
        qtype=QType.BINARY, title="Will it rain?",
        resolution_criteria="Resolves YES if rain.",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )
    as_of = datetime(2024, 3, 1, tzinfo=timezone.utc)
    ctx = AsOfContext(as_of=as_of, llm=CountingLLM())
    adapter = CalibratedEnsembleAdapter(cfg)

    art = await adapter.forecast(q, ctx)

    # Must produce an error artifact, not raise
    assert art.forecast is None
    assert art.error is not None, "error must be set when research provider='none'"
    err_lower = art.error.lower()
    assert "research" in err_lower, (
        f"error must mention 'research'; got: {art.error!r}")

    # LLM must NOT have been called at all
    assert CountingLLM.call_count == 0, (
        f"LLM was called {CountingLLM.call_count} times; "
        "must be 0 when research provider='none'")
