from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.crosswalk import build_crosswalk, market_for
from backtest.corpus.store import CorpusStore


def _q(qid, platform, title, criteria, close=None):
    return BacktestQuestion(
        qid=qid, platform=platform, native_id=qid.split(":", 1)[1],
        qtype=QType.BINARY, title=title, resolution_criteria=criteria,
        scheduled_close_time=close, resolution=True)


def _seed(store):
    store.upsert_question(_q(
        "metaculus:1", "metaculus",
        "Will the Fed cut rates at the June 2026 meeting?",
        "Resolves YES if the FOMC lowers the target range in June 2026.",
        close=datetime(2026, 6, 18, tzinfo=timezone.utc)))
    store.upsert_question(_q(
        "polymarket:9", "polymarket",
        "Fed rate cut at June 2026 meeting?",
        "Resolves Yes if the FOMC cuts rates at its June 2026 meeting.",
        close=datetime(2026, 6, 18, tzinfo=timezone.utc)))
    store.upsert_question(_q(
        "kalshi:Z", "kalshi",
        "Will it rain in Seattle tomorrow?", "NWS gauge reading."))


async def test_build_crosswalk_lexical_then_llm_gate(tmp_path):
    store = CorpusStore(tmp_path)
    _seed(store)
    prompts = []

    async def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "YES" if "Fed rate cut" in prompt else "NO"

    n = await build_crosswalk(store, llm, top_k=2)
    assert n == 1
    assert market_for(store, "metaculus:1") == "polymarket:9"
    assert market_for(store, "kalshi:Z") is None   # market side: no row
    # the gate saw resolution criteria AND dates of BOTH questions
    assert "FOMC lowers the target range" in prompts[0]
    assert "FOMC cuts rates" in prompts[0]
    assert "2026-06-18" in prompts[0]
    # lexical ranking put the near-duplicate first: one call was enough
    assert len(prompts) == 1


async def test_build_crosswalk_caches_verdicts(tmp_path):
    store = CorpusStore(tmp_path)
    _seed(store)

    async def llm(prompt: str) -> str:
        return "YES" if "Fed rate cut" in prompt else "NO"

    await build_crosswalk(store, llm, top_k=2)

    async def exploding_llm(prompt: str) -> str:
        raise AssertionError("LLM must not be called on cached pairs")

    n2 = await build_crosswalk(store, exploding_llm, top_k=2)
    assert n2 == 0
    assert market_for(store, "metaculus:1") == "polymarket:9"


async def test_rejected_candidates_are_cached_too(tmp_path):
    store = CorpusStore(tmp_path)
    _seed(store)

    async def llm_no(prompt: str) -> str:
        return "NO — dates differ."

    assert await build_crosswalk(store, llm_no, top_k=2) == 0
    assert market_for(store, "metaculus:1") is None
    calls = []

    async def llm_count(prompt: str) -> str:
        calls.append(prompt)
        return "NO"

    await build_crosswalk(store, llm_count, top_k=2)
    assert calls == []                  # NO verdicts cached as well


def test_manual_override_wins(tmp_path):
    store = CorpusStore(tmp_path)
    store.conn.execute(
        "INSERT INTO crosswalk"
        " (qid, market_qid, score, verified, manual_override)"
        " VALUES (?,?,?,?,?)",
        ("metaculus:1", "polymarket:9", 0.9, 1, "kalshi:OVERRIDE"))
    store.conn.commit()
    assert market_for(store, "metaculus:1") == "kalshi:OVERRIDE"
