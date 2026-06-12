"""Task 7: `timeseries` research provider — quantitative anchor section for
numeric questions with a target series."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from backtest.timetravel.providers import _timeseries_research
from bot.config import BotConfig

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
AS_OF = T0 + timedelta(days=99)


def _setup(tmp_path, qtype=QType.NUMERIC, with_series=True):
    store = CorpusStore(tmp_path / "corpus")
    q = BacktestQuestion(
        qid="t:num", platform="t", native_id="1", qtype=qtype, title="V?",
        open_time=T0, scheduled_close_time=T0 + timedelta(days=130),
        scaling={"range_min": 0.0, "range_max": 300.0, "unit": "widgets",
                 "cdf_size": 201})
    store.upsert_question(q)
    if with_series:
        store.put_series("t:num", "target", [
            ((T0 + timedelta(days=i)).isoformat(), float(i))
            for i in range(120)])
    ctx = AsOfContext(as_of=AS_OF, store=store)
    return ctx, {"qid": "t:num", "question_text": "V?"}


def _run(ctx, question):
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        return asyncio.run(_timeseries_research(question, BotConfig()))
    finally:
        ACTIVE_CONTEXT.reset(token)


def test_emits_anchor_with_interval_and_ledger(tmp_path):
    ctx, question = _setup(tmp_path)
    text = _run(ctx, question)
    assert "median" in text and "80%" in text and "2026-04-10" in text
    assert "widgets" in text
    assert any(e.provider == "timeseries" and e.date_trust == "snapshot"
               for e in ctx.ledger)


def test_skips_binary_and_missing_series(tmp_path):
    ctx, question = _setup(tmp_path, qtype=QType.BINARY)
    assert _run(ctx, question) == ""
    ctx2, question2 = _setup(tmp_path / "b", with_series=False)
    assert _run(ctx2, question2) == ""
    assert not ctx2.ledger


def test_requires_context():
    with pytest.raises(RuntimeError, match="AsOfContext"):
        asyncio.run(_timeseries_research({"qid": "x"}, BotConfig()))
