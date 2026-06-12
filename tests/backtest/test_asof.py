from datetime import datetime, timedelta, timezone

import pytest

from backtest.core import BacktestQuestion, LedgerEntry, QType
from backtest.timetravel.asof import (ACTIVE_CONTEXT, AsOfContext,
                                      horizon_to_asof)

OPEN = datetime(2024, 1, 1, tzinfo=timezone.utc)
CLOSE = datetime(2024, 4, 11, tzinfo=timezone.utc)   # span exactly 101 days


def _q(**kw):
    base = dict(qid="metaculus:1", platform="metaculus", native_id="1",
                qtype=QType.BINARY, title="Q?")
    base.update(kw)
    return BacktestQuestion(**base)


def test_close_horizons_subtract_from_scheduled_close():
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE)
    assert horizon_to_asof(q, "close-7d") == CLOSE - timedelta(days=7)
    assert horizon_to_asof(q, "close-30d") == CLOSE - timedelta(days=30)


def test_close_horizon_landing_before_open_is_skipped():
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE)
    assert horizon_to_asof(q, "close-365d") is None


def test_close_horizon_without_close_time_is_skipped():
    assert horizon_to_asof(_q(open_time=OPEN), "close-7d") is None


def test_spot_uses_spot_scoring_time():
    spot = datetime(2024, 3, 1, tzinfo=timezone.utc)
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE, spot_scoring_time=spot)
    assert horizon_to_asof(q, "spot") == spot
    assert horizon_to_asof(_q(open_time=OPEN), "spot") is None


def test_geo_k_matches_halawi_on_known_interval():
    # span 101 days -> offsets (101-1)^(k/5) = 100^(k/5) days from open
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE)
    t5 = horizon_to_asof(q, "geo-5")
    assert t5 == OPEN + timedelta(days=100.0)          # == close - 1 day
    assert t5 == CLOSE - timedelta(days=1)
    t1 = horizon_to_asof(q, "geo-1")
    expect = OPEN + timedelta(days=100.0 ** 0.2)       # ~2.5119 days
    assert abs((t1 - expect).total_seconds()) < 1.0
    ts = [horizon_to_asof(q, f"geo-{k}") for k in range(1, 6)]
    assert ts == sorted(ts)                            # strictly increasing
    assert all(OPEN < t < CLOSE for t in ts)           # inside [open, close)


def test_geo_k_out_of_range_or_unknown_horizon_raises():
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE)
    with pytest.raises(ValueError):
        horizon_to_asof(q, "geo-6")
    with pytest.raises(ValueError):
        horizon_to_asof(q, "geo-x")
    with pytest.raises(ValueError):
        horizon_to_asof(q, "close-tomorrow")


def test_geo_k_degenerate_interval_is_skipped():
    q = _q(open_time=OPEN, scheduled_close_time=OPEN + timedelta(hours=12))
    assert horizon_to_asof(q, "geo-3") is None


def test_context_defaults_record_and_contextvar():
    assert ACTIVE_CONTEXT.get() is None
    ctx = AsOfContext(as_of=OPEN)
    assert ctx.cutoff_mode == "strict"
    assert ctx.ledger == [] and ctx.llm is None and ctx.store is None
    ctx.record(LedgerEntry(provider="p", query="q"))
    assert len(ctx.ledger) == 1 and ctx.ledger[0].provider == "p"
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        assert ACTIVE_CONTEXT.get() is ctx
    finally:
        ACTIVE_CONTEXT.reset(token)
    assert ACTIVE_CONTEXT.get() is None
