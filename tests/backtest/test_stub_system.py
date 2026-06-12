"""StubSystem: deterministic, zero-network ForecastSystem for the CI smoke
gate. Never calls ctx.llm, never does research."""
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.systems.stub import StubSystem
from backtest.timetravel.asof import AsOfContext

AS_OF = datetime(2025, 2, 22, tzinfo=timezone.utc)


def _q(qtype, **kw):
    base = dict(qid="metaculus:1", platform="metaculus", native_id="1",
                qtype=qtype, title="t")
    base.update(kw)
    return BacktestQuestion(**base)


async def test_binary_returns_fixed_prob_clean_artifact():
    art = await StubSystem().forecast(_q(QType.BINARY),
                                      AsOfContext(as_of=AS_OF))
    assert art.system_name == "stub"
    assert art.qid == "metaculus:1"
    assert art.as_of == AS_OF
    assert art.forecast.binary_prob == 0.6
    assert art.contamination == "clean"
    assert art.error is None
    assert art.stages and art.stages[0].cost_usd == 0.0


async def test_mc_returns_uniform_over_options():
    q = _q(QType.MULTIPLE_CHOICE, options=("a", "b", "c", "d"))
    art = await StubSystem().forecast(q, AsOfContext(as_of=AS_OF))
    probs = art.forecast.mc_probs
    assert set(probs) == {"a", "b", "c", "d"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9


async def test_numeric_returns_valid_uniform_cdf():
    q = _q(QType.NUMERIC,
           scaling={"range_min": 0, "range_max": 100, "cdf_size": 201})
    art = await StubSystem().forecast(q, AsOfContext(as_of=AS_OF))
    cdf = art.forecast.cdf
    assert len(cdf) == 201
    assert cdf[0] == 0.0 and cdf[-1] == 1.0
    assert all(b > a for a, b in zip(cdf, cdf[1:]))      # strictly ascending
    assert all(b - a <= 0.2 for a, b in zip(cdf, cdf[1:]))  # per-bin mass cap


async def test_does_not_touch_llm_or_store():
    # ctx with NO llm and NO store: the stub must not need either.
    art = await StubSystem().forecast(_q(QType.BINARY),
                                      AsOfContext(as_of=AS_OF))
    assert art.forecast is not None


def test_fingerprint_is_deterministic_12_hex():
    fp = StubSystem().config_fingerprint()
    assert fp == StubSystem().config_fingerprint()
    assert len(fp) == 12
    int(fp, 16)   # raises if not hex
