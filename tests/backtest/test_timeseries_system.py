"""Task 5: TimeSeriesSystem — TS submodels as ForecastSystems over target
series. Never raises; clean by construction (no LLM sees the question)."""
import asyncio
from datetime import datetime, timedelta, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore
from backtest.systems.protocol import ForecastSystem
from backtest.systems.timeseries import TimeSeriesSystem
from backtest.timetravel.asof import AsOfContext
from bot.distributions import validate_cdf

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
AS_OF = T0 + timedelta(days=99)


def _store(tmp_path, values=None, qid="t:num"):
    store = CorpusStore(tmp_path / "corpus")
    if values is not None:
        store.put_series(qid, "target", [
            ((T0 + timedelta(days=i)).isoformat(), float(v))
            for i, v in enumerate(values)])
    return store


def _question(qtype=QType.NUMERIC, scaling=None, qid="t:num"):
    return BacktestQuestion(
        qid=qid, platform="t", native_id="1", qtype=qtype, title="V at T?",
        open_time=T0, scheduled_close_time=T0 + timedelta(days=130),
        scaling=scaling if scaling is not None else {
            "range_min": 0.0, "range_max": 300.0,
            "open_lower_bound": False, "open_upper_bound": True,
            "cdf_size": 201})


def _forecast(system, q, store):
    ctx = AsOfContext(as_of=AS_OF, store=store)
    return asyncio.run(system.forecast(q, ctx))


def test_protocol_and_valid_cdf(tmp_path):
    store = _store(tmp_path, values=range(120))
    system = TimeSeriesSystem(engine="naive_drift")
    assert isinstance(system, ForecastSystem)
    art = _forecast(system, _question(), store)
    assert art.error is None
    assert art.contamination == "clean"
    validate_cdf(art.forecast.cdf, open_lower=False, open_upper=True)
    sub = [s for s in art.stages if s.stage == "submodel"]
    assert len(sub) == 1
    assert sub[0].outputs["quantiles"]["50.0"] > 0
    assert sub[0].inputs["engine"] == "naive_drift"


def test_post_asof_points_unused(tmp_path):
    base = list(range(120))
    s1 = _store(tmp_path / "a", values=base)
    s2 = _store(tmp_path / "b", values=base + [10_000.0] * 30)  # post-T spike
    system = TimeSeriesSystem(engine="naive_drift")
    a1 = _forecast(system, _question(), s1)
    a2 = _forecast(system, _question(), s2)
    assert a1.forecast.cdf == a2.forecast.cdf


def test_error_artifacts_never_raise(tmp_path):
    system = TimeSeriesSystem(engine="naive_drift")
    store = _store(tmp_path, values=range(120))
    cases = [
        (_question(qtype=QType.BINARY), "qtype"),
        (_question(qid="t:other"), "no target series"),
        (_question(scaling={"range_min": 0, "range_max": 10,
                            "cdf_size": 101}), "cdf_size"),
        (_question(scaling={"range_min": 0, "range_max": 10,
                            "zero_point": 1.0, "cdf_size": 201}),
         "zero_point"),
    ]
    for q, needle in cases:
        art = _forecast(system, q, store)
        assert art.forecast is None and needle in art.error, (q.qid, art.error)


def test_too_few_points_is_error_artifact(tmp_path):
    store = _store(tmp_path, values=range(3))
    art = _forecast(TimeSeriesSystem(engine="naive_drift"),
                    _question(), store)
    assert art.forecast is None and "at least" in art.error


def test_fingerprint_varies_with_engine_and_params(tmp_path):
    a = TimeSeriesSystem(engine="naive_drift")
    b = TimeSeriesSystem(engine="naive_drift", engine_params={"tail_widen": 2.0})
    c = TimeSeriesSystem(engine="theta")
    assert len({s.config_fingerprint() for s in (a, b, c)}) == 3
