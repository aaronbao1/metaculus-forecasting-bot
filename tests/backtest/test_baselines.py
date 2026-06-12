"""Baseline systems — real CorpusStore in tmp_path, no network."""
from datetime import datetime, timezone

import pytest

import backtest.systems.baselines as baselines_mod
from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore
from backtest.systems.baselines import (ChanceBaseline, CPEchoBaseline,
                                        MarketEchoBaseline)
from backtest.systems.protocol import ForecastSystem
from backtest.timetravel.asof import AsOfContext

AS_OF = datetime(2024, 5, 25, tzinfo=timezone.utc)


def _q(qtype=QType.BINARY, **kw):
    defaults = dict(qid="metaculus:1", platform="metaculus", native_id="1",
                    qtype=qtype, title="Q?",
                    open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    scheduled_close_time=datetime(2024, 6, 1,
                                                  tzinfo=timezone.utc))
    defaults.update(kw)
    return BacktestQuestion(**defaults)


@pytest.fixture
def store(tmp_path):
    return CorpusStore(tmp_path / "corpus")


def test_baselines_satisfy_protocol(store):
    assert isinstance(ChanceBaseline(), ForecastSystem)
    assert isinstance(CPEchoBaseline(store), ForecastSystem)
    assert isinstance(MarketEchoBaseline(store), ForecastSystem)


async def test_chance_binary_mc_numeric():
    ctx = AsOfContext(as_of=AS_OF)
    b = ChanceBaseline()
    binary = await b.forecast(_q(), ctx)
    assert binary.forecast.binary_prob == 0.5
    assert binary.contamination == "clean"
    assert binary.system_name == "chance"

    mc = await b.forecast(_q(qtype=QType.MULTIPLE_CHOICE,
                             options=("A", "B", "C", "D")), ctx)
    assert mc.forecast.mc_probs == {o: 0.25 for o in ("A", "B", "C", "D")}

    num = await b.forecast(
        _q(qtype=QType.NUMERIC,
           scaling={"range_min": 0.0, "range_max": 10.0,
                    "open_lower_bound": False, "open_upper_bound": True,
                    "cdf_size": 201}), ctx)
    cdf = num.forecast.cdf
    assert len(cdf) == 201
    assert cdf[0] == pytest.approx(0.0)
    assert cdf[-1] == pytest.approx(0.999)
    assert all(b2 - a2 >= 5e-05 - 1e-9 for a2, b2 in zip(cdf, cdf[1:]))


async def test_cp_echo_reads_store_at_asof(store):
    store.upsert_question(_q(resolution=True))
    store.put_series("metaculus:1", "cp",
                     [("2024-05-01T00:00:00+00:00", 0.66),
                      ("2024-05-20T00:00:00+00:00", 0.72)])
    art = await CPEchoBaseline(store).forecast(
        _q(), AsOfContext(as_of=AS_OF))
    assert art.error is None
    assert art.forecast.binary_prob == pytest.approx(0.72)


async def test_cp_echo_skips_without_data(store):
    store.upsert_question(_q(resolution=True))
    store.put_series("metaculus:1", "cp",
                     [("2024-05-01T00:00:00+00:00", 0.66)])
    # as_of before the first CP point -> no data -> skip, never raise
    art = await CPEchoBaseline(store).forecast(
        _q(), AsOfContext(as_of=datetime(2024, 4, 1, tzinfo=timezone.utc)))
    assert art.forecast is None
    assert "no data" in art.error


async def test_market_echo_uses_crosswalk(store, monkeypatch):
    store.upsert_question(_q(resolution=True))
    store.upsert_question(_q(qid="polymarket:xyz", platform="polymarket",
                             native_id="xyz"))
    store.put_series("polymarket:xyz", "market_price",
                     [("2024-05-01T00:00:00+00:00", 0.41)])
    monkeypatch.setattr(baselines_mod, "market_for",
                        lambda s, qid: "polymarket:xyz")
    art = await MarketEchoBaseline(store).forecast(
        _q(), AsOfContext(as_of=AS_OF))
    assert art.error is None
    assert art.forecast.binary_prob == pytest.approx(0.41)


async def test_market_echo_skips_without_crosswalk(store, monkeypatch):
    store.upsert_question(_q(resolution=True))
    monkeypatch.setattr(baselines_mod, "market_for", lambda s, qid: None)
    art = await MarketEchoBaseline(store).forecast(
        _q(), AsOfContext(as_of=AS_OF))
    assert art.forecast is None
    assert "no data" in art.error
