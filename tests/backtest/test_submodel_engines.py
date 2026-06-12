"""Task 4: pluggable time-series engines. naive_drift is the always-available
zero-dep floor; statsforecast/chronos/timesfm engines share the same contract
and are exercised only when installed."""
from datetime import datetime, timedelta, timezone

import pytest

from backtest.submodels.engines import ENGINES, get_engine

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _series(values, step_days=1.0):
    return [(T0 + timedelta(days=i * step_days), float(v))
            for i, v in enumerate(values)]


def _assert_contract(fc, expect_engine):
    pcts = sorted(fc.quantiles)
    assert pcts[0] >= 0.0 and pcts[-1] <= 100.0
    assert 50.0 in fc.quantiles
    vals = [fc.quantiles[p] for p in pcts]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    assert fc.engine == expect_engine


def test_naive_drift_extrapolates_linear_trend():
    eng = get_engine("naive_drift")
    fc = eng.fit_predict(_series(range(100)), horizon_days=10.0)
    _assert_contract(fc, "naive_drift")
    assert fc.quantiles[50.0] == pytest.approx(109.0, abs=1.0)


def test_naive_drift_constant_series_degenerate_safe():
    eng = get_engine("naive_drift")
    fc = eng.fit_predict(_series([5.0] * 50), horizon_days=7.0)
    _assert_contract(fc, "naive_drift")
    assert fc.quantiles[50.0] == pytest.approx(5.0, abs=1e-6)


def test_naive_drift_tail_widen():
    base = get_engine("naive_drift").fit_predict(
        _series([0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7] * 4), horizon_days=5.0)
    wide = get_engine("naive_drift", tail_widen=2.0).fit_predict(
        _series([0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7] * 4), horizon_days=5.0)
    spread = base.quantiles[97.5] - base.quantiles[2.5]
    wspread = wide.quantiles[97.5] - wide.quantiles[2.5]
    assert wspread == pytest.approx(2.0 * spread, rel=1e-6)
    assert wide.quantiles[50.0] == pytest.approx(base.quantiles[50.0])


def test_naive_drift_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 8"):
        get_engine("naive_drift").fit_predict(_series([1, 2, 3]), 5.0)


def test_unknown_engine_lists_options():
    with pytest.raises(KeyError, match="naive_drift"):
        get_engine("nope")


def test_registry_contains_core_engines():
    for name in ("naive_drift", "theta", "autoarima", "autoets",
                 "chronos2", "timesfm"):
        assert name in ENGINES


def test_statsforecast_theta_contract():
    pytest.importorskip("statsforecast")
    eng = get_engine("theta")
    fc = eng.fit_predict(_series([i + (i % 3) for i in range(60)]), 10.0)
    _assert_contract(fc, "theta")
    assert 55.0 < fc.quantiles[50.0] < 85.0
