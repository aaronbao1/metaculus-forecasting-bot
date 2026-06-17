import numpy as np
import pytest

from bot.distributions import (
    CDF_SIZE, MAX_STEP, MIN_STEP, percentiles_to_cdf, repair_percentiles,
    standardize_cdf, validate_cdf,
)

DECLARED = {2.5: 10, 10: 20, 25: 30, 50: 40, 75: 50, 90: 60, 97.5: 70}


def test_happy_path_closed_bounds():
    cdf = percentiles_to_cdf(DECLARED, 0, 100, open_lower=False, open_upper=False)
    assert len(cdf) == CDF_SIZE
    assert cdf[0] == pytest.approx(0.0)
    assert cdf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cdf) >= MIN_STEP - 1e-12)
    # median knot: value 40 sits at grid index 80 of linspace(0,100,201)
    assert cdf[80] == pytest.approx(0.5, abs=0.02)


def test_open_bounds_keep_mass_outside():
    cdf = percentiles_to_cdf(DECLARED, 0, 100, open_lower=True, open_upper=True)
    assert cdf[0] >= 0.001 - 1e-9
    assert cdf[-1] <= 0.999 + 1e-9


def test_non_monotone_input_is_repaired_not_fatal():
    bad = {10: 30, 50: 20, 90: 60}   # 50th percentile below 10th
    cdf = percentiles_to_cdf(bad, 0, 100, open_lower=False, open_upper=False)
    validate_cdf(cdf, open_lower=False, open_upper=False)  # should not raise


def test_spike_distribution_respects_max_step():
    spike = {10: 49.9, 50: 50.0, 90: 50.1}
    cdf = percentiles_to_cdf(spike, 0, 100, open_lower=False, open_upper=False)
    assert np.diff(cdf).max() <= MAX_STEP + 1e-9


def test_repair_percentiles_sorts_and_strictly_increases():
    out = repair_percentiles({50: 20, 10: 20, 90: 19}, 0, 100)
    vals = list(out.values())
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_validate_rejects_garbage():
    with pytest.raises(ValueError):
        validate_cdf(np.zeros(CDF_SIZE), open_lower=False, open_upper=False)


# Metaculus' NumericDistribution rejects any CDF whose adjacent percentiles
# differ by < 5e-05 (STRICT, no tolerance). A prediction concentrated relative
# to wide bounds floors many tail steps; if our floor sits exactly at 5e-05 the
# values round-trip to 4.9999...e-05 and submission fails (Gate-C crash on
# bot-testing-area numeric group qid 43322).
PLATFORM_MIN_STEP = 5.0e-05
CONCENTRATED = {2.5: 3.15, 10: 3.25, 25: 3.33, 50: 3.4, 75: 3.47, 90: 3.55, 97.5: 3.7}


def test_concentrated_prediction_meets_platform_min_step():
    cdf = percentiles_to_cdf(CONCENTRATED, 1.0, 7.0, open_lower=True, open_upper=True)
    assert np.diff(cdf).min() >= PLATFORM_MIN_STEP


def test_concentrated_prediction_accepted_by_forecasting_tools():
    from forecasting_tools import NumericDistribution
    from forecasting_tools.data_models.numeric_report import Percentile

    cdf = percentiles_to_cdf(CONCENTRATED, 1.0, 7.0, open_lower=True, open_upper=True)
    grid = np.linspace(1.0, 7.0, len(cdf))
    NumericDistribution(   # raises ValueError if any step < 5e-05
        declared_percentiles=[Percentile(percentile=float(p), value=float(v))
                              for p, v in zip(cdf, grid)],
        open_lower_bound=True, open_upper_bound=True,
        lower_bound=1.0, upper_bound=7.0, zero_point=None, cdf_size=CDF_SIZE)


# Symmetric ceiling bug: Metaculus also rejects any CDF bin whose mass > 0.2
# (STRICT). A forecast piling mass at an open bound clips bins to MAX_STEP; with
# MAX_STEP pinned at 0.2 the bin round-trips to 0.2000...07 and is rejected.
# Real failure: bot-testing-area date q 43324 ("when will martial law be lifted
# in >=3/4 of Ukraine?") — >50% mass beyond the open upper bound.
PLATFORM_MAX_STEP = 0.2
TOP_HEAVY = {2.5: 1788220800.0, 10: 1796083200.0, 25: 1805068800.0,
             50: 1811548800.0, 75: 1811548800.0, 90: 1811548800.0,
             97.5: 1811548800.0}
TOP_LO, TOP_HI = 1653782400.0, 1811462400.0


def test_top_heavy_prediction_meets_platform_max_step():
    cdf = percentiles_to_cdf(TOP_HEAVY, TOP_LO, TOP_HI,
                             open_lower=False, open_upper=True)
    assert np.diff(cdf).max() <= PLATFORM_MAX_STEP


def test_top_heavy_prediction_accepted_by_forecasting_tools():
    from forecasting_tools import NumericDistribution
    from forecasting_tools.data_models.numeric_report import Percentile

    cdf = percentiles_to_cdf(TOP_HEAVY, TOP_LO, TOP_HI,
                             open_lower=False, open_upper=True)
    grid = np.linspace(TOP_LO, TOP_HI, len(cdf))
    NumericDistribution(   # raises ValueError if any bin mass > 0.2
        declared_percentiles=[Percentile(percentile=float(p), value=float(v))
                              for p, v in zip(cdf, grid)],
        open_lower_bound=False, open_upper_bound=True,
        lower_bound=TOP_LO, upper_bound=TOP_HI, zero_point=None,
        cdf_size=CDF_SIZE, is_date=True)
