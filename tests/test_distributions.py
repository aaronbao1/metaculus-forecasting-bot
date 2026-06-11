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
