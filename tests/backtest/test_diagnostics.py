"""Diagnostics (spec §6.3): quintile calibration curves with binomial
consistency bands, Murphy decomposition, discrimination separation.
`pairs` is list[(binary prob, outcome bool)]."""
import math

import pytest

from backtest.scoring.diagnostics import (
    calibration_curve,
    discrimination_separation,
    murphy_decomposition,
)


def test_calibration_curve_quintiles_perfectly_calibrated():
    # Five forecast levels, 20 each, hit rates exactly equal to the level:
    # every bin's frequency must equal its mean forecast and sit inside the
    # binomial consistency band.
    pairs = []
    for p, k in [(0.1, 2), (0.3, 6), (0.5, 10), (0.7, 14), (0.9, 18)]:
        pairs += [(p, True)] * k + [(p, False)] * (20 - k)
    curve = calibration_curve(pairs, n_bins=5)
    assert len(curve) == 5
    assert sum(b["n"] for b in curve) == 100
    for b in curve:
        assert set(b) == {"p_mean", "freq", "n", "lo", "hi"}
        assert b["freq"] == pytest.approx(b["p_mean"])
        assert b["lo"] <= b["freq"] <= b["hi"]
    p_means = [b["p_mean"] for b in curve]
    assert p_means == sorted(p_means)  # quantile bins ascend


def test_calibration_curve_flags_overconfidence():
    # 50 forecasts of 0.9 that all resolve No: every bin's observed frequency
    # (0.0) must fall below the lower consistency band
    # (binom.ppf(0.05, 10, 0.9)/10 = 0.7).
    pairs = [(0.9, False)] * 50
    curve = calibration_curve(pairs, n_bins=5)
    assert all(b["freq"] < b["lo"] for b in curve)


def test_calibration_curve_empty_input():
    assert calibration_curve([]) == []


def test_murphy_decomposition_identity():
    # Murphy (1973): brier = uncertainty - resolution + reliability, exact
    # when grouping by unique forecast value.
    pairs = [(0.2, False), (0.2, True), (0.7, True), (0.7, True),
             (0.7, False), (0.5, True)]
    out = murphy_decomposition(pairs)
    assert set(out) == {"brier", "uncertainty", "resolution", "reliability"}
    assert out["brier"] == pytest.approx(
        out["uncertainty"] - out["resolution"] + out["reliability"], abs=1e-12)


def test_murphy_perfect_and_constant_forecasters():
    perfect = murphy_decomposition([(1.0, True), (0.0, False)])
    assert perfect["brier"] == pytest.approx(0.0)
    assert perfect["reliability"] == pytest.approx(0.0)
    assert perfect["resolution"] == pytest.approx(perfect["uncertainty"])
    constant = murphy_decomposition([(0.5, True), (0.5, False)])
    assert constant["resolution"] == pytest.approx(0.0)
    assert constant["reliability"] == pytest.approx(0.0)
    assert constant["brier"] == pytest.approx(constant["uncertainty"])


def test_murphy_empty_raises():
    with pytest.raises(ValueError):
        murphy_decomposition([])


def test_discrimination_separation_golden():
    # spec §6.3: mean forecast on Yes-resolvers minus mean on No-resolvers
    # (the Pros-vs-bots 44pp-vs-26pp statistic).
    pairs = [(0.8, True), (0.6, True), (0.3, False), (0.1, False)]
    assert discrimination_separation(pairs) == pytest.approx(0.5)


def test_discrimination_separation_degenerate_is_nan():
    assert math.isnan(discrimination_separation([(0.5, True)]))
    assert math.isnan(discrimination_separation([]))
