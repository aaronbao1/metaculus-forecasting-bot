"""Paired statistics and difficulty adjustment (spec §6.4)."""
import pytest
from scipy import stats as scipy_stats

from backtest.scoring.stats import (
    detectable_effect,
    difficulty_adjust,
    paired_bootstrap,
    paired_t,
)


def test_paired_bootstrap_deterministic_and_brackets_mean():
    diffs = [1.0, 1.2, 0.8, 1.1, 0.9, 1.05, 0.95, 1.15, 0.85, 1.0]
    a = paired_bootstrap(diffs, n=2000, seed=0)
    b = paired_bootstrap(diffs, n=2000, seed=0)
    assert a == b                                   # same seed, same result
    assert set(a) == {"mean", "lo", "hi"}
    assert a["mean"] == pytest.approx(sum(diffs) / len(diffs))
    assert a["lo"] <= a["mean"] <= a["hi"]
    assert a["lo"] > 0                              # clearly positive effect


def test_paired_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        paired_bootstrap([])


def test_paired_t_matches_scipy_one_sample():
    diffs = [0.3, -0.1, 0.4, 0.2, 0.0, 0.5, -0.2, 0.1]
    out = paired_t(diffs)
    ref = scipy_stats.ttest_1samp(diffs, 0.0)
    assert set(out) == {"mean", "t", "p"}
    assert out["mean"] == pytest.approx(sum(diffs) / len(diffs))
    assert out["t"] == pytest.approx(float(ref.statistic))
    assert out["p"] == pytest.approx(float(ref.pvalue))


def test_paired_t_needs_two_points():
    with pytest.raises(ValueError):
        paired_t([1.0])


def test_detectable_effect_spec_power_example():
    # spec §6.4 power honesty: "~400 paired questions for a 0.01 Brier gap
    # at 2 sigma (delta ~ 0.1)" -> 2 * sd / sqrt(n).
    assert detectable_effect(400, 0.1) == pytest.approx(0.01)
    assert detectable_effect(100, 0.2) == pytest.approx(0.04)
    with pytest.raises(ValueError):
        detectable_effect(0, 0.1)


def test_difficulty_adjust_recovers_known_additive_structure():
    # b_ij = alpha_i + gamma_j with mean-zero gamma and no noise: adjusted
    # scores must equal alpha_i exactly (contract: exact recovery test).
    alpha = {"A": 1.0, "B": 3.0}
    gamma = {"q1": -2.0, "q2": 0.0, "q3": 2.0}
    scores = {(s, q): alpha[s] + gamma[q] for s in alpha for q in gamma}
    adjusted = difficulty_adjust(scores)
    assert set(adjusted) == set(scores)
    for (s, q), v in adjusted.items():
        assert v == pytest.approx(alpha[s], abs=1e-9)


def test_difficulty_adjust_preserves_paired_differences():
    # Subtracting a per-question effect never changes within-question
    # system differences.
    scores = {("A", "q1"): 5.0, ("B", "q1"): 1.0,
              ("A", "q2"): -3.0, ("B", "q2"): 0.5}
    adjusted = difficulty_adjust(scores)
    for q in ("q1", "q2"):
        assert adjusted[("A", q)] - adjusted[("B", q)] == pytest.approx(
            scores[("A", q)] - scores[("B", q)])


def test_difficulty_adjust_empty():
    assert difficulty_adjust({}) == {}
