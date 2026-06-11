import numpy as np
import pytest

from bot.aggregation import (
    aggregate_binary, aggregate_cdfs, aggregate_multiple_choice,
    extremize, geometric_mean_of_odds, median_pool, trimmed_gmo,
)


def test_gmo_known_value():
    # odds 1/9 and 2/3 -> mean log-odds -1.3013 -> p ~= 0.2139
    assert geometric_mean_of_odds([0.1, 0.4]) == pytest.approx(0.2139, abs=1e-3)


def test_gmo_single_forecast_is_identity():
    assert geometric_mean_of_odds([0.37]) == pytest.approx(0.37, abs=1e-4)


def test_trimmed_gmo_drops_extremes():
    probs = [0.05, 0.3, 0.35, 0.4, 0.95]
    assert trimmed_gmo(probs) == pytest.approx(
        geometric_mean_of_odds([0.3, 0.35, 0.4]))


def test_trimmed_gmo_small_n_falls_back_to_plain_gmo():
    probs = [0.2, 0.4, 0.6]
    assert trimmed_gmo(probs) == pytest.approx(geometric_mean_of_odds(probs))


def test_median_pool():
    assert median_pool([0.1, 0.5, 0.9]) == 0.5


def test_extremize_identity_at_a1():
    assert extremize(0.7, 1.0) == pytest.approx(0.7)


def test_extremize_pushes_away_from_half():
    assert extremize(0.7, 2.5) > 0.7
    assert extremize(0.3, 2.5) < 0.3


def test_aggregate_binary_rejects_unknown_method_and_empty():
    with pytest.raises(ValueError):
        aggregate_binary([0.5], method="nope")
    with pytest.raises(ValueError):
        aggregate_binary([], method="trimmed_gmo")


def test_aggregate_mc_floors_and_renormalizes():
    runs = [{"A": 0.998, "B": 0.001, "C": 0.001},
            {"A": 0.998, "B": 0.001, "C": 0.001}]
    agg = aggregate_multiple_choice(runs, floor=0.01)
    assert agg["B"] >= 0.009          # floored before renormalization
    assert sum(agg.values()) == pytest.approx(1.0)


def test_aggregate_cdfs_pointwise_median_stays_monotone():
    c1 = np.linspace(0, 1, 201)
    c2 = np.linspace(0, 1, 201) ** 2
    out = aggregate_cdfs([c1, c2])
    assert np.all(np.diff(out) >= 0)
    assert out[0] == pytest.approx(0.0)
