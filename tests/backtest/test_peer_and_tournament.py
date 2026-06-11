"""Peer score, head-to-head, tournament take — plus the scoring-layer
property tests required by the spec (§8): peer zero-sum, proper-scoring
sanity via grid search, clamps applied. FAQ sections cited inline."""
import math

import pytest

from backtest.core import BacktestQuestion, Forecast, QType
from backtest.scoring.metaculus_scores import (
    baseline_score,
    head_to_head,
    log_score,
    peer_score,
    tournament_take,
)


def _binary_q(resolution):
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="Will it happen?", resolution=resolution)


def _mc_q(resolution, options):
    return BacktestQuestion(
        qid="metaculus:2", platform="metaculus", native_id="2",
        qtype=QType.MULTIPLE_CHOICE, title="Which one?",
        resolution=resolution, options=options)


def _numeric_q(resolution, *, range_min=0.0, range_max=100.0,
               open_lower=False, open_upper=False, cdf_size=201):
    return BacktestQuestion(
        qid="metaculus:3", platform="metaculus", native_id="3",
        qtype=QType.NUMERIC, title="How many?", resolution=resolution,
        scaling={"range_min": range_min, "range_max": range_max,
                 "zero_point": None, "open_lower_bound": open_lower,
                 "open_upper_bound": open_upper, "cdf_size": cdf_size,
                 "unit": "units"})


def build_cdf(*, n_points=201, below=0.0, above=0.0,
              spike_bucket=None, spike_mass=0.0):
    """CDF on the standardized domain (see tests/backtest/test_log_score.py
    for the bucket convention)."""
    n_bins = n_points - 1
    inbound = 1.0 - below - above
    if spike_bucket is None:
        masses = [inbound / n_bins] * n_bins
    else:
        base = (inbound - spike_mass) / (n_bins - 1)
        masses = [spike_mass if b == spike_bucket else base
                  for b in range(1, n_bins + 1)]
    cdf = [below]
    for m in masses:
        cdf.append(cdf[-1] + m)
    cdf[-1] = 1.0 - above
    return tuple(cdf)


def _bin(p):
    return Forecast(qtype=QType.BINARY, binary_prob=p)


# ----------------------------------------------------------------- peer score

def test_peer_binary_matches_geometric_mean_formula():
    # FAQ "Peer score math": peer = 100 * (ln p - ln GM(other predictions)).
    f = _bin(0.8)
    others = [_bin(0.5), _bin(0.6)]
    expected = 100.0 * (math.log(0.8) - (math.log(0.5) + math.log(0.6)) / 2.0)
    got = peer_score(f, others, _binary_q(True))
    assert got == pytest.approx(expected)
    assert got == pytest.approx(37.8843, abs=1e-3)


def test_peer_zero_sum_binary():
    # FAQ "What is the Peer score?": "on any given question, the sum of all
    # participants' Peer scores is always 0" (worked example: log scores
    # -1, 1, 2 give peer values -2.5, +0.5, +2, summing to 0).
    fs = [_bin(p) for p in (0.9, 0.6, 0.3, 0.45)]
    q = _binary_q(True)
    total = sum(peer_score(f, [o for o in fs if o is not f], q) for f in fs)
    assert total == pytest.approx(0.0, abs=1e-9)


def test_peer_zero_sum_mc():
    opts = ("a", "b", "c")
    fs = [
        Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 0.6, "b": 0.3, "c": 0.1}),
        Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 0.2, "b": 0.5, "c": 0.3}),
        Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}),
    ]
    q = _mc_q("b", opts)
    total = sum(peer_score(f, [o for o in fs if o is not f], q) for f in fs)
    assert total == pytest.approx(0.0, abs=1e-9)


def test_peer_zero_sum_continuous():
    cdfs = [build_cdf(),
            build_cdf(spike_bucket=100, spike_mass=0.175),
            build_cdf(spike_bucket=120, spike_mass=0.10)]
    fs = [Forecast(qtype=QType.NUMERIC, cdf=c) for c in cdfs]
    q = _numeric_q(50.0)
    total = sum(peer_score(f, [o for o in fs if o is not f], q) for f in fs)
    assert total == pytest.approx(0.0, abs=1e-9)


def test_peer_continuous_is_halved():
    # FAQ "Peer score math": "Peer scores of continuous and discrete
    # questions are further divided by 2."
    sharp = Forecast(qtype=QType.NUMERIC,
                     cdf=build_cdf(spike_bucket=100, spike_mass=0.175))
    flat = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    q = _numeric_q(49.75)
    raw_gap = log_score(sharp, q) - log_score(flat, q)
    assert peer_score(sharp, [flat], q) == pytest.approx(100.0 * raw_gap / 2.0)
    # pdf 35 vs pdf 1: 100 * ln(35) / 2
    assert peer_score(sharp, [flat], q) == pytest.approx(
        100.0 * math.log(35.0) / 2.0)


def test_peer_with_no_others_is_zero():
    assert peer_score(_bin(0.8), [], _binary_q(True)) == 0.0


# --------------------------------------------------------------- head-to-head

def test_head_to_head_binary_log_ratio():
    # spec §6.2: head-to-head spot peer vs CP-at-T = 100 * ln(p_bot / p_CP)
    # on the realized side (the statistic in Metaculus Pros-vs-Bots analyses).
    q = _binary_q(True)
    assert head_to_head(_bin(0.7), _bin(0.5), q) == pytest.approx(
        100.0 * math.log(0.7 / 0.5))
    assert head_to_head(_bin(0.7), _bin(0.5), q) == pytest.approx(33.647, abs=1e-3)


def test_head_to_head_antisymmetric_and_zero_on_tie():
    q = _binary_q(False)
    a, b = _bin(0.7), _bin(0.4)
    assert head_to_head(a, b, q) == pytest.approx(-head_to_head(b, a, q))
    assert head_to_head(a, a, q) == pytest.approx(0.0)


def test_head_to_head_continuous_halved():
    sharp = Forecast(qtype=QType.NUMERIC,
                     cdf=build_cdf(spike_bucket=100, spike_mass=0.175))
    flat = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    q = _numeric_q(49.75)
    assert head_to_head(sharp, flat, q) == pytest.approx(
        100.0 * math.log(35.0) / 2.0)


# ------------------------------------------------------------ tournament take

def test_tournament_take_formula():
    # FAQ "How are my tournament Score, Take, Prize, and Rank calculated?":
    # total = sum(peer score * question weight); take = max(total, 0)^2.
    assert tournament_take([20.0, -10.0, 5.0], [1.0, 1.0, 0.5]) == pytest.approx(156.25)
    assert tournament_take([-5.0, -1.0], [1.0, 1.0]) == 0.0
    assert tournament_take([], []) == 0.0


def test_tournament_take_length_mismatch_raises():
    with pytest.raises(ValueError):
        tournament_take([1.0], [1.0, 2.0])


# ------------------------------------------------------------ property tests

def test_truth_telling_maximizes_expected_log_score():
    # Proper-scoring sanity (spec §8): over a 1%..99% grid, expected log
    # score under true belief 0.7 is maximized exactly at 0.7.
    q_yes, q_no = _binary_q(True), _binary_q(False)
    belief = 0.7

    def expected_score(pred):
        f = _bin(pred)
        return belief * log_score(f, q_yes) + (1 - belief) * log_score(f, q_no)

    grid = [i / 100 for i in range(1, 100)]
    best = max(grid, key=expected_score)
    assert best == pytest.approx(belief)


def test_baseline_expected_value_matches_faq_extremes_table():
    # FAQ "Can I get better scores by predicting extreme values?" — belief
    # 80%: expected baseline scores +24 / +28 / +21 / -34 for predictions
    # 70% / 80% / 90% / 99% (the proper-score maximum sits at the belief).
    q_yes, q_no = _binary_q(True), _binary_q(False)
    for pred, expected in [(0.7, 24.0), (0.8, 28.0), (0.9, 21.0), (0.99, -34.0)]:
        f = _bin(pred)
        ev = 0.8 * baseline_score(f, q_yes) + 0.2 * baseline_score(f, q_no)
        assert ev == pytest.approx(expected, abs=0.6)


def test_peer_score_raises_on_mixed_qtypes():
    # Guard: all forecasts must share the same qtype.  An MC 'mine' with a
    # binary entry in 'others' on an MC question must raise ValueError, not
    # silently mis-score (float resolution evaluated via truthiness).
    opts = ("a", "b", "c")
    q = _mc_q("b", opts)
    mine = Forecast(qtype=QType.MULTIPLE_CHOICE,
                    mc_probs={"a": 0.2, "b": 0.5, "c": 0.3})
    binary_interloper = Forecast(qtype=QType.BINARY, binary_prob=0.7)
    with pytest.raises(ValueError, match="qtype"):
        peer_score(mine, [binary_interloper], q)


def test_clamps_keep_all_scores_finite():
    # Degenerate probabilities never produce infinities (clamp [0.001, 0.999]).
    q_yes = _binary_q(True)
    for p in (0.0, 1.0):
        assert math.isfinite(log_score(_bin(p), q_yes))
        assert math.isfinite(baseline_score(_bin(p), q_yes))
        assert math.isfinite(peer_score(_bin(p), [_bin(0.5)], q_yes))
        assert math.isfinite(head_to_head(_bin(p), _bin(0.5), q_yes))
