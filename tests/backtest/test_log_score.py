"""Golden tests for the Metaculus log score and the Brier score.

Golden numbers come from the worked examples in the Metaculus Scores FAQ
(https://www.metaculus.com/help/scores-faq/, content verified against the
platform's open-source repo). The FAQ section is cited next to each number.
"""
import math

import pytest

from backtest.core import BacktestQuestion, Forecast, QType
from backtest.scoring.metaculus_scores import brier, log_score


def _binary_q(resolution):
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="Will it happen?", resolution=resolution)


def _mc_q(resolution, options):
    return BacktestQuestion(
        qid="metaculus:2", platform="metaculus", native_id="2",
        qtype=QType.MULTIPLE_CHOICE, title="Which one?",
        resolution=resolution, options=options)


def _numeric_q(resolution, *, qtype=QType.NUMERIC, range_min=0.0,
               range_max=100.0, zero_point=None,
               open_lower=False, open_upper=False, cdf_size=201):
    return BacktestQuestion(
        qid="metaculus:3", platform="metaculus", native_id="3", qtype=qtype,
        title="How many?", resolution=resolution,
        scaling={"range_min": range_min, "range_max": range_max,
                 "zero_point": zero_point, "open_lower_bound": open_lower,
                 "open_upper_bound": open_upper, "cdf_size": cdf_size,
                 "unit": "units"})


def build_cdf(*, n_points=201, below=0.0, above=0.0,
              spike_bucket=None, spike_mass=0.0):
    """CDF on the standardized domain. `below`/`above` are out-of-bound
    masses; `spike_mass` goes into inbound bin `spike_bucket` (1-indexed,
    platform bucket convention: bin b spans cdf[b-1]..cdf[b]); the remaining
    inbound mass is spread uniformly over the other bins."""
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


# ------------------------------------------------------------- binary and MC

def test_binary_log_score_faq_extreme_probabilities_table():
    # FAQ "What is the log score?": predicting 99% / 99.9% Yes scores
    # -0.01 / -0.001 if the outcome is Yes and -4.6 / -6.9 if it is No.
    assert log_score(_bin(0.99), _binary_q(True)) == pytest.approx(-0.01, abs=0.005)
    assert log_score(_bin(0.99), _binary_q(False)) == pytest.approx(-4.6, abs=0.01)
    assert log_score(_bin(0.999), _binary_q(True)) == pytest.approx(-0.001, abs=0.0005)
    # 1 - 0.999 = 0.001 sits exactly on the clamp floor: ln(0.001) = -6.9078
    assert log_score(_bin(0.999), _binary_q(False)) == pytest.approx(-6.9, abs=0.01)


def test_binary_log_score_faq_die_roll_table():
    # FAQ "What is a proper scoring rule?" die-roll table: log scores of
    # predicting 5% / 17% / 30% on "die lands 6" are -3 / -1.77 / -1.2 when it
    # lands 6, and -0.05 / -0.19 / -0.37 otherwise.
    assert log_score(_bin(0.05), _binary_q(True)) == pytest.approx(-3.0, abs=0.005)
    assert log_score(_bin(0.17), _binary_q(True)) == pytest.approx(-1.77, abs=0.005)
    assert log_score(_bin(0.30), _binary_q(True)) == pytest.approx(-1.2, abs=0.005)
    assert log_score(_bin(0.05), _binary_q(False)) == pytest.approx(-0.05, abs=0.002)
    assert log_score(_bin(0.17), _binary_q(False)) == pytest.approx(-0.19, abs=0.005)
    assert log_score(_bin(0.30), _binary_q(False)) == pytest.approx(-0.37, abs=0.02)


def test_binary_log_score_clamps_to_0_001_0_999():
    # FAQ notable-values tables: "Metaculus restricts binary predictions to
    # be between 0.1% and 99.9%" — degenerate inputs must stay finite.
    assert log_score(_bin(1.0), _binary_q(True)) == pytest.approx(math.log(0.999))
    assert log_score(_bin(0.0), _binary_q(True)) == pytest.approx(math.log(0.001))
    assert log_score(_bin(1.0), _binary_q(False)) == pytest.approx(math.log(0.001))


def test_mc_log_score_die_roll_as_six_options():
    # FAQ "What is a proper scoring rule?": P(six)=17% -> ln(0.17) = -1.77.
    opts = ("one", "two", "three", "four", "five", "six")
    probs = {o: 0.166 for o in opts[:5]}
    probs["six"] = 0.17
    f = Forecast(qtype=QType.MULTIPLE_CHOICE, mc_probs=probs)
    assert log_score(f, _mc_q("six", opts)) == pytest.approx(-1.77, abs=0.005)


def test_mc_log_score_unknown_resolution_raises():
    f = Forecast(qtype=QType.MULTIPLE_CHOICE, mc_probs={"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError):
        log_score(f, _mc_q("c", ("a", "b")))


def test_unresolved_question_raises():
    with pytest.raises(ValueError):
        log_score(_bin(0.5), _binary_q(None))


# ------------------------------------------------------- continuous (numeric)

def test_continuous_log_score_uniform_forecast_is_zero():
    # FAQ "What is the log score for continuous questions?": ln(pdf(outcome));
    # the uniform pdf on closed bounds is 1.0 -> ln(1) = 0.
    f = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    assert log_score(f, _numeric_q(50.0)) == pytest.approx(0.0, abs=1e-9)


def test_continuous_log_score_known_pdf_at_cap():
    # mass 0.175 in one of 200 bins -> pdf = 0.175 * 200 = 35.0 (= the cap).
    # FAQ: "Metaculus restricts ... continuous pdfs to be between 0.01 and ~35".
    f = Forecast(qtype=QType.NUMERIC,
                 cdf=build_cdf(below=0.05, above=0.05,
                               spike_bucket=100, spike_mass=0.175))
    q = _numeric_q(49.75, open_lower=True, open_upper=True)
    assert log_score(f, q) == pytest.approx(math.log(35.0))


def test_continuous_log_score_pdf_cap_applies():
    # A bin holding mass 0.999 has raw pdf 199.8; it must be capped to 35.
    f = Forecast(qtype=QType.NUMERIC,
                 cdf=build_cdf(spike_bucket=50, spike_mass=0.999))
    assert log_score(f, _numeric_q(24.75)) == pytest.approx(math.log(35.0))


def test_continuous_log_score_pdf_floor_applies():
    # Same spike forecast scored far from the spike: raw pdf ~0.001,
    # floored to 0.01 (FAQ: 0.01 uniform floor prevents extreme log scores).
    f = Forecast(qtype=QType.NUMERIC,
                 cdf=build_cdf(spike_bucket=50, spike_mass=0.999))
    assert log_score(f, _numeric_q(74.75)) == pytest.approx(math.log(0.01))


def test_out_of_bounds_resolution_scored_as_binary_on_boundary_mass():
    # FAQ "What is the log score for continuous questions?": out-of-bounds
    # resolutions are scored "as in the question 'Will the value be below the
    # lower bound?'" — i.e. ln(boundary mass).
    f = Forecast(qtype=QType.NUMERIC, cdf=build_cdf(below=0.05, above=0.05))
    q_lo = _numeric_q(-3.0, open_lower=True, open_upper=True)
    q_hi = _numeric_q(123.0, open_lower=True, open_upper=True)
    assert log_score(f, q_lo) == pytest.approx(math.log(0.05))
    assert log_score(f, q_hi) == pytest.approx(math.log(0.05))


def test_log_scaled_question_zero_point_bucket_mapping():
    # Mirrors Metaculus utils/the_math/formulas.py log scaling: range 1..1000
    # with zero_point 0 puts sqrt(1000) = 31.62 at the standardized midpoint,
    # i.e. in bucket 100 of 200.
    f = Forecast(qtype=QType.NUMERIC,
                 cdf=build_cdf(spike_bucket=100, spike_mass=0.175))
    q = _numeric_q(31.6227766, range_min=1.0, range_max=1000.0, zero_point=0.0)
    assert log_score(f, q) == pytest.approx(math.log(35.0))


def test_boundary_value_bucket_edge_rules():
    # Platform rule: exactly range_min lands in bucket 1 (not "below");
    # exactly range_max lands in bucket n_bins (not "above").
    f = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    assert log_score(f, _numeric_q(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert log_score(f, _numeric_q(100.0)) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------- discrete

def test_discrete_log_score_uses_pmf_with_floor():
    # FAQ: "For discrete continuous questions, the pmf is used in place of the
    # pdf, and the minimum value assigned to any outcome cannot go below
    # 0.01 / number of possible inbound outcomes."
    uniform = Forecast(qtype=QType.DISCRETE, cdf=build_cdf(n_points=11))
    q = _numeric_q(55.0, qtype=QType.DISCRETE, cdf_size=11)
    assert log_score(uniform, q) == pytest.approx(math.log(0.1))
    spiky = Forecast(
        qtype=QType.DISCRETE,
        cdf=build_cdf(n_points=11, spike_bucket=1, spike_mass=0.9919))
    # non-spike bins hold 0.0009 < floor 0.01/10 = 0.001
    assert log_score(spiky, q) == pytest.approx(math.log(0.001))


# ---------------------------------------------------------------------- brier

def test_brier_binary():
    assert brier(_bin(0.8), _binary_q(True)) == pytest.approx(0.04)
    assert brier(_bin(0.8), _binary_q(False)) == pytest.approx(0.64)


def test_brier_mc_standard_multiclass():
    f = Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 0.7, "b": 0.2, "c": 0.1})
    q = _mc_q("a", ("a", "b", "c"))
    # (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.14
    assert brier(f, q) == pytest.approx(0.14)


def test_brier_numeric_not_implemented():
    f = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    with pytest.raises(NotImplementedError):
        brier(f, _numeric_q(50.0))
