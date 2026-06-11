"""Golden tests for the Metaculus Baseline score, encoding the worked
examples and notable values from the Metaculus Scores FAQ
(https://www.metaculus.com/help/scores-faq/). FAQ table values are rounded
integers, hence the abs=0.6 tolerance on those assertions; exact-formula
assertions accompany them."""
import math

import pytest

from backtest.core import BacktestQuestion, Forecast, QType
from backtest.scoring.metaculus_scores import baseline_score


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


def test_binary_baseline_faq_predicting_extremes_table():
    # FAQ "Can I get better scores by predicting extreme values?":
    # prediction -> (score if Yes, score if No):
    # 70% -> (+48, -74); 80% -> (+68, -132); 90% -> (+85, -232); 99% -> (+99, -564)
    table = [(0.7, 48, -74), (0.8, 68, -132), (0.9, 85, -232), (0.99, 99, -564)]
    for p, yes, no in table:
        assert baseline_score(_bin(p), _binary_q(True)) == pytest.approx(yes, abs=0.6)
        assert baseline_score(_bin(p), _binary_q(False)) == pytest.approx(no, abs=0.6)


def test_binary_baseline_exact_formula():
    # FAQ "Baseline score math": 100 * (ln P(outcome) - ln 50%) / ln 2,
    # equivalently 100 * (log2(P) + 1).
    assert baseline_score(_bin(0.8), _binary_q(True)) == pytest.approx(
        100.0 * (math.log(0.8) - math.log(0.5)) / math.log(2.0))
    assert baseline_score(_bin(0.8), _binary_q(True)) == pytest.approx(
        100.0 * (math.log2(0.8) + 1.0))


def test_binary_baseline_chance_zero_and_notable_values():
    # FAQ "What is the Baseline score?": chance prediction scores 0; notable
    # values (binary): best +99.9 at 99.9%, worst -897 at 0.1%.
    assert baseline_score(_bin(0.5), _binary_q(True)) == pytest.approx(0.0)
    assert baseline_score(_bin(0.999), _binary_q(True)) == pytest.approx(99.9, abs=0.6)
    # clamp: a 0.0 prediction scores as 0.001
    assert baseline_score(_bin(0.0), _binary_q(True)) == pytest.approx(-897, abs=0.6)


def test_mc_baseline_exact_formula_and_uniform_zero():
    # FAQ "Baseline score math": 100 * (ln P(outcome) - ln(1/N)) / ln N.
    opts = ("a", "b", "c", "d")
    uniform = Forecast(qtype=QType.MULTIPLE_CHOICE,
                       mc_probs={o: 0.25 for o in opts})
    assert baseline_score(uniform, _mc_q("b", opts)) == pytest.approx(0.0)
    f = Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1})
    assert baseline_score(f, _mc_q("a", opts)) == pytest.approx(
        100.0 * (math.log(0.4) - math.log(0.25)) / math.log(4.0))


def test_mc_baseline_faq_notable_values_8_options():
    # FAQ "What is the Baseline score?" notable values (MC, 8 options):
    # best +99.9, worst -232.
    opts = tuple("abcdefgh")
    best = Forecast(
        qtype=QType.MULTIPLE_CHOICE,
        mc_probs={o: (0.999 if o == "a" else 0.001 / 7) for o in opts})
    worst = Forecast(
        qtype=QType.MULTIPLE_CHOICE,
        mc_probs={o: (0.001 if o == "a" else 0.999 / 7) for o in opts})
    assert baseline_score(best, _mc_q("a", opts)) == pytest.approx(99.9, abs=0.6)
    assert baseline_score(worst, _mc_q("a", opts)) == pytest.approx(-232, abs=0.6)


def test_continuous_baseline_uniform_is_zero_for_each_bound_config():
    # FAQ "Baseline score math": continuous baseline pdf is 1 / 0.95 / 0.9
    # for closed / one-open / two-open bounds — so the matching uniform
    # forecast scores exactly 0 in every configuration.
    cases = [
        (build_cdf(), dict()),
        (build_cdf(above=0.05), dict(open_upper=True)),
        (build_cdf(below=0.05, above=0.05),
         dict(open_lower=True, open_upper=True)),
    ]
    for cdf, bounds in cases:
        f = Forecast(qtype=QType.NUMERIC, cdf=cdf)
        assert baseline_score(f, _numeric_q(50.0, **bounds)) == pytest.approx(
            0.0, abs=1e-9)


def test_continuous_baseline_faq_best_and_worst_notable_values():
    # FAQ "What is the Baseline score?" notable values (continuous):
    # best +183 = 100*(ln 35 - ln 0.9)/2 (pdf cap, two open bounds);
    # worst -230 = 100*(ln 0.01 - ln 1)/2 (pdf floor, closed bounds).
    best = Forecast(qtype=QType.NUMERIC,
                    cdf=build_cdf(below=0.05, above=0.05,
                                  spike_bucket=100, spike_mass=0.175))
    q2 = _numeric_q(49.75, open_lower=True, open_upper=True)
    assert baseline_score(best, q2) == pytest.approx(183, abs=0.6)
    assert baseline_score(best, q2) == pytest.approx(
        100.0 * (math.log(35.0) - math.log(0.9)) / 2.0)
    worst = Forecast(qtype=QType.NUMERIC,
                     cdf=build_cdf(spike_bucket=50, spike_mass=0.999))
    assert baseline_score(worst, _numeric_q(74.75)) == pytest.approx(-230, abs=0.6)
    assert baseline_score(worst, _numeric_q(74.75)) == pytest.approx(
        100.0 * math.log(0.01) / 2.0)


def test_continuous_baseline_out_of_bounds_uses_boundary_mass_vs_0_05():
    # Platform implementation (Metaculus repo scoring/score_math.py,
    # evaluate_forecasts_baseline_*): out-of-bounds buckets score the
    # boundary mass against the uniform baseline's own out-of-bound mass
    # 0.05 — so the chance forecast scores 0 even out of bounds.
    q = _numeric_q(-3.0, open_lower=True, open_upper=True)
    chance = Forecast(qtype=QType.NUMERIC,
                      cdf=build_cdf(below=0.05, above=0.05))
    assert baseline_score(chance, q) == pytest.approx(0.0, abs=1e-9)
    heavier = Forecast(qtype=QType.NUMERIC,
                       cdf=build_cdf(below=0.10, above=0.05))
    assert baseline_score(heavier, q) == pytest.approx(
        100.0 * math.log(0.10 / 0.05) / 2.0)


def test_discrete_baseline_uniform_is_zero_and_floor_matches_continuous():
    # Server formula 100 * ln(pmf / ((1 - 0.05*open) / n_bins)) / 2 applies to
    # discrete questions unchanged; the FAQ floor 0.01/n_bins maps to the same
    # -230.26 worst case as continuous.
    q = _numeric_q(55.0, qtype=QType.DISCRETE, cdf_size=11)
    uniform = Forecast(qtype=QType.DISCRETE, cdf=build_cdf(n_points=11))
    assert baseline_score(uniform, q) == pytest.approx(0.0, abs=1e-9)
    spiky = Forecast(
        qtype=QType.DISCRETE,
        cdf=build_cdf(n_points=11, spike_bucket=1, spike_mass=0.9919))
    assert baseline_score(spiky, q) == pytest.approx(
        100.0 * (math.log(0.001) - math.log(0.1)) / 2.0)
