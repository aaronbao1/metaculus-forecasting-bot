"""Metaculus score family.

Formulas verified against the Metaculus Scores FAQ
(https://www.metaculus.com/help/scores-faq/) and the platform's open-source
implementation (github.com/Metaculus/metaculus: scoring/score_math.py and
utils/the_math/formulas.py). Golden tests in tests/backtest/ encode the FAQ's
worked examples; the FAQ section is cited next to each golden number.

Conventions:
- Binary/MC probabilities are clamped to [0.001, 0.999] before any log
  (FAQ: "Metaculus restricts binary predictions to be between 0.1% and 99.9%").
- Numeric ("continuous") questions are scored from the 201-point CDF:
  pmf = diff(cdf) over n_bins = len(cdf) - 1 standardized-domain bins,
  pdf = pmf * n_bins, clamped to [0.01, 35.0]
  (FAQ: "all pdfs have a uniform distribution of height 0.01 added to them";
   "continuous pdfs to be between 0.01 and ~35").
- Discrete questions score the pmf directly with floor 0.01 / n_bins
  (FAQ: "the pmf is used in place of the pdf").
- Out-of-bounds resolutions are scored as binary on the boundary mass
  (mass below = cdf[0], mass above = 1 - cdf[-1]). For the baseline score the
  chance value of that implied binary is 0.05 — the uniform baseline's own
  out-of-bound mass — per the platform implementation
  (scoring/score_math.py: boundary buckets use baseline = 0.05).
- Continuous/discrete peer and head-to-head scores are divided by 2
  (FAQ "Peer score math": factor chosen empirically).
"""
from __future__ import annotations

import math

from backtest.core import BacktestQuestion, Forecast, QType

PROB_FLOOR = 0.001
PROB_CEIL = 0.999
PDF_FLOOR = 0.01
PDF_CAP = 35.0


def _clamp_prob(p: float) -> float:
    return min(max(float(p), PROB_FLOOR), PROB_CEIL)


def _require_resolution(q: BacktestQuestion) -> None:
    if q.resolution is None:
        raise ValueError(f"{q.qid}: question is unresolved; cannot score")


def _outcome_prob(forecast: Forecast, q: BacktestQuestion) -> float:
    """Clamped probability assigned to the realized outcome (binary/MC)."""
    if forecast.qtype is QType.BINARY:
        if forecast.binary_prob is None:
            raise ValueError("binary forecast missing binary_prob")
        p = forecast.binary_prob if q.resolution else 1.0 - forecast.binary_prob
        return _clamp_prob(p)
    if forecast.qtype is QType.MULTIPLE_CHOICE:
        if forecast.mc_probs is None:
            raise ValueError("multiple_choice forecast missing mc_probs")
        if q.resolution not in forecast.mc_probs:
            raise ValueError(
                f"resolution {q.resolution!r} not among forecast options")
        return _clamp_prob(forecast.mc_probs[q.resolution])
    raise ValueError(f"not a categorical question type: {forecast.qtype}")


def _scaling(q: BacktestQuestion) -> dict:
    if not q.scaling:
        raise ValueError(f"{q.qid}: numeric/discrete question missing scaling")
    return q.scaling


def _unscaled_location(x: float, q: BacktestQuestion) -> float:
    """Actual value -> standardized [0, 1] domain. Mirror of Metaculus
    utils/the_math/formulas.py:scaled_location_to_unscaled_location
    (log scaling when zero_point is set, else linear)."""
    s = _scaling(q)
    rmin, rmax = float(s["range_min"]), float(s["range_max"])
    zero_point = s.get("zero_point")
    if zero_point is not None:
        deriv_ratio = (rmax - zero_point) / (rmin - zero_point)
        return (
            math.log((x - rmin) * (deriv_ratio - 1.0) + (rmax - rmin))
            - math.log(rmax - rmin)
        ) / math.log(deriv_ratio)
    return (x - rmin) / (rmax - rmin)


def _resolution_bucket(x: float, q: BacktestQuestion, n_bins: int) -> int:
    """0 = below lower bound, 1..n_bins = inbound bins, n_bins + 1 = above
    upper. Mirror of Metaculus
    utils/the_math/formulas.py:unscaled_location_to_bucket_index."""
    s = _scaling(q)
    rmin, rmax = float(s["range_min"]), float(s["range_max"])
    # In/out-of-bounds is intentionally value-driven (x < range_min / x > range_max).
    # open_lower_bound and open_upper_bound are NOT consulted here — they do not
    # change bucketing, mirroring the Metaculus platform implementation.
    if x < rmin:
        return 0
    if x > rmax:
        return n_bins + 1
    u = _unscaled_location(x, q)
    if u >= 1.0:
        return n_bins
    # == ceil(u*n_bins) 1-indexed; the 1e-10 absorbs fp noise at bin edges
    # (matches Metaculus u==0 -> bucket 1)
    return max(int(u * n_bins + 1 - 1e-10), 1)


def _pdf_value(forecast: Forecast, q: BacktestQuestion) -> tuple[str, float]:
    """Score input for a numeric/discrete forecast at q.resolution.

    Returns ("binary", clamped boundary mass) when the resolution is out of
    bounds, ("pmf", floored pmf) for in-bounds discrete, and
    ("pdf", clamped pdf) for in-bounds numeric.
    """
    if forecast.cdf is None:
        raise ValueError("numeric/discrete forecast missing cdf")
    cdf = forecast.cdf
    n_bins = len(cdf) - 1
    if n_bins < 1:
        raise ValueError("cdf must have at least 2 points")
    bucket = _resolution_bucket(float(q.resolution), q, n_bins)
    if bucket == 0:
        return "binary", _clamp_prob(cdf[0])
    if bucket == n_bins + 1:
        return "binary", _clamp_prob(1.0 - cdf[-1])
    pmf = float(cdf[bucket] - cdf[bucket - 1])
    if forecast.qtype is QType.DISCRETE:
        return "pmf", max(pmf, PDF_FLOOR / n_bins)
    return "pdf", min(max(pmf * n_bins, PDF_FLOOR), PDF_CAP)


def log_score(forecast: Forecast, q: BacktestQuestion) -> float:
    """FAQ "What is the log score?": ln P(outcome) for binary/MC.
    FAQ "What is the log score for continuous questions?": ln pdf(outcome)
    for numeric, ln pmf(outcome) for discrete; out-of-bounds resolutions are
    scored as binary on the boundary mass."""
    _require_resolution(q)
    if forecast.qtype in (QType.BINARY, QType.MULTIPLE_CHOICE):
        return math.log(_outcome_prob(forecast, q))
    _, value = _pdf_value(forecast, q)
    return math.log(value)


def baseline_score(forecast: Forecast, q: BacktestQuestion) -> float:
    """FAQ "Baseline score math":
    binary:     100 * (ln P(outcome) - ln 0.5) / ln 2
    MC:         100 * (ln P(outcome) - ln(1/N)) / ln N
    continuous: 100 * (ln pdf(outcome) - ln baseline) / 2, with baseline
                1 / 0.95 / 0.9 for closed / one-open / two-open bounds;
                discrete uses the pmf against the uniform pmf (identical
                formula after the pmf -> pdf change of units).
    Out-of-bounds resolutions score the boundary mass against the uniform
    baseline's own out-of-bound mass 0.05 (platform source:
    scoring/score_math.py, evaluate_forecasts_baseline_*)."""
    _require_resolution(q)
    if forecast.qtype is QType.BINARY:
        p = _outcome_prob(forecast, q)
        return 100.0 * (math.log(p) - math.log(0.5)) / math.log(2.0)
    if forecast.qtype is QType.MULTIPLE_CHOICE:
        n = len(q.options)
        if n < 2:
            raise ValueError("multiple_choice question needs >= 2 options")
        p = _outcome_prob(forecast, q)
        return 100.0 * (math.log(p) - math.log(1.0 / n)) / math.log(n)
    s = _scaling(q)
    open_count = (int(bool(s.get("open_lower_bound")))
                  + int(bool(s.get("open_upper_bound"))))
    kind, value = _pdf_value(forecast, q)
    if kind == "binary":
        return 100.0 * (math.log(value) - math.log(0.05)) / 2.0
    if kind == "pmf":  # discrete: pmf vs uniform pmf
        n_bins = len(forecast.cdf) - 1
        chance = (1.0 - 0.05 * open_count) / n_bins
        return 100.0 * (math.log(value) - math.log(chance)) / 2.0
    chance = 1.0 - 0.05 * open_count
    return 100.0 * (math.log(value) - math.log(chance)) / 2.0


def peer_score(forecast: Forecast, others: list[Forecast],
               q: BacktestQuestion) -> float:
    """FAQ "Peer score math": 100 * (ln p - ln GM(others)) — equivalently the
    average log-score difference against every other forecast (the FAQ's
    worked zero-sum example: log scores -1, 1, 2 give peer values -2.5,
    +0.5, +2, summing to 0). Continuous/discrete scores are divided by 2.
    `others` excludes the scored forecast. Empty `others` scores 0."""
    if any(o.qtype is not forecast.qtype for o in others):
        raise ValueError("peer_score requires all forecasts to share one qtype")
    if not others:
        return 0.0
    mine = log_score(forecast, q)
    mean_others = sum(log_score(o, q) for o in others) / len(others)
    score = 100.0 * (mine - mean_others)
    if forecast.qtype in (QType.NUMERIC, QType.DISCRETE):
        score /= 2.0
    return score


def head_to_head(forecast: Forecast, opponent: Forecast,
                 q: BacktestQuestion) -> float:
    """Spot head-to-head vs a single synthetic opponent:
    100 * ln(p / p_opponent) on the realized outcome (the statistic used in
    Metaculus's Pros-vs-Bots analyses); continuous/discrete divided by 2.
    Identical to a peer score with a single other participant."""
    return peer_score(forecast, [opponent], q)


def tournament_take(spot_peer_scores: list[float],
                    weights: list[float]) -> float:
    """FAQ "How are my tournament Score, Take, Prize, and Rank calculated?":
    total = sum(spot peer score * question weight); take = max(total, 0)^2."""
    if len(spot_peer_scores) != len(weights):
        raise ValueError("scores and weights must have equal length")
    total = sum(s * w for s, w in zip(spot_peer_scores, weights))
    return max(total, 0.0) ** 2


def brier(forecast: Forecast, q: BacktestQuestion) -> float:
    """Brier score (not part of the Metaculus score family; reported for
    cross-literature comparability). Binary: (p_yes - outcome)^2, the Halawi/
    ForecastBench convention. MC: standard multiclass Brier,
    sum_k (p_k - 1[k == outcome])^2. Numeric/discrete: not defined here."""
    _require_resolution(q)
    if forecast.qtype is QType.BINARY:
        if forecast.binary_prob is None:
            raise ValueError("binary forecast missing binary_prob")
        outcome = 1.0 if q.resolution else 0.0
        return (float(forecast.binary_prob) - outcome) ** 2
    if forecast.qtype is QType.MULTIPLE_CHOICE:
        if forecast.mc_probs is None:
            raise ValueError("multiple_choice forecast missing mc_probs")
        if q.resolution not in q.options:
            raise ValueError(
                f"resolution {q.resolution!r} not among question options")
        return sum(
            (float(forecast.mc_probs.get(opt, 0.0))
             - (1.0 if opt == q.resolution else 0.0)) ** 2
            for opt in q.options
        )
    raise NotImplementedError(
        "Brier score is implemented for binary and multiple_choice only")
