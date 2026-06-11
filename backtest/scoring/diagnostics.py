"""Forecast diagnostics (spec §6.3). All functions take
pairs = list[(binary prob, outcome bool)]."""
from __future__ import annotations

from scipy.stats import binom

Pair = tuple[float, bool]


def calibration_curve(pairs: list[Pair], n_bins: int = 5) -> list[dict]:
    """Quantile-binned (default quintile) calibration curve with consistency
    bands from binomial quantiles (Broecker & Smith): under perfect
    calibration the bin frequency lies within
    [binom.ppf(0.05, n, p_mean), binom.ppf(0.95, n, p_mean)] / n
    with ~90% probability. Returns one dict per non-empty bin:
    {"p_mean", "freq", "n", "lo", "hi"}."""
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda t: t[0])
    n_total = len(ordered)
    edges = [round(i * n_total / n_bins) for i in range(n_bins + 1)]
    out: list[dict] = []
    for i in range(n_bins):
        chunk = ordered[edges[i]:edges[i + 1]]
        if not chunk:
            continue
        n = len(chunk)
        p_mean = sum(p for p, _ in chunk) / n
        freq = sum(1.0 for _, o in chunk if o) / n
        lo = float(binom.ppf(0.05, n, p_mean)) / n
        hi = float(binom.ppf(0.95, n, p_mean)) / n
        out.append({"p_mean": p_mean, "freq": freq, "n": n, "lo": lo, "hi": hi})
    return out


def murphy_decomposition(pairs: list[Pair]) -> dict:
    """Murphy (1973) decomposition of the Brier score, grouped by unique
    forecast value: brier = uncertainty - resolution + reliability (exact
    identity under this grouping). Reliability ~ recalibration headroom;
    resolution ~ research/decisiveness investment signal (spec §6.3)."""
    if not pairs:
        raise ValueError("murphy_decomposition needs at least one pair")
    n = len(pairs)
    obar = sum(1.0 for _, o in pairs if o) / n
    groups: dict[float, list[bool]] = {}
    for p, o in pairs:
        groups.setdefault(float(p), []).append(o)
    reliability = 0.0
    resolution = 0.0
    for p, outcomes in groups.items():
        nk = len(outcomes)
        ok = sum(1.0 for o in outcomes if o) / nk
        reliability += nk * (p - ok) ** 2
        resolution += nk * (ok - obar) ** 2
    reliability /= n
    resolution /= n
    uncertainty = obar * (1.0 - obar)
    brier_score = sum((p - (1.0 if o else 0.0)) ** 2 for p, o in pairs) / n
    return {
        "brier": brier_score,
        "uncertainty": uncertainty,
        "resolution": resolution,
        "reliability": reliability,
    }


def discrimination_separation(pairs: list[Pair]) -> float:
    """Mean forecast on Yes-resolvers minus mean forecast on No-resolvers.
    NaN when either side is empty."""
    yes = [p for p, o in pairs if o]
    no = [p for p, o in pairs if not o]
    if not yes or not no:
        return float("nan")
    return sum(yes) / len(yes) - sum(no) / len(no)
