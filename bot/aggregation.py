"""Probability pooling for ensemble forecasts.

Default: trimmed geometric mean of odds (Samotsvety's method) — drop the most
extreme forecast on each side, then average in log-odds space. GMO beat
mean/median on ~850 resolved Metaculus questions (Sevilla 2021); trimming
fixes diversity-induced underconfidence (Jose et al. 2014).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def to_log_odds(probs) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def from_log_odds(lo) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(lo, dtype=float)))


def geometric_mean_of_odds(probs) -> float:
    return float(from_log_odds(np.mean(to_log_odds(probs))))


def median_pool(probs) -> float:
    return float(np.median(np.asarray(probs, dtype=float)))


def mean_pool(probs) -> float:
    return float(np.mean(np.asarray(probs, dtype=float)))


def trimmed_gmo(probs, n_trim: int = 1) -> float:
    """Drop n_trim lowest and highest forecasts, then GMO; plain GMO if too few."""
    p = np.sort(np.asarray(probs, dtype=float))
    if len(p) > 2 * n_trim + 1:
        p = p[n_trim:-n_trim]
    return geometric_mean_of_odds(p)


def extremize(p: float, a: float) -> float:
    """Karmarkar transform t(p) = p^a / (p^a + (1-p)^a). a=1 is identity.

    OFF by default: fitted extremization overfits in-sample and hurt on recent
    Metaculus data. Available for held-out experiments only.
    """
    p = float(np.clip(p, EPS, 1 - EPS))
    num = p ** a
    return float(num / (num + (1 - p) ** a))


def aggregate_binary(probs, method: str = "trimmed_gmo") -> float:
    methods = {
        "trimmed_gmo": trimmed_gmo,
        "gmo": geometric_mean_of_odds,
        "median": median_pool,
        "mean": mean_pool,
    }
    if method not in methods:
        raise ValueError(f"unknown aggregation method: {method}")
    probs = list(probs)
    if not probs:
        raise ValueError("no forecasts to aggregate")
    return methods[method](probs)


def aggregate_multiple_choice(option_prob_dicts, floor: float = 0.01) -> dict:
    """Per-option arithmetic mean, then floor every option and renormalize.

    'Good forecasters leave some moderate probability on most options.'
    """
    if not option_prob_dicts:
        raise ValueError("no forecasts to aggregate")
    options = list(option_prob_dicts[0].keys())
    means = {o: float(np.mean([d[o] for d in option_prob_dicts])) for o in options}
    floored = {o: max(v, floor) for o, v in means.items()}
    total = sum(floored.values())
    return {o: v / total for o, v in floored.items()}


def aggregate_cdfs(cdfs) -> np.ndarray:
    """Pointwise median across member CDFs (preserves monotonicity and the
    per-step bounds each member already satisfies)."""
    arr = np.stack([np.asarray(c, dtype=float) for c in cdfs])
    return np.median(arr, axis=0)
