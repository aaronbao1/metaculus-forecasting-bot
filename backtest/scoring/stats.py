"""Paired statistics and difficulty adjustment (spec §6.4).

All variant comparisons in the harness are paired on identical
(question, horizon) sets; these helpers consume the per-question paired
score differences."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as scipy_stats


def paired_bootstrap(diffs: list[float], n: int = 10_000, seed: int = 0) -> dict:
    """Percentile bootstrap 95% CI for the mean paired difference.
    Returns {"mean", "lo", "hi"}; deterministic for a given seed."""
    arr = np.asarray(diffs, dtype=float)
    if arr.size == 0:
        raise ValueError("paired_bootstrap needs at least one difference")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi)}


def paired_t(diffs: list[float]) -> dict:
    """Two-sided one-sample t-test of mean(diffs) == 0.
    Returns {"mean", "t", "p"}.

    When all diffs are zero the standard deviation is zero and scipy's
    ttest_1samp returns NaN for both t and p.  This function deliberately
    returns t=0.0, p=1.0 instead: a zero mean with zero spread is not
    evidence against the null, so p=1 is the correct frequentist answer.
    (all-zero diffs -> t=0, p=1, deliberately unlike scipy's NaN)
    """
    arr = np.asarray(diffs, dtype=float)
    if arr.size < 2:
        raise ValueError("paired_t needs at least two differences")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        t = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        p = 0.0 if mean != 0 else 1.0
    else:
        t = mean / (sd / math.sqrt(arr.size))
        p = 2.0 * float(scipy_stats.t.sf(abs(t), arr.size - 1))
    return {"mean": mean, "t": t, "p": p}


def detectable_effect(n: int, sd: float) -> float:
    """Minimal detectable mean paired difference at ~2 sigma: 2 * sd / sqrt(n).
    Spec §6.4 power example: n=400, sd=0.1 -> 0.01. Reports must refuse to
    bold a winner whose mean difference is inside this band."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 2.0 * float(sd) / math.sqrt(n)


def _panel_components(scores: dict) -> int:
    """Return the number of connected components in the bipartite
    system-question graph implied by *scores*.

    Nodes are systems and questions; an edge exists for every (system, qid)
    key.  We use union-find so the traversal is O(N α(N)) on the key count.
    """
    parent: dict = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, qid in scores:
        # Namespace system/question nodes so identical strings don't collide.
        union(("s", s), ("q", qid))

    roots = {find(("s", s)) for s, _ in scores}
    roots |= {find(("q", qid)) for _, qid in scores}
    return len(roots)


def difficulty_adjust(scores: dict) -> dict:
    """Two-way fixed effects b_ij = alpha_i + gamma_j + eps (system i,
    question j), solved with np.linalg.lstsq (minimum-norm solution for the
    rank-deficient dummy design). Question effects gamma are centered to mean
    zero — the centered gamma is invariant to the alpha/gamma level ambiguity
    — and the adjusted score is b~ = b - gamma_hat (spec §6.4, ForecastBench
    updated difficulty-adjustment methodology).

    `scores` maps (system, qid) -> score; returns the same keys.

    Raises ValueError if the bipartite system-question graph has more than one
    connected component (disconnected panel): in that case the lstsq solution
    is under-identified across components and the returned adjustments would be
    fabricated relative offsets.  Transitively connected panels (e.g. A shares
    q2 with B, B shares q3 with C) are accepted.
    """
    if not scores:
        return {}
    n_components = _panel_components(scores)
    if n_components > 1:
        raise ValueError(
            f"difficulty_adjust requires a connected system-question panel; "
            f"got {n_components} components"
        )
    systems = sorted({s for s, _ in scores})
    questions = sorted({qid for _, qid in scores})
    s_index = {s: i for i, s in enumerate(systems)}
    q_index = {qid: i for i, qid in enumerate(questions)}
    keys = list(scores)
    design = np.zeros((len(keys), len(systems) + len(questions)))
    y = np.zeros(len(keys))
    for row, (s, qid) in enumerate(keys):
        design[row, s_index[s]] = 1.0
        design[row, len(systems) + q_index[qid]] = 1.0
        y[row] = float(scores[(s, qid)])
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    gamma = theta[len(systems):]
    gamma = gamma - gamma.mean()
    return {
        (s, qid): float(scores[(s, qid)] - gamma[q_index[qid]])
        for s, qid in keys
    }
