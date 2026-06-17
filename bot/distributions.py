"""LLM percentiles -> validated, PCHIP-smoothed, Metaculus-compliant CDF.

Bots' worst deficit vs Pros is numeric questions (-23.2 head-to-head); much of
it is plumbing. This module: (1) repairs non-monotone elicited percentiles
(~30% of naive runs fail platform validation), (2) interpolates with PCHIP
(shape-preserving; the official template's linear interpolation wastes
information), (3) enforces the Metaculus API constraint set: 201 points,
monotone, step in [5e-05, 0.2], closed bounds pinned to 0/1, open bounds
keep >= 0.001 mass outside.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

CDF_SIZE = 201
# Metaculus' NumericDistribution rejects any CDF step < 5e-05 (strict, no
# tolerance). A step pinned at exactly 5e-05 round-trips through float64 to
# 4.9999...e-05 (e.g. (0.001 + 5e-05) - 0.001) and is rejected on submission,
# so we ENFORCE a hair above the platform floor — the margin (1e-6) dwarfs the
# ~1e-16 round-trip noise yet is immaterial to tail mass.
PLATFORM_MIN_STEP = 5.0e-05
MIN_STEP = PLATFORM_MIN_STEP + 1.0e-6
# Symmetric ceiling: Metaculus rejects any bin with mass > 0.2 (strict). A bin
# clipped to exactly 0.2 round-trips to 0.2000...07 > 0.2 and is rejected, so we
# ENFORCE a hair below the platform ceiling (margin immaterial to the forecast).
PLATFORM_MAX_STEP = 0.2
MAX_STEP = PLATFORM_MAX_STEP - 1.0e-6
MIN_OUTSIDE = 0.001


def repair_percentiles(declared: dict, range_min: float, range_max: float) -> dict:
    """Sort by percentile; clamp values inside the range (with margin); force
    strictly increasing values by minimal jitter. Raises if unrepairable."""
    if not declared:
        raise ValueError("no percentiles declared")
    span = range_max - range_min
    if span <= 0:
        raise ValueError("range_max must exceed range_min")
    eps = max(span * 1e-6, 1e-12)
    lo, hi = range_min + eps, range_max - eps
    items = sorted(declared.items())
    values = [min(max(float(v), lo), hi) for _, v in items]
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] + eps
    values[-1] = min(values[-1], hi)
    for i in range(len(values) - 2, -1, -1):
        if values[i] >= values[i + 1]:
            values[i] = values[i + 1] - eps
    if values[0] < lo - 1e-9:
        raise ValueError("percentiles cannot fit inside question range")
    return {pct: v for (pct, _), v in zip(items, values)}


def percentiles_to_cdf(declared: dict, range_min: float, range_max: float,
                       open_lower: bool, open_upper: bool) -> np.ndarray:
    d = repair_percentiles(declared, range_min, range_max)
    first = MIN_OUTSIDE if open_lower else 0.0
    last = (1.0 - MIN_OUTSIDE) if open_upper else 1.0
    xs = [range_min] + list(d.values()) + [range_max]
    ys = [first] + [min(max(p / 100.0, first + 1e-4), last - 1e-4)
                    for p in d.keys()] + [last]
    for i in range(1, len(ys)):                 # strictness after clamping
        if ys[i] <= ys[i - 1]:
            ys[i] = ys[i - 1] + 1e-6
    interp = PchipInterpolator(xs, ys)
    grid = np.linspace(range_min, range_max, CDF_SIZE)
    return standardize_cdf(interp(grid), open_lower, open_upper)


def standardize_cdf(cdf, open_lower: bool, open_upper: bool) -> np.ndarray:
    c = np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0)
    c = np.maximum.accumulate(c)
    first = MIN_OUTSIDE if open_lower else 0.0
    last = (1.0 - MIN_OUTSIDE) if open_upper else 1.0
    interior = (c - c[0]) / max(c[-1] - c[0], 1e-12)   # rescale between pins
    c = first + interior * (last - first)
    d = np.diff(c)
    total = last - first
    for _ in range(10):                                # step bounds + mass conservation
        d = np.clip(d, MIN_STEP, MAX_STEP)
        excess = d.sum() - total
        if abs(excess) < 1e-12:
            break
        if excess > 0:
            room = d - MIN_STEP
            d = d - room * min(excess / max(room.sum(), 1e-12), 1.0)
        else:
            room = MAX_STEP - d
            d = d + room * min(-excess / max(room.sum(), 1e-12), 1.0)
    c = np.concatenate([[first], first + np.cumsum(d)])
    c[-1] = last
    validate_cdf(c, open_lower, open_upper)
    return c


def validate_cdf(c, open_lower: bool, open_upper: bool) -> None:
    c = np.asarray(c, dtype=float)
    if len(c) != CDF_SIZE:
        raise ValueError(f"cdf must have {CDF_SIZE} points, got {len(c)}")
    d = np.diff(c)
    if d.min() < PLATFORM_MIN_STEP:        # constraints the platform enforces (strict, no slack)
        raise ValueError(f"cdf step below platform minimum {PLATFORM_MIN_STEP}: {d.min()}")
    if d.max() > PLATFORM_MAX_STEP:
        raise ValueError(f"cdf step above platform maximum {PLATFORM_MAX_STEP}: {d.max()}")
    lo_ok = c[0] >= MIN_OUTSIDE - 1e-9 if open_lower else abs(c[0]) < 1e-9
    hi_ok = c[-1] <= 1 - MIN_OUTSIDE + 1e-9 if open_upper else abs(c[-1] - 1) < 1e-9
    if not (lo_ok and hi_ok):
        raise ValueError("cdf endpoint violates bound rules")
