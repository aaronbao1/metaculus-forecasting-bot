"""Synthetic opponent constructors (spec §6.2): CP-at-T and
market-price-at-T. Pure constructors — looking the values up at time T
(CP history intervals, market candles) is the corpus layer's job.

cp_opponent accepts the three shapes a community prediction takes:
a scalar probability (binary), an option->probability dict (MC), or a CDF
vector (numeric/discrete). market_opponent is binary-only by contract: a
market price is a single Yes-probability, and MC/numeric questions have no
single-price equivalent — anything but a scalar raises.
"""
from __future__ import annotations

from numbers import Real

from backtest.core import Forecast, QType

_PROB_FLOOR = 0.001
_PROB_CEIL = 0.999


def _clamped_binary(p: float) -> Forecast:
    return Forecast(
        qtype=QType.BINARY,
        binary_prob=min(max(float(p), _PROB_FLOOR), _PROB_CEIL),
    )


def cp_opponent(value) -> Forecast:
    """Community-prediction-at-T opponent.

    - float in [0, 1]            -> binary forecast
    - dict option -> prob (~1)   -> multiple_choice forecast
    - sequence of floats         -> numeric forecast (ascending CDF in [0, 1])
    """
    if isinstance(value, bool):
        raise TypeError("cp value must be a probability, dict, or cdf; got bool")
    if isinstance(value, Real):
        p = float(value)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"cp probability out of [0, 1]: {p}")
        return _clamped_binary(p)
    if isinstance(value, dict):
        probs = {str(k): float(v) for k, v in value.items()}
        total = sum(probs.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"cp mc probabilities sum to {total}, expected ~1")
        return Forecast(qtype=QType.MULTIPLE_CHOICE, mc_probs=probs)
    if isinstance(value, (list, tuple)):
        cdf = tuple(float(v) for v in value)
        if len(cdf) < 2:
            raise ValueError("cp cdf needs at least 2 points")
        if cdf[0] < 0.0 or cdf[-1] > 1.0 or any(
            b < a for a, b in zip(cdf, cdf[1:])
        ):
            raise ValueError("cp cdf must be ascending within [0, 1]")
        return Forecast(qtype=QType.NUMERIC, cdf=cdf)
    raise TypeError(f"unsupported cp value type: {type(value).__name__}")


def market_opponent(price: float) -> Forecast:
    """Market-price-at-T opponent. Binary only (contract): raises TypeError
    for vectors/dicts, ValueError for prices outside [0, 1]."""
    if isinstance(price, bool) or not isinstance(price, Real):
        raise TypeError(
            f"market price must be a number, got {type(price).__name__}")
    p = float(price)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"market price out of [0, 1]: {p}")
    return _clamped_binary(p)
