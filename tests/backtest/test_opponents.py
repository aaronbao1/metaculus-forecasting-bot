"""cp_opponent / market_opponent constructors (spec §6.2). Pure constructors:
data lookup (CP history, market price at T) lives in the corpus layer."""
import math

import pytest

from backtest.core import BacktestQuestion, QType
from backtest.scoring.metaculus_scores import log_score
from backtest.scoring.opponents import cp_opponent, market_opponent


def test_cp_opponent_binary_scalar():
    f = cp_opponent(0.62)
    assert f.qtype is QType.BINARY
    assert f.binary_prob == pytest.approx(0.62)
    assert f.mc_probs is None and f.cdf is None


def test_cp_opponent_clamps_degenerate_probabilities():
    # constructors maintain the Forecast invariant binary_prob in (0, 1)
    assert cp_opponent(0.0).binary_prob == pytest.approx(0.001)
    assert cp_opponent(1.0).binary_prob == pytest.approx(0.999)


def test_cp_opponent_mc_dict_vector():
    f = cp_opponent({"a": 0.5, "b": 0.3, "c": 0.2})
    assert f.qtype is QType.MULTIPLE_CHOICE
    assert f.mc_probs == {"a": 0.5, "b": 0.3, "c": 0.2}


def test_cp_opponent_mc_bad_sum_raises():
    with pytest.raises(ValueError):
        cp_opponent({"a": 0.5, "b": 0.2})


def test_cp_opponent_numeric_cdf_vector():
    cdf = [i / 200 for i in range(201)]
    f = cp_opponent(cdf)
    assert f.qtype is QType.NUMERIC
    assert f.cdf == tuple(cdf)


def test_cp_opponent_rejects_invalid_values():
    with pytest.raises(TypeError):
        cp_opponent("0.6")
    with pytest.raises(TypeError):
        cp_opponent(True)          # bool is not a probability
    with pytest.raises(TypeError):
        cp_opponent(None)
    with pytest.raises(ValueError):
        cp_opponent(1.7)           # probability out of [0, 1]
    with pytest.raises(ValueError):
        cp_opponent([0.5, 0.2, 0.9])   # cdf not ascending
    with pytest.raises(ValueError):
        cp_opponent([0.5])         # cdf too short


def test_market_opponent_binary_scalar():
    f = market_opponent(0.35)
    assert f.qtype is QType.BINARY
    assert f.binary_prob == pytest.approx(0.35)


def test_market_opponent_raises_on_non_binary_values():
    # contract: market price is binary-only — vectors/dicts must raise
    with pytest.raises(TypeError):
        market_opponent({"a": 0.5, "b": 0.5})
    with pytest.raises(TypeError):
        market_opponent([0.1, 0.9])
    with pytest.raises(TypeError):
        market_opponent(True)
    with pytest.raises(ValueError):
        market_opponent(1.5)


def test_market_opponent_clamps_settled_prices():
    assert market_opponent(0.0).binary_prob == pytest.approx(0.001)
    assert market_opponent(1.0).binary_prob == pytest.approx(0.999)


# ---------------------------------------------------------------------------
# Hardening tests (NaN CDF, negative MC probs, flat CDF acceptance)
# ---------------------------------------------------------------------------

def test_cp_opponent_rejects_nan_in_cdf():
    """CDF containing NaN must raise — NaN comparisons silently pass ascending
    check, poisoning downstream scores."""
    with pytest.raises(ValueError, match="finite"):
        cp_opponent([0.0, float("nan"), 1.0])


def test_cp_opponent_rejects_inf_in_cdf():
    """CDF containing inf must also be rejected by the finiteness guard."""
    with pytest.raises(ValueError, match="finite"):
        cp_opponent([0.0, float("inf"), 1.0])


def test_cp_opponent_rejects_negative_mc_probability():
    """MC dict with a negative value (e.g. {a: 1.2, b: -0.2}) sums to ~1 but
    contains an illegal negative probability."""
    with pytest.raises(ValueError, match="non-negative"):
        cp_opponent({"a": 1.2, "b": -0.2})


def test_cp_opponent_accepts_flat_cdf_plateau():
    """A flat (plateau) CDF — e.g. [0.0, 0.4, 0.4, 1.0] — is non-decreasing
    and must be ACCEPTED (locks the non-strict inequality semantics)."""
    f = cp_opponent([0.0, 0.4, 0.4, 1.0])
    assert f.qtype is QType.NUMERIC
    assert f.cdf == (0.0, 0.4, 0.4, 1.0)


def test_opponents_are_scoreable():
    q = BacktestQuestion(
        qid="polymarket:x", platform="polymarket", native_id="x",
        qtype=QType.BINARY, title="Will it?", resolution=True)
    assert log_score(market_opponent(0.35), q) == pytest.approx(math.log(0.35))
    assert log_score(cp_opponent(0.62), q) == pytest.approx(math.log(0.62))
