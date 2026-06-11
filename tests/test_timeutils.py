from datetime import datetime, timezone

import pytest

from bot.timeutils import days_until, laplace_rate, remaining_window_probability


def test_days_until():
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    close = datetime(2026, 6, 11, tzinfo=timezone.utc)
    assert days_until(close, now) == pytest.approx(10.0)


def test_laplace_rule_of_succession():
    assert laplace_rate(0, 0) == pytest.approx(0.5)
    assert laplace_rate(1, 1) == pytest.approx(2 / 3)
    assert laplace_rate(0, 10) == pytest.approx(1 / 12)
    with pytest.raises(ValueError):
        laplace_rate(5, 3)


def test_constant_hazard_burndown():
    # P=0.5 over 100 days; 50 days left -> 1 - 0.5**0.5 ~= 0.2929
    assert remaining_window_probability(0.5, 100, 50) == pytest.approx(0.2929, abs=1e-3)
    assert remaining_window_probability(0.5, 100, 100) == pytest.approx(0.5)
    assert remaining_window_probability(0.5, 100, 0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        remaining_window_probability(0.5, 0, 0)
