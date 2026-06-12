"""Time/hazard math for deadline ('will X happen by DATE') questions."""
from __future__ import annotations

from datetime import datetime, timezone


def days_until(close_time: datetime, now: datetime | None = None) -> float:
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (close_time - now).total_seconds() / 86400.0


def laplace_rate(k: int, n: int) -> float:
    """Rule of succession: P(event in next period) = (k+1)/(n+2)."""
    if k < 0 or n < 0 or k > n:
        raise ValueError("need 0 <= k <= n")
    return (k + 1) / (n + 2)


def remaining_window_probability(p_full_window: float, total_days: float,
                                 remaining_days: float) -> float:
    """Constant-hazard burn-down. Solves (1-r)^T = 1-P for the per-day hazard,
    then returns P over the remaining window: 1-(1-P)^(t/T). Absent news, a
    deadline question's probability must decay — sitting still implicitly
    raises the hazard estimate daily."""
    if total_days <= 0:
        raise ValueError("total_days must be positive")
    p = min(max(p_full_window, 0.0), 1.0 - 1e-9)
    t = min(max(remaining_days, 0.0), total_days)
    return 1.0 - (1.0 - p) ** (t / total_days)
