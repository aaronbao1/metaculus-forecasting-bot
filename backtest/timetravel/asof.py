"""As-of context and horizon semantics. T is always anchored to
scheduled_close_time / spot_scoring_time — never actual_resolve_time
(anchoring to resolution leaks 'this resolves about now'; Halawi's protocol
deliberately avoids it). Every provider records what it surfaced into the
context's evidence ledger, which makes leakage auditable per question."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from backtest.core import BacktestQuestion, LedgerEntry

if TYPE_CHECKING:
    from backtest.corpus.store import CorpusStore
    from backtest.timetravel.llm_cache import CachedLLM


@dataclass
class AsOfContext:
    as_of: datetime
    cutoff_mode: str = "strict"          # "strict" | "pragmatic"
    ledger: list[LedgerEntry] = field(default_factory=list)
    llm: "CachedLLM | None" = None
    store: "CorpusStore | None" = None

    def record(self, e: LedgerEntry) -> None:
        self.ledger.append(e)


# Set by the adapter per question; providers read it at call time. Default
# None means "not inside the backtest harness" — historical providers refuse
# to run.
ACTIVE_CONTEXT: ContextVar[AsOfContext | None] = ContextVar(
    "ACTIVE_CONTEXT", default=None)

_CLOSE_DELTAS = {"close-7d": 7, "close-30d": 30, "close-365d": 365}
_GEO_N = 5


def _to_utc(dt: datetime) -> datetime:
    """Normalise a possibly-naive datetime to UTC.

    Store convention: naive = UTC (62 corpus questions have naive open_time
    + aware scheduled_close_time; mixing them causes TypeError in subtraction).
    """
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def horizon_to_asof(q: BacktestQuestion, horizon: str) -> datetime | None:
    """Map a horizon name to T for this question; None = skip this horizon.

    close-Nd: scheduled_close_time - N days; skipped (None) when that lands
    before open_time. spot: spot_scoring_time. geo-k (k=1.._GEO_N): Halawi
    geometric retrieval dates, T_k = open + (span_days - 1)^(k/n) days.

    All datetime arithmetic is done on UTC-normalised copies so that mixed
    naive/aware question datetimes (old corpus rows) never raise TypeError.
    """
    if horizon in _CLOSE_DELTAS:
        if q.scheduled_close_time is None:
            return None
        close = _to_utc(q.scheduled_close_time)
        t = close - timedelta(days=_CLOSE_DELTAS[horizon])
        if q.open_time is not None and t < _to_utc(q.open_time):
            return None              # horizon predates the question; skip
        return t
    if horizon == "spot":
        return q.spot_scoring_time
    if horizon.startswith("geo-"):
        try:
            k = int(horizon.split("-", 1)[1])
        except ValueError:
            raise ValueError(f"unknown horizon {horizon!r}")
        if not 1 <= k <= _GEO_N:
            raise ValueError(f"geo-k supports k=1..{_GEO_N}, got {horizon!r}")
        if q.open_time is None or q.scheduled_close_time is None:
            return None
        open_utc = _to_utc(q.open_time)
        close_utc = _to_utc(q.scheduled_close_time)
        span_days = (close_utc - open_utc).total_seconds() / 86400.0
        if span_days <= 1.0:
            return None              # Halawi offset undefined; skip
        return open_utc + timedelta(days=(span_days - 1.0) ** (k / _GEO_N))
    raise ValueError(f"unknown horizon {horizon!r}")
