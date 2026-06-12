"""Synthetic numeric questions generated from target series (Retrocast
extension 4 seed; ForecastBench dataset-question recipe kept numeric).

Each question asks "what will <series> be on <date>?" with the as-of anchor
T0 sampled inside the series, resolution taken from the series itself at the
close date, and scaling bounds derived from the trailing window. Leak-free
for non-LLM systems by construction (the series is dated); LLM use must
stratify by contamination as usual — these measure distribution PLUMBING,
not fine-print reading (confirm winners on the real Metaculus stratum).
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore

MIN_HISTORY = 30          # points required before the anchor
BOUND_MARGIN = 0.25       # bounds = trailing-window range ± margin * span


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def generate_questions(store: CorpusStore, source_qid: str, *, n: int,
                       horizon_days: float, seed: int = 0,
                       name: str | None = None,
                       lookback_days: float = 365.0) -> list[BacktestQuestion]:
    """Generate up to n resolved synthetic numeric questions from the
    source's target series, upserting questions + copied series into the
    store. Returns the generated questions (possibly fewer than n when the
    series is too short)."""
    pts = [(_parse(ts), float(v))
           for ts, v in store.series_points(source_qid, "target")]
    if not pts:
        return []
    label = name or source_qid

    # anchor candidates: enough history before, a resolvable point after
    horizon = timedelta(days=horizon_days)
    candidates = []
    for i in range(MIN_HISTORY, len(pts)):
        t0 = pts[i][0]
        if any(t >= t0 + horizon for t, _ in pts[i:]):
            candidates.append(i)
    if not candidates:
        return []
    rng = random.Random(seed)
    chosen = sorted(rng.sample(candidates, min(n, len(candidates))))

    out: list[BacktestQuestion] = []
    for i in chosen:
        t0, _ = pts[i]
        close = t0 + horizon
        # resolution: step-function value at close (last point <= close)
        resolution = None
        for t, v in pts:
            if t <= close:
                resolution = v
            else:
                break
        window_pts = [(t, v) for t, v in pts[: i + 1]
                      if t >= t0 - timedelta(days=lookback_days)]
        window = [v for _, v in window_pts]
        lo, hi = min(window), max(window)
        span = max(hi - lo, abs(hi) * 0.05, 1e-9)
        # widen for trend: bounds must usually contain the (unseen) outcome,
        # so project the in-window drift over the horizon — history-only, no
        # peeking at the resolution
        w_days = max((window_pts[-1][0] - window_pts[0][0]).total_seconds()
                     / 86400.0, 1.0)
        drift = (window_pts[-1][1] - window_pts[0][1]) / w_days
        trend_pad = 2.0 * abs(drift) * horizon_days
        # midnight anchors keep the short date-only tag (no cache churn for
        # daily series); intraday anchors need the time to stay unique
        ts_tag = (t0.date().isoformat()
                  if (t0.hour, t0.minute, t0.second) == (0, 0, 0)
                  else t0.strftime("%Y-%m-%dT%H%M%S"))
        q = BacktestQuestion(
            qid=f"synthetic:{source_qid.replace(':', '_')}"
                f"-{ts_tag}-{int(horizon_days)}d",
            platform="synthetic", native_id=f"{source_qid}-{i}",
            qtype=QType.NUMERIC,
            title=(f"What will {label} be on {close.date().isoformat()}? "
                   "(synthetic)"),
            description=(f"Synthetic dataset question generated from the "
                         f"{label} series. Resolves to the last observed "
                         f"value on or before {close.date().isoformat()}."),
            resolution_criteria=(f"Resolves to the value of {label} at "
                                 f"{close.date().isoformat()} (last "
                                 "observation on or before that date)."),
            # open_time = t0, NOT earlier: scaling bounds are computed from
            # data through t0, and horizon_to_asof can place as_of anywhere
            # in [open_time, close) — an earlier open would let bounds leak
            # post-as_of observations into the prompt and the scored grid
            open_time=t0,
            scheduled_close_time=close,
            actual_resolve_time=close,
            resolution=resolution,
            scaling={"range_min": lo - BOUND_MARGIN * span - trend_pad,
                     "range_max": hi + BOUND_MARGIN * span + trend_pad,
                     "open_lower_bound": True, "open_upper_bound": True,
                     "cdf_size": 201},
            provenance={"synthetic": True, "source_qid": source_qid,
                        "generator": "v1", "horizon_days": horizon_days})
        store.upsert_question(q)
        # copy the series so TimeSeriesSystem can address it by the new qid
        store.put_series(q.qid, "target",
                         [(t.isoformat(), v) for t, v in pts])
        out.append(q)
    return out
