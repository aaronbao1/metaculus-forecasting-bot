"""TimeSeriesSystem: non-LLM submodels as ForecastSystems (plan item 2).

Forecasts numeric/discrete questions that have a kind="target" series in the
corpus store: history <= ctx.as_of (leak-free by construction — the series
was snapshotted/dated at ingestion) -> engine quantiles ->
percentiles_to_cdf. contamination="clean" because no LLM ever sees the
question (same rationale as the baselines). NEVER raises: every failure is
an error artifact.

Fitness calls on these systems are free (no LLM), which is what makes them
usable inside evolutionary/optimization loops (docs/backtesting/OPTIMIZE.md).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord)
from backtest.submodels.engines import get_engine
from backtest.timetravel.asof import AsOfContext

logger = logging.getLogger(__name__)

MIN_HISTORY_DAYS_NOTE = ("horizon uses scheduled_close_time; spot semantics "
                         "are the runner's concern")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


class TimeSeriesSystem:
    """ForecastSystem over target series with a pluggable engine."""

    def __init__(self, engine: str = "naive_drift",
                 engine_params: dict | None = None,
                 series_kind: str = "target",
                 name: str | None = None):
        self.engine_name = engine
        self.engine_params = dict(engine_params or {})
        self.series_kind = series_kind
        self.name = name or f"ts_{engine}"

    def config_fingerprint(self) -> str:
        blob = json.dumps({"engine": self.engine_name,
                           "params": self.engine_params,
                           "kind": self.series_kind}, sort_keys=True)
        return "ts-" + hashlib.sha256(blob.encode()).hexdigest()[:12]

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        stages: list[StageRecord] = []
        try:
            fc = self._forecast_inner(q, ctx, stages)
            return self._artifact(q, ctx, fc, stages)
        except Exception as e:
            logger.warning("timeseries forecast failed for %s: %s", q.qid, e)
            return self._artifact(q, ctx, None, stages,
                                  error=f"{type(e).__name__}: {e}")

    def _forecast_inner(self, q: BacktestQuestion, ctx: AsOfContext,
                        stages: list[StageRecord]) -> Forecast:
        if q.qtype not in (QType.NUMERIC, QType.DISCRETE):
            raise ValueError(f"unsupported qtype for timeseries: "
                             f"{q.qtype.value}")
        scaling = q.scaling or {}
        if "range_min" not in scaling or "range_max" not in scaling:
            raise ValueError(f"numeric question {q.qid} lacks scaling bounds")
        if int(scaling.get("cdf_size") or 201) != 201:
            raise ValueError("cdf_size != 201 unsupported (parity with "
                             "ensemble MVP)")
        if scaling.get("zero_point") is not None:
            raise ValueError("zero_point (log-scaled) unsupported (parity "
                             "with ensemble MVP)")
        if ctx.store is None:
            raise ValueError("AsOfContext.store is not set")
        raw = ctx.store.series_points(q.qid, self.series_kind,
                                      until=ctx.as_of)
        if not raw:
            raise ValueError(f"no target series for {q.qid} at/before as_of")
        points = []
        for ts, v in raw:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            points.append((_to_utc(dt), float(v)))
        if q.scheduled_close_time is None:
            raise ValueError("question has no scheduled_close_time")
        horizon_days = ((_to_utc(q.scheduled_close_time) - ctx.as_of)
                        .total_seconds() / 86400.0)
        if horizon_days <= 0:
            raise ValueError(f"non-positive horizon ({horizon_days:.1f}d)")

        started = _now_iso()
        engine = get_engine(self.engine_name, **self.engine_params)
        ef = engine.fit_predict(points, horizon_days)
        stages.append(StageRecord(
            stage="submodel",
            inputs={"engine": self.engine_name,
                    "params": self.engine_params,
                    "n_points": len(points),
                    "horizon_days": round(horizon_days, 3),
                    "series_kind": self.series_kind},
            outputs={"quantiles": {str(p): v
                                   for p, v in ef.quantiles.items()},
                     "engine_params": ef.params},
            started_at=started, ended_at=_now_iso()))

        from bot.distributions import percentiles_to_cdf
        cdf = percentiles_to_cdf(
            dict(ef.quantiles), scaling["range_min"], scaling["range_max"],
            bool(scaling.get("open_lower_bound")),
            bool(scaling.get("open_upper_bound")))
        return Forecast(qtype=q.qtype, cdf=tuple(float(v) for v in cdf))

    def _artifact(self, q: BacktestQuestion, ctx: AsOfContext,
                  fc: Forecast | None, stages: list[StageRecord],
                  error: str | None = None) -> ForecastArtifact:
        return ForecastArtifact(
            qid=q.qid, horizon="", as_of=ctx.as_of, system_name=self.name,
            config_fingerprint=self.config_fingerprint(), forecast=fc,
            stages=stages, contamination="clean",
            cutoff_mode=ctx.cutoff_mode, error=error)
