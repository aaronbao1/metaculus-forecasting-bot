"""Pluggable time-series engines (incorporation plan item 2).

Contract: ``engine.fit_predict(points, horizon_days) -> EngineForecast`` where
points is ``[(datetime, float)]`` sorted ascending and the result's quantiles
map percentile (0-100) -> strictly increasing values. Consumers feed these
straight into bot.distributions.percentiles_to_cdf.

naive_drift is the always-available zero-dependency floor (random walk with
drift + empirical residual quantiles — the M-competition-respectable naive
baseline). statsforecast (theta/autoarima/autoets) and the TS foundation
models (chronos2/timesfm) are optional extras resolved lazily; a missing
package raises ImportError at fit_predict time, which TimeSeriesSystem
converts into an error artifact (never-raise contract lives there, not here).

Per the TSFM-in-finance literature (research-notes/07 §3a): point these at
indicator/count series, not martingale market prices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

PERCENTILES = (2.5, 10.0, 25.0, 50.0, 75.0, 90.0, 97.5)
MIN_POINTS = 8


@dataclass(frozen=True)
class EngineForecast:
    quantiles: dict[float, float]      # percentile (0-100) -> value, increasing
    engine: str
    params: dict = field(default_factory=dict)


def _to_arrays(points: list[tuple[datetime, float]]):
    if len(points) < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} points, got {len(points)}")
    ts = np.array([p[0].timestamp() for p in points]) / 86400.0   # days
    vals = np.array([float(p[1]) for p in points])
    if np.any(np.diff(ts) <= 0):
        order = np.argsort(ts)
        ts, vals = ts[order], vals[order]
    return ts, vals


def offsets_from_quantile_grid(quantiles: dict[float, float],
                               median: float) -> dict[float, float]:
    """Map an arbitrary quantile grid (e.g. TimesFM deciles 10..90) onto the
    PERCENTILES grid as offsets-from-median, interpolating interior points
    and clamping beyond the grid's ends. Without this, percentiles absent
    from the source grid would default to offset 0 in _finalize — collapsing
    them onto the median and (via the monotonicity pass) destroying the
    lower half of the distribution."""
    grid = sorted(quantiles)
    vals = [quantiles[p] for p in grid]
    return {p: float(np.interp(p, grid, vals)) - median
            for p in PERCENTILES}


def _finalize(median: float, offsets: dict[float, float], engine: str,
              params: dict, tail_widen: float) -> EngineForecast:
    """Apply tail widening around the median, enforce a minimum spread and
    strict monotonicity, and package the forecast."""
    eps = max(1e-9, 1e-6 * abs(median))
    q = {}
    for pct in PERCENTILES:
        off = offsets.get(pct, 0.0) * tail_widen
        floor_off = (pct - 50.0) / 47.5 * eps      # degenerate-safe spread
        q[pct] = median + (off if abs(off) > abs(floor_off) else floor_off)
    vals = [q[p] for p in PERCENTILES]
    for i in range(1, len(vals)):                  # strictness
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] + eps
    return EngineForecast(
        quantiles={p: float(v) for p, v in zip(PERCENTILES, vals)},
        engine=engine, params=dict(params, tail_widen=tail_widen))


class NaiveDriftEngine:
    """Random walk with drift; uncertainty from empirical horizon-step
    residuals of the drift model on history."""

    name = "naive_drift"

    def __init__(self, tail_widen: float = 1.0):
        self.tail_widen = float(tail_widen)

    def fit_predict(self, points, horizon_days: float) -> EngineForecast:
        ts, vals = _to_arrays(points)
        span = ts[-1] - ts[0]
        drift = (vals[-1] - vals[0]) / span if span > 0 else 0.0
        median = float(vals[-1] + drift * horizon_days)

        dt_med = float(np.median(np.diff(ts))) or 1.0
        steps = max(1, int(round(horizon_days / dt_med)))
        n = len(vals)
        if steps < n - 4:
            resid = vals[steps:] - (vals[:-steps] + drift * (ts[steps:] - ts[:-steps]))
        else:                                       # short series: scale 1-step
            one = vals[1:] - (vals[:-1] + drift * np.diff(ts))
            resid = one * np.sqrt(steps)
        offsets = {p: float(np.quantile(resid, p / 100.0)) for p in PERCENTILES}
        center = offsets[50.0]
        offsets = {p: v - center for p, v in offsets.items()}
        return _finalize(median, offsets, self.name, {}, self.tail_widen)


class StatsForecastEngine:
    """statsforecast models (optional extra `timeseries`). Irregular spacing
    is treated as regular steps of the median observed spacing."""

    _LEVELS = (50, 80, 95)
    _PCT_FROM = {2.5: ("lo", 95), 10.0: ("lo", 80), 25.0: ("lo", 50),
                 75.0: ("hi", 50), 90.0: ("hi", 80), 97.5: ("hi", 95)}

    def __init__(self, name: str, tail_widen: float = 1.0):
        self.name = name
        self.tail_widen = float(tail_widen)

    def _model(self):
        from statsforecast import models as sf_models
        cls = {"theta": sf_models.Theta,
               "autoarima": sf_models.AutoARIMA,
               "autoets": sf_models.AutoETS}[self.name]
        return cls()

    def fit_predict(self, points, horizon_days: float) -> EngineForecast:
        ts, vals = _to_arrays(points)
        dt_med = float(np.median(np.diff(ts))) or 1.0
        steps = max(1, int(round(horizon_days / dt_med)))
        model = self._model()
        out = model.forecast(y=vals, h=steps, level=list(self._LEVELS))
        median = float(np.asarray(out["mean"])[-1])
        offsets = {}
        for pct, (side, level) in self._PCT_FROM.items():
            arr = out.get(f"{side}-{level}")
            offsets[pct] = (float(np.asarray(arr)[-1]) - median
                            if arr is not None else 0.0)
        return _finalize(median, offsets, self.name, {"steps": steps},
                         self.tail_widen)


class ChronosEngine:
    """Amazon Chronos-Bolt / Chronos-2 (optional extra `tsfm`). Quantile-grid
    outputs map directly onto our percentile contract."""

    name = "chronos2"

    def __init__(self, model_id: str = "amazon/chronos-bolt-small",
                 tail_widen: float = 1.0):
        self.model_id = model_id
        self.tail_widen = float(tail_widen)

    def fit_predict(self, points, horizon_days: float) -> EngineForecast:
        import torch
        from chronos import BaseChronosPipeline
        ts, vals = _to_arrays(points)
        dt_med = float(np.median(np.diff(ts))) or 1.0
        steps = max(1, int(round(horizon_days / dt_med)))
        pipeline = BaseChronosPipeline.from_pretrained(
            self.model_id, device_map="cpu", torch_dtype=torch.float32)
        # Bolt's quantile head is trained on [0.1, 0.9]; out-of-range
        # requests are CLAMPED (95% interval would silently equal the 80%).
        # Request only the supported band; percentiles_to_cdf does the tails.
        inner = (10.0, 25.0, 50.0, 75.0, 90.0)
        quantiles, _mean = pipeline.predict_quantiles(
            context=torch.tensor(vals, dtype=torch.float32),
            prediction_length=steps,
            quantile_levels=[p / 100.0 for p in inner])
        last = quantiles[0, -1, :].tolist()
        median = float(last[inner.index(50.0)])
        offsets = {p: float(v) - median for p, v in zip(inner, last)}
        fc = _finalize(median, offsets, self.name,
                       {"model_id": self.model_id, "steps": steps},
                       self.tail_widen)
        kept = {p: v for p, v in fc.quantiles.items() if p in inner}
        return EngineForecast(quantiles=kept, engine=self.name,
                              params=fc.params)


class TimesFMEngine:
    """Google TimesFM 2.x (optional extra `tsfm`); deciles q10-q90 from the
    quantile head — percentiles_to_cdf supplies the tails beyond them."""

    name = "timesfm"

    def __init__(self, tail_widen: float = 1.0):
        self.tail_widen = float(tail_widen)

    def fit_predict(self, points, horizon_days: float) -> EngineForecast:
        import timesfm
        ts, vals = _to_arrays(points)
        dt_med = float(np.median(np.diff(ts))) or 1.0
        steps = max(1, int(round(horizon_days / dt_med)))
        model = timesfm.TimesFm_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch")
        model.compile(timesfm.ForecastConfig(
            max_context=max(64, len(vals)), max_horizon=steps,
            use_continuous_quantile_head=True, fix_quantile_crossing=True))
        point, quantiles = model.forecast(
            horizon=steps, inputs=[np.asarray(vals, dtype=np.float32)])
        # quantiles shape: (1, steps, 10) — mean + q10..q90 per release docs
        qrow = np.asarray(quantiles)[0, -1, :]
        deciles = {float(10 * i): float(qrow[i]) for i in range(1, 10)}
        median = deciles.get(50.0, float(np.asarray(point)[0, -1]))
        # interpolate the decile grid onto PERCENTILES (q25/q75 from
        # q20/q30 and q70/q80; q2.5/q97.5 clamp to q10/q90)
        offsets = offsets_from_quantile_grid(deciles, median)
        fc = _finalize(median, offsets, self.name, {"steps": steps},
                       self.tail_widen)
        # emit only the band the model genuinely supports (clamped 2.5/97.5
        # would be false precision); percentiles_to_cdf supplies the tails
        kept = {p: v for p, v in fc.quantiles.items()
                if p in (10.0, 25.0, 50.0, 75.0, 90.0)}
        return EngineForecast(quantiles=kept, engine=self.name,
                              params=fc.params)


ENGINES = {
    "naive_drift": NaiveDriftEngine,
    "theta": lambda **kw: StatsForecastEngine("theta", **kw),
    "autoarima": lambda **kw: StatsForecastEngine("autoarima", **kw),
    "autoets": lambda **kw: StatsForecastEngine("autoets", **kw),
    "chronos2": ChronosEngine,
    "timesfm": TimesFMEngine,
}


def get_engine(name: str, **params):
    factory = ENGINES.get(name)
    if factory is None:
        raise KeyError(f"unknown engine {name!r} "
                       f"(choose from {sorted(ENGINES)})")
    return factory(**params)
