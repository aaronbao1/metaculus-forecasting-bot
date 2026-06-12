"""Floor systems every report contextualizes variants against (spec §5.3):
chance (0.5 / uniform / uniform CDF), CP echo, market echo.

No LLM, no research — artifacts carry empty stages and contamination="clean".
Echo baselines read the corpus series at ctx.as_of and skip (forecast=None,
error="no data: ...") when nothing is available; they never raise.
"""
from __future__ import annotations

import logging

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType)
from backtest.corpus.crosswalk import market_for
from backtest.scoring.opponents import cp_opponent, market_opponent
from backtest.timetravel.asof import AsOfContext

logger = logging.getLogger(__name__)

# canonical baseline system names (runner factories and the failure
# taxonomy's default exclusion both key on this)
BASELINE_NAMES = frozenset({"chance", "cp_echo", "market_echo"})


def _artifact(q: BacktestQuestion, ctx: AsOfContext, name: str,
              fingerprint: str, fc: Forecast | None,
              error: str | None = None) -> ForecastArtifact:
    return ForecastArtifact(
        qid=q.qid, horizon="", as_of=ctx.as_of, system_name=name,
        config_fingerprint=fingerprint, forecast=fc,
        contamination="clean", cutoff_mode=ctx.cutoff_mode, error=error)


class ChanceBaseline:
    name = "chance"

    def config_fingerprint(self) -> str:
        return "chance-v1"

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        if q.qtype is QType.BINARY:
            fc = Forecast(qtype=QType.BINARY, binary_prob=0.5)
        elif q.qtype is QType.MULTIPLE_CHOICE:
            n = len(q.options)
            if n == 0:
                return _artifact(q, ctx, self.name, "chance-v1", None,
                                 error="no data: MC question has no options")
            fc = Forecast(qtype=q.qtype,
                          mc_probs={o: 1.0 / n for o in q.options})
        else:
            scaling = q.scaling or {}
            open_lo = bool(scaling.get("open_lower_bound"))
            open_hi = bool(scaling.get("open_upper_bound"))
            size = int(scaling.get("cdf_size") or 201)
            first = 0.001 if open_lo else 0.0
            last = 0.999 if open_hi else 1.0
            step = (last - first) / (size - 1)
            fc = Forecast(qtype=q.qtype,
                          cdf=tuple(first + i * step for i in range(size)))
        return _artifact(q, ctx, self.name, "chance-v1", fc)


class CPEchoBaseline:
    name = "cp_echo"

    def __init__(self, store):
        self.store = store

    def config_fingerprint(self) -> str:
        return "cp-echo-v1"

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        try:
            v = self.store.value_at(q.qid, "cp", ctx.as_of)
            if v is None:
                return _artifact(q, ctx, self.name, "cp-echo-v1", None,
                                 error="no data: no CP at as_of")
            fc = cp_opponent(v)
        except (ValueError, TypeError) as e:
            logger.warning("cp value unusable for %s: %s", q.qid, e)
            return _artifact(q, ctx, self.name, "cp-echo-v1", None,
                             error=f"no data: cp value unusable: {e}")
        except Exception as e:
            logger.warning("cp lookup failed for %s: %s", q.qid, e)
            return _artifact(q, ctx, self.name, "cp-echo-v1", None,
                             error=f"no data: cp lookup failed: {e}")
        return _artifact(q, ctx, self.name, "cp-echo-v1", fc)


class MarketEchoBaseline:
    name = "market_echo"

    def __init__(self, store):
        self.store = store

    def config_fingerprint(self) -> str:
        return "market-echo-v1"

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        try:
            mqid = market_for(self.store, q.qid)
            price = (self.store.value_at(mqid, "market_price", ctx.as_of)
                     if mqid else None)
        except Exception as e:
            logger.warning("market lookup failed for %s: %s", q.qid, e)
            return _artifact(q, ctx, self.name, "market-echo-v1", None,
                             error=f"no data: market lookup failed: {e}")
        if price is None:
            return _artifact(
                q, ctx, self.name, "market-echo-v1", None,
                error="no data: no crosswalked market price at as_of")
        return _artifact(q, ctx, self.name, "market-echo-v1",
                         market_opponent(float(price)))
