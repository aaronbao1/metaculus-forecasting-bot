"""The compatibility contract every system under test implements (spec §5).

Semantics (mirrored in docs/backtesting/COMPATIBILITY.md):
- honor ctx.as_of — no wall-clock reads;
- route every model call through ctx.llm (cache/probe instrumentation);
- no unledgered network research;
- return a ForecastArtifact with per-stage inputs/outputs populated;
- NEVER raise from forecast() — failures return an artifact with error set.

Core types are re-exported so future systems can depend on this one module.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           LedgerEntry, QType, StageRecord)
from backtest.timetravel.asof import AsOfContext

__all__ = ["ForecastSystem", "BacktestQuestion", "Forecast",
           "ForecastArtifact", "LedgerEntry", "QType", "StageRecord",
           "AsOfContext"]


@runtime_checkable
class ForecastSystem(Protocol):
    name: str

    def config_fingerprint(self) -> str:
        ...

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        ...
