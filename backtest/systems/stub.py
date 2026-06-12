"""Deterministic no-network ForecastSystem for the CI smoke gate.

Fixed outputs: binary 0.6, MC uniform over options, numeric a uniform
201-point CDF. No research, no LLM calls (ctx.llm is never touched), no
randomness — so the smoke run's scores are bit-stable and the committed
regression baseline (tests/backtest/fixtures/smoke_baseline.json) is exact.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord)
from backtest.timetravel.asof import AsOfContext

_BINARY_P = 0.6
_CDF_SIZE = 201


class StubSystem:
    name = "stub"

    def config_fingerprint(self) -> str:
        return hashlib.sha256(b"stub-v1").hexdigest()[:12]

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        if q.qtype == QType.BINARY:
            fc = Forecast(qtype=q.qtype, binary_prob=_BINARY_P)
        elif q.qtype == QType.MULTIPLE_CHOICE:
            n = len(q.options) or 1
            fc = Forecast(qtype=q.qtype,
                          mc_probs={o: 1.0 / n for o in q.options})
        else:  # NUMERIC / DISCRETE: uniform CDF, valid under 2026 constraints
            size = (q.scaling or {}).get("cdf_size", _CDF_SIZE)
            fc = Forecast(qtype=q.qtype,
                          cdf=tuple(i / (size - 1) for i in range(size)))
        now = datetime.now(timezone.utc).isoformat()
        return ForecastArtifact(
            qid=q.qid,
            horizon="",   # stamped by the experiment runner, which owns it
            as_of=ctx.as_of,
            system_name=self.name,
            config_fingerprint=self.config_fingerprint(),
            forecast=fc,
            stages=[StageRecord(stage="stub",
                                outputs={"note": "fixed deterministic forecast"},
                                started_at=now, ended_at=now)],
            contamination="clean",   # no model knowledge is used at all
            cutoff_mode=ctx.cutoff_mode,
        )
