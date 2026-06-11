"""CalibratedEnsembleAdapter: the live CalibratedEnsembleBot run under an
AsOfContext — research via bot.research (historical providers are installed
into its namespace by the harness), every LLM call routed through ctx.llm
(CachedLLM), full stage provenance recorded into the ForecastArtifact.

forecast() NEVER raises: any failure returns an artifact with error set and
forecast=None. The horizon label is unknown at this level — artifacts carry
horizon="" and the experiment runner stamps it.

Cost note: CachedLLM.invoke() returns only the completion string, so
member_llm StageRecords carry cost_usd=0 / tokens=0 for now; report cost
totals sum whatever StageRecords contain.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timezone

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord)
from backtest.corpus.model_releases import model_set_hash
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot import research as bot_research
from bot.config import BotConfig, config_hash
from bot.distributions import percentiles_to_cdf
from bot.ensemble_bot import CalibratedEnsembleBot

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_set_hash(cfg: BotConfig) -> str:
    """Delegate to the canonical corpus implementation so the adapter and
    the corpus stratifier always agree on the hash (model_set_hash is total
    over unknown slugs — RELEASE_DATES.get(m, "unknown"))."""
    return model_set_hash([m.model for m in cfg.ensemble])


def _question_dict(q: BacktestQuestion) -> dict:
    return dict(question_text=q.title,
                resolution_criteria=q.resolution_criteria,
                fine_print=q.fine_print,
                qid=q.qid)


def _common_fields(q: BacktestQuestion) -> dict:
    return dict(
        question_text=q.title,
        background_info=q.description,
        resolution_criteria=q.resolution_criteria,
        fine_print=q.fine_print,
        close_iso=(q.scheduled_close_time.isoformat()
                   if q.scheduled_close_time else ""))


class _BacktestEnsembleBot(CalibratedEnsembleBot):
    """Subclass whose only LLM seam is ctx.llm.

    The stock _fan_out (bot/ensemble_bot.py) does not pass a run index, so
    it is overridden here to enumerate (member, run_idx) pairs — the
    completion-cache key needs run_index to keep same-prompt same-member
    runs distinct."""

    def __init__(self, bot_config: BotConfig, ctx: AsOfContext,
                 stages: list[StageRecord]):
        super().__init__(bot_config=bot_config)
        self._ctx = ctx
        self._stages = stages

    async def _invoke_model(self, member_and_prompt) -> str:
        member, prompt = member_and_prompt
        return await self._invoke_member(member, prompt, 0)

    async def _invoke_member(self, member, prompt: str, run_idx: int) -> str:
        if self._ctx.llm is None:
            raise RuntimeError("AsOfContext.llm is not set")
        started = _now_iso()
        try:
            text = await self._ctx.llm.invoke(
                member.model, prompt, member.temperature, run_index=run_idx)
        except Exception as e:
            self._stages.append(StageRecord(
                stage="member_llm", model=member.model,
                inputs={"prompt": prompt, "temperature": member.temperature,
                        "run_index": run_idx},
                outputs={"error": f"{type(e).__name__}: {e}"},
                started_at=started, ended_at=_now_iso()))
            raise
        self._stages.append(StageRecord(
            stage="member_llm", model=member.model,
            inputs={"prompt": prompt, "temperature": member.temperature,
                    "run_index": run_idx},
            outputs={"completion": text},
            started_at=started, ended_at=_now_iso()))
        return text

    async def _fan_out(self, prompt: str):
        jobs, models = [], []
        for member in self.cfg.ensemble:
            for run_idx in range(member.runs):
                jobs.append(self._invoke_member(member, prompt, run_idx))
                models.append(member.model)
        results = await asyncio.gather(*jobs, return_exceptions=True)
        texts, kept = [], []
        for r, m in zip(results, models):
            if isinstance(r, BaseException):
                logger.warning("member %s failed: %s", m, r)
            else:
                texts.append(r)
                kept.append(m)
        return texts, kept


async def _forecast_binary(bot: _BacktestEnsembleBot, q: BacktestQuestion,
                           research_text: str,
                           stages: list[StageRecord]) -> Forecast:
    r = await bot.run_binary_ensemble(research=research_text,
                                      **_common_fields(q))
    stages.append(StageRecord(
        stage="aggregation",
        inputs={"member_probs": list(r.member_probs),
                "member_models": list(r.member_models),
                "method": bot.cfg.aggregation_method},
        outputs={"aggregate_raw": r.aggregate_raw, "spread": r.spread}))
    stages.append(StageRecord(
        stage="calibration",
        inputs={"aggregate_raw": r.aggregate_raw,
                "cap_lo": bot.cfg.cap_lo, "cap_hi": bot.cfg.cap_hi},
        outputs={"final": r.final}))
    return Forecast(qtype=q.qtype, binary_prob=r.final)


async def _forecast_mc(bot: _BacktestEnsembleBot, q: BacktestQuestion,
                       research_text: str,
                       stages: list[StageRecord]) -> Forecast:
    r = await bot.run_mc_ensemble(research=research_text,
                                  options=list(q.options),
                                  **_common_fields(q))
    stages.append(StageRecord(
        stage="aggregation",
        inputs={"member_dicts": list(r.member_dicts),
                "member_models": list(r.member_models),
                "mc_floor": bot.cfg.mc_floor},
        outputs={"final": dict(r.final)}))
    stages.append(StageRecord(
        stage="calibration",
        inputs={"note": "MC floor+renormalize happens in aggregation; "
                        "no Platt layer for MC"},
        outputs={"final": dict(r.final)}))
    return Forecast(qtype=q.qtype, mc_probs=dict(r.final))


async def _forecast_numeric(bot: _BacktestEnsembleBot, q: BacktestQuestion,
                            research_text: str,
                            stages: list[StageRecord]) -> Forecast:
    scaling = q.scaling or {}
    if "range_min" not in scaling or "range_max" not in scaling:
        raise ValueError(f"numeric question {q.qid} lacks scaling bounds")
    if int(scaling.get("cdf_size") or 201) != 201:
        raise NotImplementedError(
            "cdf_size != 201 (discrete grids) unsupported in MVP")
    lo, hi = scaling["range_min"], scaling["range_max"]
    open_lo = bool(scaling.get("open_lower_bound"))
    open_hi = bool(scaling.get("open_upper_bound"))
    lo_msg = (f"The value is likely not lower than {lo}." if open_lo
              else f"The outcome can not be lower than {lo}.")
    hi_msg = (f"The value is likely not higher than {hi}." if open_hi
              else f"The outcome can not be higher than {hi}.")
    r = await bot.run_numeric_ensemble(
        research=research_text, units=scaling.get("unit") or "",
        lower_bound_message=lo_msg, upper_bound_message=hi_msg,
        **_common_fields(q))
    stages.append(StageRecord(
        stage="aggregation",
        inputs={"member_percentile_dicts": list(r.member_percentile_dicts),
                "member_models": list(r.member_models)},
        outputs={"median_percentiles": dict(r.median_percentiles)}))
    # NOTE: percentiles_to_cdf builds a linear-grid PCHIP CDF; log-scaled
    # questions (zero_point set) are scored on that approximation in MVP.
    cdf = percentiles_to_cdf(r.median_percentiles, lo, hi, open_lo, open_hi)
    cdf_t = tuple(float(v) for v in cdf)
    stages.append(StageRecord(
        stage="calibration",
        inputs={"median_percentiles": dict(r.median_percentiles),
                "scaling": dict(scaling)},
        outputs={"cdf_len": len(cdf_t), "cdf_first": cdf_t[0],
                 "cdf_last": cdf_t[-1]}))
    return Forecast(qtype=q.qtype, cdf=cdf_t)


_HANDLERS = {
    QType.BINARY: _forecast_binary,
    QType.MULTIPLE_CHOICE: _forecast_mc,
    QType.NUMERIC: _forecast_numeric,
    QType.DISCRETE: _forecast_numeric,   # 201-grid only; else error artifact
}


class CalibratedEnsembleAdapter:
    """ForecastSystem implementation wrapping the live ensemble bot."""

    def __init__(self, cfg: BotConfig | None = None):
        self.cfg = cfg or BotConfig()
        self.name = "calibrated_ensemble"

    @classmethod
    def variant(cls, base: BotConfig | None = None,
                **patch) -> "CalibratedEnsembleAdapter":
        return cls(replace(base or BotConfig(), **patch))

    def config_fingerprint(self) -> str:
        # config_hash pops as_of before hashing, so the fingerprint is
        # stable across horizons of the same configuration.
        return config_hash(self.cfg)

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        stages: list[StageRecord] = []
        token = ACTIVE_CONTEXT.set(ctx)
        try:
            run_cfg = replace(self.cfg, as_of=ctx.as_of)
            bot = _BacktestEnsembleBot(run_cfg, ctx, stages)
            started = _now_iso()
            res = await bot_research.run_research(_question_dict(q), run_cfg)
            stages.append(StageRecord(
                stage="research",
                inputs={"providers": list(run_cfg.research_providers)},
                outputs={"text": res.text, "provider": res.provider},
                started_at=started, ended_at=_now_iso()))
            fc = await self._dispatch(bot, q, res.text, stages)
            return self._artifact(q, ctx, fc, stages)
        except Exception as e:
            logger.warning("forecast failed for %s: %s", q.qid, e)
            return self._artifact(q, ctx, None, stages,
                                  error=f"{type(e).__name__}: {e}")
        finally:
            ACTIVE_CONTEXT.reset(token)

    async def _dispatch(self, bot: _BacktestEnsembleBot, q: BacktestQuestion,
                        research_text: str,
                        stages: list[StageRecord]) -> Forecast:
        handler = _HANDLERS.get(q.qtype)
        if handler is None:
            raise NotImplementedError(f"unsupported qtype: {q.qtype!r}")
        return await handler(bot, q, research_text, stages)

    def _artifact(self, q: BacktestQuestion, ctx: AsOfContext,
                  fc: Forecast | None, stages: list[StageRecord],
                  error: str | None = None) -> ForecastArtifact:
        return ForecastArtifact(
            qid=q.qid, horizon="", as_of=ctx.as_of,
            system_name=self.name,
            config_fingerprint=self.config_fingerprint(),
            forecast=fc, stages=stages,
            evidence_ledger=list(ctx.ledger),
            contamination=self._contamination(q, ctx),
            cutoff_mode=ctx.cutoff_mode, error=error)

    def _contamination(self, q: BacktestQuestion, ctx: AsOfContext) -> str:
        if ctx.store is None:
            return "contaminated"
        try:
            return (ctx.store.get_contamination(
                q.qid, _model_set_hash(self.cfg)) or "contaminated")
        except Exception as e:
            logger.warning("contamination lookup failed for %s: %s",
                           q.qid, e)
            return "contaminated"
