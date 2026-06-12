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
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import replace
from datetime import datetime, timezone

from pathlib import Path

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord)
from backtest.corpus.model_releases import model_set_hash
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot import research as bot_research
from bot.aggregation import aggregate_binary, extremize
from bot.calibration import calibrate_binary, load_platt
from bot.config import BotConfig, config_hash
from bot.distributions import percentiles_to_cdf
from bot.ensemble_bot import CalibratedEnsembleBot

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# I3 — content-addressed research cache
# ---------------------------------------------------------------------------

# Providers whose output depends on BotConfig beyond the provider list —
# their config knobs must salt the research cache key or variants A/B-ing
# those knobs would silently share dossiers (guaranteed null result).
_CFG_DEPENDENT_PROVIDERS = {"asknews_rated"}


def _research_cache_key(qid: str, as_of: datetime,
                        providers: list[str], cfg_salt: str = "") -> str:
    """sha256(qid | as_of_iso | comma-joined-providers [| cfg_salt]).

    Stable for any ordering: providers are sorted before joining so that
    differently-ordered provider lists with the same members share a cache.
    """
    payload = f"{qid}|{as_of.isoformat()}|{','.join(sorted(providers))}"
    if cfg_salt:
        payload += f"|{cfg_salt}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _research_cfg_salt(cfg: BotConfig) -> str:
    """Cache-key salt for config-dependent providers ("" otherwise)."""
    if _CFG_DEPENDENT_PROVIDERS & set(cfg.research_providers):
        return f"parser_model={cfg.parser_model}"
    return ""


def _research_cache_dir(llm_cache_dir: Path) -> Path:
    """research_cache lives as a sibling of the LLM cache dir."""
    return llm_cache_dir.parent / "research_cache"


def _load_research_cache(cache_dir: Path, key: str):
    """Return cached dict {"text": ..., "provider": ..., "providers": [...]}
    or None on miss / corrupt entry."""
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_research_cache(cache_dir: Path, key: str, record: dict) -> None:
    """Atomic write: tmp + os.replace, mirroring llm_cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(record))
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _model_set_hash(cfg: BotConfig) -> str:
    """Delegate to the canonical corpus implementation so the adapter and
    the corpus stratifier always agree on the hash (model_set_hash is total
    over unknown slugs — RELEASE_DATES.get(m, "unknown"))."""
    return model_set_hash([m.model for m in cfg.ensemble])


def _question_dict(q: BacktestQuestion) -> dict:
    # resolution_criteria falls back to description so that market-native
    # questions (polymarket etc.) whose substance lives in description are
    # seen by the research stage.
    return dict(question_text=q.title,
                resolution_criteria=q.resolution_criteria or q.description,
                fine_print=q.fine_print,
                description=q.description,
                qid=q.qid,
                # Halawi-style research window start (providers may use it):
                open_time_iso=(q.open_time.isoformat()
                               if q.open_time else None))


def _common_fields(q: BacktestQuestion) -> dict:
    return dict(
        question_text=q.title,
        background_info=q.description,
        resolution_criteria=q.resolution_criteria,
        fine_print=q.fine_print,
        close_iso=str(q.scheduled_close_time or ""))


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
                "method": bot.cfg.aggregation_method,
                "n_trim": bot.cfg.n_trim},
        outputs={"aggregate_raw": r.aggregate_raw, "spread": r.spread}))
    stages.append(StageRecord(
        stage="calibration",
        inputs={"aggregate_raw": r.aggregate_raw,
                "cap_lo": bot.cfg.cap_lo, "cap_hi": bot.cfg.cap_hi,
                "extremize_a": bot.cfg.extremize_a},
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
    if scaling.get("zero_point") is not None:
        raise NotImplementedError(
            "log-scaled question: linear-grid CDF not faithful; deferred")
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
    """ForecastSystem implementation wrapping the live ensemble bot.

    allow_calibration: when False (default) and a fitted binary.json exists in
    cfg.calibration_dir, the adapter replaces calibration_dir with a sentinel
    nonexistent path so load_platt() returns the identity map — preventing a
    future-fitted Platt from contaminating backtest results.  Set True only
    when the caller has verified the Platt snapshot is as-of-correct.
    """

    def __init__(self, cfg: BotConfig | None = None,
                 allow_calibration: bool = False):
        self.cfg = cfg or BotConfig()
        self.allow_calibration = allow_calibration
        self.name = "calibrated_ensemble"

    @classmethod
    def variant(cls, base: BotConfig | None = None,
                **patch) -> "CalibratedEnsembleAdapter":
        return cls(replace(base or BotConfig(), **patch))

    def config_fingerprint(self) -> str:
        # config_hash pops as_of before hashing, so the fingerprint is
        # stable across horizons of the same configuration.
        return config_hash(self.cfg)

    def _guard_calibration(self, run_cfg: "BotConfig",
                           stages: list[StageRecord]) -> "BotConfig":
        """Calibration leakage guard.

        If a fitted Platt file exists in run_cfg.calibration_dir but the
        caller has not explicitly opted in (allow_calibration=True), redirect
        calibration_dir to a guaranteed-nonexistent sentinel path so that
        load_platt() returns the identity map — preventing future-fitted Platt
        parameters from contaminating backtest results.  Also appends an
        explanatory StageRecord so the guard is always visible in the artifact.

        Returns the (possibly modified) run_cfg.
        """
        platt_file = Path(run_cfg.calibration_dir) / "binary.json"
        if platt_file.exists() and not self.allow_calibration:
            sentinel = run_cfg.calibration_dir + ".disabled-for-backtest"
            logger.warning(
                "calibration leakage guard: platt file present at %s "
                "but allow_calibration=False; redirecting calibration_dir "
                "to nonexistent sentinel %s (identity Platt will be used)",
                platt_file, sentinel)
            run_cfg = replace(run_cfg, calibration_dir=sentinel)
            stages.append(StageRecord(
                stage="calibration",
                outputs={"note": (
                    "platt disabled for backtest "
                    "(file present but allow_calibration=False); "
                    "identity Platt applied")}))
        return run_cfg

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        stages: list[StageRecord] = []
        token = ACTIVE_CONTEXT.set(ctx)
        try:
            run_cfg = replace(self.cfg, as_of=ctx.as_of)
            run_cfg = self._guard_calibration(run_cfg, stages)
            bot = _BacktestEnsembleBot(run_cfg, ctx, stages)

            # I3 — content-addressed research cache.
            # Only cache when ctx.llm is set AND has a cache_dir attribute
            # (CachedLLM provides this; stub LLMs in tests may not).
            research_text: str | None = None
            cached_provider: str | None = None
            cache_hit = False
            r_cache_dir = None
            r_key = None
            if ctx.llm is not None and hasattr(ctx.llm, "cache_dir"):
                r_key = _research_cache_key(
                    q.qid, ctx.as_of,
                    list(run_cfg.research_providers),
                    cfg_salt=_research_cfg_salt(run_cfg))
                r_cache_dir = _research_cache_dir(ctx.llm.cache_dir)
                cached = _load_research_cache(r_cache_dir, r_key)
                if cached is not None:
                    research_text = cached["text"]
                    cached_provider = cached.get("provider", "cached")
                    cache_hit = True

            if cache_hit:
                stages.append(StageRecord(
                    stage="research",
                    inputs={"providers": list(run_cfg.research_providers),
                            "cache_key": r_key},
                    outputs={"text": research_text,
                             "provider": cached_provider,
                             "cache_hit": True},
                    started_at=_now_iso(), ended_at=_now_iso()))
            else:
                started = _now_iso()
                res = await bot_research.run_research(_question_dict(q), run_cfg)
                ended = _now_iso()
                stages.append(StageRecord(
                    stage="research",
                    inputs={"providers": list(run_cfg.research_providers)},
                    outputs={"text": res.text, "provider": res.provider},
                    started_at=started, ended_at=ended))
                research_text = res.text
                if res.provider == "none":
                    raise RuntimeError(
                        "no research available; refusing LLM spend "
                        "(all configured research providers returned nothing)")
                # Persist to cache when cache_dir is available
                if r_cache_dir is not None and r_key is not None:
                    try:
                        _save_research_cache(r_cache_dir, r_key, {
                            "text": res.text,
                            "provider": res.provider,
                            "providers": list(run_cfg.research_providers),
                        })
                    except Exception as cache_err:
                        logger.warning("research cache write failed: %s",
                                       cache_err)

            fc = await self._dispatch(bot, q, research_text, stages)
            return self._artifact(q, ctx, fc, stages)
        except Exception as e:
            logger.warning("forecast failed for %s: %s", q.qid, e)
            return self._artifact(q, ctx, None, stages,
                                  error=f"{type(e).__name__}: {e}")
        finally:
            ACTIVE_CONTEXT.reset(token)

    async def from_research_text(self, q: BacktestQuestion, ctx: AsOfContext,
                                 research_text: str) -> ForecastArtifact:
        """Stage entry point: skip run_research entirely and forecast from
        the given (frozen) research text — isolates downstream stages."""
        stages: list[StageRecord] = []
        token = ACTIVE_CONTEXT.set(ctx)
        try:
            run_cfg = replace(self.cfg, as_of=ctx.as_of)
            run_cfg = self._guard_calibration(run_cfg, stages)
            bot = _BacktestEnsembleBot(run_cfg, ctx, stages)
            stages.append(StageRecord(
                stage="research",
                inputs={"injected": True},
                outputs={"text": research_text, "provider": "injected"},
                started_at=_now_iso(), ended_at=_now_iso()))
            fc = await self._dispatch(bot, q, research_text, stages)
            return self._artifact(q, ctx, fc, stages)
        except Exception as e:
            logger.warning("from_research_text failed for %s: %s", q.qid, e)
            return self._artifact(q, ctx, None, stages,
                                  error=f"{type(e).__name__}: {e}")
        finally:
            ACTIVE_CONTEXT.reset(token)

    async def from_member_forecasts(self, q: BacktestQuestion,
                                    ctx: AsOfContext,
                                    member_outputs: list) -> ForecastArtifact:
        """Stage entry point: bypass research and the LLM entirely; run
        aggregation + calibration on pre-computed member outputs.

        BINARY-ONLY for MVP — member_outputs is a list[float] of member
        probabilities. Other question types raise NotImplementedError
        (a programming error, not a runtime failure)."""
        if q.qtype is not QType.BINARY:
            raise NotImplementedError(
                "from_member_forecasts is binary-only for MVP")
        stages: list[StageRecord] = []
        try:
            run_cfg = replace(self.cfg, as_of=ctx.as_of)
            run_cfg = self._guard_calibration(run_cfg, stages)
            probs = [float(p) for p in member_outputs]
            started = _now_iso()
            raw = aggregate_binary(probs, run_cfg.aggregation_method,
                                   n_trim=run_cfg.n_trim)
            stages.append(StageRecord(
                stage="aggregation",
                inputs={"member_probs": probs,
                        "method": run_cfg.aggregation_method,
                        "n_trim": run_cfg.n_trim},
                outputs={"aggregate_raw": raw},
                started_at=started, ended_at=_now_iso()))
            pre_cal = (extremize(raw, run_cfg.extremize_a)
                       if run_cfg.extremize_a != 1.0 else raw)
            platt = load_platt(Path(run_cfg.calibration_dir) / "binary.json")
            final = calibrate_binary(pre_cal, platt,
                                     run_cfg.cap_lo, run_cfg.cap_hi)
            stages.append(StageRecord(
                stage="calibration",
                inputs={"aggregate_raw": raw, "cap_lo": run_cfg.cap_lo,
                        "cap_hi": run_cfg.cap_hi,
                        "extremize_a": run_cfg.extremize_a},
                outputs={"final": final}))
            return self._artifact(q, ctx,
                                  Forecast(qtype=QType.BINARY,
                                           binary_prob=final), stages)
        except Exception as e:
            logger.warning("from_member_forecasts failed for %s: %s",
                           q.qid, e)
            return self._artifact(q, ctx, None, stages,
                                  error=f"{type(e).__name__}: {e}")

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
