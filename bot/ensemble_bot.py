"""CalibratedEnsembleBot: research -> K runs across model families ->
trimmed-GMO aggregation -> Platt + caps -> validated submission.

The pure ensemble core (CalibratedEnsembleBot) is framework-free and fully
unit-testable. build_framework_bot() wraps it in a forecasting_tools
ForecastBot subclass with research_reports_per_question=1 and
predictions_per_research_report=1 — the ensemble fan-out is OURS, so
aggregation and calibration stay under our control and our tests.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from bot.aggregation import (aggregate_binary, aggregate_multiple_choice,
                             extremize)
from bot.calibration import calibrate_binary, load_platt
from bot.config import BotConfig, DEFAULT_CONFIG, config_hash
from bot.logging_store import append_forecast
from bot.parsing import (ParseError, parse_binary, parse_multiple_choice,
                         parse_percentiles)
from bot.prompts import BINARY_PROMPT, MC_PROMPT, NUMERIC_PROMPT
from bot.research import run_research
from bot.timeutils import days_until

logger = logging.getLogger(__name__)


@dataclass
class BinaryEnsembleResult:
    final: float
    aggregate_raw: float
    member_probs: list = field(default_factory=list)
    member_models: list = field(default_factory=list)
    spread: float = 0.0                  # max(member) - min(member); logged
    # provenance — high-spread questions are the designed seam for the
    # post-MVP System 2 supervisor (plan roadmap #1).


@dataclass
class MCEnsembleResult:
    final: dict
    member_dicts: list = field(default_factory=list)
    member_models: list = field(default_factory=list)


@dataclass
class NumericEnsembleResult:
    median_percentiles: dict          # percentile -> value, medianed across members
    member_percentile_dicts: list = field(default_factory=list)
    member_models: list = field(default_factory=list)


def _days_remaining(close_iso: str, now: datetime | None = None) -> float:
    try:
        close = datetime.fromisoformat(str(close_iso).replace("Z", "+00:00"))
        return max(days_until(close, now=now), 0.0)
    except Exception:
        return 0.0


def _median_per_percentile(dicts: list[dict]) -> dict:
    import numpy as np
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    if len(keys) < 5:
        raise RuntimeError("ensemble members share too few percentiles")
    return {k: float(np.median([d[k] for d in dicts])) for k in sorted(keys)}


class CalibratedEnsembleBot:
    """Core engine (no forecasting-tools imports at module scope)."""

    def __init__(self, bot_config: BotConfig | None = None, **kwargs):
        self.cfg = bot_config or DEFAULT_CONFIG
        self.cfg_hash = config_hash(self.cfg)
        self._platt_binary = load_platt(
            Path(self.cfg.calibration_dir) / "binary.json")

    def _template(self, qtype: str) -> str:
        """Prompt template for a question type: cfg.prompt_overrides wins
        (the optimization harness ships winners as config), else the module
        constant. Overrides may use any subset of the standard placeholders —
        str.format ignores unused kwargs."""
        override = (self.cfg.prompt_overrides or {}).get(qtype)
        if override:
            return override
        return {"binary": BINARY_PROMPT, "mc": MC_PROMPT,
                "numeric": NUMERIC_PROMPT}[qtype]

    def _playbook_section(self) -> str:
        """Playbook block, or "" — in which case rendered prompts are
        byte-identical to the pre-playbook templates (llm_cache safety)."""
        if not self.cfg.playbook_path:
            return ""
        from bot.playbook import load_playbook
        # cfg.as_of threads the temporal guard: under a backtest, bullets
        # distilled from outcomes after T must not reach the prompt
        text = load_playbook(self.cfg.playbook_path, as_of=self.cfg.as_of)
        if not text:
            return ""
        return ("\nFORECASTING PLAYBOOK (distilled lessons from past "
                "resolved forecasts; apply only where relevant)\n"
                f"{text}\n")

    async def _invoke_model(self, member_and_prompt) -> str:
        """Single seam for all LLM calls (monkeypatch target in tests)."""
        member, prompt = member_and_prompt
        from forecasting_tools import GeneralLlm
        llm = GeneralLlm(model=f"openrouter/{member.model}",
                         temperature=member.temperature, timeout=900)
        return await llm.invoke(prompt)

    async def _fan_out(self, prompt: str):
        """Run every ensemble member; return (texts, models) of survivors."""
        jobs, models = [], []
        for member in self.cfg.ensemble:
            for _ in range(member.runs):
                jobs.append(self._invoke_model((member, prompt)))
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

    def _parse_survivors(self, texts, models, parse_fn):
        parsed, kept = [], []
        for t, m in zip(texts, models):
            try:
                parsed.append(parse_fn(t))
                kept.append(m)
            except ParseError as e:
                logger.warning("parse failure from %s: %s", m, e)
        if len(parsed) < self.cfg.min_members:
            raise RuntimeError(
                f"only {len(parsed)} ensemble members survived "
                f"(min {self.cfg.min_members}); skipping question")
        return parsed, kept

    async def run_binary_ensemble(self, *, question_text, background_info,
                                  resolution_criteria, fine_print, research,
                                  close_iso) -> BinaryEnsembleResult:
        prompt = self._template("binary").format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research, playbook_section=self._playbook_section(),
            today=(self.cfg.as_of.date() if self.cfg.as_of
                   else date.today()).isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso, now=self.cfg.as_of))
        texts, models = await self._fan_out(prompt)
        probs, kept = self._parse_survivors(texts, models, parse_binary)
        raw = aggregate_binary(probs, self.cfg.aggregation_method,
                               n_trim=self.cfg.n_trim)
        spread = max(probs) - min(probs)
        pre_cal = (extremize(raw, self.cfg.extremize_a)
                   if self.cfg.extremize_a != 1.0 else raw)
        final = calibrate_binary(pre_cal, self._platt_binary,
                                 self.cfg.cap_lo, self.cfg.cap_hi)
        return BinaryEnsembleResult(final=final, aggregate_raw=raw,
                                    member_probs=probs, member_models=kept,
                                    spread=spread)

    async def run_mc_ensemble(self, *, question_text, background_info,
                              resolution_criteria, fine_print, research,
                              close_iso, options) -> MCEnsembleResult:
        prompt = self._template("mc").format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research, playbook_section=self._playbook_section(),
            today=(self.cfg.as_of.date() if self.cfg.as_of
                   else date.today()).isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso, now=self.cfg.as_of),
            options=", ".join(options))
        texts, models = await self._fan_out(prompt)
        dicts, kept = self._parse_survivors(
            texts, models, lambda t: parse_multiple_choice(t, list(options)))
        final = aggregate_multiple_choice(dicts, floor=self.cfg.mc_floor)
        return MCEnsembleResult(final=final, member_dicts=dicts,
                                member_models=kept)

    async def run_numeric_ensemble(self, *, question_text, background_info,
                                   resolution_criteria, fine_print, research,
                                   close_iso, units, lower_bound_message,
                                   upper_bound_message) -> NumericEnsembleResult:
        prompt = self._template("numeric").format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research, playbook_section=self._playbook_section(),
            today=(self.cfg.as_of.date() if self.cfg.as_of
                   else date.today()).isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso, now=self.cfg.as_of),
            units=units or "(see question)",
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message)
        texts, models = await self._fan_out(prompt)
        dicts, kept = self._parse_survivors(texts, models, parse_percentiles)
        median_pcts = _median_per_percentile(dicts)
        return NumericEnsembleResult(median_percentiles=median_pcts,
                                     member_percentile_dicts=dicts,
                                     member_models=kept)


_pchip_distribution_class = None


def build_pchip_distribution_class():
    """Cached forecasting_tools.NumericDistribution subclass whose get_cdf()
    uses our PCHIP+standardized construction. Module-level (not closed over
    inside build_framework_bot) so tests can exercise it directly.

    get_cdf() is the method the framework actually calls when aggregating and
    publishing (the .cdf property is deprecated and never read by the
    framework — overriding only .cdf silently ships the default CDF).
    Delegates to the framework construction for log-scaled (zero_point) and
    discrete (cdf_size != 201) questions, whose grids our linear-grid PCHIP
    math does not model, and on any failure (stopgap chain)."""
    global _pchip_distribution_class
    if _pchip_distribution_class is not None:
        return _pchip_distribution_class

    import numpy as np
    from forecasting_tools import NumericDistribution
    from forecasting_tools.data_models.numeric_report import Percentile

    from bot.distributions import CDF_SIZE, percentiles_to_cdf

    class PchipNumericDistribution(NumericDistribution):

        def get_cdf(self) -> list[Percentile]:
            try:
                if self.zero_point is not None:
                    logger.info("log-scaled question (zero_point set); "
                                "using framework CDF construction")
                    return super().get_cdf()
                if (self.cdf_size or CDF_SIZE) != CDF_SIZE:
                    logger.info("discrete question (cdf_size=%s); "
                                "using framework CDF construction",
                                self.cdf_size)
                    return super().get_cdf()
                declared = {p.percentile * 100: p.value
                            for p in self.declared_percentiles}
                cdf_probs = percentiles_to_cdf(
                    declared, self.lower_bound, self.upper_bound,
                    self.open_lower_bound, self.open_upper_bound)
                grid = np.linspace(self.lower_bound, self.upper_bound,
                                   len(cdf_probs))
                return [Percentile(percentile=float(p), value=float(v))
                        for p, v in zip(cdf_probs, grid)]
            except Exception as e:
                logger.warning("PCHIP CDF failed (%s); falling back to "
                               "framework construction", e)
                return super().get_cdf()

    _pchip_distribution_class = PchipNumericDistribution
    return PchipNumericDistribution


def build_framework_bot(cfg: BotConfig | None = None):
    """Factory returning a forecasting_tools.ForecastBot subclass instance
    wrapping CalibratedEnsembleBot. Inside a factory so importing this module
    never requires forecasting-tools at test time."""
    from forecasting_tools import (BinaryQuestion, ForecastBot,
                                   MultipleChoiceQuestion, NumericQuestion,
                                   PredictedOptionList, ReasonedPrediction)
    from forecasting_tools.data_models.multiple_choice_report import \
        PredictedOption
    from forecasting_tools.data_models.numeric_report import Percentile

    cfg = cfg or DEFAULT_CONFIG
    engine = CalibratedEnsembleBot(bot_config=cfg)
    PchipNumericDistribution = build_pchip_distribution_class()

    def _common_fields(question):
        return dict(
            question_text=question.question_text,
            background_info=question.background_info or "",
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
            close_iso=str(question.scheduled_resolution_time
                          or question.close_time or ""))

    research_providers_by_qid: dict = {}

    def _qid(question):
        """Unique key: group-question sub-questions share id_of_post and run
        concurrently — id_of_question disambiguates them."""
        return getattr(question, "id_of_question", None) or question.id_of_post

    def _log(question, qtype, final, members, models, **extra):
        try:
            append_forecast(
                {"question_id": question.id_of_post,
                 "id_of_question": getattr(question, "id_of_question", None),
                 "page_url": getattr(question, "page_url", None),
                 "type": qtype, "final": final, "members": members,
                 "models": models, "config_hash": engine.cfg_hash,
                 "research_providers": research_providers_by_qid.pop(
                     _qid(question), ""),
                 **extra},
                engine.cfg.log_path)
        except Exception as e:
            # the log is the self-improvement substrate, but a logging failure
            # must never cost the question after the full ensemble spend
            logger.error("failed to write forecast log: %s", e)

    class FrameworkBot(ForecastBot):
        async def run_research(self, question) -> str:
            res = await run_research(
                dict(question_text=question.question_text,
                     resolution_criteria=question.resolution_criteria or "",
                     fine_print=question.fine_print or ""), engine.cfg)
            research_providers_by_qid[_qid(question)] = res.provider
            if res.provider == "none":
                logger.error("forecasting %s with NO research",
                             question.page_url)
            return res.text

        async def forecast_questions(self, questions, return_exceptions=False):
            limit = engine.cfg.max_questions_per_run
            if limit and len(questions) > limit:
                logger.warning(
                    "cost brake: %d questions exceeds max_questions_per_run="
                    "%d; forecasting only the first %d",
                    len(questions), limit, limit)
                questions = list(questions)[:limit]
            return await super().forecast_questions(
                questions, return_exceptions=return_exceptions)

        async def _run_forecast_on_binary(
                self, question: BinaryQuestion, research: str
        ) -> ReasonedPrediction:
            r = await engine.run_binary_ensemble(
                research=research, **_common_fields(question))
            _log(question, "binary", r.final, r.member_probs, r.member_models,
                 raw=r.aggregate_raw, spread=r.spread)
            return ReasonedPrediction(
                prediction_value=r.final,
                reasoning=(f"ensemble={[round(p, 3) for p in r.member_probs]} "
                           f"-> {engine.cfg.aggregation_method}="
                           f"{r.aggregate_raw:.3f} (spread {r.spread:.2f}) "
                           f"-> calibrated+capped={r.final:.3f}"))

        async def _run_forecast_on_multiple_choice(
                self, question: MultipleChoiceQuestion, research: str
        ) -> ReasonedPrediction:
            r = await engine.run_mc_ensemble(
                research=research, options=list(question.options),
                **_common_fields(question))
            _log(question, "multiple_choice", r.final, r.member_dicts,
                 r.member_models)
            option_list = PredictedOptionList(predicted_options=[
                PredictedOption(option_name=o, probability=p)
                for o, p in r.final.items()])
            return ReasonedPrediction(
                prediction_value=option_list,
                reasoning=f"per-option mean of {len(r.member_dicts)} members, "
                          f"floored at {engine.cfg.mc_floor}: {r.final}")

        async def _run_forecast_on_numeric(
                self, question: NumericQuestion, research: str
        ) -> ReasonedPrediction:
            lo_msg = (f"The outcome can not be lower than {question.lower_bound}."
                      if not question.open_lower_bound else
                      f"The value is likely not lower than {question.lower_bound}.")
            hi_msg = (f"The outcome can not be higher than {question.upper_bound}."
                      if not question.open_upper_bound else
                      f"The value is likely not higher than {question.upper_bound}.")
            r = await engine.run_numeric_ensemble(
                research=research, units=question.unit_of_measure,
                lower_bound_message=lo_msg, upper_bound_message=hi_msg,
                **_common_fields(question))
            _log(question, "numeric", r.median_percentiles,
                 r.member_percentile_dicts, r.member_models)
            dist = PchipNumericDistribution(
                declared_percentiles=[
                    Percentile(percentile=pct / 100.0, value=val)
                    for pct, val in sorted(r.median_percentiles.items())],
                open_lower_bound=question.open_lower_bound,
                open_upper_bound=question.open_upper_bound,
                lower_bound=question.lower_bound,
                upper_bound=question.upper_bound,
                zero_point=question.zero_point,
                cdf_size=question.cdf_size)
            return ReasonedPrediction(
                prediction_value=dist,
                reasoning=f"median percentiles across "
                          f"{len(r.member_percentile_dicts)} members: "
                          f"{r.median_percentiles}")

    return FrameworkBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=cfg.submit_predictions,
        skip_previously_forecasted_questions=True,
    )
