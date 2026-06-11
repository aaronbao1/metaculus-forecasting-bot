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
                             from_log_odds, to_log_odds)
from bot.calibration import calibrate_binary, load_platt
from bot.config import BotConfig, DEFAULT_CONFIG, EnsembleMember, config_hash
from bot.logging_store import append_forecast
from bot.parsing import (ParseError, parse_binary, parse_multiple_choice,
                         parse_percentiles)
from bot.prompts import (BINARY_PROMPT, MC_PROMPT, NUMERIC_PROMPT,
                         SUPERVISOR_PROMPT)
from bot.research import run_research
from bot.timeutils import days_until

logger = logging.getLogger(__name__)

# Tail of each member's scratchpad shown to the supervisor — the final
# calibration/premortem section is where the disagreement's crux surfaces.
SUPERVISOR_EXCERPT_CHARS = 1200


@dataclass
class BinaryEnsembleResult:
    final: float
    aggregate_raw: float
    member_probs: list = field(default_factory=list)
    member_models: list = field(default_factory=list)
    spread: float = 0.0                  # max(member) - min(member)
    supervisor_prob: float | None = None  # set only when supervisor ran + parsed


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


def _days_remaining(close_iso: str) -> float:
    try:
        close = datetime.fromisoformat(str(close_iso).replace("Z", "+00:00"))
        return max(days_until(close), 0.0)
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
            if isinstance(r, Exception):
                logger.warning("member %s failed: %s", m, r)
            else:
                texts.append(r)
                kept.append(m)
        return texts, kept

    def _parse_survivors(self, texts, models, parse_fn):
        parsed, kept, kept_texts = [], [], []
        for t, m in zip(texts, models):
            try:
                parsed.append(parse_fn(t))
                kept.append(m)
                kept_texts.append(t)
            except ParseError as e:
                logger.warning("parse failure from %s: %s", m, e)
        if len(parsed) < self.cfg.min_members:
            raise RuntimeError(
                f"only {len(parsed)} ensemble members survived "
                f"(min {self.cfg.min_members}); skipping question")
        return parsed, kept, kept_texts

    async def _run_supervisor(self, *, question_text, background_info,
                              resolution_criteria, fine_print, research,
                              close_iso, probs, models, texts) -> float | None:
        """One frontier call reconciling a divergent panel (AIA-style).
        Returns a probability, or None on any failure — the raw aggregate is
        always a safe fallback; the supervisor must never cost the question."""
        members_block = "\n\n".join(
            f"--- Forecaster {i + 1} ({m}) — final answer {p:.0%}.\n"
            f"Reasoning excerpt: ...{t[-SUPERVISOR_EXCERPT_CHARS:]}"
            for i, (m, p, t) in enumerate(zip(models, probs, texts)))
        prompt = SUPERVISOR_PROMPT.format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research, today=date.today().isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso),
            member_count=len(probs), members_block=members_block)
        member = EnsembleMember(self.cfg.supervisor_model, runs=1,
                                temperature=0.2)
        try:
            return parse_binary(await self._invoke_model((member, prompt)))
        except Exception as e:
            logger.warning("supervisor failed (%s); using raw aggregate", e)
            return None

    async def run_binary_ensemble(self, *, question_text, background_info,
                                  resolution_criteria, fine_print, research,
                                  close_iso) -> BinaryEnsembleResult:
        prompt = BINARY_PROMPT.format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research, today=date.today().isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso))
        texts, models = await self._fan_out(prompt)
        probs, kept, kept_texts = self._parse_survivors(
            texts, models, parse_binary)
        raw = aggregate_binary(probs, self.cfg.aggregation_method)
        spread = max(probs) - min(probs)
        supervisor_prob = None
        if (self.cfg.supervisor_enabled
                and spread > self.cfg.supervisor_trigger_spread):
            supervisor_prob = await self._run_supervisor(
                question_text=question_text, background_info=background_info,
                resolution_criteria=resolution_criteria,
                fine_print=fine_print, research=research, close_iso=close_iso,
                probs=probs, models=kept, texts=kept_texts)
        blended = raw
        if supervisor_prob is not None:
            w = self.cfg.supervisor_weight
            blended = float(from_log_odds(
                w * to_log_odds([supervisor_prob])[0]
                + (1 - w) * to_log_odds([raw])[0]))
        final = calibrate_binary(blended, self._platt_binary,
                                 self.cfg.cap_lo, self.cfg.cap_hi)
        return BinaryEnsembleResult(final=final, aggregate_raw=raw,
                                    member_probs=probs, member_models=kept,
                                    spread=spread,
                                    supervisor_prob=supervisor_prob)

    async def run_mc_ensemble(self, *, question_text, background_info,
                              resolution_criteria, fine_print, research,
                              close_iso, options) -> MCEnsembleResult:
        prompt = MC_PROMPT.format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research, today=date.today().isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso),
            options=", ".join(options))
        texts, models = await self._fan_out(prompt)
        dicts, kept, _ = self._parse_survivors(
            texts, models, lambda t: parse_multiple_choice(t, list(options)))
        final = aggregate_multiple_choice(dicts, floor=self.cfg.mc_floor)
        return MCEnsembleResult(final=final, member_dicts=dicts,
                                member_models=kept)

    async def run_numeric_ensemble(self, *, question_text, background_info,
                                   resolution_criteria, fine_print, research,
                                   close_iso, units, lower_bound_message,
                                   upper_bound_message) -> NumericEnsembleResult:
        prompt = NUMERIC_PROMPT.format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research, today=date.today().isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso),
            units=units or "(see question)",
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message)
        texts, models = await self._fan_out(prompt)
        dicts, kept, _ = self._parse_survivors(texts, models, parse_percentiles)
        median_pcts = _median_per_percentile(dicts)
        return NumericEnsembleResult(median_percentiles=median_pcts,
                                     member_percentile_dicts=dicts,
                                     member_models=kept)


def build_framework_bot(cfg: BotConfig | None = None):
    """Factory returning a forecasting_tools.ForecastBot subclass instance
    wrapping CalibratedEnsembleBot. Inside a factory so importing this module
    never requires forecasting-tools at test time."""
    import numpy as np
    from forecasting_tools import (BinaryQuestion, ForecastBot,
                                   MultipleChoiceQuestion, NumericDistribution,
                                   NumericQuestion, PredictedOptionList,
                                   ReasonedPrediction)
    from forecasting_tools.data_models.multiple_choice_report import \
        PredictedOption
    from forecasting_tools.data_models.numeric_report import Percentile

    from bot.distributions import percentiles_to_cdf

    cfg = cfg or DEFAULT_CONFIG
    engine = CalibratedEnsembleBot(bot_config=cfg)

    class PchipNumericDistribution(NumericDistribution):
        """Framework distribution with our PCHIP+standardized CDF; falls back
        to the framework's linear construction if ours fails (stopgap chain)."""

        @property
        def cdf(self) -> list[Percentile]:
            try:
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
                               "framework linear construction", e)
                return super().cdf

    def _common_fields(question):
        return dict(
            question_text=question.question_text,
            background_info=question.background_info or "",
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
            close_iso=str(question.scheduled_resolution_time
                          or question.close_time or ""))

    def _log(question, qtype, final, members, models):
        append_forecast({"question_id": question.id_of_post,
                         "page_url": getattr(question, "page_url", None),
                         "type": qtype, "final": final, "members": members,
                         "models": models, "config_hash": engine.cfg_hash},
                        engine.cfg.log_path)

    class FrameworkBot(ForecastBot):
        async def run_research(self, question) -> str:
            res = await run_research(
                dict(question_text=question.question_text,
                     resolution_criteria=question.resolution_criteria or "",
                     fine_print=question.fine_print or ""), engine.cfg)
            if res.provider == "none":
                logger.error("forecasting %s with NO research",
                             question.page_url)
            return res.text

        async def _run_forecast_on_binary(
                self, question: BinaryQuestion, research: str
        ) -> ReasonedPrediction:
            r = await engine.run_binary_ensemble(
                research=research, **_common_fields(question))
            _log(question, "binary", r.final, r.member_probs, r.member_models)
            return ReasonedPrediction(
                prediction_value=r.final,
                reasoning=(f"ensemble={[round(p, 3) for p in r.member_probs]} "
                           f"-> {engine.cfg.aggregation_method}="
                           f"{r.aggregate_raw:.3f} -> calibrated+capped="
                           f"{r.final:.3f}"))

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
