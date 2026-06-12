"""Candidate/fitness framework (plan items 4-6's shared seam).

A Candidate is a typed payload an optimizer proposes; a fitness object
evaluates it over cached backtest artifacts and returns a score plus
per-question textual feedback (the reflection substrate GEPA-style loops
need). Two evaluators ship now:

- ReplayFitness — kind "params": replay cached member probabilities through
  aggregation+calibration (from_member_forecasts). $0 — no LLM.
- PromptFitness — kind "prompt:binary": replay cached research through the
  member LLM stage (from_research_text) with prompt_overrides patched.
  Identical prompts hit the shared llm_cache at $0; mutated prompts spend —
  guarded by a hard budget cap.

Reserved kinds "code:*" are the plug point for evolutionary program search
(OpenEvolve/ShinkaEvolve — NOT implemented; see docs/backtesting/OPTIMIZE.md).

Promotion discipline (non-negotiable): optimizers select on the TRAIN side
of a temporal split; anything that looks like a winner must then pass the
experiments paired-stats gate on a temporally LATER window before touching
live config. Scores here are mean Metaculus baseline score (higher better).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    kind: str                 # "params" | "prompt:binary" | "code:*" (reserved)
    payload: dict


@dataclass
class EvalResult:
    score: float              # mean baseline score over evaluated items
    per_item: list = field(default_factory=list)
    n: int = 0
    cost_usd: float = 0.0


class BudgetExceeded(RuntimeError):
    """Raised by PromptFitness when fresh-LLM spend would exceed the cap."""


def temporal_split(questions, train_frac: float = 0.6):
    """Split questions by scheduled_close_time: (train, val) with train
    strictly the earlier-closing fraction. Optimizers select on train only;
    val is reported, never selected on."""
    qs = sorted(questions,
                key=lambda q: (q.scheduled_close_time is None,
                               q.scheduled_close_time, q.qid))
    cut = int(len(qs) * train_frac)
    return qs[:cut], qs[cut:]


def latest_artifacts(run_dir: Path, variant: str | None = None) -> dict:
    """(qid, horizon) -> latest ForecastArtifact for one variant (or
    (qid, horizon, variant) -> artifact when variant is None)."""
    from backtest.core import artifact_from_dict
    latest: dict = {}
    path = Path(run_dir) / "artifacts.jsonl"
    if not path.exists():
        return latest
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            a = artifact_from_dict(json.loads(line))
            if variant is None:
                latest[(a.qid, a.horizon, a.system_name)] = a
            elif a.system_name == variant:
                latest[(a.qid, a.horizon)] = a
    return latest


def _stage_value(artifact, stage: str, attr: str, key: str):
    for s in artifact.stages:
        if s.stage == stage:
            val = getattr(s, attr).get(key)
            if val is not None:
                return val
    return None


def _feedback(qid: str, final: float, q, score: float, brier_v) -> str:
    return (f"question={qid!r} title={q.title[:90]!r} "
            f"forecast={final:.3f} resolved={q.resolution} "
            f"baseline_score={score:.1f}"
            + (f" brier={brier_v:.3f}" if brier_v is not None else ""))


class CountingLLM:
    """Budget-enforcing wrapper around a CachedLLM-compatible llm: cache
    hits are free; before any FRESH call it checks the projected spend
    against the cap and raises BudgetExceeded. Actual cost is read back from
    the cache record when available; litellm prices many slugs at 0, so the
    estimate is the conservative floor."""

    def __init__(self, inner, budget_usd: float,
                 est_cost_per_fresh: float = 0.02):
        self.inner = inner
        self.budget_usd = float(budget_usd)
        self.est = float(est_cost_per_fresh)
        self.cost_usd = 0.0
        self.fresh_calls = 0
        self.cached_calls = 0

    def _cache_path(self, model, prompt, temperature, run_index):
        cache_dir = getattr(self.inner, "cache_dir", None)
        if cache_dir is None:
            return None
        from backtest.timetravel.llm_cache import cache_key
        return (Path(cache_dir)
                / f"{cache_key(model, prompt, temperature, run_index)}.json")

    async def invoke(self, model, prompt, temperature, run_index=0) -> str:
        path = self._cache_path(model, prompt, temperature, run_index)
        cached = path is not None and path.exists()
        if not cached and self.cost_usd + self.est > self.budget_usd:
            raise BudgetExceeded(
                f"projected spend ${self.cost_usd + self.est:.2f} exceeds "
                f"budget ${self.budget_usd:.2f} "
                f"({self.fresh_calls} fresh calls so far)")
        text = await self.inner.invoke(model, prompt, temperature,
                                       run_index=run_index)
        if cached:
            self.cached_calls += 1
            return text
        self.fresh_calls += 1
        actual = 0.0
        if path is not None and path.exists():
            try:
                actual = float(json.loads(path.read_text())
                               .get("cost_usd", 0.0))
            except Exception:
                actual = 0.0
        self.cost_usd += actual if actual > 0 else self.est
        return text


class PromptFitness:
    """Fitness for kind="prompt:binary" candidates: cached research replayed
    through the member-LLM stage with the candidate template patched into
    prompt_overrides. The shared llm_cache makes the un-mutated template $0;
    every mutated prompt costs member calls — hard-capped by budget_usd
    across the lifetime of this fitness object."""

    kinds = ("prompt:binary",)

    def __init__(self, run_dir: Path, store, variant: str, llm,
                 budget_usd: float, base_config=None,
                 est_cost_per_fresh: float = 0.02):
        from bot.config import BotConfig
        self.run_dir = Path(run_dir)
        self.store = store
        self.variant = variant
        self.counting = CountingLLM(llm, budget_usd, est_cost_per_fresh)
        sandbox = str(self.run_dir / "optimize-no-calibration")
        self.base_config = base_config or BotConfig(calibration_dir=sandbox)
        if Path(self.base_config.calibration_dir, "binary.json").exists():
            raise ValueError("base_config.calibration_dir holds a fitted "
                             "Platt — use a sandbox dir (leakage guard)")
        self._items = self._load_items()

    @property
    def cost_usd(self) -> float:
        return self.counting.cost_usd

    def _load_items(self) -> list[dict]:
        from backtest.core import QType
        items = []
        for (qid, horizon), a in sorted(
                latest_artifacts(self.run_dir, self.variant).items()):
            research = _stage_value(a, "research", "outputs", "text")
            q = self.store.get_question(qid)
            if (research is None or q is None or q.resolution is None
                    or q.qtype is not QType.BINARY):
                continue
            items.append({"qid": qid, "horizon": horizon, "as_of": a.as_of,
                          "research": research, "question": q})
        return items

    @property
    def questions(self):
        return [it["question"] for it in self._items]

    async def evaluate(self, candidate: Candidate,
                       qids: set | None = None) -> EvalResult:
        if candidate.kind not in self.kinds:
            raise ValueError(
                f"PromptFitness handles kinds {self.kinds}, got "
                f"{candidate.kind!r}")
        template = candidate.payload.get("template")
        if not template:
            raise ValueError("prompt:binary candidate needs payload"
                             "['template']")
        from dataclasses import replace

        from backtest.scoring.metaculus_scores import baseline_score, brier
        from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
        from backtest.timetravel.asof import AsOfContext

        overrides = dict(self.base_config.prompt_overrides or {})
        overrides["binary"] = template
        cfg = replace(self.base_config, prompt_overrides=overrides)
        adapter = CalibratedEnsembleAdapter(cfg)
        start_cost = self.counting.cost_usd
        per_item, scores = [], []
        for it in self._items:
            if qids is not None and it["qid"] not in qids:
                continue
            # loop-level brake: _fan_out swallows per-member exceptions, so
            # BudgetExceeded inside member calls surfaces as a failed item —
            # the reliable enforcement point is before each question
            if self.counting.cost_usd + self.counting.est \
                    > self.counting.budget_usd:
                raise BudgetExceeded(
                    f"spent ${self.counting.cost_usd:.2f} of "
                    f"${self.counting.budget_usd:.2f} budget after "
                    f"{self.counting.fresh_calls} fresh calls")
            ctx = AsOfContext(as_of=it["as_of"], store=self.store,
                              llm=self.counting)
            art = await adapter.from_research_text(it["question"], ctx,
                                                   it["research"])
            if art.error is not None or art.forecast is None:
                logger.warning("prompt replay failed for %s: %s",
                               it["qid"], art.error)
                continue
            q = it["question"]
            s = baseline_score(art.forecast, q)
            try:
                b = brier(art.forecast, q)
            except Exception:
                b = None
            scores.append(s)
            per_item.append({
                "qid": it["qid"], "horizon": it["horizon"], "score": s,
                "brier": b,
                "feedback": _feedback(it["qid"],
                                      art.forecast.binary_prob, q, s, b)})
        cost = self.counting.cost_usd - start_cost
        if not scores:
            return EvalResult(score=float("-inf"), per_item=[], n=0,
                              cost_usd=cost)
        return EvalResult(score=sum(scores) / len(scores),
                          per_item=per_item, n=len(scores), cost_usd=cost)


class ReplayFitness:
    """$0 fitness for kind="params" candidates: cached member probabilities
    replayed through aggregation+calibration with the candidate's BotConfig
    patch. Binary-only (from_member_forecasts MVP contract)."""

    kinds = ("params",)

    def __init__(self, run_dir: Path, store, variant: str,
                 sandbox_dir: Path | None = None):
        self.run_dir = Path(run_dir)
        self.store = store
        self.variant = variant
        # calibration sandbox: never read a repo-level fitted Platt during
        # optimization (leakage guard parity with the adapter)
        self.calibration_dir = str(
            (sandbox_dir or self.run_dir) / "optimize-no-calibration")
        self._items = self._load_items()

    def _load_items(self) -> list[dict]:
        from backtest.core import QType
        items = []
        for (qid, horizon), a in sorted(
                latest_artifacts(self.run_dir, self.variant).items()):
            if a.forecast is None or a.forecast.qtype is not QType.BINARY:
                continue
            probs = _stage_value(a, "aggregation", "inputs", "member_probs")
            q = self.store.get_question(qid)
            if not probs or q is None or q.resolution is None:
                continue
            items.append({"qid": qid, "horizon": horizon, "as_of": a.as_of,
                          "member_probs": [float(p) for p in probs],
                          "question": q})
        return items

    @property
    def questions(self):
        return [it["question"] for it in self._items]

    async def evaluate(self, candidate: Candidate,
                       qids: set | None = None) -> EvalResult:
        if candidate.kind not in self.kinds:
            raise ValueError(
                f"ReplayFitness handles kinds {self.kinds}, got "
                f"{candidate.kind!r} (prompt candidates need PromptFitness)")
        from backtest.scoring.metaculus_scores import baseline_score, brier
        from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
        from backtest.timetravel.asof import AsOfContext
        from bot.config import BotConfig

        adapter = CalibratedEnsembleAdapter.variant(
            base=BotConfig(calibration_dir=self.calibration_dir),
            **candidate.payload)
        per_item, scores = [], []
        for it in self._items:
            if qids is not None and it["qid"] not in qids:
                continue
            ctx = AsOfContext(as_of=it["as_of"], store=self.store)
            art = await adapter.from_member_forecasts(
                it["question"], ctx, it["member_probs"])
            if art.error is not None or art.forecast is None:
                logger.warning("replay failed for %s: %s", it["qid"],
                               art.error)
                continue
            q = it["question"]
            s = baseline_score(art.forecast, q)
            try:
                b = brier(art.forecast, q)
            except Exception:
                b = None
            scores.append(s)
            per_item.append({
                "qid": it["qid"], "horizon": it["horizon"], "score": s,
                "brier": b,
                "feedback": _feedback(it["qid"],
                                      art.forecast.binary_prob, q, s, b)})
        if not scores:
            return EvalResult(score=float("-inf"), per_item=[], n=0)
        return EvalResult(score=sum(scores) / len(scores),
                          per_item=per_item, n=len(scores))
