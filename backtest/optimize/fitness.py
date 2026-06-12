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
