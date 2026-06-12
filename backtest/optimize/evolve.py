"""Native GEPA-style reflective prompt evolution (plan item 4, scoped).

The GEPA mechanism (arXiv 2507.19457) without the dependency: sample
rollouts, REFLECT in natural language on the worst per-question feedback,
propose full revised templates, keep a Pareto frontier of (train score,
template length), iterate. Selection happens on the TRAIN split only — the
val score is reported for the record and a winner still has to pass the
experiments paired-stats gate on a later window before going live
(docs/backtesting/OPTIMIZE.md).

Known limits, priced in (research-notes/07 §8): prompt optimization has
diminishing returns on frontier models and overfits with large train sets —
point this at cheap ensemble members with small train sets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backtest.optimize.fitness import BudgetExceeded, Candidate

logger = logging.getLogger(__name__)

CONTRACT_LINE = "Probability:"

REFLECT_PROMPT = """You are improving the prompt template of a binary-forecasting bot.

CURRENT TEMPLATE (delimited below):
<<<
{template}
>>>

WORST-SCORED QUESTIONS UNDER THIS TEMPLATE:
{feedback}

Propose ONE revised full template that addresses these failures. Hard
requirements:
- Keep every {{placeholder}} that appears in the current template.
- Keep the final-line output contract ending in "Probability: ZZ%".
- Reasoning-step changes only; do not add facts about specific questions
  (that would overfit to the training examples).

Respond with ONLY the complete revised template, no commentary."""


@dataclass
class EvolveResult:
    best_template: str
    best_train_score: float
    val_score: float | None
    history: list = field(default_factory=list)   # candidates.jsonl rows
    stopped_reason: str = "completed"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        t = "\n".join(lines)
    return t


def _valid_child(template: str, parent: str) -> bool:
    if CONTRACT_LINE not in template:
        return False
    # every placeholder of the parent must survive (format() would KeyError
    # only on unknown placeholders; missing ones silently drop content)
    import string
    parent_fields = {f for _, f, _, _ in string.Formatter().parse(parent)
                     if f}
    child_fields = {f for _, f, _, _ in string.Formatter().parse(template)
                    if f}
    return parent_fields <= child_fields


def _pareto_best(frontier: list[dict]) -> dict:
    """Highest train score; ties broken by shorter template."""
    return max(frontier, key=lambda c: (c["train_score"], -len(c["template"])))


async def reflective_evolve(seed_template: str, fitness, reflect_llm,
                            reflect_model: str, *, train_qids: set,
                            val_qids: set | None = None,
                            generations: int = 3, children: int = 2,
                            k_worst: int = 5) -> EvolveResult:
    history: list[dict] = []
    stopped = "completed"

    async def _eval(template: str, gen: int, parent_gen: int | None):
        res = await fitness.evaluate(
            Candidate("prompt:binary", {"template": template}),
            qids=train_qids)
        row = {"gen": gen, "parent_gen": parent_gen, "template": template,
               "train_score": res.score, "n": res.n,
               "cost_usd": res.cost_usd,
               "worst": sorted(res.per_item, key=lambda i: i["score"])
               [:k_worst]}
        history.append(row)
        return row

    try:
        best = await _eval(seed_template, gen=0, parent_gen=None)
        frontier = [best]
        for gen in range(1, generations + 1):
            parent = _pareto_best(frontier)
            feedback = "\n".join(
                f"- {item['feedback']}" for item in parent["worst"])
            prompt = REFLECT_PROMPT.format(template=parent["template"],
                                           feedback=feedback)
            for child_idx in range(children):
                try:
                    raw = await reflect_llm.invoke(reflect_model, prompt,
                                                   0.7, run_index=child_idx)
                except BudgetExceeded:
                    raise
                except Exception as e:
                    logger.warning("reflection call failed: %s", e)
                    continue
                template = _strip_fences(raw)
                if not _valid_child(template, parent["template"]):
                    logger.warning(
                        "gen %d child %d violates the template contract; "
                        "skipped", gen, child_idx)
                    continue
                child = await _eval(template, gen=gen,
                                    parent_gen=parent["gen"])
                # Pareto keep on (train_score, -len): drop dominated entries
                frontier = [c for c in frontier + [child]
                            if not any(
                                o["train_score"] >= c["train_score"]
                                and len(o["template"]) <= len(c["template"])
                                and o is not c
                                and (o["train_score"] > c["train_score"]
                                     or len(o["template"])
                                     < len(c["template"]))
                                for o in frontier + [child])]
    except BudgetExceeded as e:
        logger.warning("evolution stopped on budget: %s", e)
        stopped = "budget"

    if not history:
        raise RuntimeError("no candidate was ever evaluated")
    best = _pareto_best([c for c in history])
    val_score = None
    if val_qids and stopped != "budget":
        try:
            val_res = await fitness.evaluate(
                Candidate("prompt:binary", {"template": best["template"]}),
                qids=val_qids)
            val_score = val_res.score
        except BudgetExceeded:
            stopped = "budget-on-val"
    return EvolveResult(best_template=best["template"],
                        best_train_score=best["train_score"],
                        val_score=val_score, history=history,
                        stopped_reason=stopped)
