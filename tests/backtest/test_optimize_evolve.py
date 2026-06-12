"""Task 15: reflective prompt evolution — all-fake, deterministic."""
import asyncio

import pytest

from backtest.optimize.evolve import reflective_evolve
from backtest.optimize.fitness import BudgetExceeded, EvalResult

SEED = "Forecast {question_text} using {research}.\nProbability: ZZ%"
TRAIN = {"t:0", "t:1"}
VAL = {"t:9"}


class FakeFitness:
    """train score = 10 * count('MAGIC'); val score = -count('MAGIC')
    (diverges, to prove selection is train-only)."""

    def __init__(self, fail_after=None):
        self.evals = 0
        self.fail_after = fail_after

    async def evaluate(self, candidate, qids=None):
        self.evals += 1
        if self.fail_after is not None and self.evals > self.fail_after:
            raise BudgetExceeded("cap")
        magic = candidate.payload["template"].count("MAGIC")
        score = 10.0 * magic if qids == TRAIN else -1.0 * magic
        per_item = [{"qid": q, "score": score, "brier": 0.1,
                     "feedback": f"q={q} fb"} for q in sorted(qids)]
        return EvalResult(score=score, per_item=per_item, n=len(per_item))


class FakeReflect:
    """Each reflection appends one MAGIC token; child 1 of gen 1 returns a
    contract-violating template (must be skipped)."""

    def __init__(self):
        self.calls = 0

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls += 1
        start = prompt.index("<<<") + 4
        end = prompt.index(">>>")
        parent = prompt[start:end].strip()
        if self.calls == 2:
            return "no contract line and no placeholders"
        return parent + "\nMAGIC hint."


def test_evolve_improves_monotonically_and_selects_on_train():
    fit = FakeFitness()
    res = asyncio.run(reflective_evolve(
        SEED, fit, FakeReflect(), "fake-model", train_qids=TRAIN,
        val_qids=VAL, generations=3, children=2))
    assert res.stopped_reason == "completed"
    assert res.best_template.count("MAGIC") == 3
    assert res.best_train_score == 30.0
    assert res.val_score == -3.0              # reported, NOT selected on
    train_bests = [max(h["train_score"] for h in res.history[:i + 1])
                   for i in range(len(res.history))]
    assert train_bests == sorted(train_bests)  # monotone best-so-far


def test_contract_violating_children_skipped():
    fit = FakeFitness()
    res = asyncio.run(reflective_evolve(
        SEED, fit, FakeReflect(), "fake-model", train_qids=TRAIN,
        generations=1, children=2))
    # seed + 1 valid child (the call-2 child was invalid)
    assert len(res.history) == 2


def test_budget_exceeded_returns_best_so_far():
    fit = FakeFitness(fail_after=2)            # seed + first child only
    res = asyncio.run(reflective_evolve(
        SEED, fit, FakeReflect(), "fake-model", train_qids=TRAIN,
        val_qids=VAL, generations=5, children=2))
    assert res.stopped_reason == "budget"
    assert res.best_train_score == 10.0
    assert res.val_score is None


def test_placeholder_dropping_child_rejected():
    class DropReflect:
        async def invoke(self, model, prompt, temperature, run_index=0):
            return "Just say something.\nProbability: ZZ%"   # drops fields

    fit = FakeFitness()
    res = asyncio.run(reflective_evolve(
        SEED, fit, DropReflect(), "fake-model", train_qids=TRAIN,
        generations=2, children=1))
    assert len(res.history) == 1               # only the seed survived
    assert res.best_template == SEED
