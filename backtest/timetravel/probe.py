"""Memorization probe (Halawi App. A.2 analogue): ask each model for the
outcome directly, zero research. A confident-correct answer upgrades the
question's stratum to contaminated-confirmed for that model. Deterministic
(temperature=0, run_index=0) and routed through CachedLLM, so re-probing a
corpus is free after the first pass. Numeric/discrete questions have no
crisp confident-correct criterion and are never flagged by the probe."""
from __future__ import annotations

import logging

from backtest.core import BacktestQuestion, QType
from backtest.timetravel.llm_cache import CachedLLM

logger = logging.getLogger(__name__)

CONFIDENT = 0.95

_BINARY_PROBE_PROMPT = """Answer from memory only — do not reason about what might happen; report what
you already know happened. The question below has already resolved.

QUESTION
{title}

RESOLUTION CRITERIA
{resolution_criteria}

If you know how this resolved, state it. The very last line of your answer
must be exactly:
Probability: ZZ%
where ZZ is the probability that it resolved Yes."""

_MC_PROBE_PROMPT = """Answer from memory only — do not reason about what might happen; report what
you already know happened. The question below has already resolved.

QUESTION
{title}

RESOLUTION CRITERIA
{resolution_criteria}

The options were: {options}

If you know how this resolved, the very last line of your answer must be
exactly the winning option's name, verbatim, and nothing else."""


def _binary_confident_correct(completion: str, resolution: bool) -> bool:
    from bot.parsing import ParseError, parse_binary
    try:
        p = parse_binary(completion)
    except ParseError:
        return False
    return p >= CONFIDENT if resolution else p <= 1.0 - CONFIDENT


def _mc_confident_correct(completion: str, resolution: str) -> bool:
    lines = completion.strip().splitlines()
    last = lines[-1].strip().lower() if lines else ""
    return last == str(resolution).strip().lower()


async def memorization_probe(q: BacktestQuestion, models: list[str],
                             llm: CachedLLM) -> dict[str, bool]:
    """True = contaminated-confirmed for that model on this question."""
    if q.resolution is None:
        return {m: False for m in models}
    if q.qtype == QType.BINARY:
        prompt = _BINARY_PROBE_PROMPT.format(
            title=q.title, resolution_criteria=q.resolution_criteria)
        resolved = bool(q.resolution)

        def check(c: str) -> bool:
            return _binary_confident_correct(c, resolved)
    elif q.qtype == QType.MULTIPLE_CHOICE:
        prompt = _MC_PROBE_PROMPT.format(
            title=q.title, resolution_criteria=q.resolution_criteria,
            options=", ".join(q.options))
        winning = str(q.resolution)

        def check(c: str) -> bool:
            return _mc_confident_correct(c, winning)
    else:
        return {m: False for m in models}    # numeric/discrete: not probeable
    out: dict[str, bool] = {}
    for m in models:
        try:
            completion = await llm.invoke(m, prompt, temperature=0.0,
                                          run_index=0)
        except Exception as e:
            logger.warning("memorization probe failed for %s: %s", m, e)
            out[m] = False
            continue
        out[m] = check(completion)
    return out
