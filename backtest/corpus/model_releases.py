"""Model release dates — the contamination boundary (spec §3.4).

clean iff question.open_time > max RELEASE DATE across the ensemble's
models. Release date, NOT stated training cutoff: cutoff labels
demonstrably lie; release dates are the trustworthy bound.

!!! PLACEHOLDER DATES — IMPLEMENTER MUST VERIFY BEFORE USE !!!
The dates below are set to 2026-06-11 (the design date) as the MAXIMALLY
CONSERVATIVE placeholder: with these values NOTHING in the corpus can be
labeled "clean", so a forgotten verification fails safe (no false clean
stratum) rather than silently contaminating the headline metric. Before the
first real stratified report, replace each with the model's verified public
release date (OpenRouter model page / provider announcement) and re-run
stratify over the corpus. The slugs mirror bot/config.py:_default_ensemble().
"""
from __future__ import annotations

import hashlib

from backtest.core import BacktestQuestion
from backtest.corpus.store import CorpusStore

RELEASE_DATES: dict[str, str] = {
    # OpenRouter slug -> iso release date. PLACEHOLDERS — see module docstring.
    "openai/gpt-5.4": "2026-06-11",
    "anthropic/claude-opus-4-6": "2026-06-11",
    "google/gemini-3.1-pro-preview": "2026-06-11",
}


def model_set_hash(model_set: list[str]) -> str:
    """sha256 of the sorted release dates (contract formula). Two model sets
    with identical release-date multisets intentionally share strata.
    Total: unknown slugs hash as "unknown" — never raises."""
    dates = sorted(RELEASE_DATES.get(m, "unknown") for m in model_set)
    return hashlib.sha256("|".join(dates).encode()).hexdigest()[:12]


def stratify(store: CorpusStore, q: BacktestQuestion,
             model_set: list[str]) -> str:
    """Write and return the contamination stratum for q under model_set.
    The memorization probe (backtest/timetravel/probe.py) may later upgrade
    "contaminated" to "contaminated-confirmed" via set_contamination."""
    h = model_set_hash(model_set)
    latest = max(RELEASE_DATES.get(m, "unknown") for m in model_set)
    if q.open_time is not None and q.open_time.date().isoformat() > latest:
        stratum = "clean"
    else:
        stratum = "contaminated"   # unknown open_time fails safe
    store.set_contamination(q.qid, h, stratum)
    return stratum
