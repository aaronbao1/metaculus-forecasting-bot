"""Model release dates — the contamination boundary (spec §3.4).

clean iff question.open_time > max RELEASE DATE across the ensemble's
models. Release date, NOT stated training cutoff: cutoff labels
demonstrably lie; release dates are the trustworthy bound.

Dates VERIFIED 2026-06-11 against provider announcements, cross-checked
with the OpenRouter models API `created` timestamps (which anchor when the
slug became callable by this bot). Where the two differed, the LATER date
was chosen (conservative: fewer questions labeled "clean").

ANY NEW ENSEMBLE MEMBER needs the same verification before its first
stratified report; unknown slugs fail safe (hash as "unknown" and the
"u" > digit string-compare makes every question "contaminated").
The slugs mirror bot/config.py:_default_ensemble(). Note OpenRouter
aliases the configured "claude-opus-4-6" to canonical "claude-opus-4.6".
"""
from __future__ import annotations

import hashlib

from backtest.core import BacktestQuestion
from backtest.corpus.store import CorpusStore

RELEASE_DATES: dict[str, str] = {
    # OpenRouter slug -> iso public release date. Verified 2026-06-11:
    # GPT-5.4: openai.com/index/introducing-gpt-5-4 (TechCrunch 2026-03-05);
    # OpenRouter created 2026-03-05 — agrees.
    "openai/gpt-5.4": "2026-03-05",
    # Claude Opus 4.6: anthropic.com/news/claude-opus-4-6 (TechCrunch
    # 2026-02-05); OpenRouter created 2026-02-04 — later date kept.
    "anthropic/claude-opus-4-6": "2026-02-05",
    # Gemini 3.1 Pro preview: blog.google .../gemini-3-1-pro (9to5Google
    # 2026-02-19); OpenRouter created 2026-02-19 — agrees.
    "google/gemini-3.1-pro-preview": "2026-02-19",
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
