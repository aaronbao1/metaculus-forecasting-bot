"""Question <-> market crosswalk (spec §3.3).

Candidate generation is LEXICAL: difflib title ratio + token Jaccard over
title+resolution_criteria, top_k candidates per question.
EMBEDDING SEAM: replace _lexical_score/_candidates with an embedding
index (top-20 NN recall >99.9% of true equivalents, arXiv 2601.01706)
without touching the LLM gate, the table schema, or market_for() — kept
lexical in the MVP to avoid heavy dependencies.

Verification: an injected async LLM callable gates each candidate on
resolution criteria AND dates — date mismatch is the dominant false
positive (arXiv 2512.02436). Verdicts (YES and NO) are cached in the
crosswalk table; re-runs never re-ask. A non-null manual_override column
always wins in market_for().

Note for reports: live bot market evidence is keyword search
(bot/markets.py market_research()); backtest market evidence is this
crosswalk — different in kind, stated in every report's comparability
section.
"""
from __future__ import annotations

import difflib
import re

from backtest.core import BacktestQuestion
from backtest.corpus.store import CorpusStore

MARKET_PLATFORMS = ("polymarket", "kalshi", "manifold")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_score(a: BacktestQuestion, b: BacktestQuestion) -> float:
    ratio = difflib.SequenceMatcher(
        None, a.title.lower(), b.title.lower()).ratio()
    ta = _tokens(a.title + " " + a.resolution_criteria)
    tb = _tokens(b.title + " " + b.resolution_criteria)
    union = ta | tb
    jacc = len(ta & tb) / len(union) if union else 0.0
    return 0.5 * ratio + 0.5 * jacc


def _fmt(q: BacktestQuestion) -> str:
    close = (q.scheduled_close_time.date().isoformat()
             if q.scheduled_close_time else "unknown")
    return (f"TITLE: {q.title}\n"
            f"RESOLUTION CRITERIA: {q.resolution_criteria}\n"
            f"CLOSE DATE: {close}")


def _verify_prompt(q: BacktestQuestion, m: BacktestQuestion) -> str:
    return (
        "You are verifying whether a forecasting question and a prediction"
        " market are equivalent. Answer YES only if BOTH hold:\n"
        "1. The resolution criteria describe the same underlying event"
        " with the same resolution conditions.\n"
        "2. The dates match: same resolution/close timeframe. A date"
        " mismatch means NO even when the topic is identical.\n\n"
        f"QUESTION:\n{_fmt(q)}\n\nMARKET:\n{_fmt(m)}\n\n"
        "Answer with YES or NO on the first line.")


async def build_crosswalk(store: CorpusStore, llm, *,
                          top_k: int = 50) -> int:
    """llm: async (prompt: str) -> str. Returns count of NEW verified
    matches. Verdict cache: any (qid, market_qid) row is final."""
    questions = store.query(platforms=["metaculus"], resolved_only=False)
    markets: list[BacktestQuestion] = []
    for p in MARKET_PLATFORMS:
        markets.extend(store.query(platforms=[p], resolved_only=False))
    cached = {(r["qid"], r["market_qid"]) for r in store.conn.execute(
        "SELECT qid, market_qid FROM crosswalk").fetchall()}
    matched = {r["qid"] for r in store.conn.execute(
        "SELECT qid FROM crosswalk WHERE verified = 1").fetchall()}

    new_matches = 0
    for q in questions:
        if q.qid in matched:
            continue                       # already has a verified market
        candidates = sorted(
            ((_lexical_score(q, m), m) for m in markets),
            key=lambda x: x[0], reverse=True)[:top_k]
        for score, m in candidates:
            if (q.qid, m.qid) in cached:
                continue                   # cached NO verdict is final
            answer = await llm(_verify_prompt(q, m))
            verified = answer.strip().upper().startswith("YES")
            store.conn.execute(
                "INSERT OR REPLACE INTO crosswalk"
                " (qid, market_qid, score, verified, manual_override)"
                " VALUES (?,?,?,?,NULL)",
                (q.qid, m.qid, score, int(verified)))
            store.conn.commit()
            cached.add((q.qid, m.qid))
            if verified:
                new_matches += 1
                break                      # one verified market suffices
    return new_matches


def market_for(store: CorpusStore, qid: str) -> str | None:
    """Best market qid for a question. manual_override wins."""
    row = store.conn.execute(
        "SELECT manual_override FROM crosswalk"
        " WHERE qid = ? AND manual_override IS NOT NULL", (qid,)).fetchone()
    if row:
        return row["manual_override"]
    row = store.conn.execute(
        "SELECT market_qid FROM crosswalk"
        " WHERE qid = ? AND verified = 1"
        " ORDER BY score DESC LIMIT 1", (qid,)).fetchone()
    return row["market_qid"] if row else None
