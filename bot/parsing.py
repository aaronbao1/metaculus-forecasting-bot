"""Robust extraction of forecasts from LLM text. One units/parse bug cost a
Fall-2025 bot ~80 points; parsers raise ParseError so callers can reprompt
once, then drop the ensemble member (never submit garbage)."""
from __future__ import annotations

import re
from datetime import datetime, timezone


class ParseError(ValueError):
    pass


_NUM = r"(-?[\d,]+(?:\.\d+)?)"


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def parse_binary(text: str) -> float:
    """Last 'Probability: ZZ%' in the text, as a probability in [0.001, 0.999]."""
    matches = re.findall(r"Probability:\s*" + _NUM + r"\s*%", text, re.IGNORECASE)
    if not matches:
        raise ParseError("no 'Probability: ZZ%' line found")
    p = _to_float(matches[-1]) / 100.0
    return min(max(p, 0.001), 0.999)


def parse_multiple_choice(text: str, options: list[str]) -> dict:
    """Last 'OptionName: XX%' per option (case-insensitive); renormalized."""
    out = {}
    for opt in options:
        pattern = re.escape(opt) + r"\s*:\s*" + _NUM + r"\s*%"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            raise ParseError(f"option not found in output: {opt!r}")
        out[opt] = max(_to_float(matches[-1]) / 100.0, 0.0)
    total = sum(out.values())
    if total <= 0:
        raise ParseError("all options parsed as zero")
    return {k: v / total for k, v in out.items()}


def parse_percentiles(text: str) -> dict:
    """All 'Percentile P: value' lines; requires >= 5 distinct percentiles."""
    matches = re.findall(
        r"Percentile\s+(\d+(?:\.\d+)?)\s*:\s*" + _NUM, text, re.IGNORECASE)
    out = {}
    for pct, val in matches:
        out[float(pct)] = _to_float(val)   # later occurrences overwrite earlier
    if len(out) < 5:
        raise ParseError(f"only {len(out)} percentiles found, need >= 5")
    return out


def to_utc_timestamp(dt: datetime) -> float:
    """Unix seconds. A naive datetime is assumed UTC (NOT machine-local):
    naive datetime.timestamp() uses the local TZ, which is non-deterministic
    across machines and would corrupt llm_cache replay and the alignment
    between question bounds and elicited percentiles."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_DATE = r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?Z?)?)"


def _date_to_timestamp(s: str) -> float:
    s = s.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise ParseError(f"unparseable date: {s!r}") from e
    return to_utc_timestamp(dt)


def parse_date_percentiles(text: str) -> dict:
    """All 'Percentile P: <date>' lines as {percentile: unix_timestamp}.
    Dates may be YYYY-MM-DD or YYYY-MM-DD[ T]HH:MM[:SS][Z]; any value without an
    explicit offset is assumed UTC (bare dates -> midnight UTC). Requires >= 5
    distinct percentiles (else ParseError, so the member is dropped rather than
    submitting garbage)."""
    matches = re.findall(
        r"Percentile\s+(\d+(?:\.\d+)?)\s*:\s*" + _DATE, text, re.IGNORECASE)
    out = {}
    for pct, datestr in matches:
        out[float(pct)] = _date_to_timestamp(datestr)   # later overwrite earlier
    if len(out) < 5:
        raise ParseError(f"only {len(out)} date percentiles found, need >= 5")
    return out
