"""Robust extraction of forecasts from LLM text. One units/parse bug cost a
Fall-2025 bot ~80 points; parsers raise ParseError so callers can reprompt
once, then drop the ensemble member (never submit garbage)."""
from __future__ import annotations

import re


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


TIERS = ("near_certain", "goldilocks", "noise")


def parse_tier(text: str) -> str:
    """Last 'Tier: X' line; X in TIERS (case-insensitive)."""
    matches = re.findall(r"Tier:\s*(near_certain|goldilocks|noise)",
                         text, re.IGNORECASE)
    if not matches:
        raise ParseError("no 'Tier: X' line found")
    return matches[-1].lower()
