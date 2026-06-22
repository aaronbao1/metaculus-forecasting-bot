"""Collectors for the live dashboard.

Pure parsers (``parse_*``) transform already-fetched JSON and are unit-tested
against fixtures. Network fetchers (``fetch_*``) do I/O and return the parser
output, or an ``{"available": False, ...}`` sentinel on any failure so one dead
source never breaks the page. Collectors return ONLY derived numbers — never
secrets (API keys, account ids).
"""
from __future__ import annotations

from typing import Any

import httpx


def parse_openrouter_budget(payload: Any) -> dict:
    """Derive the cost panel from a GET /api/v1/auth/key response body."""
    data = (payload or {}).get("data") if isinstance(payload, dict) else None
    if not data:
        return {"available": False}
    reset = data.get("limit_reset")
    window_used = data.get(f"usage_{reset}") if reset else None
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    pct_used = None
    if isinstance(limit, (int, float)) and limit and isinstance(remaining, (int, float)):
        pct_used = (limit - remaining) / limit * 100.0
    return {
        "available": True,
        "limit": limit,
        "limit_reset": reset,
        "remaining": remaining,
        "window_used": window_used,
        "today": data.get("usage_daily"),
        "week": data.get("usage_weekly"),
        "month": data.get("usage_monthly"),
        "lifetime": data.get("usage"),
        "pct_used": pct_used,
    }


def fetch_openrouter_budget(api_key: str, timeout: float = 8.0) -> dict:
    """GET the OpenRouter key usage; never raise — return a sentinel on failure."""
    if not api_key:
        return {"available": False, "error": "no api key"}
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return {"available": False, "error": f"http {resp.status_code}"}
        return parse_openrouter_budget(resp.json())
    except Exception as exc:  # noqa: BLE001 - one dead source must not break the page
        return {"available": False, "error": str(exc)}
