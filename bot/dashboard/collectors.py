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


def _unscale(scaled: float, scaling: dict) -> float:
    """Map a scaled [0,1] value to real units (linear; log if zero_point set)."""
    lo = scaling.get("range_min")
    hi = scaling.get("range_max")
    zero = scaling.get("zero_point")
    if lo is None or hi is None:
        return scaled
    if zero is None:
        return lo + scaled * (hi - lo)
    # log scaling: value = lo + (hi-lo) * ((deriv**scaled) - 1) / (deriv - 1)
    deriv = (hi - zero) / (lo - zero)
    return lo + (hi - lo) * ((deriv ** scaled) - 1.0) / (deriv - 1.0)


def extract_own_forecast(post: dict) -> dict:
    """Read mopsus's own latest forecast + score from a post JSON (all types)."""
    q = post.get("question") or {}
    qtype = q.get("type")
    mf = q.get("my_forecasts") or {}
    latest = mf.get("latest") or {}
    out = {
        "post_id": post.get("id"),
        "question_id": q.get("id"),
        "type": qtype,
        "status": post.get("status"),
        "resolution": q.get("resolution"),
        "start_time": latest.get("start_time"),
        "score": mf.get("score_data"),
        "summary": None,
    }
    fv = latest.get("forecast_values")
    if not fv:
        return out
    scaling = q.get("scaling") or {}
    if qtype == "binary":
        out["summary"] = fv[1]
    elif qtype in ("numeric", "date"):
        centers = latest.get("centers") or []
        lows = latest.get("interval_lower_bounds") or []
        highs = latest.get("interval_upper_bounds") or []
        if centers:
            out["median"] = _unscale(centers[0], scaling)
            out["summary"] = out["median"]
        if lows:
            out["q1"] = _unscale(lows[0], scaling)
        if highs:
            out["q3"] = _unscale(highs[0], scaling)
    elif qtype == "multiple_choice":
        options = q.get("options") or []
        pairs = list(zip(options, fv))
        out["options"] = pairs
        if pairs:
            top = max(pairs, key=lambda p: p[1])
            out["top_option"] = top[0]
            out["summary"] = top[0]
    return out


def _is_question_post(post: dict) -> bool:
    return post.get("question") is not None


def parse_coverage(open_posts: list[dict], forecasted_ids: set) -> dict:
    """open questions minus the ones mopsus has forecast = the coverage gap."""
    questions = [p for p in open_posts if _is_question_post(p)]
    gap = [
        {"post_id": p["id"], "title": p.get("title"),
         "type": (p.get("question") or {}).get("type")}
        for p in questions if p["id"] not in forecasted_ids
    ]
    return {
        "open_count": len(questions),
        "forecasted_count": sum(1 for p in questions if p["id"] in forecasted_ids),
        "gap_count": len(gap),
        "gap": gap,
    }
