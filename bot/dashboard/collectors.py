"""Collectors for the live dashboard.

Pure parsers (``parse_*``) transform already-fetched JSON and are unit-tested
against fixtures. Network fetchers (``fetch_*``) do I/O and return the parser
output, or an ``{"available": False, ...}`` sentinel on any failure so one dead
source never breaks the page. Collectors return ONLY derived numbers — never
secrets (API keys, account ids).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
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
    if zero is None or lo == zero:
        return lo + scaled * (hi - lo)
    # log scaling: value = lo + (hi-lo) * ((deriv**scaled) - 1) / (deriv - 1)
    deriv = (hi - zero) / (lo - zero)
    if deriv == 1:
        return lo + scaled * (hi - lo)
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
        if len(fv) > 1:
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


_METACULUS_POSTS = "https://www.metaculus.com/api/posts/"

# A refresh makes ~10 Metaculus calls; the API rate-limits bursts (429). Space
# the calls and back off once on 429. This runs in the background refresh thread,
# so the small sleeps never block the server's request handling.
_MIN_METACULUS_SPACING = 0.5
_last_metaculus_call = 0.0


def _metaculus_get(url: str, token: str, params: dict | None = None,
                   timeout: float = 15.0):
    """Throttled GET vs Metaculus with a single 429 backoff-retry."""
    global _last_metaculus_call
    resp = None
    for attempt in range(2):
        gap = _MIN_METACULUS_SPACING - (time.monotonic() - _last_metaculus_call)
        if gap > 0:
            time.sleep(gap)
        _last_metaculus_call = time.monotonic()
        resp = httpx.get(url, headers={"Authorization": f"Token {token}"},
                         params=params, timeout=timeout)
        if resp.status_code == 429 and attempt == 0:
            time.sleep(2.0)
            continue
        break
    return resp


def fetch_posts(token: str, tournament: str, forecaster_id: int | None,
                statuses: str = "open", page_size: int = 100,
                max_pages: int = 50, timeout: float = 15.0) -> list[dict]:
    """Paginate /api/posts/ for one tournament (count is always null → follow offset)."""
    results: list[dict] = []
    for page in range(max_pages):
        params = {"tournaments": tournament, "statuses": statuses,
                  "limit": page_size, "offset": page * page_size}
        if forecaster_id:
            params["forecaster_id"] = forecaster_id
        resp = _metaculus_get(_METACULUS_POSTS, token, params, timeout)
        if resp.status_code != 200:
            break
        batch = resp.json().get("results") or []
        results.extend(batch)
        if not batch:
            break
    return results


def fetch_metaculus_coverage(token: str, tournaments: list[str]) -> dict:
    """Per-tournament open-vs-forecasted coverage + the gap list."""
    if not token:
        return {"available": False, "error": "no token"}
    try:
        per: dict[str, dict] = {}
        for t in tournaments:
            open_posts = fetch_posts(token, t, forecaster_id=None)
            mine = fetch_posts(token, t, forecaster_id=303527)
            per[t] = parse_coverage(open_posts, {p["id"] for p in mine})
        return {"available": True, "tournaments": per}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def fetch_post_detail(token: str, post_id: int, timeout: float = 15.0) -> dict:
    """GET one post (detail endpoint carries my_forecasts + resolution + score)."""
    resp = _metaculus_get(f"{_METACULUS_POSTS}{post_id}/", token, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


_RESOLVED_CACHE = "logs/dashboard_resolved_cache.json"


def _load_resolved_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - a corrupt cache must never break the panel
        return {}


def _save_resolved_cache(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:  # noqa: BLE001
        pass


def fetch_recent_resolved(token: str, tournaments: list[str], limit: int = 8,
                          cache_path: str = _RESOLVED_CACHE) -> dict:
    """Resolved questions mopsus forecast: our value, the outcome, our score.
    Resolved questions are immutable, so detail is fetched once and cached to disk;
    steady-state this makes ~0 detail calls (the calls that rate-limit us)."""
    if not token:
        return {"available": False, "error": "no token"}
    try:
        cache = _load_resolved_cache(cache_path)
        ids: list[int] = []
        for t in tournaments:
            ids += [p["id"] for p in
                    fetch_posts(token, t, forecaster_id=303527, statuses="resolved")]
        new = [pid for pid in ids if str(pid) not in cache]
        for pid in new:
            detail = fetch_post_detail(token, pid)
            entry = extract_own_forecast(detail)
            entry["title"] = detail.get("title")
            cache[str(pid)] = entry
        if new:
            _save_resolved_cache(cache_path, cache)
        entries = [cache[str(pid)] for pid in ids if str(pid) in cache]
        entries.sort(key=lambda e: e.get("start_time") or 0, reverse=True)
        return {"available": True, "resolved": entries[:limit], "total": len(cache)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


_REPO = "aaronbao1/metaculus-forecasting-bot"


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def parse_github_runs(stdout: str, now_iso: str | None = None) -> dict:
    """Shape `gh run list --json ...` output into the run-health panel."""
    runs_raw = (json.loads(stdout) if stdout.strip() else [])
    now = _parse_iso(now_iso) if now_iso else _now()
    runs = []
    last_success_dt = None
    failures = 0
    for r in runs_raw:
        started = _parse_iso(r["startedAt"]) if r.get("startedAt") else None
        updated = _parse_iso(r["updatedAt"]) if r.get("updatedAt") else None
        dur = int((updated - started).total_seconds()) if started and updated else None
        if r.get("conclusion") == "failure":
            failures += 1
        if r.get("conclusion") == "success" and last_success_dt is None:
            last_success_dt = updated
        runs.append({
            "id": r.get("databaseId"), "event": r.get("event"),
            "conclusion": r.get("conclusion"), "status": r.get("status"),
            "created_at": r.get("createdAt"), "duration_s": dur,
        })
    last_min = (int((now - last_success_dt).total_seconds() // 60)
                if last_success_dt else None)
    return {
        "available": True, "runs": runs,
        "recent_failures": failures,
        "last_success_minutes_ago": last_min,
        "cadence_stale": (last_min is not None and last_min > 40),
    }


def fetch_github_runs(limit: int = 15, now_iso: str | None = None) -> dict:
    """Run `gh run list` for the run-bot workflow; sentinel on any failure."""
    try:
        cp = subprocess.run(
            ["gh", "run", "list", "--workflow", "run-bot", "--repo", _REPO,
             "-L", str(limit), "--json",
             "databaseId,status,conclusion,createdAt,startedAt,updatedAt,event"],
            capture_output=True, text=True, timeout=20,
        )
        if cp.returncode != 0:
            return {"available": False, "error": "gh nonzero"}
        return parse_github_runs(cp.stdout, now_iso=now_iso)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def fetch_armed_state() -> dict:
    """Workflow active?, SUBMIT_PREDICTION var, secret names present."""
    try:
        def _gh(args):
            return subprocess.run(["gh", *args, "--repo", _REPO],
                                  capture_output=True, text=True, timeout=20)
        wf = _gh(["workflow", "list"])
        var = _gh(["variable", "list"])
        sec = _gh(["secret", "list"])
        submit = None
        for line in var.stdout.splitlines():
            if line.startswith("SUBMIT_PREDICTION"):
                submit = line.split()[1] if len(line.split()) > 1 else None
        return {
            "available": True,
            "run_bot_active": any(
                l.startswith("run-bot") and "active" in l for l in wf.stdout.splitlines()),
            "submit_prediction": submit,
            "secrets": [l.split()[0] for l in sec.stdout.splitlines() if l.strip()],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def fetch_run_summary(path: str = "logs/runs.jsonl") -> dict:
    """Aggregate the run-summary JSONL (truncation/retry signal)."""
    if not os.path.exists(path):
        return {"available": False}
    try:
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if not rows:
            return {"available": False}
        return {
            "available": True,
            "total_truncated": sum(r.get("truncated_count", 0) for r in rows),
            "total_recovered": sum(r.get("retry_recovered", 0) for r in rows),
            "latest": rows[-1],
            "count": len(rows),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
