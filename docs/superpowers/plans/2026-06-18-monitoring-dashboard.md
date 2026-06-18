# Monitoring Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `scripts/dashboard.py` that reads the bot's local logs/artifacts (and makes one live OpenRouter usage call) and writes a self-contained `dashboard.html` covering API costs, questions/forecasts, progress, backtest performance, and the improvement roadmap.

**Architecture:** One module with three layers kept separate for testability — (1) **pure aggregators/parsers** over already-loaded data (no I/O), (2) **pure renderers** that turn aggregated data + inline-SVG charts into HTML strings, (3) a thin **`main()`** that loads files, makes the OpenRouter call, and writes the file. Every data source is best-effort: a missing/failing source renders an "unavailable" panel, never crashes the page. The HTML embeds inline CSS + Python-generated inline SVG — no JS, no CDN, fully offline.

**Tech Stack:** Python 3.12, `httpx` (already a dep) for the OpenRouter call, `python-dotenv` for the key, stdlib for everything else. Tests via `pytest` (`uv run pytest`).

---

## File Structure

- **Create `scripts/dashboard.py`** — the whole dashboard: aggregators, parsers, SVG helpers, renderers, `main()`. One focused file (~350 lines); it is a leaf tool with no importers besides its test.
- **Create `tests/test_dashboard.py`** — unit tests for every pure function + an HTML smoke test. No network, no browser.
- **Modify `.gitignore`** — add `dashboard.html` (generated snapshot embedding live balance).

Data contracts (verified against real files):
- `logs/forecasts.jsonl` rows: `{type, final, config_hash, research_providers (comma-str), page_url, logged_at (ISO-8601), question_id, ...}`
- `logs/backtest/<spec>/<run>/scores.jsonl` rows: `{brier, log_score, baseline_score, h2h_market, h2h_cp, stratum, platform, qid, resolution, ...}`
- `logs/backtest/<spec>/<run>/manifest.json`: `{run_id, spec_name, actual_cost_usd, corpus_root, statuses}`
- `research-notes/08-improvement-roadmap.md` headings: `### N. <title> — **<tag>**` (tag optional)
- OpenRouter `GET /api/v1/key` → `{"data": {"usage", "limit", "limit_remaining", "usage_daily", "usage_weekly", "usage_monthly", "limit_reset"}}`

---

## Task 1: Forecast-log aggregator

**Files:**
- Create: `scripts/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py
from scripts.dashboard import summarize_forecasts


def test_summarize_forecasts_counts_types_providers_and_timeline():
    rows = [
        {"type": "binary", "final": 0.2, "config_hash": "aaa",
         "research_providers": "asknews,search_llm,markets",
         "logged_at": "2026-06-11T16:32:02.7+00:00"},
        {"type": "binary", "final": 0.9, "config_hash": "aaa",
         "research_providers": "search_llm,markets",
         "logged_at": "2026-06-11T17:00:00+00:00"},
        {"type": "numeric", "final": {"50": 3.0}, "config_hash": "bbb",
         "research_providers": "", "logged_at": "2026-06-12T09:00:00+00:00"},
    ]
    s = summarize_forecasts(rows)
    assert s["total"] == 3
    assert s["by_type"] == {"binary": 2, "numeric": 1}
    assert s["provider_coverage"]["markets"] == 2
    assert s["provider_coverage"]["asknews"] == 1
    assert s["provider_coverage"]["none"] == 1
    assert s["by_config"] == {"aaa": 2, "bbb": 1}
    assert s["binary_finals"] == [0.2, 0.9]
    assert s["timeline"] == [("2026-06-11", 2), ("2026-06-12", 1)]
    assert s["last_forecast"] == "2026-06-12T09:00:00+00:00"


def test_summarize_forecasts_empty():
    s = summarize_forecasts([])
    assert s["total"] == 0
    assert s["binary_finals"] == []
    assert s["last_forecast"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.dashboard'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/dashboard.py
"""Self-contained monitoring dashboard for the Metaculus bot.

Reads local logs/artifacts (+ one live OpenRouter usage call) and writes a
self-contained, offline dashboard.html (inline CSS + inline SVG, no JS/CDN).
Run: uv run python scripts/dashboard.py [--out dashboard.html]
"""
from __future__ import annotations

from collections import Counter

RESEARCH_PROVIDERS = ("asknews", "search_llm", "markets")


def summarize_forecasts(rows: list[dict]) -> dict:
    by_type = Counter(r.get("type") for r in rows if r.get("type"))
    by_config = Counter(r.get("config_hash") for r in rows if r.get("config_hash"))
    provider_coverage: Counter = Counter()
    for r in rows:
        provs = [p for p in (r.get("research_providers") or "").split(",") if p]
        if not provs:
            provider_coverage["none"] += 1
        for p in provs:
            provider_coverage[p] += 1
    binary_finals = [float(r["final"]) for r in rows
                     if r.get("type") == "binary"
                     and isinstance(r.get("final"), (int, float))]
    day_counts: Counter = Counter()
    last = None
    for r in rows:
        ts = r.get("logged_at")
        if not ts:
            continue
        day_counts[str(ts)[:10]] += 1
        if last is None or ts > last:
            last = ts
    return {
        "total": len(rows),
        "by_type": dict(by_type),
        "by_config": dict(by_config),
        "provider_coverage": dict(provider_coverage),
        "binary_finals": binary_finals,
        "timeline": sorted(day_counts.items()),
        "last_forecast": last,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): forecast-log aggregator"
```

---

## Task 2: Backtest-run aggregator

**Files:**
- Modify: `scripts/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py  (append)
import json
from scripts.dashboard import summarize_backtest_scores, load_backtest_runs


def test_summarize_backtest_scores_means_and_counts():
    scores = [
        {"brier": 0.10, "log_score": -0.3, "baseline_score": 12.0,
         "h2h_market": 5.0, "h2h_cp": None},
        {"brier": 0.20, "log_score": -0.5, "baseline_score": 8.0,
         "h2h_market": 3.0, "h2h_cp": 2.0},
        {"brier": None, "log_score": None, "baseline_score": None,
         "h2h_market": None, "h2h_cp": None},  # unscored row ignored
    ]
    manifest = {"run_id": "current-bot-clean-0-91c43068",
                "spec_name": "current-bot-clean", "actual_cost_usd": 1.23}
    s = summarize_backtest_scores(scores, manifest)
    assert s["run_id"] == "current-bot-clean-0-91c43068"
    assert s["n_scored"] == 2
    assert s["mean_brier"] == 0.15
    assert s["mean_log_score"] == -0.4
    assert s["mean_h2h_market"] == 4.0
    assert s["mean_h2h_cp"] == 2.0          # only the one non-null
    assert s["actual_cost_usd"] == 1.23


def test_load_backtest_runs_discovers_and_sorts(tmp_path):
    root = tmp_path / "backtest"
    run = root / "spec" / "spec-0-abc"
    run.mkdir(parents=True)
    (run / "scores.jsonl").write_text(
        json.dumps({"brier": 0.1, "log_score": -0.2, "baseline_score": 1.0,
                    "h2h_market": None, "h2h_cp": None}) + "\n")
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "spec-0-abc", "spec_name": "spec",
                    "actual_cost_usd": 0.5}))
    runs = load_backtest_runs(root)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "spec-0-abc"
    assert runs[0]["n_scored"] == 1


def test_load_backtest_runs_missing_root_returns_empty(tmp_path):
    assert load_backtest_runs(tmp_path / "nope") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: FAIL — `ImportError: cannot import name 'summarize_backtest_scores'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/dashboard.py  (add imports at top)
import json
from pathlib import Path


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize_backtest_scores(scores: list[dict], manifest: dict) -> dict:
    scored = [r for r in scores if r.get("log_score") is not None]
    return {
        "run_id": manifest.get("run_id"),
        "spec_name": manifest.get("spec_name"),
        "n_scored": len(scored),
        "n_total": len(scores),
        "mean_brier": _mean([r.get("brier") for r in scores]),
        "mean_log_score": _mean([r.get("log_score") for r in scores]),
        "mean_baseline": _mean([r.get("baseline_score") for r in scores]),
        "mean_h2h_market": _mean([r.get("h2h_market") for r in scores]),
        "mean_h2h_cp": _mean([r.get("h2h_cp") for r in scores]),
        "actual_cost_usd": manifest.get("actual_cost_usd"),
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def load_backtest_runs(root: Path) -> list[dict]:
    """Discover <root>/<spec>/<run>/ dirs with scores.jsonl + manifest.json."""
    root = Path(root)
    if not root.exists():
        return []
    runs = []
    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        run_dir = manifest_path.parent
        scores_path = run_dir / "scores.jsonl"
        if not scores_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            scores = _read_jsonl(scores_path)
        except (json.JSONDecodeError, OSError):
            continue
        summary = summarize_backtest_scores(scores, manifest)
        summary["report_path"] = str(run_dir / "report.md") \
            if (run_dir / "report.md").exists() else None
        runs.append(summary)
    return sorted(runs, key=lambda r: r.get("run_id") or "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): backtest-run aggregator"
```

---

## Task 3: Roadmap parser

**Files:**
- Modify: `scripts/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py  (append)
from scripts.dashboard import parse_roadmap


def test_parse_roadmap_extracts_rank_title_and_tag():
    md = (
        "# heading noise\n"
        "### 1. Ship a calibration layer now: Platt vs extremization "
        "— **IMMEDIATE, $0 replay**\n"
        "some body text\n"
        "### 2. Swap regressed ensemble members — newest != best\n"
        "### 10. Research composition: restore outside-view track\n"
    )
    items = parse_roadmap(md)
    assert [i["rank"] for i in items] == [1, 2, 10]
    assert items[0]["title"].startswith("Ship a calibration layer now")
    assert items[0]["tag"] == "IMMEDIATE, $0 replay"
    assert items[1]["tag"] == ""          # no bold tag
    assert "Swap regressed ensemble members" in items[1]["title"]


def test_parse_roadmap_empty_text():
    assert parse_roadmap("") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: FAIL — `ImportError: cannot import name 'parse_roadmap'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/dashboard.py  (add import at top)
import re

_ROADMAP_RE = re.compile(r"^###\s+(\d+)\.\s+(.*)$")
_TAG_RE = re.compile(r"\s*—\s*\*\*(.+?)\*\*\s*$")


def parse_roadmap(md_text: str) -> list[dict]:
    items = []
    for line in md_text.splitlines():
        m = _ROADMAP_RE.match(line.strip())
        if not m:
            continue
        rank = int(m.group(1))
        rest = m.group(2).strip()
        tag = ""
        tm = _TAG_RE.search(rest)
        if tm:
            tag = tm.group(1).strip()
            rest = rest[:tm.start()].strip()
        items.append({"rank": rank, "title": rest, "tag": tag})
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): roadmap parser"
```

---

## Task 4: OpenRouter usage (pure parser + network fetch)

**Files:**
- Modify: `scripts/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py  (append)
from scripts.dashboard import parse_openrouter_key, fetch_openrouter_usage


def test_parse_openrouter_key_maps_fields():
    data = {"data": {"usage": 120.37, "limit": 100, "limit_remaining": 29.29,
                     "usage_daily": 0.5, "usage_weekly": 70.7,
                     "usage_monthly": 120.37, "limit_reset": "weekly"}}
    u = parse_openrouter_key(data)
    assert u["usage"] == 120.37
    assert u["limit"] == 100
    assert u["limit_remaining"] == 29.29
    assert u["usage_weekly"] == 70.7
    assert u["limit_reset"] == "weekly"


def test_parse_openrouter_key_tolerates_missing_fields():
    u = parse_openrouter_key({"data": {"usage": 5.0}})
    assert u["usage"] == 5.0
    assert u["limit"] is None
    assert u["usage_weekly"] is None


def test_fetch_openrouter_usage_no_key_returns_none():
    assert fetch_openrouter_usage(None) is None
    assert fetch_openrouter_usage("") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: FAIL — `ImportError: cannot import name 'parse_openrouter_key'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/dashboard.py  (add)
def parse_openrouter_key(data: dict) -> dict:
    d = (data or {}).get("data", {}) or {}
    keys = ("usage", "limit", "limit_remaining", "usage_daily",
            "usage_weekly", "usage_monthly", "limit_reset")
    return {k: d.get(k) for k in keys}


def fetch_openrouter_usage(api_key: str | None, *, timeout: float = 10.0) -> dict | None:
    """Live OpenRouter key usage. Returns None on any failure (offline, 4xx)."""
    if not api_key:
        return None
    try:
        import httpx
        resp = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return parse_openrouter_key(resp.json())
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): OpenRouter usage parser + fetch"
```

---

## Task 5: Inline-SVG chart + numeric helpers

**Files:**
- Modify: `scripts/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py  (append)
from scripts.dashboard import svg_bars, svg_gauge, histogram, esc


def test_histogram_buckets_into_ten_bins():
    h = histogram([0.05, 0.15, 0.15, 0.99])
    assert len(h) == 10
    assert h[0] == ("0.0–0.1", 1)
    assert h[1] == ("0.1–0.2", 2)
    assert h[9] == ("0.9–1.0", 1)


def test_svg_bars_renders_each_pair():
    out = svg_bars([("binary", 2), ("numeric", 1)])
    assert out.startswith("<svg")
    assert "binary" in out and "numeric" in out
    assert out.count("<rect") == 2


def test_svg_bars_empty_is_safe():
    assert svg_bars([]).startswith("<svg")


def test_svg_gauge_clamps_and_colors():
    assert svg_gauge(50, 100).startswith("<svg")
    assert svg_gauge(200, 100).startswith("<svg")   # over-limit clamps, no crash
    assert svg_gauge(1, 0).startswith("<svg")        # zero limit, no div-by-zero


def test_esc_escapes_html():
    assert esc("<a>&\"") == "&lt;a&gt;&amp;&quot;"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: FAIL — `ImportError: cannot import name 'svg_bars'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/dashboard.py  (add)
from html import escape as _escape


def esc(s) -> str:
    return _escape(str(s), quote=True)


def _fmt_num(x) -> str:
    if isinstance(x, float):
        return f"{x:.2f}".rstrip("0").rstrip(".") if abs(x) < 1000 else f"{x:.0f}"
    return str(x)


def histogram(values: list[float], lo: float = 0.0, hi: float = 1.0,
              bins: int = 10) -> list[tuple[str, int]]:
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / width)
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    return [(f"{lo + i*width:.1f}–{lo + (i+1)*width:.1f}", counts[i])
            for i in range(bins)]


def svg_bars(pairs, *, width: int = 460, bar_h: int = 22, gap: int = 6,
             color: str = "#5b8def", label_w: int = 140) -> str:
    pairs = list(pairs)
    if not pairs:
        return '<svg width="10" height="10" xmlns="http://www.w3.org/2000/svg"></svg>'
    mx = max((v for _, v in pairs), default=1) or 1
    track_w = max(1, width - label_w - 50)
    height = len(pairs) * (bar_h + gap) + gap
    parts = []
    for i, (label, value) in enumerate(pairs):
        y = gap + i * (bar_h + gap)
        w = max(1, int(track_w * (value / mx)))
        ty = y + int(bar_h * 0.7)
        parts.append(
            f'<text x="0" y="{ty}" font-size="13" fill="#cdd">{esc(label)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{w}" height="{bar_h}" rx="3" fill="{color}"/>'
            f'<text x="{label_w + w + 6}" y="{ty}" font-size="12" fill="#9ab">{_fmt_num(value)}</text>'
        )
    return (f'<svg width="{width}" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg" role="img">{"".join(parts)}</svg>')


def svg_gauge(used, limit, *, width: int = 260, height: int = 18) -> str:
    frac = 0.0
    if limit and limit > 0 and used is not None:
        frac = max(0.0, min(1.0, used / limit))
    fill_w = int(width * frac)
    color = "#e5534b" if frac > 0.85 else "#d9a441" if frac > 0.6 else "#3fb950"
    return (f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<rect x="0" y="0" width="{width}" height="{height}" rx="4" fill="#21262d"/>'
            f'<rect x="0" y="0" width="{fill_w}" height="{height}" rx="4" fill="{color}"/>'
            f'</svg>')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: PASS (15 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): inline-SVG chart + numeric helpers"
```

---

## Task 6: HTML page renderer

**Files:**
- Modify: `scripts/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py  (append)
from scripts.dashboard import render_page


def _ctx():
    return {
        "generated_at": "2026-06-18T00:00:00+00:00",
        "status": {"workflow_state": "disabled", "submit_prediction": "false",
                   "origin": "in sync", "last_forecast": "2026-06-12T09:00:00+00:00",
                   "total": 58},
        "costs": {"usage": 120.37, "limit": 100, "limit_remaining": 29.29,
                  "usage_weekly": 70.7, "usage_daily": 0.5, "usage_monthly": 120.37,
                  "limit_reset": "weekly"},
        "forecasts": {"total": 58, "by_type": {"binary": 20, "numeric": 24},
                      "by_config": {"aaa": 58},
                      "provider_coverage": {"asknews": 30, "markets": 58, "none": 2},
                      "binary_finals": [0.2, 0.9], "timeline": [("2026-06-11", 58)],
                      "last_forecast": "2026-06-12T09:00:00+00:00"},
        "backtests": [{"run_id": "current-bot-clean-0-91c43068",
                       "spec_name": "current-bot-clean", "n_scored": 121,
                       "n_total": 485, "mean_brier": 0.094, "mean_log_score": -0.3,
                       "mean_baseline": 12.0, "mean_h2h_market": 5.31,
                       "mean_h2h_cp": None, "actual_cost_usd": 1.23,
                       "report_path": "logs/backtest/x/report.md"}],
        "roadmap": [{"rank": 1, "title": "Ship calibration layer", "tag": "IMMEDIATE, $0"}],
    }


def test_render_page_is_self_contained_html_with_all_panels():
    html = render_page(_ctx())
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower()        # no JS
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")  # no external links
    for marker in ("API costs", "Questions", "Backtest", "Future directions",
                   "current-bot-clean-0-91c43068", "Ship calibration layer", "29.29"):
        assert marker in html


def test_render_page_handles_missing_costs():
    ctx = _ctx()
    ctx["costs"] = None
    html = render_page(ctx)
    assert "unavailable" in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: FAIL — `ImportError: cannot import name 'render_page'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/dashboard.py  (add)
_CSS = """
:root{color-scheme:dark}
body{margin:0;background:#0d1117;color:#c9d1d9;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:#8b949e;font-size:12px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px}
.card h2{font-size:15px;margin:0 0 12px;color:#e6edf3}
.kv{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d}
.kv b{color:#e6edf3;font-weight:600}
.big{font-size:26px;font-weight:700;color:#e6edf3}
.muted{color:#8b949e;font-size:12px}
.status{display:flex;flex-wrap:wrap;gap:18px;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;margin-bottom:16px}
.status div span{display:block}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{text-align:left;padding:4px 6px;border-bottom:1px solid #21262d;vertical-align:top}
.tag{font-size:11px;color:#d9a441}
a{color:#58a6ff}
"""


def _kv(label: str, value) -> str:
    return f'<div class="kv"><span>{esc(label)}</span><b>{esc(value)}</b></div>'


def render_status(s: dict) -> str:
    items = [
        ("Cron workflow", s.get("workflow_state", "?")),
        ("SUBMIT_PREDICTION", s.get("submit_prediction", "?")),
        ("Git origin", s.get("origin", "?")),
        ("Forecasts logged", s.get("total", 0)),
        ("Last forecast", s.get("last_forecast") or "—"),
    ]
    cells = "".join(
        f'<div><span class="muted">{esc(k)}</span><span><b>{esc(v)}</b></span></div>'
        for k, v in items)
    return f'<div class="status">{cells}</div>'


def render_costs(c: dict | None) -> str:
    if not c:
        return ('<div class="card"><h2>💸 API costs</h2>'
                '<p class="muted">OpenRouter usage unavailable '
                '(offline or key error).</p></div>')
    used_wk = c.get("usage_weekly")
    limit = c.get("limit")
    gauge = svg_gauge(used_wk, limit) if limit else ""
    rows = [
        f'<div class="big">${_fmt_num(c.get("usage_weekly") or 0)}<span class="muted"> used this week</span></div>',
        gauge,
        _kv("Weekly limit", f'${_fmt_num(limit)}' if limit else "none"),
        _kv("Weekly remaining", f'${_fmt_num(c.get("limit_remaining"))}'
            if c.get("limit_remaining") is not None else "—"),
        _kv("Today", f'${_fmt_num(c.get("usage_daily") or 0)}'),
        _kv("This month", f'${_fmt_num(c.get("usage_monthly") or 0)}'),
        _kv("All-time usage", f'${_fmt_num(c.get("usage") or 0)}'),
        '<p class="muted">Key-level totals from OpenRouter (not per-question).</p>',
    ]
    return f'<div class="card"><h2>💸 API costs</h2>{"".join(rows)}</div>'


def render_forecasts(f: dict) -> str:
    by_type = sorted(f.get("by_type", {}).items(), key=lambda kv: -kv[1])
    prov = sorted(f.get("provider_coverage", {}).items(), key=lambda kv: -kv[1])
    hist = histogram(f.get("binary_finals", []))
    timeline = f.get("timeline", [])
    body = [
        f'<div class="big">{f.get("total", 0)}<span class="muted"> forecasts</span></div>',
        "<h3 class='muted'>By type</h3>", svg_bars(by_type),
        "<h3 class='muted'>Research-provider coverage</h3>", svg_bars(prov, color="#3fb950"),
        "<h3 class='muted'>Binary prediction distribution</h3>", svg_bars(hist, color="#a371f7"),
        "<h3 class='muted'>Forecasts per day</h3>", svg_bars(timeline, color="#d9a441"),
    ]
    return f'<div class="card"><h2>📊 Questions / forecasts</h2>{"".join(body)}</div>'


def render_progress(s: dict) -> str:
    return ('<div class="card"><h2>🚦 Progress</h2>'
            f'{_kv("Cron workflow", s.get("workflow_state", "?"))}'
            f'{_kv("Submit flag", s.get("submit_prediction", "?"))}'
            f'{_kv("Recent CI runs", s.get("ci_summary", "—"))}'
            '<p class="muted">Go live: <code>gh workflow enable run-bot</code> '
            '+ <code>gh variable set SUBMIT_PREDICTION --body true</code>.</p></div>')


def render_backtests(runs: list[dict]) -> str:
    if not runs:
        return ('<div class="card"><h2>🔬 Backtest performance</h2>'
                '<p class="muted">No runs under logs/backtest/.</p></div>')
    rows = []
    for r in runs:
        link = (f' · <a href="{esc(r["report_path"])}">report</a>'
                if r.get("report_path") else "")
        rows.append(
            f'<tr><td><b>{esc(r.get("run_id"))}</b>{link}</td>'
            f'<td>{r.get("n_scored", 0)}/{r.get("n_total", 0)}</td>'
            f'<td>{_fmt_num(r.get("mean_brier")) if r.get("mean_brier") is not None else "—"}</td>'
            f'<td>{_fmt_num(r.get("mean_h2h_market")) if r.get("mean_h2h_market") is not None else "—"}</td>'
            f'<td>${_fmt_num(r.get("actual_cost_usd")) if r.get("actual_cost_usd") is not None else "—"}</td></tr>')
    return ('<div class="card" style="grid-column:1/-1"><h2>🔬 Backtest performance</h2>'
            '<table><tr><th>Run</th><th>scored</th><th>mean Brier</th>'
            '<th>h2h vs market</th><th>cost</th></tr>'
            f'{"".join(rows)}</table></div>')


def render_roadmap(items: list[dict]) -> str:
    if not items:
        return ('<div class="card"><h2>🧭 Future directions</h2>'
                '<p class="muted">Roadmap doc not found.</p></div>')
    rows = "".join(
        f'<tr><td>{i["rank"]}</td><td>{esc(i["title"])}</td>'
        f'<td class="tag">{esc(i["tag"])}</td></tr>' for i in items)
    return ('<div class="card" style="grid-column:1/-1"><h2>🧭 Future directions</h2>'
            '<p class="muted">From research-notes/08-improvement-roadmap.md</p>'
            f'<table><tr><th>#</th><th>Improvement</th><th>cost/status</th></tr>{rows}</table></div>')


def render_page(ctx: dict) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Forecasting bot dashboard</title><style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        "<h1>🤖 Forecasting bot dashboard</h1>"
        f"<div class='sub'>generated {esc(ctx.get('generated_at'))}</div>"
        f"{render_status(ctx.get('status', {}))}"
        "<div class='grid'>"
        f"{render_costs(ctx.get('costs'))}"
        f"{render_forecasts(ctx.get('forecasts', {}))}"
        f"{render_progress(ctx.get('status', {}))}"
        f"{render_backtests(ctx.get('backtests', []))}"
        f"{render_roadmap(ctx.get('roadmap', []))}"
        "</div></div></body></html>"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: PASS (17 passed). Note: the `test_render_page_is_self_contained_html_with_all_panels` "no external links" assertion allows the SVG namespace URI; keep all other URLs out of rendered HTML.

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): self-contained HTML page renderer"
```

---

## Task 7: `main()` wiring, status gathering, gitignore, manual run

**Files:**
- Modify: `scripts/dashboard.py`
- Modify: `.gitignore`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py  (append)
from scripts.dashboard import build_context


def test_build_context_assembles_all_sections(tmp_path, monkeypatch):
    # forecasts log
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "forecasts.jsonl").write_text(
        json.dumps({"type": "binary", "final": 0.3, "config_hash": "aaa",
                    "research_providers": "asknews,markets",
                    "logged_at": "2026-06-11T16:32:02+00:00"}) + "\n")
    # backtest run
    run = tmp_path / "logs" / "backtest" / "spec" / "spec-0-abc"
    run.mkdir(parents=True)
    (run / "scores.jsonl").write_text(
        json.dumps({"brier": 0.1, "log_score": -0.2, "baseline_score": 1.0,
                    "h2h_market": 2.0, "h2h_cp": None}) + "\n")
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "spec-0-abc", "spec_name": "spec",
                    "actual_cost_usd": 0.4}))
    # roadmap
    notes = tmp_path / "research-notes"
    notes.mkdir()
    (notes / "08-improvement-roadmap.md").write_text("### 1. Do the thing — **$0**\n")

    # no network: force costs None
    monkeypatch.setattr("scripts.dashboard.fetch_openrouter_usage", lambda *a, **k: None)
    ctx = build_context(root=tmp_path, openrouter_key=None, generated_at="T")
    assert ctx["forecasts"]["total"] == 1
    assert ctx["backtests"][0]["run_id"] == "spec-0-abc"
    assert ctx["roadmap"][0]["title"] == "Do the thing"
    assert ctx["costs"] is None
    assert ctx["generated_at"] == "T"


def test_build_context_missing_files_is_safe(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.dashboard.fetch_openrouter_usage", lambda *a, **k: None)
    ctx = build_context(root=tmp_path, openrouter_key=None, generated_at="T")
    assert ctx["forecasts"]["total"] == 0
    assert ctx["backtests"] == []
    assert ctx["roadmap"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_context'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/dashboard.py  (add)
import argparse
import os
import subprocess
from datetime import datetime, timezone


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def gather_status(root: Path) -> dict:
    """Best-effort live status via gh/git. Any failure -> 'unknown'."""
    wf = _run(["gh", "workflow", "list", "--all"]) or ""
    workflow_state = "unknown"
    for line in wf.splitlines():
        if "run-bot" in line:
            workflow_state = "disabled" if "disabled" in line else "active"
    submit = _run(["gh", "variable", "list"]) or ""
    submit_pred = "unknown"
    for line in submit.splitlines():
        if line.startswith("SUBMIT_PREDICTION"):
            submit_pred = line.split()[1] if len(line.split()) > 1 else "?"
    ahead_behind = _run(["git", "-C", str(root), "rev-list",
                         "--left-right", "--count", "origin/main...HEAD"])
    origin = "unknown"
    if ahead_behind and len(ahead_behind.split()) == 2:
        behind, ahead = ahead_behind.split()
        origin = "in sync" if behind == "0" == ahead else f"ahead {ahead}, behind {behind}"
    ci = _run(["gh", "run", "list", "--workflow", "run-bot", "--limit", "3",
               "--json", "status,conclusion"])
    return {"workflow_state": workflow_state, "submit_prediction": submit_pred,
            "origin": origin, "ci_summary": (ci or "none")[:120]}


def build_context(root: Path, openrouter_key: str | None,
                  generated_at: str) -> dict:
    root = Path(root)
    fc_path = root / "logs" / "forecasts.jsonl"
    rows = _read_jsonl(fc_path) if fc_path.exists() else []
    forecasts = summarize_forecasts(rows)
    backtests = load_backtest_runs(root / "logs" / "backtest")
    rm_path = root / "research-notes" / "08-improvement-roadmap.md"
    roadmap = parse_roadmap(rm_path.read_text()) if rm_path.exists() else []
    costs = fetch_openrouter_usage(openrouter_key)
    status = gather_status(root)
    status["total"] = forecasts["total"]
    status["last_forecast"] = forecasts["last_forecast"]
    return {"generated_at": generated_at, "status": status, "costs": costs,
            "forecasts": forecasts, "backtests": backtests, "roadmap": roadmap}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate dashboard.html")
    parser.add_argument("--out", default="dashboard.html")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    key = os.getenv("OPENROUTER_API_KEY")
    ctx = build_context(Path(args.root), key,
                        datetime.now(timezone.utc).isoformat(timespec="seconds"))
    html = render_page(ctx)
    Path(args.out).write_text(html)
    print(f"wrote {args.out} "
          f"({ctx['forecasts']['total']} forecasts, "
          f"{len(ctx['backtests'])} backtest runs, "
          f"costs={'live' if ctx['costs'] else 'unavailable'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py -q`
Expected: PASS (19 passed)

- [ ] **Step 5: Add the generated file to .gitignore**

Append to `.gitignore`:

```
# generated monitoring dashboard (embeds live OpenRouter balance)
dashboard.html
```

- [ ] **Step 6: Generate the real dashboard and eyeball it**

Run: `uv run python scripts/dashboard.py`
Expected stdout: `wrote dashboard.html (58 forecasts, N backtest runs, costs=live)`
Then open `dashboard.html` in a browser (`open dashboard.html` on macOS) and confirm all six panels render with real numbers (costs may show "unavailable" if the OpenRouter call fails — that's acceptable).

- [ ] **Step 7: Run the full suite + commit**

```bash
uv run pytest -q          # expect: 527 passed (508 + 19), 1 skipped
git add scripts/dashboard.py tests/test_dashboard.py .gitignore
git commit -m "feat(dashboard): main() wiring, live status, gitignore"
```

---

## Self-Review

**1. Spec coverage** (against the approved design):
- API costs (live OpenRouter) → Task 4 + `render_costs` (Task 6) ✓
- Questions/forecasts → Task 1 + `render_forecasts` ✓
- Progress (workflow/submit/CI health) → `gather_status` + `render_progress` (Tasks 6–7) ✓
- Backtest performance → Task 2 + `render_backtests` ✓
- Future directions → Task 3 + `render_roadmap` ✓
- Self-contained HTML, offline, no JS/CDN → `render_page` + inline SVG (Tasks 5–6), asserted in test ✓
- Tested parsing helpers → Tasks 1–5 all TDD ✓
- Gitignored output → Task 7 Step 5 ✓
- Non-goal (Metaculus live standing) → excluded; Progress uses local log + Actions health ✓

**2. Placeholder scan:** none — every step has complete code/commands.

**3. Type consistency:** `summarize_forecasts`→`render_forecasts`, `summarize_backtest_scores`/`load_backtest_runs`→`render_backtests`, `parse_roadmap`→`render_roadmap`, `parse_openrouter_key`/`fetch_openrouter_usage`→`render_costs`, `build_context` keys (`generated_at/status/costs/forecasts/backtests/roadmap`) match `render_page` reads. `gather_status` adds `total`/`last_forecast` consumed by `render_status`. Consistent.

**Note for the implementer:** add each task's new code to the existing `scripts/dashboard.py` (don't recreate the file); imports listed per-task should be hoisted to the top of the module.
