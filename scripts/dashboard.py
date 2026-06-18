"""Self-contained monitoring dashboard for the Metaculus bot.

Reads local logs/artifacts (+ one live OpenRouter usage call) and writes a
self-contained, offline dashboard.html (inline CSS + inline SVG, no JS/CDN).
Run: uv run python scripts/dashboard.py [--out dashboard.html]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from html import escape as _escape
from pathlib import Path

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
