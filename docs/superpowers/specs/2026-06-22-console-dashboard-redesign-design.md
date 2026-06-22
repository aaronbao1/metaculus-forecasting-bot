# Console Dashboard Redesign + Metaculus Request Reduction — Design Spec

**Date:** 2026-06-22
**Status:** Approved (brainstormed 2026-06-22)

Redesign the local monitoring dashboard's UI into a polished dark "Console", and
cut its Metaculus request volume (which is rate-limiting itself with 429s).

The `/api/status` JSON contract and `bot/dashboard/server.py` are unchanged, so the
UI and the data layer evolve independently.

---

## Component A — Console UI (polished dark)

### Decision
Console layout (dense, technical, ops-feel), **polished dark theme** (chosen over
auto/light). It's a standalone page, so the theme is a fixed hand-tuned dark palette
— not bound by the chat-widget constraints.

### What it renders
`bot/dashboard/page.py`'s `PAGE_HTML` is rewritten. The page JS fetches `/api/status`
every 10s and renders **structured panels** (no more `JSON.stringify` dump):

- **Header** — title, a live status dot (subtle CSS pulse, colored by overall health),
  "updated Ns ago" freshness counter.
- **Metric strip** (4 KPI tiles) — Status (Live / Degraded / Down), Cost·week
  (`window_used` / `limit` + `pct_used`), Coverage gap (0 = green, >0 = red), Last run
  (time + conclusion + duration).
- **Panels** — Coverage (per tournament: open · gap, lists the gap items if >0),
  Recent runs (time · conclusion dot · duration; cadence-stale flag in the header),
  Resolved · our score (question → outcome, our forecast, peer-score chip), Armed
  state (workflow/submit/secrets), Cost detail (today/week/month/lifetime + a budget
  bar), and a full-width **Alerts** strip that lists only real signals (cadence stale,
  truncation > 0, budget high, gap > 0, any unavailable panel) or "all nominal".
- **Per-panel states** — each render fn shows an amber "unavailable" when its collector
  returns `available:false`, so one dead source never blanks the page.

### Palette / type
Fixed dark CSS variables (`--bg:#0b0e14`, `--panel:#121826`, hairline `--line`, cyan
`--accent:#5ad1c4`, semantic `--ok/--warn/--bad`), tabular-numeric monospace for all
numbers. **No external assets** — no icon webfont, no CDN (works offline); status is
carried by colored dots + typography.

### Constraints
- Single self-contained `PAGE_HTML` string (CSS + JS inline) — keeps `server.py`'s
  single-route serving trivial.
- No secrets ever rendered (already guaranteed by the collectors).

---

## Component B — Metaculus request reduction (the 429 fix)

A refresh currently fires ~12–14 Metaculus calls; the API 429s the burst.

### Per-panel cadence (`state.py`)
`DashboardState` takes collectors as `{name: (fn, interval_seconds)}` (a bare `fn` still
works, defaulting). The background loop ticks every ~15s but re-runs a collector only
when *its* interval has elapsed:

| Collector | Interval | Why |
|---|---|---|
| cost, runs, run_summary | 60s | cheap (OpenRouter / `gh` / local file) |
| armed | 120s | cheap, rarely changes |
| coverage | 300s | rounds open over hours |
| resolved | 1800s | resolved data barely changes |

`refresh_once()` force-runs all collectors at startup (page has full data immediately);
the loop calls a new `refresh_due()` that runs only elapsed collectors.

### Persistent resolved cache (`collectors.py`)
Resolved questions are immutable. `fetch_recent_resolved` reads/writes
`logs/dashboard_resolved_cache.json` (`{post_id: extracted+title}`); each cycle it lists
the resolved post **ids** (cheap, cached-free) and fetches **detail only for ids not
already cached**. Steady-state ≈ 0 detail calls — and the detail calls are exactly what
429'd. The throttle + 429-backoff added earlier stays as a safety net.

Net effect: ~85% fewer Metaculus calls in steady state.

> Note: we keep full id-pagination for resolved (it's a couple of cheap list calls at a
> 30-min cadence) rather than a single-page fetch — full pagination reliably catches
> newly-resolved questions without depending on a specific server-side sort order.

---

## Files

| File | Change |
|---|---|
| `bot/dashboard/collectors.py` | `fetch_recent_resolved` gains disk cache + title; cached ids skip detail |
| `bot/dashboard/state.py` | per-collector cadence (`refresh_due`, `(fn, interval)` specs) |
| `scripts/dashboard.py` | `build_collectors_map` returns `(fn, interval)`; `--tick` arg |
| `bot/dashboard/page.py` | full rewrite — dark Console UI |
| `tests/test_dashboard_collectors.py` | resolved-cache tests; update the existing resolved test to a tmp cache path |
| `tests/test_dashboard_server.py` | cadence tests; update `DashboardState(...)` constructions; update `build_collectors_map` test |

## Out of scope
- Theme switching / light mode (fixed dark by decision).
- Per-panel freshness timestamps (global "updated Ns ago" only).
- Any change to `server.py` or the `/api/status` JSON shape.
- Changes to the live bot.

## Testing
- Pure/logic units (resolved cache gating, per-collector cadence) unit-tested offline
  with monkeypatched network + tmp cache files.
- Page is verified by a light assertion (contains panel containers + polls
  `/api/status`) plus a live launch at the end (panels render; Metaculus call count per
  cycle is materially lower).
- Full suite (currently 557 passing) stays green.
