# Console Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dashboard's JSON-dump UI with a polished dark "Console", and cut its Metaculus request volume (per-panel cadence + a persistent resolved cache) so it stops 429-ing itself.

**Architecture:** `bot/dashboard/page.py` becomes a self-contained dark Console page that renders structured panels from `/api/status`. `state.py` gains per-collector refresh cadence (`{name: (fn, interval)}` + `refresh_due()`). `collectors.py`'s resolved fetcher gains a disk cache so immutable resolved questions are fetched once. `/api/status` and `server.py` are unchanged.

**Tech Stack:** Python 3.12 stdlib (`http.server`, `threading`, `time`, `json`), `httpx`, vanilla HTML/CSS/JS, pytest + monkeypatch.

---

## Conventions (match these)
- Run one test: `.venv/bin/python -m pytest tests/test_dashboard_server.py -v`. Full suite: `.venv/bin/python -m pytest -q` (baseline **557 passed / 1 skipped**, must stay green).
- `asyncio_mode=auto`; stub network via `monkeypatch`; `tmp_path` for files.
- Bot account is mopsus, id `303527`. Tournaments `["33022", "minibench"]`.

## File structure
- Modify `bot/dashboard/collectors.py` — resolved disk cache (`fetch_recent_resolved`, `_load_resolved_cache`, `_save_resolved_cache`).
- Modify `bot/dashboard/state.py` — `DashboardState` per-collector cadence (`refresh_due`).
- Modify `scripts/dashboard.py` — `build_collectors_map` returns `(fn, interval)`; `--tick` arg.
- Modify `bot/dashboard/page.py` — full rewrite to the dark Console UI.
- Modify `tests/test_dashboard_collectors.py`, `tests/test_dashboard_server.py`.

---

### Task 1: Persistent resolved cache

**Files:**
- Modify: `bot/dashboard/collectors.py` (`fetch_recent_resolved` and helpers)
- Test: `tests/test_dashboard_collectors.py`

> The current `fetch_recent_resolved(token, tournaments, limit=6)` fetches post detail for every recent resolved id on every call — the calls that 429'd. Resolved questions never change, so cache them to disk and only fetch detail for ids not already cached. Also capture the post `title` for the UI.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_collectors.py
def test_fetch_recent_resolved_caches_detail_calls(monkeypatch, tmp_path):
    cache = tmp_path / "resolved.json"
    monkeypatch.setattr(collectors, "_MIN_METACULUS_SPACING", 0.0)

    def fake_posts(token, t, forecaster_id, statuses="open", **kw):
        assert statuses == "resolved" and forecaster_id == 303527
        return [{"id": 43928}] if t == "33022" else []

    detail_calls = []
    def fake_detail(token, pid, **kw):
        detail_calls.append(pid)
        return {"id": 43928, "title": "Ceasefire?",
                "question": {"id": 43954, "type": "binary", "resolution": "yes",
                             "scaling": {"range_min": None, "range_max": None, "zero_point": None},
                             "my_forecasts": {"latest": {"forecast_values": [0.18, 0.82],
                                                         "start_time": 100.0},
                                              "score_data": {"peer_score": 47.2}}}}
    monkeypatch.setattr(collectors, "fetch_posts", fake_posts)
    monkeypatch.setattr(collectors, "fetch_post_detail", fake_detail)

    out1 = collectors.fetch_recent_resolved("TOK", ["33022", "minibench"], cache_path=str(cache))
    assert out1["available"] is True
    assert out1["resolved"][0]["title"] == "Ceasefire?"
    assert out1["resolved"][0]["resolution"] == "yes"
    assert out1["resolved"][0]["summary"] == pytestapprox(0.82)
    assert len(detail_calls) == 1                     # fetched the one uncached id

    out2 = collectors.fetch_recent_resolved("TOK", ["33022", "minibench"], cache_path=str(cache))
    assert out2["available"] is True
    assert len(detail_calls) == 1                     # cached -> no new detail call
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k resolved_caches -v`
Expected: FAIL — `fetch_recent_resolved()` got an unexpected keyword argument `cache_path` (and the second-call assertion fails: no caching).

- [ ] **Step 3: Write minimal implementation**

In `bot/dashboard/collectors.py`, replace the whole `fetch_recent_resolved` function with the cache version, and add the two cache helpers just above it:

```python
_RESOLVED_CACHE = "logs/dashboard_resolved_cache.json"


def _load_resolved_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k resolved_caches -v`
Expected: PASS (2 assertions on detail-call counts).

- [ ] **Step 5: Fix the older resolved test to use a temp cache path**

The pre-existing `test_fetch_recent_resolved_maps_extractor` now writes to the real default cache. Point it at a temp file. Edit that test: change its signature to `def test_fetch_recent_resolved_maps_extractor(monkeypatch, tmp_path):` and its call to:

```python
    out = collectors.fetch_recent_resolved("TOK", ["33022", "minibench"], limit=10,
                                           cache_path=str(tmp_path / "rc.json"))
```

(Leave its assertions unchanged — `extract_own_forecast` output is unchanged.)

- [ ] **Step 6: Run the dashboard collector tests**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -q`
Expected: PASS (all, including the updated older test).

- [ ] **Step 7: Commit**

```bash
git add bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "feat(dashboard): cache immutable resolved questions to disk

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Per-collector refresh cadence

**Files:**
- Modify: `bot/dashboard/state.py` (rewrite `DashboardState`)
- Test: `tests/test_dashboard_server.py`

- [ ] **Step 1: Write the failing test + migrate existing constructions**

Replace the three state-construction call sites and add a cadence test. In `tests/test_dashboard_server.py`:

Change `test_state_refresh_isolates_failures` construction to:
```python
    st = state_mod.DashboardState({"good": good, "bad": bad})
```
Change `test_state_get_is_copy` construction to:
```python
    st = state_mod.DashboardState({"x": lambda: {"a": 1}})
```
Change all three `route` tests' constructions (drop `interval=999`):
```python
    st = state_mod.DashboardState({"x": lambda: {"a": 1}})
```
```python
    st = state_mod.DashboardState({})
```
```python
    st = state_mod.DashboardState({})
```

Then append the cadence test:
```python
def test_per_collector_cadence():
    calls = {"slow": 0, "fast": 0}
    def slow():
        calls["slow"] += 1
        return {"available": True}
    def fast():
        calls["fast"] += 1
        return {"available": True}
    st = state_mod.DashboardState({"slow": (slow, 1000), "fast": (fast, 0)})
    st.refresh_once()                       # force all
    assert calls == {"slow": 1, "fast": 1}
    st.refresh_due()                        # only elapsed collectors
    assert calls["fast"] == 2               # interval 0 -> always due
    assert calls["slow"] == 1               # interval 1000 -> not due yet


def test_refresh_due_isolates_failures():
    def boom():
        raise RuntimeError("x")
    st = state_mod.DashboardState({"bad": (boom, 0)})
    st.refresh_due()
    assert st.get()["panels"]["bad"]["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k "cadence or refresh_due" -v`
Expected: FAIL — `DashboardState` has no `refresh_due`; tuple specs not handled.

- [ ] **Step 3: Write the implementation**

Replace the entire body of `bot/dashboard/state.py` (keep the module docstring line) with:

```python
"""In-memory dashboard state with a per-collector background refresh.

Collectors are ``{name: (fn, interval_seconds)}`` (a bare ``fn`` defaults to
``default_interval``). The loop ticks every ``tick`` seconds and re-runs a
collector only when its own interval has elapsed — so expensive sources (Metaculus)
poll far less often than cheap ones.
"""
from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timezone


class DashboardState:
    def __init__(self, collectors: dict, tick: float = 15.0,
                 default_interval: float = 60.0):
        self._collectors: dict[str, tuple] = {}
        for name, spec in collectors.items():
            fn, interval = spec if isinstance(spec, tuple) else (spec, default_interval)
            self._collectors[name] = (fn, interval)
        self._tick = tick
        self._last_run: dict[str, float | None] = {n: None for n in self._collectors}
        self._lock = threading.Lock()
        self._snapshot: dict = {"panels": {}, "generated_at": None}
        self._stop = threading.Event()

    def _run(self, name: str) -> dict:
        fn, _ = self._collectors[name]
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - one dead source must not break others
            return {"available": False, "error": str(exc)}

    def _commit(self, panels: dict) -> None:
        snap = {"panels": panels,
                "generated_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._snapshot = snap

    def refresh_once(self) -> None:
        """Force-run every collector (startup)."""
        now = time.monotonic()
        panels = {}
        for name in self._collectors:
            panels[name] = self._run(name)
            self._last_run[name] = now
        self._commit(panels)

    def refresh_due(self) -> None:
        """Run only collectors whose interval has elapsed."""
        now = time.monotonic()
        with self._lock:
            panels = dict(self._snapshot["panels"])
        changed = False
        for name, (_, interval) in self._collectors.items():
            last = self._last_run[name]
            if last is None or (now - last) >= interval:
                panels[name] = self._run(name)
                self._last_run[name] = now
                changed = True
        if changed:
            self._commit(panels)

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.refresh_due()
            self._stop.wait(self._tick)

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -q`
Expected: PASS (cadence + the migrated state/route tests).

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/state.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): per-collector refresh cadence (refresh_due)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Wire per-collector intervals in the entrypoint

**Files:**
- Modify: `scripts/dashboard.py`
- Test: `tests/test_dashboard_server.py`

- [ ] **Step 1: Update the failing test**

Replace `test_build_collectors_map_wires_sources`'s body assertions (the values are now `(fn, interval)` tuples):
```python
def test_build_collectors_map_wires_sources(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "KEY")
    monkeypatch.setenv("METACULUS_TOKEN", "TOK")
    mod = _load_dashboard_script()
    cmap = mod.build_collectors_map()
    assert set(cmap) >= {"cost", "coverage", "resolved", "runs", "armed", "run_summary"}
    for spec in cmap.values():
        fn, interval = spec
        assert callable(fn) and interval > 0
    # Metaculus collectors poll less often than the cheap ones
    assert cmap["resolved"][1] > cmap["cost"][1]
    assert cmap["coverage"][1] > cmap["runs"][1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k build_collectors_map -v`
Expected: FAIL — values are bare lambdas, not `(fn, interval)` tuples (unpack raises).

- [ ] **Step 3: Write the implementation**

In `scripts/dashboard.py`: update the docstring usage line to `[--tick 15]`, rewrite `build_collectors_map`, and update `main`.

Replace `build_collectors_map`:
```python
def build_collectors_map() -> dict:
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    mc_token = os.getenv("METACULUS_TOKEN", "")
    # (collector, refresh interval seconds). Metaculus polls least often.
    return {
        "cost": (lambda: collectors.fetch_openrouter_budget(or_key), 60),
        "runs": (lambda: collectors.fetch_github_runs(), 60),
        "run_summary": (lambda: collectors.fetch_run_summary(), 60),
        "armed": (lambda: collectors.fetch_armed_state(), 120),
        "coverage": (lambda: collectors.fetch_metaculus_coverage(mc_token, TOURNAMENTS), 300),
        "resolved": (lambda: collectors.fetch_recent_resolved(mc_token, TOURNAMENTS), 1800),
    }
```

Replace the arg + `DashboardState` construction in `main`:
```python
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--tick", type=float, default=15.0,
                    help="how often the loop checks for due collectors")
    args = ap.parse_args(argv)

    state = DashboardState(build_collectors_map(), tick=args.tick)
    state.refresh_once()
    state.start_background()
    logger.info("dashboard at http://127.0.0.1:%d  (tick %ss)", args.port, args.tick)
```

(Remove the old `--refresh-interval` argument and the `interval=args.refresh_interval`
construction.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k build_collectors_map -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): per-collector intervals (Metaculus polled least often)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Console dark UI

**Files:**
- Modify: `bot/dashboard/page.py` (rewrite `PAGE_HTML`)
- Test: `tests/test_dashboard_server.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_server.py
from bot.dashboard import page as page_mod


def test_page_is_console_shell():
    html = page_mod.PAGE_HTML
    assert "/api/status" in html                 # still polls the endpoint
    assert 'id="strip"' in html and 'id="grid"' in html   # metric strip + panels
    assert "renderalerts" in html.lower() or "alerts_body" in html.lower() or "alerts" in html.lower()
    assert "JSON.stringify" not in html          # no raw dump anymore
    assert "#0b0e14" in html                      # the dark canvas color
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k console_shell -v`
Expected: FAIL — current `PAGE_HTML` still contains `JSON.stringify` and lacks `id="strip"` / the dark color.

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `bot/dashboard/page.py` with:

```python
"""Polished dark "Console" dashboard page (polls /api/status, no secrets, no CDN)."""

PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forecasting bot · monitor</title>
<style>
:root{--bg:#0b0e14;--panel:#121826;--line:rgba(255,255,255,.07);--text:#e7ebf2;
--muted:#8b95a7;--dim:#5d6678;--accent:#5ad1c4;--ok:#52d39a;--warn:#e6b15e;--bad:#e0667a;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-size:14px;
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:22px 20px 48px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
.brand{display:flex;align-items:center;gap:11px}
.brand .title{font-size:16px;font-weight:600}
.brand .sub{color:var(--dim);font-size:12px}
.pulse{width:9px;height:9px;border-radius:50%;background:var(--ok);animation:p 2s infinite}
@keyframes p{0%{box-shadow:0 0 0 0 rgba(90,209,196,.45)}70%{box-shadow:0 0 0 7px rgba(90,209,196,0)}100%{box-shadow:0 0 0 0 rgba(90,209,196,0)}}
.fresh{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin-bottom:14px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 15px}
.kpi .l{color:var(--muted);font-size:12px;margin-bottom:7px}
.kpi .v{font-size:23px;font-weight:600;font-family:var(--mono);letter-spacing:-.5px}
.kpi .s{color:var(--dim);font-size:12px;margin-top:3px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:720px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.panel h2{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;
letter-spacing:.06em;margin:0 0 10px;display:flex;justify-content:space-between;align-items:center}
.panel h2 .meta{font-weight:400;text-transform:none;letter-spacing:0;color:var(--dim);font-size:12px}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:7px 0;
border-top:1px solid var(--line);font-size:13px}
.row:first-of-type{border-top:none}
.r{font-family:var(--mono);color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:7px;vertical-align:1px}
.chip{font-size:11px;font-weight:600;padding:2px 8px;border-radius:6px;font-family:var(--mono)}
.unavail{color:var(--warn);font-size:13px}
.muted{color:var(--muted)}
.span2{grid-column:1/-1}
.bar{height:5px;border-radius:3px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:10px}
.bar>span{display:block;height:100%;background:var(--accent)}
</style></head><body>
<div class="wrap">
 <header>
  <div class="brand"><span class="pulse" id="pulse"></span>
   <div><div class="title">Forecasting bot</div><div class="sub">mopsus · Summer 2026 FutureEval</div></div></div>
  <div class="fresh" id="fresh">connecting…</div>
 </header>
 <div class="strip" id="strip"></div>
 <div class="grid" id="grid"></div>
</div>
<script>
const $=(id)=>document.getElementById(id);
const usd=(n)=>n==null?'—':'$'+Number(n).toFixed(2);
const pct=(n)=>n==null?'—':Math.round(n)+'%';
const DOT={ok:'var(--ok)',warn:'var(--warn)',bad:'var(--bad)',idle:'var(--dim)'};
const dot=(s)=>'<span class="dot" style="background:'+(DOT[s]||DOT.idle)+'"></span>';
const esc=(s)=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmtZ=(iso)=>{if(!iso)return '—';const d=new Date(iso);return String(d.getUTCHours()).padStart(2,'0')+':'+String(d.getUTCMinutes()).padStart(2,'0')+'Z';};
const dur=(s)=>{if(s==null)return '—';const m=Math.floor(s/60),x=s%60;return m?m+'m'+String(x).padStart(2,'0'):x+'s';};
function kpi(label,val,sub,color){return '<div class="kpi"><div class="l">'+label+'</div><div class="v"'+(color?' style="color:'+color+'"':'')+'>'+val+'</div><div class="s">'+(sub||'')+'</div></div>';}
function panel(title,meta,body){return '<div class="panel"><h2>'+title+(meta?'<span class="meta">'+meta+'</span>':'')+'</h2>'+body+'</div>';}
const unavail=(p)=>'<div class="unavail">unavailable'+(p&&p.error?(' · '+esc(p.error)):'')+'</div>';
const covTotal=(c,k)=>(!c||!c.available)?0:Object.values(c.tournaments||{}).reduce((a,t)=>a+(t[k]||0),0);
const NAMES={'33022':'FutureEval 33022','minibench':'MiniBench'};
let lastGen=null;
function render(s){
 const P=s.panels||{};lastGen=s.generated_at?new Date(s.generated_at):null;
 const armed=P.armed||{},cost=P.cost||{},runs=P.runs||{};
 const degraded=Object.values(P).some(p=>p&&p.available===false)||runs.cadence_stale;
 const down=armed.available&&!armed.run_bot_active;
 const st=down?'bad':(degraded?'warn':'ok');$('pulse').style.background=DOT[st];
 const gap=covTotal(P.coverage,'gap_count'),open=covTotal(P.coverage,'open_count');
 const last=(runs.runs&&runs.runs[0])||null;
 const costColor=cost.pct_used>95?'var(--bad)':(cost.pct_used>80?'var(--warn)':null);
 $('strip').innerHTML=
   kpi('Status',down?'Down':(degraded?'Degraded':'Live'),
       armed.run_bot_active?(armed.submit_prediction==='true'?'armed · submitting':'armed · dry-run'):'workflow off',DOT[st])
  +kpi('Cost · week',usd(cost.window_used),cost.available?('of '+usd(cost.limit)+' · '+pct(cost.pct_used)):'unavailable',costColor)
  +kpi('Coverage gap',(P.coverage&&P.coverage.available)?String(gap):'—',
       (P.coverage&&P.coverage.available)?(open+' open now'):'unavailable',gap>0?'var(--bad)':'var(--ok)')
  +kpi('Last run',last?fmtZ(last.created_at):'—',last?((last.conclusion||'?')+' · '+dur(last.duration_s)):'no runs',
       last&&last.conclusion&&last.conclusion!=='success'?'var(--bad)':null);
 let g='';
 g+=panel('Coverage','',covBody(P.coverage));
 g+=panel('Recent runs',runsMeta(runs),runsBody(runs));
 g+=panel('Resolved · our score',resolvedMeta(P.resolved),resolvedBody(P.resolved));
 g+=panel('Armed state','',armedBody(armed));
 g+=panel('Cost detail','',costBody(cost));
 g+='<div class="span2">'+panel('Alerts','',alertsBody(P))+'</div>';
 $('grid').innerHTML=g;
}
function covBody(c){
 if(!c||c.available===false)return unavail(c);
 const T=c.tournaments||{};let h='';
 for(const id in T){const t=T[id],g=t.gap_count||0;
  h+='<div class="row"><span>'+dot(g>0?'bad':'ok')+(NAMES[id]||id)+'</span><span class="r">'+t.open_count+' open · '+g+' gap</span></div>';
  if(g>0)(t.gap||[]).slice(0,4).forEach(q=>{h+='<div class="row"><span class="muted" style="padding-left:15px">'+esc(q.title||q.post_id)+'</span><span class="r">'+esc(q.type||'')+'</span></div>';});}
 return h||'<div class="muted">no tournaments</div>';
}
function runsMeta(r){if(!r||!r.available)return '';if(r.cadence_stale)return 'stale '+r.last_success_minutes_ago+'m';return r.last_success_minutes_ago!=null?(r.last_success_minutes_ago+'m ago'):'';}
function runsBody(r){if(!r||r.available===false)return unavail(r);
 return (r.runs||[]).slice(0,6).map(x=>'<div class="row"><span class="r">'+fmtZ(x.created_at)+'</span><span>'+dot(x.conclusion==='success'?'ok':(x.conclusion?'bad':'idle'))+(x.conclusion||x.status||'?')+' · '+dur(x.duration_s)+'</span></div>').join('')||'<div class="muted">no runs</div>';}
function resolvedMeta(r){if(!r||!r.available)return '';const sc=(r.resolved||[]).map(x=>x.score&&x.score.peer_score).filter(v=>v!=null);if(!sc.length)return (r.total||0)+' total';const a=sc.reduce((x,y)=>x+y,0)/sc.length;return 'peer avg '+(a>=0?'+':'')+a.toFixed(0);}
function fmtOurs(x){if(x.type==='binary'&&x.summary!=null)return 'we said '+Math.round(x.summary*100)+'%';
 if((x.type==='numeric'||x.type==='date')&&x.median!=null)return 'we said '+Number(x.median).toLocaleString();
 if(x.type==='multiple_choice'&&x.top_option)return 'we said '+esc(x.top_option);return '';}
function resolvedBody(r){if(!r||r.available===false)return unavail(r);
 const rows=(r.resolved||[]).slice(0,6);if(!rows.length)return '<div class="muted">none resolved yet</div>';
 return rows.map(x=>{const peer=x.score&&x.score.peer_score;const c=peer==null?'idle':(peer>=0?'ok':'bad');const ours=fmtOurs(x);
  return '<div class="row"><span>'+dot(x.resolution?'ok':'idle')+esc(x.title||x.question_id||x.post_id)+'</span><span>'+(ours?'<span class="muted" style="margin-right:8px">'+ours+'</span>':'')+(peer!=null?'<span class="chip" style="background:rgba(255,255,255,.06);color:'+DOT[c]+'">peer '+(peer>=0?'+':'')+peer.toFixed(0)+'</span>':'')+'</span></div>';}).join('');}
function armedBody(a){if(!a||a.available===false)return unavail(a);
 return '<div class="row"><span>run-bot workflow</span><span style="color:'+(a.run_bot_active?'var(--ok)':'var(--bad)')+'">'+(a.run_bot_active?'active':'off')+'</span></div>'
  +'<div class="row"><span>submit_prediction</span><span style="color:'+(a.submit_prediction==='true'?'var(--ok)':'var(--warn)')+'">'+esc(a.submit_prediction)+'</span></div>'
  +'<div class="row"><span>secrets</span><span class="r">'+((a.secrets||[]).length)+' set</span></div>';}
function costBody(c){if(!c||c.available===false)return unavail(c);
 const p=Math.min(100,Math.round(c.pct_used||0));
 return '<div class="row"><span>today</span><span class="r">'+usd(c.today)+'</span></div>'
  +'<div class="row"><span>this week</span><span class="r">'+usd(c.week)+' / '+usd(c.limit)+'</span></div>'
  +'<div class="row"><span>this month</span><span class="r">'+usd(c.month)+'</span></div>'
  +'<div class="row"><span>lifetime</span><span class="r">'+usd(c.lifetime)+'</span></div>'
  +'<div class="bar"><span style="width:'+p+'%;background:'+(p>95?'var(--bad)':(p>80?'var(--warn)':'var(--accent)'))+'"></span></div>';}
function alertsBody(P){const a=[];const runs=P.runs||{};
 if(runs.cadence_stale)a.push(['warn','No run-bot run in '+runs.last_success_minutes_ago+'m']);
 if(runs.recent_failures>0)a.push(['bad',runs.recent_failures+' recent run failure(s)']);
 const rs=P.run_summary||{};if(rs.available&&rs.total_truncated>0)a.push(['bad',rs.total_truncated+' question(s) truncated by the per-run cap']);
 const c=P.cost||{};if(c.available&&c.pct_used>80)a.push(['warn','Weekly budget '+pct(c.pct_used)+' used']);
 const gap=covTotal(P.coverage,'gap_count');if(gap>0)a.push(['bad',gap+' open question(s) not yet forecast']);
 Object.entries(P).forEach(([k,p])=>{if(p&&p.available===false)a.push(['warn',k+' data unavailable']);});
 if(!a.length)return '<div class="row"><span>'+dot('ok')+'No alerts — all nominal</span></div>';
 return a.map(([s,m])=>'<div class="row"><span>'+dot(s)+esc(m)+'</span></div>').join('');}
async function tick(){try{const r=await fetch('/api/status');render(await r.json());}catch(e){$('fresh').textContent='fetch error';}}
function freshTick(){if(!lastGen)return;const sec=Math.max(0,Math.round((Date.now()-lastGen.getTime())/1000));$('fresh').textContent='updated '+(sec<60?sec+'s':Math.floor(sec/60)+'m')+' ago';}
tick();setInterval(tick,10000);setInterval(freshTick,1000);
</script></body></html>"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k "console_shell or route" -v`
Expected: PASS (the console-shell test + the route tests that serve `PAGE_HTML`).

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/page.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): polished dark Console UI (structured panels)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Full-suite + live verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: green (baseline 557 + the new cache/cadence/console tests), 1 skipped, 0 failed.

- [ ] **Step 2: Live smoke (needs real `.env`)**

Start the dashboard, confirm it serves and the Metaculus call volume per cycle is low:
```bash
.venv/bin/python scripts/dashboard.py --port 8803 > /tmp/console.log 2>&1 &
PID=$!
curl -s --max-time 60 --retry 40 --retry-delay 2 --retry-connrefused http://127.0.0.1:8803/ -o /tmp/console.html
curl -s http://127.0.0.1:8803/api/status -o /tmp/console_status.json
kill $PID
```
Expected: `/tmp/console.html` contains `id="strip"` and `id="grid"`; `/tmp/console_status.json` has populated `panels`. Confirm no secret values leaked:
```bash
.venv/bin/python - <<'PY'
import json
TOK=open('.env').read()
d=json.load(open('/tmp/console_status.json'))
blob=json.dumps(d)
for bad in ['sk-or-v1','creator_user_id','Bearer']:
    assert bad not in blob, bad
print('panels:', list(d['panels']), '| no secret values leaked')
PY
```

- [ ] **Step 3: Confirm cache file behavior**

Run: `wc -l logs/dashboard_resolved_cache.json 2>/dev/null && echo "cache present (gitignored)"`
Expected: the cache exists after a live run (and `logs/` is gitignored, so it won't be committed).

No commit (verification only).

---

## Self-Review checklist (run after all tasks)
- [ ] Full suite green: `.venv/bin/python -m pytest -q`
- [ ] `/api/status` JSON shape and `server.py` unchanged.
- [ ] No secret values rendered in the page or `/api/status`.
- [ ] Live: page renders the Console (strip + panels), and a refresh cycle makes far fewer Metaculus calls than before (resolved detail calls ≈ 0 after warmup).
- [ ] `logs/dashboard_resolved_cache.json` is covered by the existing `logs/` gitignore rule.
