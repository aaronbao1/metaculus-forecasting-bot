# Forecast Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "All forecasts" dashboard panel tracking every question mopsus has forecast (both tournaments, all statuses) with our value, outcome, and peer score — via a progressively-warmed shared cache.

**Architecture:** A new `fetch_forecast_ledger` collector builds the full inventory from cheap list calls and warms a bounded number of per-question detail calls per cycle into a shared disk cache (also used by the resolved panel). A new full-width scrollable panel renders it. Dashboard-only; `/api/status` gains one panel key, nothing else changes.

**Tech Stack:** Python 3.12 stdlib + `httpx`, vanilla JS, pytest + monkeypatch.

---

## Conventions
- Run one file: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -v`. Full suite: `.venv/bin/python -m pytest -q` (baseline **561 passed / 1 skipped**, keep green).
- `tests/test_dashboard_collectors.py` already has `from pytest import approx as pytestapprox` and `import collectors`. mopsus id `303527`. Tournaments `["33022","minibench"]`.

## File structure
- Modify `bot/dashboard/collectors.py` — rename cache to shared `_FORECAST_CACHE`/`_load_cache`/`_save_cache`; add `fetch_forecast_ledger`.
- Modify `scripts/dashboard.py` — wire the `forecasts` collector (120s).
- Modify `bot/dashboard/page.py` — `.scroll`/`.tag` CSS, `ledgerMeta`/`ledgerBody`, the "All forecasts" panel call.
- Modify `tests/test_dashboard_collectors.py`, `tests/test_dashboard_server.py`.

---

### Task 1: Generalize the cache (resolved → shared forecast cache)

**Files:**
- Modify: `bot/dashboard/collectors.py`
- Test: `tests/test_dashboard_collectors.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_collectors.py
def test_cache_helpers_roundtrip(tmp_path):
    p = str(tmp_path / "fc.json")
    assert collectors._load_cache(p) == {}                 # missing -> {}
    collectors._save_cache(p, {"1": {"x": 1}})
    assert collectors._load_cache(p) == {"1": {"x": 1}}
    with open(p, "w") as f:
        f.write("[1,2,3]")
    assert collectors._load_cache(p) == {}                 # non-dict JSON self-heals to {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k cache_helpers_roundtrip -v`
Expected: FAIL — `collectors` has no attribute `_load_cache` / `_save_cache`.

- [ ] **Step 3: Write the implementation**

In `bot/dashboard/collectors.py`, rename the resolved-cache constant and helpers to shared names. Replace the constant line:
```python
_RESOLVED_CACHE = "logs/dashboard_resolved_cache.json"
```
with:
```python
_FORECAST_CACHE = "logs/dashboard_forecast_cache.json"
```
Rename `def _load_resolved_cache(path: str) -> dict:` to `def _load_cache(path: str) -> dict:` and `def _save_resolved_cache(path: str, data: dict) -> None:` to `def _save_cache(path: str, data: dict) -> None:` (bodies unchanged). Then in `fetch_recent_resolved`, change the default `cache_path: str = _RESOLVED_CACHE` to `cache_path: str = _FORECAST_CACHE`, and its two calls `_load_resolved_cache(cache_path)` → `_load_cache(cache_path)` and `_save_resolved_cache(cache_path, cache)` → `_save_cache(cache_path, cache)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -q`
Expected: PASS (the new roundtrip test + all existing resolved tests, which pass their own `cache_path`).

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "refactor(dashboard): shared forecast cache (_load_cache/_save_cache)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `fetch_forecast_ledger` collector

**Files:**
- Modify: `bot/dashboard/collectors.py` (add after `fetch_recent_resolved`)
- Test: `tests/test_dashboard_collectors.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_collectors.py
def test_fetch_forecast_ledger_inventory_and_bounded_warmup(monkeypatch, tmp_path):
    cache = str(tmp_path / "fc.json")
    monkeypatch.setattr(collectors, "_MIN_METACULUS_SPACING", 0.0)
    LIST = {
        "33022": [
            {"id": 10, "title": "A open", "status": "open", "question": {"id": 110, "type": "binary"}},
            {"id": 11, "title": "B closed", "status": "closed", "question": {"id": 111, "type": "binary"}},
            {"id": 12, "title": "C resolved", "status": "resolved", "question": {"id": 112, "type": "binary"}},
            {"id": 99, "title": "notebook", "status": "open", "question": None},
        ],
        "minibench": [
            {"id": 20, "title": "D closed", "status": "closed", "question": {"id": 120, "type": "binary"}},
        ],
    }
    def fake_posts(token, t, forecaster_id, statuses="open", **kw):
        assert forecaster_id == 303527 and statuses == ["open", "closed", "resolved"]
        return LIST[t]
    detail_calls = []
    def fake_detail(token, pid, **kw):
        detail_calls.append(pid)
        return {"id": pid, "title": f"detail{pid}",
                "question": {"id": pid + 100, "type": "binary",
                             "resolution": ("yes" if pid == 12 else None),
                             "scaling": {"range_min": None, "range_max": None, "zero_point": None},
                             "my_forecasts": {"latest": {"forecast_values": [0.3, 0.7], "start_time": float(pid)},
                                              "score_data": ({"peer_score": 5.0} if pid == 12 else None)}}}
    monkeypatch.setattr(collectors, "fetch_posts", fake_posts)
    monkeypatch.setattr(collectors, "fetch_post_detail", fake_detail)

    out = collectors.fetch_forecast_ledger("TOK", ["33022", "minibench"],
                                           cache_path=cache, detail_budget=2)
    assert out["available"] is True
    assert out["total"] == 4                                # notebook excluded
    assert out["by_status"] == {"open": 1, "closed": 2, "resolved": 1}
    assert len(detail_calls) == 2                           # bounded by detail_budget
    assert out["warmed"] == 2
    warmed = [it for it in out["items"] if it.get("summary") is not None]
    assert len(warmed) == 2 and warmed[0]["summary"] == pytestapprox(0.7)
    # items sorted open -> closed -> resolved
    assert [it["status"] for it in out["items"]] == ["open", "closed", "closed", "resolved"]

    out2 = collectors.fetch_forecast_ledger("TOK", ["33022", "minibench"],
                                            cache_path=cache, detail_budget=2)
    assert len(detail_calls) == 4                           # warms the remaining 2
    assert out2["warmed"] == 4

    out3 = collectors.fetch_forecast_ledger("TOK", ["33022", "minibench"],
                                            cache_path=cache, detail_budget=2)
    assert len(detail_calls) == 4                           # all cached -> no new detail calls
    resolved = [it for it in out3["items"] if it["status"] == "resolved"][0]
    assert resolved["resolution"] == "yes" and resolved["peer_score"] == pytestapprox(5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k forecast_ledger -v`
Expected: FAIL — `collectors` has no attribute `fetch_forecast_ledger`.

- [ ] **Step 3: Write the implementation**

In `bot/dashboard/collectors.py`, add after `fetch_recent_resolved`:

```python
_LEDGER_STATUSES = ["open", "closed", "resolved"]


def fetch_forecast_ledger(token: str, tournaments: list[str],
                          cache_path: str = _FORECAST_CACHE,
                          detail_budget: int = 15) -> dict:
    """Every question mopsus has forecast (all statuses), with our value + outcome.
    The inventory is built from cheap list calls every cycle; per-question detail
    (value/outcome/score) is warmed into the shared cache a bounded number per cycle,
    so steady-state makes ~0 detail calls."""
    if not token:
        return {"available": False, "error": "no token"}
    try:
        cache = _load_cache(cache_path)
        items: list[dict] = []
        for t in tournaments:
            for p in fetch_posts(token, t, forecaster_id=303527, statuses=_LEDGER_STATUSES):
                if not _is_question_post(p):
                    continue
                items.append({"post_id": p["id"], "title": p.get("title"),
                              "tournament": t,
                              "type": (p.get("question") or {}).get("type"),
                              "status": p.get("status")})
        order = {"open": 0, "closed": 1, "resolved": 2}
        items.sort(key=lambda x: (order.get(x["status"], 3), x["tournament"],
                                  (x.get("title") or "")))
        # bounded warmup of uncached items
        fetched = 0
        for it in items:
            if fetched >= detail_budget:
                break
            key = str(it["post_id"])
            if key not in cache:
                detail = fetch_post_detail(token, it["post_id"])
                entry = extract_own_forecast(detail)
                entry["title"] = detail.get("title")
                cache[key] = entry
                fetched += 1
        if fetched:
            _save_cache(cache_path, cache)
        # merge cached value/outcome onto each item
        for it in items:
            c = cache.get(str(it["post_id"]))
            if isinstance(c, dict):
                for k in ("summary", "median", "top_option"):
                    if c.get(k) is not None:
                        it[k] = c.get(k)
                it["resolution"] = c.get("resolution")
                sc = c.get("score")
                it["peer_score"] = sc.get("peer_score") if isinstance(sc, dict) else None
        by_status: dict[str, int] = {}
        for it in items:
            by_status[it["status"]] = by_status.get(it["status"], 0) + 1
        warmed = sum(1 for it in items if str(it["post_id"]) in cache)
        return {"available": True, "total": len(items), "by_status": by_status,
                "warmed": warmed, "items": items}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k forecast_ledger -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "feat(dashboard): forecast-ledger collector (inventory + bounded warmup)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Wire the `forecasts` collector

**Files:**
- Modify: `scripts/dashboard.py`
- Test: `tests/test_dashboard_server.py`

- [ ] **Step 1: Update the failing test**

In `tests/test_dashboard_server.py`, change `test_build_collectors_map_wires_sources`'s required-set assertion to include `forecasts`:
```python
    assert set(cmap) >= {"cost", "coverage", "resolved", "forecasts", "runs", "armed", "run_summary"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k build_collectors_map -v`
Expected: FAIL — `forecasts` not in `cmap`.

- [ ] **Step 3: Write the implementation**

In `scripts/dashboard.py` `build_collectors_map`, add the `forecasts` entry to the returned dict (after `resolved`):
```python
        "forecasts": (lambda: collectors.fetch_forecast_ledger(mc_token, TOURNAMENTS), 120),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k build_collectors_map -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): wire forecast-ledger collector (120s)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: "All forecasts" panel

**Files:**
- Modify: `bot/dashboard/page.py`
- Test: `tests/test_dashboard_server.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_server.py
def test_page_has_forecast_ledger_panel():
    html = page_mod.PAGE_HTML
    assert "ledgerBody" in html and "All forecasts" in html
    assert ".scroll{" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k forecast_ledger_panel -v`
Expected: FAIL — current `PAGE_HTML` has no `ledgerBody` / "All forecasts" / `.scroll`.

- [ ] **Step 3: Write the implementation (three targeted edits to `bot/dashboard/page.py`)**

Edit 1 — add CSS. Replace:
```css
.span2{grid-column:1/-1}
```
with:
```css
.span2{grid-column:1/-1}
.scroll{max-height:360px;overflow-y:auto;margin-top:4px}
.tag{font-size:11px;color:var(--dim);font-family:var(--mono);margin-left:6px}
```

Edit 2 — add the panel to `render`. Replace:
```javascript
 g+='<div class="span2">'+panel('Alerts','',alertsBody(P))+'</div>';
```
with:
```javascript
 g+='<div class="span2">'+panel('Alerts','',alertsBody(P))+'</div>';
 g+='<div class="span2">'+panel('All forecasts',ledgerMeta(P.forecasts),ledgerBody(P.forecasts))+'</div>';
```

Edit 3 — add the render functions. Replace:
```javascript
async function tick(){try{const r=await fetch('/api/status');render(await r.json());}catch(e){$('fresh').textContent='fetch error';}}
```
with:
```javascript
const TAG={'33022':'FE','minibench':'MB'};
function ledgerMeta(l){if(!l||!l.available)return '';return (l.total||0)+' · warmed '+(l.warmed||0)+'/'+(l.total||0);}
function ledgerBody(l){
 if(!l||l.available===false)return unavail(l);
 const bs=l.by_status||{};
 const head='<div class="row"><span class="muted">forecast history</span><span class="r">'+['open','closed','resolved'].map(s=>(bs[s]||0)+' '+s).join(' · ')+'</span></div>';
 const rows=(l.items||[]).map(it=>{
  const ours=fmtOurs(it)||'<span class="muted">…</span>';
  const out=(it.status==='resolved'&&it.resolution)?(' → '+esc(it.resolution)):'';
  const peer=(it.peer_score!=null)?(' <span class="chip" style="background:rgba(255,255,255,.06);color:'+(it.peer_score>=0?'var(--ok)':'var(--bad)')+'">peer '+(it.peer_score>=0?'+':'')+it.peer_score.toFixed(0)+'</span>'):'';
  const sd=it.status!=='resolved'?'idle':(it.peer_score==null?'idle':(it.peer_score>=0?'ok':'bad'));
  return '<div class="row"><span>'+dot(sd)+esc(it.title||it.post_id)+'<span class="tag">'+(TAG[it.tournament]||it.tournament)+'</span></span><span class="r">'+ours+out+peer+'</span></div>';
 }).join('');
 return head+'<div class="scroll">'+(rows||'<div class="muted">none</div>')+'</div>';
}
async function tick(){try{const r=await fetch('/api/status');render(await r.json());}catch(e){$('fresh').textContent='fetch error';}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k "forecast_ledger_panel or console_shell or route" -v`
Expected: PASS (new panel test + existing console-shell/route tests still pass).

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/page.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): All forecasts panel (scrollable ledger)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Full-suite + live verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: green (561 baseline + new cache/ledger/panel tests), 1 skipped, 0 failed.

- [ ] **Step 2: Structural smoke (no network)**

Run:
```bash
.venv/bin/python -c "from bot.dashboard.page import PAGE_HTML as h; assert 'ledgerBody' in h and 'All forecasts' in h and '.scroll{' in h; from bot.dashboard import collectors as c; assert hasattr(c,'fetch_forecast_ledger') and hasattr(c,'_load_cache'); print('structural smoke ok')"
```
Expected: `structural smoke ok`.

- [ ] **Step 3 (optional, needs `.env`): live warm check**

```bash
.venv/bin/python scripts/dashboard.py --port 8804 > /tmp/ledger.log 2>&1 &
PID=$!
curl -s --max-time 90 --retry 45 --retry-delay 2 --retry-connrefused http://127.0.0.1:8804/api/status -o /tmp/ledger.json
.venv/bin/python -c "import json; d=json.load(open('/tmp/ledger.json')); f=d['panels']['forecasts']; print('available:', f.get('available'), '| total:', f.get('total'), '| warmed:', f.get('warmed'), '| by_status:', f.get('by_status'))"
kill $PID
```
Expected: `forecasts` panel available with a non-zero `total` and `warmed` ≤ `detail_budget` on the first cycle.

No commit (verification only).

---

## Self-Review checklist
- [ ] Full suite green.
- [ ] `/api/status` unchanged except the added `forecasts` panel key; `server.py` untouched.
- [ ] No secret values rendered.
- [ ] Resolved panel still works (shares the renamed cache).
- [ ] Warmup is bounded per cycle (`detail_budget`), so a refresh pass stays responsive.
- [ ] `logs/dashboard_forecast_cache.json` covered by the `logs/` gitignore rule.
