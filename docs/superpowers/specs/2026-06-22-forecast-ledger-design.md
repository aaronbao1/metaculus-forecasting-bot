# Forecast Ledger — "track all questions forecasted on" — Design Spec

**Date:** 2026-06-22
**Status:** Approved (scope chosen: full ledger)

Add a dashboard panel that tracks **every** question the bot (mopsus, id 303527) has
forecast — across both tournaments and all statuses — with our forecast value, the
outcome, and our peer score. Dashboard-only; no bot-path changes.

---

## Key constraint (verified)
The Metaculus **list** endpoint does NOT include `my_forecasts` or `resolution` — those
are only in the per-question **detail** endpoint. So the value/outcome for each of the
~150 forecast questions needs one detail call. We make that affordable with a
**progressively-warmed disk cache**.

## Approach

### New collector `fetch_forecast_ledger(token, tournaments, cache_path, detail_budget)`
1. **Inventory (cheap, every cycle):** for each tournament, list `forecaster_id=303527`
   posts across `["open","closed","resolved"]` (httpx repeats the `statuses` param).
   From the list, build the full item set: `{post_id, title, tournament, type, status}`
   (skip notebook/group posts where `question is None`).
2. **Bounded warmup:** for up to `detail_budget` (~15) items **not yet in cache**, fetch
   detail → `extract_own_forecast(detail)` + `title` → write to cache. This bounds each
   refresh (~15 × 0.5s throttle ≈ 7.5s) so the loop stays responsive; the rest warm over
   subsequent cycles (~150 warm in ~15–20 min, then 0 detail calls forever).
3. **Merge + return:** attach cached value fields (`summary`/`median`/`top_option`,
   `resolution`, `score.peer_score`) onto each item where cached; sort open → closed →
   resolved then by tournament/title. Return
   `{available, total, by_status:{open,closed,resolved}, warmed, items:[...]}`.

### Shared cache
Generalize the resolved cache into one `logs/dashboard_forecast_cache.json`
(`{post_id: extract_own_forecast(...) + title}`). Both `fetch_recent_resolved` and
`fetch_forecast_ledger` read/write it — resolved questions are a subset, so the ledger
reuses entries the resolved fetcher already cached and vice versa (collectors run
sequentially in the refresh thread, so no write race). Rename `_RESOLVED_CACHE` →
`_FORECAST_CACHE`, helpers `_load_resolved_cache`/`_save_resolved_cache` →
`_load_cache`/`_save_cache`.

### New "All forecasts" panel (page.py)
A full-width panel below Alerts:
- Header meta: `N total · warmed M/N` and a status breakdown (`open · closed · resolved`).
- A **scrollable** list (max-height ~360px): each row = status dot + title + tournament
  tag (left); our forecast value + (if resolved) `→ outcome` + `peer ±N` (right).
  Unwarmed rows show `…` for the value until the cache fills.

### Wiring (scripts/dashboard.py)
Add `"forecasts": (lambda: collectors.fetch_forecast_ledger(mc_token, TOURNAMENTS), 120)`
to `build_collectors_map` (120s cadence — frequent enough to warm in ~15–20 min, light
once warm).

## Files
| File | Change |
|---|---|
| `bot/dashboard/collectors.py` | rename cache to shared `_FORECAST_CACHE`/`_load_cache`/`_save_cache`; add `fetch_forecast_ledger` |
| `bot/dashboard/page.py` | add `.scroll`/`.tag` CSS, `ledgerBody`, and the "All forecasts" panel call |
| `scripts/dashboard.py` | wire the `forecasts` collector (120s) |
| tests | ledger inventory + bounded-warmup + cache-sharing tests; page panel marker; build_collectors_map includes `forecasts` |

## Out of scope
- Per-question re-fetch of values for already-cached questions (forecast-once → immutable).
- Resolution-time ordering (we sort by status then tournament/title).
- Changes to the live bot or `/api/status` JSON contract shape (only a new panel key added).

## Testing
- `fetch_forecast_ledger` unit-tested offline: inventory across statuses from a fake list,
  bounded warmup (≤ budget detail calls/cycle), cache-sharing (a resolved id already in
  cache is not re-fetched), value/outcome merge.
- Page: assert the panel container + `ledgerBody` exist and the page still polls
  `/api/status`.
- Full suite (561) stays green; live launch confirms the panel renders and warms.
