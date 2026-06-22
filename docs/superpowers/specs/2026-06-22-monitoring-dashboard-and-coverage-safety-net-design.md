# Monitoring Dashboard + Coverage Safety-Net — Design Spec

**Date:** 2026-06-22
**Status:** Approved (brainstormed 2026-06-22)

Two independent operational improvements for the live Metaculus forecasting bot
(`mopsus`, user id 303527), brainstormed together because they share one small
integration point (a run-summary record).

---

## Component A — Live monitoring dashboard

### Problem
The existing `dashboard.html` is a **frozen, hand-generated static snapshot**
(generated 2026-06-18, hard-coded values, no generator script, untracked). It was
built from the local `logs/forecasts.jsonl`, which **only contains local gate-test
runs** — the real live forecasts post from inside GitHub Actions and never touch
that file. So the current artifact cannot monitor live activity at all.

### Goal
A local, **auto-refreshing** dashboard that pulls **live** state and answers "how
is the bot doing right now?" — operational health, coverage, cost, and the bot's
own forecasts/scores.

### Decisions (from brainstorming)
- **Auto-refreshing local server** (chosen over regenerate-on-demand / scheduled regen).
- **Python stdlib `http.server` only** — no new dependencies (repo has no web
  framework; `httpx` is already a dependency for HTTP client calls).
- Realized **peer/baseline scores ARE readable** per-post on the plain token
  (`question.my_forecasts.score_data`); only the **community aggregate** is blocked
  by the March-2026 lockdown. So the dashboard CAN show our own realized scores on
  resolved questions — richer than originally assumed.

### Architecture
- Server binds `127.0.0.1:<port>`, holds an in-memory `state` dict refreshed by a
  background thread every ~60s (configurable). Browser page polls `/api/status`
  (JSON) every ~10s and re-renders, showing a "data as of HH:MM:SS" stamp.
- Each data source is collected behind its own `try/except`; a failing source shows
  a per-panel "unavailable" with last-good timestamp instead of breaking the page.
- **Secrets never reach the browser** — collectors return only derived numbers; the
  server binds to localhost only.

### Data sources (all live)
1. **OpenRouter** `GET /api/v1/auth/key` (Bearer `OPENROUTER_API_KEY`) →
   `data.limit`, `data.limit_remaining`, `data.limit_reset`, `data.usage` (lifetime),
   `data.usage_daily|weekly|monthly`. Pair `limit_remaining` with the
   `usage_{window}` matching `limit_reset` (do not hardcode "weekly"). Non-200 →
   "cost unavailable". Redact `data.label` / `data.creator_user_id`.
2. **Metaculus** `GET /api/posts/` (Token auth), per tournament (33022 + `minibench`):
   - Open set: `?tournaments=<id>&statuses=open&limit=100&offset=…`
   - Forecast subset: add `&forecaster_id=303527`
   - **`count` is always null** → must paginate following `next`/offset until empty.
   - **Gap = open ids − forecast ids** (computed client-side).
   - Exclude notebook/group posts: keep only `post.get("question") is not None`.
   - Own forecast + score per resolved post: `question.my_forecasts.latest`
     (value by type; numeric/date unscale via `question.scaling`) and
     `question.my_forecasts.score_data` (peer/baseline/coverage). `question.resolution`
     gives the outcome.
3. **GitHub Actions** via `gh` CLI (keyring auth): `gh run list --workflow run-bot
   --json …` → recent run conclusions/durations/timestamps, time-since-last-run,
   cadence-stale warning; `gh workflow list` / `gh variable list` / `gh secret list`
   → armed-state (workflow active?, `SUBMIT_PREDICTION`, secret names present).
4. **Local run-summary** `logs/runs.jsonl` (written by Component B) → truncation/retry
   counts for the coverage card. Optional; card degrades gracefully if absent.

### Panels (YAGNI-trimmed)
① Run health & cadence ② Armed-state ③ **Coverage** (open vs forecast, gap
highlighted; per tournament) ④ Recent forecasts + by-type, and resolved-question
outcomes with our realized score ⑤ Cost/budget (window limit + today/week/month)
⑥ Errors/alerts (cadence stale, truncation>0 from runs.jsonl, cost-window low).
A single "community rank/CP — locked pending Data Needs Form" note for what's blocked.

### Files
- `bot/dashboard/collectors.py` — pure parsers + network fetchers (split so parsers unit-test against fixtures).
- `bot/dashboard/state.py` — `DashboardState`: TTL cache + background refresh thread.
- `bot/dashboard/server.py` — `ThreadingHTTPServer` + a pure `route()` function.
- `bot/dashboard/page.py` — HTML/JS template string.
- `scripts/dashboard.py` — entrypoint (`--port`, `--refresh-interval`).
- `deploy/com.aaron.metaculus-bot-dashboard.plist` — optional launchd KeepAlive.
- `tests/test_dashboard_collectors.py`, `tests/test_dashboard_server.py`.

### "Deploy"
`.venv/bin/python scripts/dashboard.py` → open `http://127.0.0.1:8787`. Optional
launchd KeepAlive plist mirrors the existing export job for always-on.

---

## Component B — Coverage safety net

### Problem
The bot forecasts whatever the framework's tournament fetch returns, but three
limits can silently drop questions it is able to forecast:
1. **Single-page fetch** — `get_all_open_questions_from_tournament` passes
   `num_questions=None`, so only the first 100-question page is ever fetched.
2. **`max_questions_per_run=50`** (`config.py:65`) silently truncates to the first 50
   and only `logger.warning`s — the dropped count is never surfaced.
3. **No failure-retry** — a question that fails (e.g. `<3` ensemble members survive a
   transient 402/timeout) is skipped that run; it relies on the next 20-min cron.

`skip_previously_forecasted_questions=True` is correct and stays (it prevents
re-forecasting, not first-time coverage).

### Goal
Guarantee no open question the bot is able to forecast is silently dropped, and make
any truncation/failure **visible** (run-summary record the dashboard reads). Keep
forecast-once semantics (no refresh/update — explicitly out of scope).

### Decisions
- **Safety-net only** (chosen over also re-forecasting/refreshing existing questions).
- All changes live in **our** code (`bot/ensemble_bot.py`, `bot/config.py`,
  `scripts/run_tournament.py`, `bot/logging_store.py`) — the framework's publish +
  skip + `return_exceptions` semantics are never touched.

### Approach
Override `forecast_on_tournament` in our `FrameworkBot` to:
1. **Paginate**: build `ApiFilter(allowed_tournaments=[tid], allowed_statuses=["open"],
   group_question_mode="unpack_subquestions")` and call
   `get_questions_matching_filter(api_filter, num_questions=cfg.max_questions_to_fetch,
   randomly_sample=False, error_if_question_target_missed=False)` (awaited directly;
   `error_if_question_target_missed=False` is required so it doesn't raise when the
   tournament has fewer questions than the target).
2. **Tracked cap**: raise `max_questions_per_run` default to 200 and, when it binds,
   record `truncated_count` into a stats object instead of only logging. Move the cap
   here (remove the silent slice from the `forecast_questions` override).
3. **Forecast**: call `self.forecast_questions(questions, return_exceptions=True)` so
   the skip filter and per-question publish run unchanged.
4. **Retry**: re-run the failed (`isinstance(r, BaseException)`) subset up to
   `cfg.forecast_retry_attempts` times via `self.forecast_questions`, splicing
   recovered reports back by index. Order is preserved by `asyncio.gather`; publish is
   the last step inside a question, so retried questions never double-publish.
5. **Stash stats** on `self._last_run_stats`; `run_tournament.py` appends a record to
   `logs/runs.jsonl` after each tournament.

### Run-summary record (`logs/runs.jsonl`, via `logging_store`)
`{tournament, config_hash, questions_fetched, questions_after_skip, cap,
truncated_count, forecast_ok, forecast_failed, retried, retry_recovered, logged_at}`.
New `append_run` / `read_runs` mirror the existing `append_forecast` / `read_forecasts`
pattern. Written in its own file (not `forecasts.jsonl`) so per-type forecast readers
are unaffected.

### Config
Add plain fields `max_questions_to_fetch: int = 500` and
`forecast_retry_attempts: int = 2`; raise `max_questions_per_run` 50 → 200. These are
in `config_hash` (consistent with `max_questions_per_run`), so the live `config_hash`
churns **once** (cosmetic; matches prior field-addition precedent).

### Files
- `bot/config.py` — new fields + raised cap.
- `bot/logging_store.py` — `append_run` + `read_runs`.
- `bot/ensemble_bot.py` — `forecast_on_tournament` override; remove cap slice from `forecast_questions`.
- `scripts/run_tournament.py` — append run-summary after each tournament.
- `tests/test_config_new_fields.py` (extend), `tests/test_logging_store.py` (extend),
  `tests/test_ensemble_bot_coverage.py` (new).

---

## Out of scope
- Re-forecasting / updating existing forecasts as new info arrives.
- Community-relative rank / CP (blocked by lockdown; pending Metaculus Data Needs Form).
- Hosting the dashboard anywhere non-local.
- Per-question cost attribution (OpenRouter gives key-level totals only).

## Testing strategy
- Pure parsers (`parse_openrouter_budget`, `extract_own_forecast`, `parse_coverage`,
  retry/cap logic) unit-tested against the verified fixture JSON — offline, no network.
- Network fetchers tested by monkeypatching `httpx` / `subprocess.run` with canned responses.
- Coverage changes TDD'd with a fake metaculus client + fake `forecast_questions`;
  full suite (currently 509 passing) must stay green.
