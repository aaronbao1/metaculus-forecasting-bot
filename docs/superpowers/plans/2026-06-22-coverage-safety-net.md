# Coverage Safety-Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee the bot forecasts every open question it is able to — no silent drops from the single-page fetch, the per-run cap, or transient failures — and surface any drop in a run-summary record.

**Architecture:** Override `forecast_on_tournament` in our nested `FrameworkBot` (`bot/ensemble_bot.py`) to (1) paginate the open-question fetch past the framework's first-page-only limit, (2) apply the skip filter ourselves so the report list aligns 1:1 with our question list, (3) apply a raised, *tracked* per-run cap, (4) retry transient per-question failures, and (5) stash coverage stats that `scripts/run_tournament.py` appends to `logs/runs.jsonl`. The framework's publish + `SUBMIT_PREDICTION` gate + `return_exceptions` semantics are never touched. Keep forecast-once semantics (`skip_previously_forecasted_questions=True` stays).

**Tech Stack:** Python 3.12, `forecasting_tools` (framework), pytest with `asyncio_mode=auto`.

---

## Context the implementer needs

- `FrameworkBot(ForecastBot)` is defined **inside** `build_framework_bot(cfg)` in `bot/ensemble_bot.py` (class starts at line 341), closing over the local `engine = CalibratedEnsembleBot(bot_config=cfg)`. So override methods can reference `engine.cfg`, `engine.cfg_hash`, `engine.cfg.log_path`, and module-level `logger`.
- The **current** `forecast_questions` override is `bot/ensemble_bot.py:353-362` — it ONLY does the silent cap. This plan **removes** it (the cap moves into `forecast_on_tournament`).
- The framework's `forecast_questions` (called via `super()` / `self.forecast_questions` once our override is gone) applies the skip filter and per-question publish.
- `get_questions_matching_filter(api_filter, num_questions=None, randomly_sample=False, error_if_question_target_missed=True)` is `async`, on `self.metaculus_client`. With `num_questions=None` it fetches ONE page (≤100). Passing an explicit `num_questions` walks pages — but you MUST also pass `error_if_question_target_missed=False` or it raises `ValueError` when the tournament has fewer questions than the target.
- `ApiFilter` import: `from forecasting_tools.helpers.metaculus_client import ApiFilter`.
- Each question object exposes `.already_forecasted` (bool) and `.id_of_post`.
- Run full suite: `.venv/bin/python -m pytest -q` (baseline 509 passed / 1 skipped). Run one file: `.venv/bin/python -m pytest tests/test_ensemble_bot_coverage.py -v`.
- Tests: flat `tests/test_<module>.py`, `asyncio_mode=auto` (bare `async def`, no marker). New-config-field tests follow `tests/test_config_new_fields.py`.

## File structure
- Modify `bot/config.py` — add `max_questions_to_fetch`, `forecast_retry_attempts`; raise `max_questions_per_run` default 50→200.
- Modify `bot/logging_store.py` — extract shared `_append_jsonl`; add `append_run` + `read_runs`.
- Modify `bot/ensemble_bot.py` — remove `forecast_questions` override; add `forecast_on_tournament` override.
- Modify `scripts/run_tournament.py` — extract `_run_one_tournament` helper that appends the run-summary.
- Tests: extend `tests/test_config_new_fields.py`, `tests/test_logging_store.py`; new `tests/test_ensemble_bot_coverage.py`, `tests/test_run_tournament.py`.

---

### Task 1: Config fields (fetch target, retry attempts, raised cap)

**Files:**
- Modify: `bot/config.py:65`
- Test: `tests/test_config_new_fields.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config_new_fields.py
from bot.config import BotConfig, config_hash


def test_coverage_fields_defaults():
    cfg = BotConfig()
    assert cfg.max_questions_per_run == 200        # raised from 50
    assert cfg.max_questions_to_fetch == 500
    assert cfg.forecast_retry_attempts == 2


def test_coverage_fields_perturb_config_hash():
    base = config_hash(BotConfig())
    assert config_hash(BotConfig(max_questions_to_fetch=999)) != base
    assert config_hash(BotConfig(forecast_retry_attempts=5)) != base
    assert config_hash(BotConfig(max_questions_per_run=50)) != base
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_new_fields.py -k coverage_fields -v`
Expected: FAIL — `AttributeError: 'BotConfig' object has no attribute 'max_questions_to_fetch'` (and the `== 200` assertion fails).

- [ ] **Step 3: Write minimal implementation**

In `bot/config.py`, replace line 65:

```python
    max_questions_per_run: int = 50                # cost brake
```

with:

```python
    # Cost-brake cap on questions forecast PER RUN. Raised from 50 so realistic
    # tournament rounds are never silently truncated; if it ever binds, the
    # forecast_on_tournament override records truncated_count to logs/runs.jsonl.
    max_questions_per_run: int = 200
    # Open-question fetch target. The framework default (num_questions=None) grabs
    # only the first 100-question page; this drives full pagination so a round
    # with >100 open questions is fully fetched.
    max_questions_to_fetch: int = 500
    # Immediate in-run retries for questions that fail transiently (e.g. a 402 or
    # timeout left <min_members survivors). 0 disables; failures then wait for the
    # next cron. Publish is the last per-question step, so a retried question never
    # double-publishes.
    forecast_retry_attempts: int = 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_new_fields.py -k coverage_fields -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot/config.py tests/test_config_new_fields.py
git commit -m "feat(config): coverage knobs (max_questions_to_fetch, retry, raised cap)"
```

---

### Task 2: Run-summary store (`append_run` / `read_runs`)

**Files:**
- Modify: `bot/logging_store.py`
- Test: `tests/test_logging_store.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_logging_store.py
from bot.logging_store import append_run, read_runs


def test_append_and_read_runs_roundtrip(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_run({"tournament": "33022", "truncated_count": 0}, p)
    append_run({"tournament": "minibench", "truncated_count": 3}, p)
    rows = read_runs(p)
    assert [r["tournament"] for r in rows] == ["33022", "minibench"]
    assert rows[1]["truncated_count"] == 3
    assert "logged_at" in rows[0]          # timestamp injected like append_forecast


def test_read_runs_missing_file(tmp_path):
    assert read_runs(tmp_path / "nope.jsonl") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_logging_store.py -k runs -v`
Expected: FAIL — `ImportError: cannot import name 'append_run'`.

- [ ] **Step 3: Write minimal implementation**

Refactor `bot/logging_store.py` to share the append core (DRY), then add the run helpers. Replace the body (keep the module docstring) so it reads:

```python
import json
from datetime import datetime, timezone
from pathlib import Path


def _append_jsonl(record: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "logged_at": datetime.now(timezone.utc).isoformat()}
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_forecast(record: dict, path) -> None:
    _append_jsonl(record, path)


def read_forecasts(path) -> list[dict]:
    return _read_jsonl(path)


def append_run(record: dict, path="logs/runs.jsonl") -> None:
    """Append-only per-tournament run-summary (coverage/truncation/retry signal).
    Separate file from forecasts.jsonl so per-type forecast readers are unaffected."""
    _append_jsonl(record, path)


def read_runs(path="logs/runs.jsonl") -> list[dict]:
    return _read_jsonl(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_logging_store.py -v`
Expected: PASS (new tests + existing `append_forecast`/`read_forecasts` tests still green).

- [ ] **Step 5: Commit**

```bash
git add bot/logging_store.py tests/test_logging_store.py
git commit -m "feat(logging): append_run/read_runs run-summary store (DRY _append_jsonl)"
```

---

### Task 3: `forecast_on_tournament` override — pagination + skip + tracked cap

**Files:**
- Modify: `bot/ensemble_bot.py` (remove `forecast_questions` override at 353-362; add `forecast_on_tournament` to the `FrameworkBot` class)
- Test: `tests/test_ensemble_bot_coverage.py`

> First grep for anything relying on the old cap location: `grep -rn "max_questions_per_run\|cost brake" tests/`. If a test asserts `forecast_questions` truncates, migrate it to call `forecast_on_tournament` (the cap moved). The existing `forecast_questions` override does nothing but cap, so removing it is safe otherwise.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ensemble_bot_coverage.py
import pytest
from bot.config import BotConfig
from bot.ensemble_bot import build_framework_bot


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # build_framework_bot constructs the framework's Metaculus client, which reads
    # these at construction time; set dummies so the bot builds offline.
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


class FakeQ:
    def __init__(self, qid, already=False):
        self.id_of_post = qid
        self.already_forecasted = already


def _bot(cfg):
    return build_framework_bot(cfg)


async def test_paginates_and_caps(monkeypatch):
    cfg = BotConfig(max_questions_to_fetch=500, max_questions_per_run=3,
                    forecast_retry_attempts=0)
    bot = _bot(cfg)
    captured = {}

    async def fake_get(api_filter, num_questions=None, randomly_sample=False,
                       error_if_question_target_missed=True):
        captured.update(num=num_questions, err=error_if_question_target_missed,
                        tourneys=api_filter.allowed_tournaments,
                        statuses=api_filter.allowed_statuses)
        return [FakeQ(i) for i in range(5)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)

    async def fake_fq(questions, return_exceptions=False):
        return [f"ok-{q.id_of_post}" for q in questions]
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)

    reports = await bot.forecast_on_tournament(33022, return_exceptions=True)
    assert len(reports) == 3                      # capped
    assert captured["num"] == 500                 # explicit fetch target -> pagination
    assert captured["err"] is False               # must not raise on shortfall
    assert captured["tourneys"] == [33022]
    assert captured["statuses"] == ["open"]
    s = bot._last_run_stats
    assert s["questions_fetched"] == 5
    assert s["truncated_count"] == 2
    assert s["forecast_ok"] == 3
    assert s["tournament"] == "33022"


async def test_skips_already_forecasted_before_forecasting(monkeypatch):
    cfg = BotConfig(max_questions_per_run=50, forecast_retry_attempts=0)
    bot = _bot(cfg)

    async def fake_get(api_filter, **kw):
        return [FakeQ(1, already=True), FakeQ(2), FakeQ(3, already=True), FakeQ(4)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)

    seen = {}
    async def fake_fq(questions, return_exceptions=False):
        seen["ids"] = [q.id_of_post for q in questions]
        return [f"ok-{q.id_of_post}" for q in questions]
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)

    await bot.forecast_on_tournament(33022, return_exceptions=True)
    assert seen["ids"] == [2, 4]                   # skip-filtered, aligns with reports
    assert bot._last_run_stats["questions_after_skip"] == 2


async def test_raises_when_not_return_exceptions(monkeypatch):
    cfg = BotConfig(forecast_retry_attempts=0)
    bot = _bot(cfg)
    async def fake_get(api_filter, **kw):
        return [FakeQ(1)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)
    async def fake_fq(questions, return_exceptions=False):
        return [RuntimeError("boom")]
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)
    with pytest.raises(RuntimeError, match="boom"):
        await bot.forecast_on_tournament(33022, return_exceptions=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ensemble_bot_coverage.py -v`
Expected: FAIL — the framework's `forecast_on_tournament` ignores `num_questions`/stats, so `captured["num"]` is unset and `bot._last_run_stats` doesn't exist (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

In `bot/ensemble_bot.py`, **delete** the `forecast_questions` override (lines 353-362). **Add** this method to the `FrameworkBot` class (e.g. immediately after `run_research`):

```python
        async def forecast_on_tournament(self, tournament_id,
                                         return_exceptions=False):
            from forecasting_tools.helpers.metaculus_client import ApiFilter
            cfg = engine.cfg
            # (1) Paginated fetch. The framework's get_all_open_questions_from_
            # tournament passes num_questions=None -> first 100-question page only.
            # An explicit target walks every page; error_if_question_target_missed
            # =False so it does NOT raise when the round has fewer questions.
            api_filter = ApiFilter(
                allowed_tournaments=[tournament_id],
                allowed_statuses=["open"],
                group_question_mode="unpack_subquestions")
            questions = await self.metaculus_client.get_questions_matching_filter(
                api_filter, num_questions=cfg.max_questions_to_fetch,
                randomly_sample=False, error_if_question_target_missed=False)
            fetched = len(questions)
            # (2) Apply the skip filter HERE so our list aligns 1:1 with reports
            # (the framework re-applies it idempotently). Required for the retry
            # index mapping in Task 4.
            if self.skip_previously_forecasted_questions:
                questions = [q for q in questions if not q.already_forecasted]
            after_skip = len(questions)
            # (3) Tracked cost-brake cap (replaces the old silent slice).
            cap = cfg.max_questions_per_run
            truncated = 0
            if cap and len(questions) > cap:
                truncated = len(questions) - cap
                logger.warning(
                    "cost brake: %d forecastable questions exceeds "
                    "max_questions_per_run=%d; forecasting first %d (%d dropped)",
                    len(questions), cap, cap, truncated)
                questions = list(questions)[:cap]
            # (4) Forecast: framework fan-out + per-question publish (SUBMIT_
            # PREDICTION gate). return_exceptions=True -> failures as objects.
            reports = list(await self.forecast_questions(
                questions, return_exceptions=True))
            # (5) Stash coverage stats (retry added in the next task).
            ok = sum(1 for r in reports if not isinstance(r, BaseException))
            self._last_run_stats = {
                "tournament": str(tournament_id),
                "config_hash": engine.cfg_hash,
                "questions_fetched": fetched,
                "questions_after_skip": after_skip,
                "cap": cap,
                "truncated_count": truncated,
                "questions_forecasted": len(questions),
                "forecast_ok": ok,
                "forecast_failed": len(reports) - ok,
                "retried": 0,
                "retry_recovered": 0,
            }
            # (6) Preserve the framework contract: re-raise if the caller did not
            # ask for exceptions (run_tournament always passes True).
            if not return_exceptions:
                for r in reports:
                    if isinstance(r, BaseException):
                        raise r
            return reports
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ensemble_bot_coverage.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite to confirm removing the old override broke nothing**

Run: `.venv/bin/python -m pytest -q`
Expected: green (update any test that asserted the old `forecast_questions` cap, per the grep note above).

- [ ] **Step 6: Commit**

```bash
git add bot/ensemble_bot.py tests/test_ensemble_bot_coverage.py
git commit -m "feat(bot): paginated forecast_on_tournament with tracked cap + skip"
```

---

### Task 4: Failure-retry pass

**Files:**
- Modify: `bot/ensemble_bot.py` (the new `forecast_on_tournament`)
- Test: `tests/test_ensemble_bot_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ensemble_bot_coverage.py
async def test_retries_transient_failures(monkeypatch):
    cfg = BotConfig(max_questions_per_run=50, forecast_retry_attempts=2)
    bot = _bot(cfg)
    async def fake_get(api_filter, **kw):
        return [FakeQ(1), FakeQ(2), FakeQ(3)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)

    calls = {"n": 0}
    async def fake_fq(questions, return_exceptions=False):
        calls["n"] += 1
        out = []
        for q in questions:
            if q.id_of_post == 2 and calls["n"] == 1:   # q2 fails only first time
                out.append(RuntimeError("transient"))
            else:
                out.append(f"ok-{q.id_of_post}")
        return out
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)

    reports = await bot.forecast_on_tournament(33022, return_exceptions=True)
    assert reports == ["ok-1", "ok-2", "ok-3"]      # q2 recovered, spliced in order
    s = bot._last_run_stats
    assert s["retried"] == 1 and s["retry_recovered"] == 1
    assert s["forecast_failed"] == 0


async def test_unrecovered_failure_stays_exception(monkeypatch):
    cfg = BotConfig(forecast_retry_attempts=1)
    bot = _bot(cfg)
    async def fake_get(api_filter, **kw):
        return [FakeQ(1), FakeQ(2)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)
    async def fake_fq(questions, return_exceptions=False):
        return [RuntimeError("always") if q.id_of_post == 2 else f"ok-{q.id_of_post}"
                for q in questions]
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)
    reports = await bot.forecast_on_tournament(33022, return_exceptions=True)
    assert reports[0] == "ok-1"
    assert isinstance(reports[1], BaseException)
    assert bot._last_run_stats["forecast_failed"] == 1
    assert bot._last_run_stats["retried"] == 1 and bot._last_run_stats["retry_recovered"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ensemble_bot_coverage.py -k retr -v`
Expected: FAIL — `retried`/`retry_recovered` stay 0; `reports[1]` is still the first-attempt exception.

- [ ] **Step 3: Write minimal implementation**

In `forecast_on_tournament`, insert the retry loop between the `reports = list(...)` line (step 4) and the `ok = sum(...)` stats line, and change the two hardcoded `0`s in the stats dict to the new `retried` / `recovered` variables:

```python
            # Retry transient failures (e.g. a 402/timeout left <min_members
            # survivors). Publish is the LAST per-question step, so a failed
            # question never published -> retrying cannot double-publish. Index
            # alignment holds because we skip-filtered before forecasting.
            retried = 0
            recovered = 0
            for _ in range(cfg.forecast_retry_attempts):
                failed_idx = [i for i, r in enumerate(reports)
                              if isinstance(r, BaseException)]
                if not failed_idx:
                    break
                retried += len(failed_idx)
                retry_reports = await self.forecast_questions(
                    [questions[i] for i in failed_idx], return_exceptions=True)
                for j, i in enumerate(failed_idx):
                    if not isinstance(retry_reports[j], BaseException):
                        reports[i] = retry_reports[j]
                        recovered += 1
```

Then in the stats dict replace `"retried": 0,` with `"retried": retried,` and `"retry_recovered": 0,` with `"retry_recovered": recovered,`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ensemble_bot_coverage.py -v`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add bot/ensemble_bot.py tests/test_ensemble_bot_coverage.py
git commit -m "feat(bot): in-run retry pass for transiently-failed questions"
```

---

### Task 5: Persist the run-summary from `run_tournament.py`

**Files:**
- Modify: `scripts/run_tournament.py`
- Test: `tests/test_run_tournament.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_tournament.py
import importlib.util, pathlib


def _load():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_tournament.py"
    spec = importlib.util.spec_from_file_location("run_tournament_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def test_run_one_tournament_appends_run_summary(monkeypatch):
    mod = _load()
    captured = []
    monkeypatch.setattr(mod, "append_run", lambda rec: captured.append(rec))

    class FakeBot:
        _last_run_stats = {"tournament": "T", "truncated_count": 0, "forecast_ok": 1}
        async def forecast_on_tournament(self, t, return_exceptions=True):
            return ["ok"]

    reports = await mod._run_one_tournament(FakeBot(), "T")
    assert reports == ["ok"]
    assert captured == [{"tournament": "T", "truncated_count": 0, "forecast_ok": 1}]


async def test_run_one_tournament_survives_append_failure(monkeypatch):
    mod = _load()
    def boom(rec):
        raise IOError("disk full")
    monkeypatch.setattr(mod, "append_run", boom)

    class FakeBot:
        _last_run_stats = {"tournament": "T"}
        async def forecast_on_tournament(self, t, return_exceptions=True):
            return ["ok"]

    # append failure must NOT propagate (run already succeeded)
    assert await mod._run_one_tournament(FakeBot(), "T") == ["ok"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_run_tournament.py -v`
Expected: FAIL — `AttributeError: module has no attribute '_run_one_tournament'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/run_tournament.py`, change the import on line 25 and refactor the loop body. Replace line 25:

```python
from bot.logging_store import read_forecasts
```

with:

```python
from bot.logging_store import append_run, read_forecasts
```

Add this helper above `main` (after `post_run_snapshot`):

```python
async def _run_one_tournament(bot, tournament) -> list:
    """Forecast one tournament, print the summary, and persist the coverage
    run-summary. An append failure never fails the run (it already succeeded)."""
    reports = await bot.forecast_on_tournament(tournament, return_exceptions=True)
    errors = [r for r in reports if isinstance(r, Exception)]
    print(f"{tournament}: {len(reports) - len(errors)} ok, {len(errors)} failed")
    for e in errors:
        print(f"  ERROR: {e}")
    stats = getattr(bot, "_last_run_stats", None)
    if stats:
        try:
            append_run(stats)
        except Exception:
            logger.exception("run-summary append failed (run unaffected)")
    return reports
```

Replace the loop body inside `main` (lines 62-70) so it delegates to the helper:

```python
    for tournament in (MetaculusApi.CURRENT_AI_COMPETITION_ID,
                       MetaculusApi.CURRENT_MINIBENCH_ID):
        await _run_one_tournament(bot, tournament)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_run_tournament.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_tournament.py tests/test_run_tournament.py
git commit -m "feat(run): persist per-tournament coverage run-summary to logs/runs.jsonl"
```

---

### Task 6: Full-suite regression + config-hash churn check

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (baseline 509 + new coverage/logging/config/run tests), 1 skipped.

- [ ] **Step 2: Confirm the one-time config-hash churn is intended**

Run: `.venv/bin/python -c "from bot.config import DEFAULT_CONFIG, config_hash; print(config_hash(DEFAULT_CONFIG))"`
The live `config_hash` changes once because three hashed fields were added/changed (`max_questions_per_run` 50→200, plus the two new fields). This is expected and cosmetic for live (it only labels new forecast-log rows). If any test or backtest fixture pinned the old hash literal, update it to the new value. Search: `grep -rn "config_hash" tests/ backtest/ | grep -i "==\|assert"`.

- [ ] **Step 3: Commit any fixture updates**

```bash
git add -A
git commit -m "test: update pinned config_hash after coverage-field additions"
```

(If nothing needed updating, skip this commit.)

---

## Self-Review checklist (run after all tasks)
- [ ] Full suite green: `.venv/bin/python -m pytest -q`
- [ ] The framework files under `.venv/` were NOT modified.
- [ ] `publish_report_to_metaculus` is never called directly — publish still flows only through the framework's `_run_individual_question` (so the `SUBMIT_PREDICTION` gate is intact).
- [ ] `skip_previously_forecasted_questions=True` is still passed at construction (`bot/ensemble_bot.py:462`).
- [ ] Retry only ever re-runs questions that returned exceptions (never-published) — no double-publish path.
- [ ] `logs/runs.jsonl` is gitignored like `logs/forecasts.jsonl` (confirm `logs/` is in `.gitignore`).
