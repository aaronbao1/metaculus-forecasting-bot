# Incorporation Implementations — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement plan items 2 (TS submodels), 3 (GABRIEL rating + Clio failure clustering), 4 (GEPA-style prompt optimization, scoped), 5 (playbook loop, limited) from `research-notes/07-incorporation-research.md`, leaving item 6 as an open framework.

**Architecture:** All new capability lands behind existing seams: `ForecastSystem` protocol for submodels, research-provider dispatch for rated/anchored research, `from_member_forecasts`/`from_research_text` stage entry points for $0/cheap fitness functions, config-driven prompt/playbook injection (byte-identical when off). One deliberate `BotConfig` hash perturbation batches every new field.

**Tech stack:** Python 3.11+, numpy/scipy (present); optional extras: `statsforecast` (timeseries), `chronos-forecasting`/`timesfm` (tsfm). No `gepa`/`openai-gabriel`/`openevolve` deps — methodology in-repo. TDD; offline tests only; heavy deps behind `pytest.importorskip`.

**Spec:** `docs/superpowers/specs/2026-06-11-incorporation-implementations-design.md`

**Standard TDD loop per task** (steps below name only the test + code): write failing test → `pytest <file> -q` (expect fail) → minimal implementation → `pytest <file> -q` (pass) → `pytest -q` (no regressions) → commit.

---

### Task 1: BotConfig fields + aggregation/extremize wiring (foundation)

**Files:** Modify `bot/config.py`, `bot/aggregation.py`, `bot/ensemble_bot.py`, `backtest/systems/ensemble_adapter.py`. Test: `tests/test_config_new_fields.py`.

- [ ] Tests: (a) defaults `n_trim=1, extremize_a=1.0, prompt_overrides={}, playbook_path=None`; (b) `config_hash` differs when any is non-default; (c) `aggregate_binary([...], "trimmed_gmo", n_trim=0)` == `gmo`; n_trim=2 drops 2/side; (d) `run_binary_ensemble` with `extremize_a=2.0` returns `cap(extremize(raw, 2.0))` (fake `_invoke_model`); identity at 1.0 — final unchanged vs current behavior golden; (e) `from_member_forecasts` honors `n_trim`/`extremize_a` from patch.
- [ ] Implementation: add the 4 fields to `BotConfig`; `aggregate_binary(probs, method, n_trim=1)` threads to `trimmed_gmo` only; in `run_binary_ensemble` and `from_member_forecasts`: `raw → (extremize(raw, a) if a != 1.0) → calibrate_binary`; record `extremize_a` in the calibration StageRecord inputs.
- [ ] Commit: `feat(config): n_trim/extremize_a/prompt_overrides/playbook_path — single deliberate config-hash perturbation (roadmap #8)`

### Task 2: Playbook loader + prompt injection (byte-identical off)

**Files:** Create `bot/playbook.py`, `skills/forecasting-playbook.md`. Modify `bot/prompts.py`, `bot/ensemble_bot.py`. Test: `tests/test_playbook.py`.

- [ ] Tests: (a) `load_playbook(missing_path)` == ""; (b) promoted bullets extracted, Candidates section excluded, `max_chars` truncation at bullet boundary; (c) **byte-identity golden**: rendered BINARY/MC/NUMERIC prompts with `playbook_path=None` equal pre-change rendering (golden strings captured from current templates); (d) with a playbook file: prompt contains "FORECASTING PLAYBOOK" block between research and date line; (e) `prompt_overrides={"binary": tmpl}` replaces the binary template (and only binary).
- [ ] Implementation: `{playbook_section}` slot in `_HEADER` rendered as "" when off (`{research}\n{playbook_section}\nToday's date:` keeps bytes identical — current text has `{research}\n\nToday's date:`); `CalibratedEnsembleBot._template("binary")` resolves `cfg.prompt_overrides` else module constant; `_playbook_section()` builds block from `load_playbook(cfg.playbook_path)`.
- [ ] Commit: `feat(bot): forecasting playbook injection + per-qtype prompt overrides (off by default, byte-identical prompts)`

### Task 3: `CorpusStore.series_points` + CSV series importer

**Files:** Modify `backtest/corpus/store.py`. Create `backtest/corpus/series_csv.py`. Test: `tests/backtest/test_series_target.py`.

- [ ] Tests: `series_points` returns all points sorted; `until=` filters inclusive ≤; missing series → []; `import_series_csv(store, path, qid, kind="target")` parses 2-col CSV (header optional), normalizes timestamps, returns FetchReport.
- [ ] Commit: `feat(corpus): target series kind — full-series reads + CSV importer`

### Task 4: Submodel engines

**Files:** Create `backtest/submodels/__init__.py`, `backtest/submodels/engines.py`. Test: `tests/backtest/test_submodel_engines.py`. Modify `pyproject.toml` (extras `timeseries=["statsforecast>=1.7"]`, `tsfm=["chronos-forecasting>=1.4","timesfm[torch]>=2.0"]`).

- [ ] Tests (naive_drift, offline): linear-trend series → median ≈ extrapolation; quantiles strictly increasing; `tail_widen=2.0` widens (q97.5−q2.5) ×~2 around median; <8 points raises `ValueError`; constant series → degenerate-safe (nonzero min spread). `get_engine("nope")` → KeyError listing options. statsforecast/chronos/timesfm tests: `pytest.importorskip`, same contract assertions.
- [ ] Implementation: `EngineForecast(quantiles: dict[float,float], engine: str, params: dict)`; `fit_predict(points, horizon_days)` protocol; `NaiveDriftEngine` (drift = mean diff/day; empirical residual quantiles of h-step-equivalent changes, min spread `1e-9 + 1e-6*|median|`); statsforecast engines map `level` intervals → percentiles; lazy imports inside factories; `ENGINES` registry + `get_engine(name, **params)`; all engines apply `tail_widen` post-hoc around q50.
- [ ] Commit: `feat(submodels): pluggable TS engines — naive_drift core, statsforecast/chronos/timesfm optional`

### Task 5: TimeSeriesSystem

**Files:** Create `backtest/systems/timeseries.py`. Test: `tests/backtest/test_timeseries_system.py`.

- [ ] Tests: numeric question + target series → artifact with valid 201-CDF (`validate_cdf` passes), `contamination="clean"`, `stage=="submodel"` with engine/quantiles/n_points; series truncated at as_of (post-T points provably unused: append absurd post-T point, forecast unchanged); error artifacts (never raise) for: binary qtype, no series, <8 points, cdf_size≠201, zero_point set; `config_fingerprint` differs across engines/params.
- [ ] Implementation: horizon_days from `scheduled_close_time − as_of`; quantiles → `percentiles_to_cdf(q, range_min, range_max, open_lo, open_hi)`; fingerprint = sha256 of (engine, params) prefix `ts-`.
- [ ] Commit: `feat(backtest): TimeSeriesSystem — TS submodels as ForecastSystems over target series`

### Task 6: System registry + spec `systems:` key + runner integration

**Files:** Create `backtest/systems/registry.py`. Modify `backtest/experiments/spec.py`, `backtest/experiments/runner.py`. Test: `tests/backtest/test_system_registry.py`.

- [ ] Tests: spec YAML with `systems: [{name: ts_theta, kind: timeseries, params: {engine: naive_drift}}]` parses; unknown kind → ValueError at build; `_build_systems` returns the system keyed by name; runner cost projection counts only `spec.variants` (registry systems free); name collision with variant/baseline → ValueError.
- [ ] Implementation: `SYSTEM_FACTORIES = {"timeseries": …}` (factory(store, params) → system); spec dataclass gains `systems: list = []` with per-entry validation (name+kind required).
- [ ] Commit: `feat(experiments): registry-backed extra systems in specs (timeseries first)`

### Task 7: `timeseries` research provider (hybrid anchor)

**Files:** Modify `backtest/timetravel/providers.py`. Test: `tests/backtest/test_timeseries_provider.py`.

- [ ] Tests: numeric question with series → section text contains median + 80% interval + as-of date, ledger entry `date_trust="snapshot"`; binary/no-series → "" (skip, no ledger); no context → RuntimeError.
- [ ] Implementation: `_timeseries_research(question, cfg)` reading `ACTIVE_CONTEXT`; engine `naive_drift` default, `getattr(cfg, "ts_anchor_engine", "naive_drift")` knob-free for now; register in `HISTORICAL_PROVIDERS` + section title "Statistical time-series anchor".
- [ ] Commit: `feat(backtest): time-series anchor research provider for numeric questions`

### Task 8: Synthetic numeric question generator + CLI

**Files:** Create `backtest/corpus/synthetic.py`. Modify `scripts/backtest.py`. Test: `tests/backtest/test_synthetic_questions.py`.

- [ ] Tests: from a stored target series, `generate_questions(store, source_qid, n=5, horizon_days=30, seed=1)` → 5 resolved numeric `BacktestQuestion`s, platform "synthetic", `resolution == series value at close (step-function)`, `open_time < as-of candidates < close`, scaling bounds contain trailing-window values with margin, deterministic under seed, target series copied to each new qid; CLI `synth-questions --corpus … --source-qid … --n …` prints count (CliRunner-style via `main(argv)`).
- [ ] Implementation: sample T0 anchors uniformly (seeded) over the series interior, require ≥ `min_history=30` points before T0 and a point ≥ horizon after; title "What will {name} be on {date}? (synthetic)"; bounds = [min,max] of trailing window ± 25% span margin (open bounds); provenance `{"synthetic": True, "source_qid": …, "generator": "v1"}`.
- [ ] Commit: `feat(corpus): synthetic numeric questions from target series (Retrocast ext. 4 seed)`

### Task 9: FRED fetcher

**Files:** Create `backtest/corpus/fred.py`. Modify `scripts/backtest.py` (build-corpus choice "fred"). Test: `tests/backtest/test_corpus_fred.py`.

- [ ] Tests: monkeypatched `_http_get_json` returns FRED observations payload → series stored as `kind="target"` under `fred:<series_id>` with a question row (qtype numeric, unresolved, provenance license note); "." missing values skipped; no API key → FetchReport error, no raise.
- [ ] Implementation: `fetch_fred(store, series_ids, api_key=env)` GET `https://api.stlouisfed.org/fred/series/observations` (+ `/fred/series` for title/units); upsert minimal question row so series have an addressable qid.
- [ ] Commit: `feat(corpus): FRED target-series fetcher`

### Task 10: GABRIEL-style rater

**Files:** Create `bot/rating.py`. Test: `tests/test_rating.py`.

- [ ] Tests (fake invoke): `rate_items` returns 0–100 means + agreement std over n_samples (distinct sample index → distinct invoke calls); malformed JSON for one item → that item `score=None`, others fine (never raise); `split_dossier_items` splits the `- [date] title — summary (url)` dossier format and rejoins losslessly; `filter_dossier` drops items < threshold and prepends a one-line filter note; everything-filtered → returns "" (compositor skip semantics).
- [ ] Implementation: rubric prompt (relevance to resolution criteria, 0–100, JSON `{"scores": {"relevance": N}}`), one call per item per sample; `invoke: Callable[[str, int], Awaitable[str]]` (prompt, sample_idx) so backtest can map sample_idx → run_index.
- [ ] Commit: `feat(bot): GABRIEL-style evidence rater (rate/consensus/filter; LLM-seam injected)`

### Task 11: `asknews_rated` provider

**Files:** Modify `backtest/timetravel/providers.py`. Test: `tests/backtest/test_asknews_rated.py`.

- [ ] Tests: monkeypatch `_asknews_historical_research` to a fixed dossier and ctx.llm to a fake scoring high/low per item → filtered section retains high item, drops low, title shows "relevance-filtered"; rating calls routed through ctx.llm with `cfg.parser_model`; empty dossier → "".
- [ ] Commit: `feat(backtest): asknews_rated provider — relevance-filtered dossiers (Halawi filter)`

### Task 12: Failure clustering (Clio-style) + CLI

**Files:** Create `backtest/reports/failures.py`. Modify `scripts/backtest.py`. Test: `tests/backtest/test_failures.py`.

- [ ] Tests: tmp run_dir with scores.jsonl/artifacts.jsonl fixtures + fake LLM → `analyze_failures(run_dir, store, llm, model, top=3, variant=…)` selects 3 worst baseline_score rows, facets parsed (summary/primary_cause/evidence_was_available/wrong_side_of_market), clusters assigned, `failures.json` + `failures.md` written; LLM JSON garbage for one item → item kept with `primary_cause="unparsed"`; CLI `failures RUN_DIR --corpus … --top 3` runs replay-mode (CachedLLM live=False) against cached fixture.
- [ ] Implementation: facet prompt per loss (title, final prob, resolution, member probs, price_at_T, research excerpt ≤1200 chars); cluster prompt over numbered summaries → JSON clusters; CLI flag `--live` to allow fresh LLM calls, default replay-only.
- [ ] Commit: `feat(backtest): Clio-style failure clustering over run artifacts + failures CLI`

### Task 13: Optimization framework — Candidate/fitness/ReplayFitness

**Files:** Create `backtest/optimize/__init__.py`, `backtest/optimize/fitness.py`. Test: `tests/backtest/test_optimize_fitness.py`.

- [ ] Tests: `temporal_split` splits by close time, train strictly earlier; `ReplayFitness` over a fixture run_dir (binary artifacts with member_probs stages) + store: params candidate `{n_trim: 0}` → scores recomputed via `from_member_forecasts`, EvalResult.n == question count, per_item feedback strings contain forecast/resolution; unknown candidate kind → ValueError; $0 (no llm needed).
- [ ] Commit: `feat(optimize): candidate/fitness framework + $0 ReplayFitness (item-6 plug point)`

### Task 14: PromptFitness with budget cap

**Files:** Modify `backtest/optimize/fitness.py`. Test: `tests/backtest/test_optimize_promptfitness.py`.

- [ ] Tests: fixture run_dir with research-stage texts + CachedLLM stub (counts invocations, fixed cost per fresh call) → candidate `prompt:binary` evaluates via `from_research_text` with `prompt_overrides` patched; identical template → 0 fresh calls; mutated template → fresh calls counted; `budget_usd` exceeded → `BudgetExceeded` raised before further spend; per_item feedback includes brier + resolution.
- [ ] Implementation: research text source = latest artifact research-stage output per (qid,horizon); cost accounting via a `CountingLLM` wrapper around ctx.llm recording fresh-call cost (cache hit = $0).
- [ ] Commit: `feat(optimize): PromptFitness — cached-research prompt evaluation with hard budget cap`

### Task 15: Reflective evolver (GEPA-mechanism, native)

**Files:** Create `backtest/optimize/evolve.py`. Test: `tests/backtest/test_optimize_evolve.py`.

- [ ] Tests (all fake): reflection LLM proposes deterministic mutations; fitness = closeness of template marker to target → `reflective_evolve(seed, fitness, reflect, generations=3, children=2, seed_rng=1)` returns history with monotone non-decreasing best-train score, Pareto keep (score, len) verified, stops on `BudgetExceeded` gracefully returning best-so-far; val score reported but never used for selection (selection-on-train asserted via a fitness that diverges train/val).
- [ ] Implementation: per generation: take K worst per_item feedbacks of current best → reflection prompt ("you are improving a forecasting prompt; here are failures…; propose a revised full template, preserving the required output-format contract lines") → M children → evaluate train → Pareto front (max score, min length) → next parent = best train; record candidates.jsonl rows.
- [ ] Commit: `feat(optimize): native GEPA-style reflective prompt evolution (budget-capped, train/val split)`

### Task 16: optimize_prompt CLI

**Files:** Create `scripts/optimize_prompt.py`. Test: `tests/backtest/test_optimize_cli.py`.

- [ ] Tests: `main(argv)` with fixture run_dir + fake evolve (monkeypatch) writes out_dir/candidates.jsonl + best_prompt.txt + summary.json and prints the experiments-variant YAML snippet; refuses missing run_dir with exit 2.
- [ ] Commit: `feat(scripts): optimize-prompt CLI — binary prompt evolution against cached backtest runs`

### Task 17: Playbook update reflector + seeded playbook

**Files:** Create `scripts/playbook_update.py` (+ `skills/forecasting-playbook.md` if not in Task 2). Test: `tests/test_playbook_update.py`.

- [ ] Tests: failures.json fixture + fake reflection LLM proposing 3 bullets (one duplicating an existing bullet) → exactly 2 new bullets appended under `## Candidates` with `<!-- id source date -->` provenance; `## Promoted` untouched; idempotent re-run adds nothing (dedupe by normalized text); `--max-new` honored.
- [ ] Commit: `feat(scripts): playbook reflector — failure facets to candidate playbook deltas (never auto-promotes)`

### Task 18: Docs + extras

**Files:** Create `docs/backtesting/OPTIMIZE.md`. Modify `docs/backtesting/README.md`, `docs/backtesting/DATA.md` (target series kind, synthetic platform), `pyproject.toml` (if not in Task 4).

- [ ] OPTIMIZE.md: fitness/candidate contract, temporal-split + experiments-gate promotion rules, the OpenEvolve plug-in recipe (`kind="code:aggregation"`) marked NOT IMPLEMENTED, budget discipline, noise-floor table from research-notes/07 §1.
- [ ] Commit: `docs(backtest): optimization framework contract + target-series/synthetic corpus docs`

### Task 19: Full verification

- [ ] `pytest -q` full suite green; `python scripts/backtest.py --help` lists new subcommands; smoke spec still parses (`ExperimentSpec.from_yaml("experiments/smoke.yaml")`).
- [ ] Final commit if stragglers; update memory.

## Self-review notes

- Spec coverage: A1→T3/T9, A2→T4, A3→T5, A4→T6, A5→T7, A6→T8, B1→T10, B2→T11, B3→T12, C1→T13/T14, C2→T15, C3→T16, D1-D3→T2, D4→T17, cross-cutting→T1, item-6 docs→T18. No gaps.
- Type consistency: `EngineForecast.quantiles: dict[float,float]` consumed by T5/T7; `Candidate(kind, payload)` consumed by T13–16; `invoke(prompt, sample_idx)` rater seam consumed by T11.
- Executor context: this plan is executed in-session by the author with the full design spec; code blocks are intentionally interface-level where the spec already pins behavior.
