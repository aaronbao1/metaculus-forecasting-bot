# Incorporation Implementations — Design (2026-06-11)

Implements the ranked plan in `research-notes/07-incorporation-research.md` per
owner scoping: item 1 (calibration flywheel) already covered by existing
backtesting + `backtest/experiments/export_calibration.py`; **item 2 and item 3
in full; item 4 scoped; item 5 limited; item 6 framework-only (no
implementation).** Companion docs: research-notes/08 (roadmap),
research-notes/09 (run-1 evaluation).

## Constraints (repo doctrine, non-negotiable)

- Everything offline-unit-testable; no network, no keys in tests; heavy deps
  optional with graceful skip.
- Backtest LLM calls route through `ctx.llm` (CachedLLM) — cached, cost-capped,
  replayable. New spend paths carry explicit budget caps.
- Systems never raise from `forecast()`; research providers obey the
  never-raise compositor contract.
- Nothing self-promotes to the live config: the experiments paired-stats gate
  on temporally held-out clean questions is the only promotion path.
- `bot/` live path stays bit-identical when new features are off.

## Workstream A — Time-series submodels (plan item 2, full)

**A1. Corpus: `"target"` series kind + sources.** The `series` table already
accepts any kind string; we add the convention `kind="target"` = the underlying
quantity a numeric/discrete question asks about. New:
- `CorpusStore.series_points(qid, kind, until=None) -> list[(iso, value)]` —
  full-series read (value_at only returns the last point).
- `backtest/corpus/fred.py` — FRED observations fetcher (`FRED_API_KEY`),
  stores `kind="target"` against a synthetic-or-real qid.
- `backtest/corpus/series_csv.py` — generic two-column (timestamp, value) CSV
  importer (offline tests use this).

**A2. Engines: `backtest/submodels/engines.py`.** Protocol:
`fit_predict(points: list[(datetime, float)], horizon_days: float) ->
EngineForecast(quantiles: dict[float, float], engine: str, params: dict)`.
Quantile keys are percentiles (2.5–97.5 typical). Registry `ENGINES`:
- `naive_drift` — numpy-only; random walk with drift, empirical h-step
  residual quantiles. Always available; the M-competition-respectable floor.
- `theta`, `autoarima`, `autoets` — statsforecast (optional extra
  `timeseries`), lazy import.
- `chronos2`, `timesfm` — optional extra `tsfm`, lazy import; quantile-head
  outputs mapped to percentile dicts.
- All engines accept `tail_widen: float = 1.0` (multiply quantile distances
  from the median; roadmap #5b arm).

**A3. `backtest/systems/timeseries.py` — `TimeSeriesSystem`.** ForecastSystem
returning CDF forecasts for numeric/discrete questions that have a `target`
series: load points ≤ `ctx.as_of` (leak-free by construction), horizon =
`scheduled_close_time − as_of`, engine quantiles →
`bot.distributions.percentiles_to_cdf` honoring `q.scaling`. Error artifact
(never raise) for: non-numeric qtype, missing series, <8 points,
`cdf_size != 201`, `zero_point` set (matches ensemble MVP limits). StageRecord
`stage="submodel"` with engine/n_points/horizon/quantiles. `contamination=
"clean"` (no LLM touches the question — same rationale as baselines).

**A4. Experiments integration.** New optional spec key `systems:` —
`[{name, kind, params}]` — resolved via `SYSTEM_FACTORIES` in new
`backtest/systems/registry.py` (`kind="timeseries"` → TimeSeriesSystem).
Spec-level validation mirrors baselines. Cost projection treats registry
systems as free (no LLM), like baselines; their StageRecord costs still
accumulate into actual spend.

**A5. Hybrid anchor provider.** `_timeseries_research` in
`backtest/timetravel/providers.py`: numeric/discrete questions with a target
series get a research section ("Statistical time-series anchor (theta): median
X, 80% interval [a, b] as of T…"), ledger `date_trust="snapshot"`. Engine
configurable via `BotConfig`-patch-free provider default (`naive_drift`) —
variants choose by provider name `timeseries` in `research_providers`.

**A6. Synthetic numeric questions (Retrocast ext. 4 seed).**
`backtest/corpus/synthetic.py`: `generate_questions(store, qid, n, horizon_days,
seed, …)` — for a stored target series, sample as-of points T0, build
`BacktestQuestion(platform="synthetic", qtype=numeric)` titled "What will
<name> be on <date>?", `open_time=T0−lookback`, `scheduled_close_time=T`,
`resolution=value_at(T)`, scaling bounds from trailing-window range ± margin,
provenance `{"synthetic": true, "source_qid": …}`. Zero contamination concern
for non-LLM systems; LLM use must stratify as usual. CLI: `backtest
synth-questions`.

## Workstream B — GABRIEL evidence rating + Clio failure clustering (item 3, full)

**B1. `bot/rating.py`** — GABRIEL-pattern primitives, LLM-seam-injected
(`invoke: async (prompt) -> str`), JSON-output prompts, never-raise per item:
- `rate_items(items, attributes, question_ctx, invoke, n_samples=1)` →
  per-item per-attribute mean score 0–100 + agreement (std across samples) —
  GABRIEL's "consensus columns".
- `split_dossier_items(text)` — splits the bullet-list dossiers our providers
  emit back into items.
- `filter_dossier(text, question_ctx, invoke, threshold=40)` → (filtered_text,
  ratings) keeping items rated ≥ threshold on `relevance`.
Decision: in-repo, NOT the `openai-gabriel` package — its client binds to the
OpenAI API directly; ours must route through CachedLLM/OpenRouter for cached,
reproducible, cost-tracked backtests. We take the methodology (rubric prompts,
consensus sampling), not the dependency.

**B2. `asknews_rated` provider** (backtest): compose `_asknews_historical_
research` → `filter_dossier` via `ctx.llm` with `cfg.parser_model` (cheap),
distinct cache `run_index` per consensus sample. Section title makes the
filtering visible. Live-path adoption deferred until backtests show value.

**B3. Failure clustering: `backtest/reports/failures.py` + CLI
`backtest failures RUN_DIR`.** Clio's pipeline at our scale (N≈10–100
losses, no embeddings dep):
1. Select: scores.jsonl rows with worst `baseline_score` (`--top N`, default
   25, variant-filterable).
2. Facet-extract per loss (LLM, JSON): one-sentence failure summary,
   `primary_cause` ∈ {research-miss, elicitation, aggregation, calibration,
   genuinely-hard, data-defect, other}, `evidence_was_available: bool`,
   wrong-side-of-market flag (price_at_T present).
3. Cluster: single LLM pass proposes ≤8 clusters + assignments from the facet
   summaries.
4. Emit `failures.json` (facets, clusters) + `failures.md`; LLM = the run's
   CachedLLM dir (replay-friendly), model `--model` (default parser model).
This automates the manual run-1 post-mortem and feeds Workstream D.

## Workstream C — GEPA-style prompt optimization (item 4, scoped)

**C1. `backtest/optimize/fitness.py`** — the shared optimization seam (and the
item-6 framework):
- `Candidate(kind: str, payload: dict)` — kinds now: `"params"` (BotConfig
  patch: n_trim/extremize_a/caps/…), `"prompt:binary"` (template text).
  Reserved namespace `"code:*"` for future evolutionary search (item 6).
- `EvalResult(score, per_item, n, cost_usd)` — score = mean baseline_score
  (repo's headline metric); per_item carries textual feedback for reflection.
- `ReplayFitness(run_dir, store, variant)` — replays cached member probs
  through `from_member_forecasts` with the candidate's config patch. $0,
  binary-only (the MVP seam's contract).
- `PromptFitness(run_dir, store, variant, budget_usd)` — replays cached
  research through `from_research_text` with `prompt_overrides` patched;
  member calls hit the shared llm_cache (identical prompt = $0; mutated
  prompt = real spend) — counts cost via cache-record deltas and raises
  `BudgetExceeded` at the cap.
- `temporal_split(questions, train_frac)` — by `scheduled_close_time`; all
  optimization selects on train, reports val; promotion requires the
  experiments gate on a later window.

**C2. `backtest/optimize/evolve.py` — `reflective_evolve()`.** Native minimal
GEPA-mechanism loop (no `gepa` dep): seed candidate → evaluate → reflect on
the K worst per-item feedbacks with a reflection model (through the same
CachedLLM) → propose M mutated prompts → evaluate → Pareto-keep (train score,
prompt length) → iterate G generations or until budget. Deterministic given
seed + cache. The `gepa` PyPI package can replace this loop later behind the
same FitnessFn seam.

**C3. CLI `scripts/optimize_prompt.py`** — args: run-dir, corpus, --budget-usd
(default 25), --generations, --train-frac, --seed, --out; writes
`candidates.jsonl` + `best_prompt.txt` + summary; prints the experiments-gate
variant snippet needed to validate the winner. Never writes bot config.

## Workstream D — Self-improving playbook (item 5, limited)

**D1. `skills/forecasting-playbook.md`** — `## Promoted` (injected) and
`## Candidates` (never injected) sections; each bullet carries an HTML comment
`<!-- id:… source:… date:… -->`.

**D2. `bot/playbook.py`** — `load_playbook(path, max_chars=4000)` → promoted
bullets as a text block ("" when file/section absent).

**D3. Config injection.** `BotConfig.playbook_path: str | None = None`; prompt
templates gain a `{playbook_section}` slot placed so that the rendered prompt
is **byte-identical** to today's when off (cache-hit preservation — tested).
When set: a "FORECASTING PLAYBOOK" block between research and the date line.

**D4. `scripts/playbook_update.py`** — reads `failures.json` (B3) → reflection
model proposes ≤K candidate bullets (lesson + applicability) → dedupe against
existing bullets → append to `## Candidates` with provenance. Promotion is
manual + experiments-gated, by design.

## Cross-cutting — one deliberate BotConfig change (single hash perturbation)

Batched per roadmap §1.8 ("perturbs config_hash — do once, deliberately"):
- `n_trim: int = 1` — threaded into `trimmed_gmo`.
- `extremize_a: float = 1.0` — fixed-α Karmarkar applied between aggregation
  and calibration when ≠ 1.0 (pre-registered roadmap #1 arm).
- `prompt_overrides: dict = {}` — keys "binary" (MVP), later "mc"/"numeric".
- `playbook_path: str | None = None`.
All defaults reproduce current behavior exactly (tested, including prompt
byte-identity). Run-ids churn once; the content-addressed llm_cache makes
re-runs ~$0.

## Item 6 — framework left open, nothing built

`docs/backtesting/OPTIMIZE.md` documents the contract an evolutionary search
(OpenEvolve/ShinkaEvolve) plugs into: `Candidate(kind="code:aggregation",
payload={"source": …})` evaluated by ReplayFitness over cached artifacts, with
the same temporal-split + experiments-gate promotion rules. No dep, no
implementation, no spec key.

## Testing

TDD throughout; pytest offline only. Fake-LLM seams everywhere (rating,
failures, evolve, playbook update). statsforecast/chronos/timesfm tests behind
`pytest.importorskip`. Golden tests: prompt byte-identity with new config
fields at defaults; synthetic-question resolution correctness; CDF validity of
engine outputs; budget-cap enforcement; temporal-split monotonicity;
registry/spec validation errors.

## Out of scope (explicit)

- Running paid optimization/annotation/backtest experiments (quotas drained;
  resume run-1 first).
- Live-bot adoption of rated research or TS anchors (backtest validation
  first).
- TimesFM/Chronos fine-tuning (07 §3b, phase 6); OpenEvolve integration
  (item 6); persona simulation (refuted, on hold).
