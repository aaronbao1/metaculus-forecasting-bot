# Retrocast — Backtesting System for the Forecasting Bot

**Date:** 2026-06-11
**Status:** Approved design (user-approved 2026-06-11); implementation plan to follow
**Repo context:** branch `perf/post-mvp-buildout`; live bot competing in Summer FutureEval 2026

## 0. Decisions log (user-set constraints)

| Decision | Choice |
|---|---|
| Budget | All three tiers as named run profiles: `smoke` / `standard` / `full` |
| Cutoff strictness | Both modes; **strict is default and headline**; pragmatic behind a config flag, leakage flagged per question |
| LLM contamination | Full corpus with prompt-masking applied, **stratified reporting** (clean post-cutoff vs contaminated), clean stratum is the headline metric |
| Architecture | **A: sidecar harness** (`backtest/` package injecting into existing bot seams) absorbing replay-first ideas (forward snapshot accumulator, stage-level replay). Environment-interface refactor documented as a post-MVP seam |

## 1. Goals and non-goals

**Goals**

1. Replay the full forecasting pipeline — or any single stage of it — on historical questions as if "today" were a past date T, with enforceable, auditable information cutoffs.
2. Score the way Metaculus scores bots (log-score family, baseline/peer/spot variants, head-to-head vs crowd) so backtest numbers are predictive of tournament behavior.
3. Compare arbitrary system variants on identical question sets with proper paired statistics; refuse to declare winners inside the noise band.
4. Evaluate at multiple horizons relative to question close (default ladder: −7d, −30d, −365d; plus `spot` and Halawi-geometric modes).
5. Define a documented compatibility contract (`docs/backtesting/COMPATIBILITY.md`) such that future systems — including ones sharing no code with the current bot — plug into the same harness.

**Non-goals (doctrine, per FORECASTING-BOT-RESEARCH.md §9.6)**

- The backtest is a **relative comparison instrument plus an upper-bound absolute estimate** — never ground truth for live skill. Prospective evaluation (MiniBench, bot testing area 32977, ForecastBench submission) remains the absolute yardstick; the harness feeds those, it does not replace them.
- No live-tournament behavior changes in this project; the live bot with `as_of=None` must be bit-identical to today.

## 2. Architecture

Sidecar package, four layers with one-way dependencies; each layer usable standalone (this is what makes "backtest just part of the system" work):

```
backtest/
├── corpus/        # Layer 1: questions, resolutions, CP history, market price
│   │              #   history, crosswalk, forward snapshots
│   ├── store.py           # SQLite metadata + JSONL/Parquet blobs; schema_version
│   ├── metaculus.py       # tier-aware fetcher (records withheld fields, upgrades later)
│   ├── halawi.py          # HF dataset importer (YuehHanChen/forecasting[_raw])
│   ├── forecastbench.py   # forecastbench-datasets importer (nightly JSON)
│   ├── polymarket.py      # Gamma enumeration + CLOB prices-history
│   ├── kalshi.py          # settled markets + candlesticks (+ /historical/* archive)
│   ├── manifold.py        # bet-chain reconstruction (exclude redemption bets)
│   ├── crosswalk.py       # question ↔ market matching (embed → top-k → LLM gate → table)
│   └── snapshot.py        # forward accumulator (archive every live-touched question)
├── timetravel/    # Layer 2: as-of context, historical research, LLM cache, leakage
│   ├── asof.py            # AsOfContext: T, cutoff mode, evidence ledger
│   ├── providers.py       # historical research providers (register into bot registry)
│   ├── leakage.py         # per-doc LLM leak-screen; canary test plumbing
│   ├── llm_cache.py       # content-addressed completion cache around _invoke_model
│   └── probe.py           # memorization probe (per question × model, cached)
├── systems/       # Layer 3: System protocol + adapters
│   ├── protocol.py        # ForecastSystem protocol, BacktestQuestion, ForecastArtifact
│   ├── ensemble_adapter.py# CalibratedEnsembleAdapter + variant() + stage entry points
│   └── baselines.py       # always-0.5, CP-echo, market-echo, raw single model
├── scoring/       # Layer 4a
│   ├── metaculus_scores.py# log/baseline/peer/spot for binary, MC, numeric, discrete
│   ├── opponents.py       # CP-at-T and market-price-at-T synthetic opponents
│   ├── diagnostics.py     # calibration curves, Murphy decomposition, discrimination,
│   │                      #   Paleka consistency battery
│   └── stats.py           # paired t/bootstrap, cluster weights, difficulty adjustment,
│                          #   detectable-effect-size reporting
├── experiments/   # Layer 4b
│   ├── spec.py            # YAML experiment spec (corpus filter × horizons × variants × profile)
│   ├── runner.py          # cost estimate gate, cost brake, per-question failure isolation,
│   │                      #   resumable run manifest
│   └── profiles.py        # smoke / standard / full
└── reports/       # Layer 4c
    └── report.py          # markdown/HTML: leaderboard, strata, diagnostics, ledger
                           #   summary, cost/latency, worst-10 post-mortems
scripts/backtest.py        # CLI: build-corpus | run | replay | compare | report
docs/backtesting/          # README.md, COMPATIBILITY.md, DATA.md
data/backtest/             # corpus.db + blobs + completion cache + run artifacts (gitignored)
```

### 2.1 Changes to `bot/` (exhaustive — nothing else is touched)

1. Add `as_of: datetime | None = None` to `BotConfig` (`bot/config.py`).
2. Replace three hard-coded `date.today().isoformat()` at `bot/ensemble_bot.py:128, 147, 165` with `(cfg.as_of or date.today())`.
3. Thread `now=cfg.as_of` into `_days_remaining()` (`bot/ensemble_bot.py:56–61`) → `timeutils.days_until(close, now=)` — the `now=` parameter already exists and is currently never passed.
4. Same `as_of` substitution in `_prompt_args()` (`bot/research.py:34–40`).
5. Historical research providers live in `backtest/timetravel/providers.py` and are installed into `bot.research`'s namespace by an explicit `install_historical_providers()` call at harness startup, where the existing `globals()`-dispatch (`bot/research.py:89–95`) resolves them by name — additive runtime registration; no `bot/` file changes and no live code path changes.

With `as_of=None`, all branches reduce to current behavior (regression-tested).

### 2.2 Data flow of one experiment

corpus query (platform, type, date range, stratum filters) → for each (question, horizon): build `AsOfContext` → time-locked research via historical providers + market-price-at-T → run System under test through LLM cache → persist `ForecastArtifact` → score vs resolution + synthetic opponents → paired stats across variants → report.

## 3. Layer 1 — Corpus

### 3.1 Store

SQLite (`corpus.db`) tables: `questions`, `resolutions`, `cp_history`, `market_history`, `crosswalk`, `snapshots`, `contamination`. Bulky series (CP/candle history, archived docs) as JSONL/Parquet blobs referenced by path. Every row: `provenance` (source, fetch_time, license note) and `schema_version`. All fetches cached; "withheld" recorded distinctly from "missing."

`BacktestQuestion` (the protocol's question type): platform + native IDs, type (binary/MC/numeric/discrete), title, description, resolution_criteria, fine_print, open/close/`spot_scoring_time`/actual_resolve times, resolution value, options/scaling (incl. `zero_point`, open bounds), `question_weight`, source provenance, per-model-set contamination stratum.

### 3.2 Sources (priority order)

1. **Metaculus via project token.** Post-March-2026 lockdown reality (live-verified): anonymous = 403; plain token sees IDs/titles/types/timestamps/tournament membership but `resolution: null`, `description: null`, null aggregation history — EXCEPT questions this bot ever forecast, which stay fully visible to the token. Enumeration: `GET /api/posts/` with `statuses=resolved`, `forecast_type=`, `tournaments=` (aibq1–4, fall-aib-2025, spring-aib-2026, summer-futureeval-2026, minibench), Django-style `scheduled_resolve_time__gt/lt` etc., `limit≤100` cursor pagination, `include_descriptions=true` (tier-dependent). Detail: `GET /api/posts/{id}/` → all timestamps, `resolution`, `options`, `scaling`, `question_weight`, `default_score_type`, and `aggregations.recency_weighted.history` (tier-dependent). Self-throttle ~3.5s + jitter (forecasting-tools convention). The fetcher is **tier-aware**: stores what's visible, records withheld fields, re-fetches/upgrades when the Bot Benchmarking Access Tier (~250 open + ~250 resolved) or Data Needs Form approval lands. **User action: submit the Data Needs Form** (also the Terms-of-Use legitimacy path for AI-eval use).
2. **Pre-lockdown datasets:** Halawi HF `YuehHanChen/forecasting` (5,516 curated binary) / `forecasting_raw` (48,754; 5 platforms, 2015–2024); `forecastingresearch/forecastbench-datasets` (nightly, CC BY-SA 4.0, cross-platform IDs + freeze values). Question text may drift from current Metaculus copies — store dataset text verbatim with provenance.
3. **Markets as first-class questions** (and opponent price series):
   - *Polymarket:* Gamma `GET /markets?closed=true` (question, conditionId, clobTokenIds aligned with outcomes, outcomePrices, endDate, volume); CLOB `GET /prices-history?market=<clobTokenId>&startTs=&endTs=&fidelity=` — no auth; **resolved markets return empty below fidelity≈720 (12h)** — 12h–1d granularity is the design assumption for price-at-T.
   - *Kalshi:* `GET /markets?status=settled&min/max_settled_ts=`; candlesticks `GET /series/{series}/markets/{ticker}/candlesticks?start_ts=&end_ts=&period_interval=` (1/60/1440 min, `include_latest_before_start=true`); rolling ~3-month live window (`GET /historical/cutoff`) then `/historical/*` endpoints.
   - *Manifold:* `GET /v0/bets?contractId=&beforeTime=` — chain `probBefore`/`probAfter` into an exact step function; **exclude redemption bets**; filter `answerId` for MC; play-money caveat recorded in provenance.
   - Bulk shortcuts when crawling is silly: jon-becker/prediction-market-analysis (Polymarket+Kalshi Parquet), pmxt archive (hourly orderbooks, CC BY 4.0), Manifold dumps (≤2024-07, non-commercial).
   - **Snapshot-on-sight policy:** market history is decaying (Kalshi window rolls; Polymarket pruning is undocumented; `/orderbook-history` already died). Anything fetched is archived locally, permanently.
4. **Forward snapshot accumulator** (`snapshot.py`): hook after every live tournament run; archive every touched question at full fidelity (the token sees them forever). The live bot becomes a corpus-building machine; the post-lockdown corpus problem decays over months.

### 3.3 Crosswalk (Metaculus question ↔ equivalent market)

Embed title+resolution criteria → top-k≈20 nearest neighbors over indexed Gamma+Kalshi(+Manifold) corpus (>99.9% recall of true equivalents in top-20, arXiv 2601.01706) → LLM verification gate on resolution criteria **and dates** (date mismatch = dominant false positive, arXiv 2512.02436) → cached `crosswalk` table with a manual-override column. Backtest analogue of the live bot's keyword `market_research()` — note in reports that live/backtest market evidence differ in kind (search vs crosswalk).

### 3.4 Contamination strata

Per question × model-set: `clean` iff `open_time >` max **release date** (not stated cutoff — release dates are the trustworthy bound; cutoff labels demonstrably lie) across ensemble models; else `contaminated`; refined to `contaminated-confirmed` by the memorization probe. Every report stratifies; headline = clean. Resolution-date validation runs on every question at import (≥3.8% of even Halawi's curated set had resolved-before-forecast-date issues).

## 4. Layer 2 — Time travel

### 4.1 `AsOfContext`

Carries: T, cutoff mode (`strict` | `pragmatic`), and the **evidence ledger** — every provider must record (provider, query, doc URL, doc date, date-trust class) for each item it surfaces. The ledger makes leakage auditable and is summarized in every report.

### 4.2 Horizon semantics

- T anchored to `scheduled_close_time` — **never** to actual resolve time (anchoring to resolution leaks "this resolves about now"; Halawi's protocol deliberately avoids it).
- Default ladder: close −7d / −30d / −365d, clamped to `open_time` (horizon skipped if T < open). Additional modes: `spot` (T = `spot_scoring_time`, tournament realism) and `halawi_geometric` (n=5 geometric points in [open, close), research-protocol comparability).
- Prompt-masking `today={T}` injected everywhere (date arithmetic correctness), explicitly **not** counted as a contamination defense (closes only ~48% of the gap; reasoning models leak more — arXiv 2601.13717).

### 4.3 Historical research providers

Registered as `_<name>_research` in the existing registry; each declares a `cutoff_class` consumed by strict/pragmatic mode:

| Provider | Mechanism | Mode | Notes |
|---|---|---|---|
| `asknews_historical` | `historical=true`, `end_timestamp=T` | strict | Archive ≥2023; ingestion-time snapshots (leak-low by architecture, unaudited); 5× quota cost; **maximizes parity with the live `asknews` provider** |
| `gdelt_wayback` | GDELT DOC 2.0 URL enumeration in 3-month windows ending T (ArtList only searches trailing 3 months — page windows) → fetch Wayback snapshot ≤ T (`id_` raw mode) | strict | Reaches 2017; slowest; strongest snapshot guarantee; DIY RetroSearch/Pastcasting |
| `exa_historical` | `endCrawlDate=T` | strict (semi-trusted flag) | Crawl date ≫ publication metadata; unaudited |
| `markets_historical` | crosswalk + price-at-T | strict | Leak-free by construction; replaces live `market_research()` |
| `nyt_archive`, `guardian` | first-party archive APIs | strict | Leak-proof date spines; narrow coverage |
| `serper_dated`, `tavily_dated` | date-filtered live search + per-doc LLM leak-screen | **pragmatic only** | Measured leakage profile (71–81% of questions affected via Google/DDG filters; Brier inflated 0.244→0.108 — arXiv 2602.00758); expect ~30% screen discard; results stamped `date_trust=low` |
| *(Perplexity `_search_llm_research`)* | — | **excluded** | No historical mode exists; backtests structurally under-represent this live provider — stated in every report's comparability section |

### 4.4 LLM layer

- **Completion cache** (`llm_cache.py`): wraps `CalibratedEnsembleBot._invoke_model()` (`bot/ensemble_bot.py:83–89`); content-addressed on (model, prompt hash, temperature, run-index). Re-running with only post-LLM changes (aggregation, caps, calibration) costs $0; a prompt change invalidates exactly its downstream stages.
- **Memorization probe** (`probe.py`): zero-research direct outcome query per (contaminated-stratum question × model); confident-correct ⇒ `contaminated-confirmed` (cached). Analog of Halawi App. A.2.
- **Stage-level replay:** `ForecastArtifact` persists research text, full prompts, raw completions, per-member parses, aggregation inputs/outputs, final forecast — everything `logs/forecasts.jsonl` currently lacks — so any run replays from any stage.

## 5. Layer 3 — Systems under test

```python
class ForecastSystem(Protocol):
    name: str
    def config_fingerprint(self) -> str          # hashed into every artifact
    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact
```

Ships with:

1. **`CalibratedEnsembleAdapter`** — wraps the real bot via a parallel factory (does not mutate the live one; neutralizes skip/publish kwargs). `variant(**config_patch)` makes experiments config patches: aggregation method, provider set, caps, ensemble membership, calibration dir (pointed at an as-of-T snapshot or empty), etc.
2. **Stage entry points** on the adapter — `from_research_text(...)`, `from_member_forecasts(...)` — to backtest *part* of the system: e.g. identical frozen research into two aggregation variants isolates that stage causally.
3. **Baselines** — always-0.5 (chance), CP-at-T echo, market-price echo, raw single-model-no-research. Every report contextualizes variants against these floors (Metaculus/ForecastBench convention).

Future systems implement the three members and inherit the entire harness. The contract lives in `docs/backtesting/COMPATIBILITY.md` (see §9).

## 6. Layer 4 — Scoring, statistics, experiments, reports

### 6.1 Metaculus score family (exact; golden-tested vs Scores FAQ worked examples)

- **Log score:** binary/MC `ln P(outcome)` with clamps [0.001, 0.999]; continuous `ln pdf(resolution)` with +0.01 uniform floor and ~35 pdf cap on the 201-point CDF (step ≥ 5e-05, per-bin mass ≤ 0.2 — already produced by `bot/distributions.py:percentiles_to_cdf()`); discrete pmf floor `0.01/N`; out-of-bounds resolutions scored as binary on boundary mass.
- **Baseline score:** binary `100×(log₂p + 1)`; MC `100×(ln p − ln(1/N))/ln N`; continuous `100×(ln pdf − ln baseline)/2`, baseline 1.0/0.95/0.9 for closed/one-open/two-open bounds. (This rescaling is what makes mixed-type aggregation legitimate.)
- **Peer score:** `100×(ln p − ln GM(others))`; continuous/discrete ÷2; zero-sum per question (property-tested).
- **Spot variants:** evaluated at `spot_scoring_time` from the API per question (do not assume close time; FAQ default differs from bot-tournament convention).
- **Tournament aggregation:** `total = Σ(spot peer × question_weight)`; `take = max(total, 0)²` — reported for tournament-shaped corpora.
- The scorer scores the **submitted (capped) forecast** — the bot's `[0.02, 0.98]` self-caps become a measured tunable, not an invisible distortion.

### 6.2 Synthetic opponents (no other bots exist in a backtest)

- **Head-to-head spot peer vs CP-at-T:** `100 × ln(p_bot/p_CP)` on Yes (the exact statistic in Metaculus's Pros-vs-Bots analyses). CP-at-T = interval lookup over `aggregations.recency_weighted.history` (`start_time ≤ T < end_time`), matching forecasting-tools `get_cp_at_time` semantics; serialized history is minimized (~400 points) — adequate for spot/horizon lookups; full resolution needs restricted endpoints (Data Needs Form). Respect `include_bots_in_aggregates` and `cp_reveal_time` when interpreting.
- **Head-to-head vs market-price-at-T** via crosswalk (12h–1d granularity).
- **Expected baseline vs CP** (resolution-free, cheap) — computed but labeled with its documented failure mode: a CP-optimized prompt won offline and lost live (+2.76 vs +4.71/question).
- **ForecastBench-style interim truth** for unresolved market questions: score against platform aggregate at evaluation date; switch to resolution when available.

### 6.3 Diagnostics (every run, resolution-free where possible)

Quintile calibration curves with consistency bands (Bröcker & Smith); Murphy decomposition per question type (recalibration vs research/decisiveness investment signal); discrimination separation (mean forecast on Yes-resolvers − No-resolvers; the Pros-vs-bots 44pp-vs-26pp gap); Paleka consistency battery (negation pairs, MC sum-to-one, monotone scope ladders — zero resolutions needed); RMS calibration error (Halawi convention).

### 6.4 Statistics

- All variant comparisons **paired on identical (question, horizon) sets**: weighted paired t-tests + bootstrap CIs over questions; cluster-aware weighting for correlated question groups (mirrors Metaculus question weights; i.i.d. SEs understate).
- **Power honesty:** report detectable effect size at achieved N; ~400 paired questions for a 0.01 Brier gap at 2σ (δ≈0.1); ~100 questions ⇒ ~30% chance the worse bot wins. The report refuses to bold a winner inside the noise band.
- **Difficulty adjustment** for cross-subset comparison: two-way fixed effects `b_ij = α_i + γ_j + ε`, adjusted `b̃ = b − γ̂`; market questions' difficulty anchored at the market price's own Brier (ForecastBench updated methodology; validation Spearman 0.91).
- **Multi-seed noise floor:** rerun a config with resampled completions to measure within-config variance; comparisons are made against measured noise.

### 6.5 Experiments

Declarative YAML spec: corpus filter × horizon set × variant grid × budget profile × cutoff mode. Profiles: `smoke` (~30 q, one member model, 1 run — CI-friendly), `standard` (~150–300 q, full ensemble), `full` (all available corpus incl. strata, full ensemble, all horizons). Runner: **cost estimate printed before spending** + hard cost brake (extends `max_questions_per_run` pattern at `ensemble_bot.py:294–303`); per-question failure isolation (skip + log, never crash a run); resumable run manifest (re-invoke continues from cache); run ID + config fingerprints recorded in every artifact.

### 6.6 Reports

Per-run markdown + HTML: leaderboard vs baselines; scores stratified by (stratum × type × horizon × cutoff-mode); diagnostics; evidence-ledger leakage summary; cost/latency per stage; **worst-10 post-mortem appendix** with full transcripts tagged against the failure-mode taxonomy (FORECASTING-BOT-RESEARCH.md §8).

## 7. Error handling

External fetches: retry with backoff, disk-cache, degrade to partial corpus with explicit withheld/missing markers. Question import: resolution-date validation, malformed records quarantined with reasons. Runs: per-question isolation; manifest-based resume; cost brake aborts cleanly with partial artifacts intact and scoreable. Providers: a provider failure downgrades the research compositor exactly as the live two-track salvage does — recorded in the ledger.

## 8. Testing

- **Scorer golden tests** vs Metaculus Scores FAQ worked examples; CP-at-T parity vs forecasting-tools `get_cp_at_time`.
- **Property tests:** peer score zero-sum; proper-scoring sanity (truth-telling optimal); CDF validity under 2026 constraints; baseline-score sign conventions.
- **Canary leakage suite:** plant questions whose answer exists only post-T; ledger + leak-screen must catch the plant; run per provider.
- **`as_of=None` regression:** bot output paths bit-identical to pre-change behavior.
- **Integration:** tiny frozen corpus (~5 questions, full snapshots) committed to repo; whole pipeline with stubbed LLM in CI (`smoke` profile).

## 9. Compatibility contract (`docs/backtesting/COMPATIBILITY.md`)

The document future systems build against. Contents:

1. **`ForecastSystem` protocol** (§5) with semantics: must honor `ctx.as_of` (no wall-clock reads), must route all model calls through the supplied LLM client (cache/probe instrumentation), must not perform unledgered network research, must return a `ForecastArtifact` with every stage's inputs/outputs populated.
2. **`BacktestQuestion` and `ForecastArtifact` JSON schemas** (versioned, `schema_version` field; additive evolution policy).
3. **Research provider contract:** signature `_<name>_research(question, prompt_args) -> ResearchResult`, registry naming, required ledger writes, `cutoff_class` declaration.
4. **Seam inventory of the current bot** (the table in §2.1 + `_invoke_model`, `run_research`, `calibration_dir`, `log_path`) — the surface the sidecar depends on; changes to these in `bot/` require a matching harness update (enforced by the integration test).
5. **Scoring interface:** a system is scoreable iff its artifact contains a final forecast in platform submission format (binary p, MC vector, 201-point CDF).
6. **Cost accounting:** systems report token/call counts per stage so Pareto reports work.

Companion docs in the same directory: `README.md` (how to build a corpus, define an experiment YAML, run/replay/compare/report via `scripts/backtest.py`) and `DATA.md` (corpus store schema, blob layout, provenance/license fields per source, snapshot-on-sight policy, schema evolution rules).

## 10. Extensions (ranked; 1–3 in scope for initial build)

1. **Forward snapshot accumulator** — every live run archives its questions (token sees them forever); converts the live bot into a corpus factory. Highest leverage given the lockdown. *(In scope: `corpus/snapshot.py`.)*
2. **CI regression gate** — `smoke` profile on PRs vs pinned baseline run; fail on significant paired regression. *(In scope.)*
3. **Calibration flywheel** — backtest artifacts feed `scripts/fit_calibration.py` (hundreds of resolved points instead of waiting on live history). *(In scope: artifact→fitter export.)*
4. **Auto-generated dataset questions** — ForecastBench-style synthesis from FRED/Wikipedia/ACLED time series: unlimited, contamination-free by construction, continuously resolving; the escape hatch from the narrow clean window, and the practice most correlated with tournament winning (+2,216 pts). Biggest post-MVP win.
5. **Live-vs-backtest concordance** — when live results land, test whether the backtest predicted variant *ranking* (BTF validated pastcast↔live within 0.004 Brier); this validates the instrument itself.
6. **Leakage audit harness** — scheduled canary runs per provider to *measure* empirical leak rates (AskNews historical's rate is currently inferred, not audited).
7. **Pinned-old-model mode** — 2024-era model snapshots via OpenRouter widen the clean window into Halawi's corpus years; measures scaffolding under old models (clearly labeled as such).
8. **Horizon-skill curves** — head-to-head vs CP/market as f(horizon): is the edge far from close (thin crowds) or near close (sharp markets)? Informs live forecast timing.
9. **Scope-sensitivity probes** — paired question variants as a standing battery (bots documented badly scope-insensitive: 1.24× vs Pros' 1.02×).
10. **ForecastBench submission pipeline** — free external prospective eval through the same System adapter.
11. **Pareto dashboards** — score vs cost per variant (volume × take² makes cheap-and-slightly-worse strategically relevant).
12. **Environment-interface refactor (Architecture B)** — documented post-MVP seam, consistent with the System-2 precedent.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Metaculus data access approval is the critical path for post-2024 Metaculus corpus; timeline unknown; ToU requires asking | Tier-aware fetcher works in degraded mode today; datasets + markets + own-question loophole + accumulator; submit Data Needs Form now |
| Clean-corpus window may stay below the ~400-question power threshold | Detectable-effect-size honesty in every report; market questions; accumulator; extension 4 (dataset questions); pinned-old-model mode |
| AskNews historical leak rate unaudited; archive starts 2023; 5× quota | Canary audits (ext. 6); GDELT+Wayback as the verifiable channel; ledger transparency |
| Market history decays (Kalshi 3-month roll; Polymarket pruning undocumented) | Snapshot-on-sight; bulk archives; local permanence |
| Backtest↔live divergence (Perplexity track missing; crosswalk vs keyword market evidence) | Stated per-report comparability caveats; concordance tracking (ext. 5) |
| Bot seam drift breaks the sidecar | COMPATIBILITY.md seam inventory + CI integration test pins the contract |

## 12. Build order (for the implementation plan)

1. **Scoring layer** (standalone, golden-testable, immediately useful on existing logs).
2. **Corpus layer** (importers + store + crosswalk + accumulator; user submits Data Needs Form in parallel).
3. **Time-travel layer + bot `as_of` threading** (providers, cache, probe, ledger).
4. **Systems + experiments + reports** (protocol, adapter, baselines, runner, profiles, report).
5. **Extensions 1–3** (accumulator hook into live runs, CI gate, calibration flywheel) + `docs/backtesting/` docs.

## 13. Key references

- Metaculus Scores FAQ — https://www.metaculus.com/help/scores-faq/ ; openapi spec — https://github.com/Metaculus/metaculus/blob/main/docs/openapi.yml ; lockdown PR #4493; Data Needs Form (api-requests@metaculus.com)
- forecasting-tools (dependency) — https://github.com/Metaculus/forecasting-tools (`get_cp_at_time`, `ApiFilter`, Benchmarker)
- Halawi et al. 2024 — arXiv 2402.18563; repo dannyallover/llm_forecasting; HF datasets YuehHanChen/forecasting[_raw]
- ForecastBench — arXiv 2409.19839; github.com/forecastingresearch/forecastbench[-datasets]; difficulty-adjustment methodology PDF
- Search-leakage audit — arXiv 2602.00758 (ACL 2026); cutoff-prompt failure — arXiv 2601.13717; eval pitfalls — arXiv 2506.00723
- BTF/RetroSearch — arXiv 2506.21558; Autocast — arXiv 2206.15474; Paleka consistency — arXiv 2412.18544
- Polymarket docs — https://docs.polymarket.com (Gamma, prices-history, rate limits); Kalshi — https://docs.kalshi.com (candlesticks, historical); Manifold — https://docs.manifold.markets/api
- AskNews historical — https://docs.asknews.app/en/news ; GDELT DOC 2.0 — blog.gdeltproject.org/updates-doc-2-0-api/ ; Wayback CDX — archive.org/help/wayback_api.php ; Exa crawl-date filters — docs.exa.ai/reference/search
- Market↔question matching — arXiv 2601.01706, arXiv 2512.02436; metaforecast; Adjacent News — docs.adj.news
- FutureEval leaderboard methodology — metaculus.com/notebooks/42076; Pros-vs-bots quarterly analyses (Q4'24 LW, Q1/Q2'25 EA Forum)
