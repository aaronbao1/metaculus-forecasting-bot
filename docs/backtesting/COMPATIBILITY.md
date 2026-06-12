# Retrocast Compatibility Contract

Contract version 1 (`schema_version: 1`) — 2026-06-11.

Audience: (a) authors of future forecasting systems — including ones sharing
no code with the current bot — that want to run under this harness; (b)
maintainers of `bot/`, whose seams the sidecar depends on. This document is
normative; the CI smoke gate (`.github/workflows/backtest-smoke.yml`) and
the integration tests under `tests/backtest/` pin the behaviors described
here. A change that breaks them must update this document in the same PR.

## 1. The `ForecastSystem` protocol

Defined in `backtest/systems/protocol.py`:

```python
class ForecastSystem(Protocol):
    name: str
    def config_fingerprint(self) -> str
    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact
```

Semantics — all four are hard requirements:

1. **Honor `ctx.as_of`.** No wall-clock reads anywhere on the forecast path
   (`date.today()`, `datetime.now()`, "current"-state API endpoints). All
   date arithmetic derives from `ctx.as_of`. The shipped adapter is pinned
   by an `as_of=None` bit-identity regression test; new systems must provide
   the equivalent guarantee.
2. **Route every model call through `ctx.llm`**
   (`backtest/timetravel/llm_cache.py:CachedLLM.invoke(model, prompt,
   temperature, run_index)`). This is what makes runs content-addressed-
   cacheable ($0 re-runs of post-LLM stages), reproducible, cost-accounted,
   and memorization-probe-able. A system that side-channels its own LLM
   client produces artifacts the harness cannot replay or audit.
3. **No unledgered network research.** Every retrieved document/answer is
   recorded via `ctx.record(LedgerEntry(provider, query, url, doc_date,
   date_trust, note))` with `date_trust` in `{"snapshot", "publisher",
   "low"}`. In `strict` cutoff mode only mechanically time-locked providers
   may run; `pragmatic`-only providers must raise `RuntimeError` when the
   active mode forbids them.
4. **Artifact completeness.** The returned `ForecastArtifact` carries every
   stage's inputs/outputs (`stages: list[StageRecord]`) — research text,
   full prompts, raw completions, per-member parses, aggregation
   inputs/outputs — sufficient to replay any single stage. A skipped or
   failed question returns `forecast=None` with a non-None `error`; the
   `forecast()` coroutine itself never raises for per-question failures.

`config_fingerprint()` must change iff behavior-relevant configuration
changes; it is hashed into every artifact and into the deterministic run id.

## 2. JSON schemas

Both types serialize via `backtest/core.py` (`artifact_to_dict` /
`artifact_from_dict`): datetimes as iso8601 strings, enums by value.
Versioned by the integer `schema_version` field. Evolution policy: strictly
additive within a version (new optional fields only); any rename, removal,
or semantic change bumps the version, and readers accept current + previous.

### `BacktestQuestion`

| Field | Type | Notes |
|---|---|---|
| `qid` | str | `"<platform>:<native_id>"`, e.g. `"metaculus:31678"` |
| `platform` | str | metaculus, polymarket, kalshi, manifold, halawi, forecastbench |
| `native_id` | str | source-native id |
| `qtype` | str | `binary` \| `multiple_choice` \| `numeric` \| `discrete` |
| `title`, `description`, `resolution_criteria`, `fine_print` | str | dataset text stored verbatim |
| `open_time`, `scheduled_close_time`, `spot_scoring_time`, `actual_resolve_time` | iso8601 or null | |
| `resolution` | bool \| str \| float \| null | binary / MC option / numeric value |
| `options` | list[str] | MC only |
| `scaling` | object or null | numeric/discrete: `range_min`, `range_max`, `zero_point`, `open_lower_bound`, `open_upper_bound`, `cdf_size`, `unit` |
| `question_weight` | float | tournament weight (default 1.0) |
| `provenance` | object | source, fetch_time, license note (see `DATA.md`) |

### `ForecastArtifact` (`artifacts.jsonl`, one object per line)

| Field | Type | Notes |
|---|---|---|
| `qid` | str | |
| `horizon` | str | `close-7d` \| `close-30d` \| `close-365d` \| `spot` \| `geo-k` |
| `as_of` | iso8601 | the simulated "today" T |
| `system_name` | str | `ForecastSystem.name` |
| `config_fingerprint` | str | from `config_fingerprint()` |
| `forecast` | object or null | null = skipped/failed (see `error`) |
| `forecast.qtype` | str | as above |
| `forecast.binary_prob` | float or null | in (0,1) |
| `forecast.mc_probs` | object or null | option -> prob, sums to ~1 |
| `forecast.cdf` | list[float] or null | ascending, in [0,1], len == `cdf_size` (201 default) |
| `stages` | list[StageRecord] | `stage`, `inputs`, `outputs`, `model`, `tokens_in`, `tokens_out`, `cost_usd`, `started_at`, `ended_at` |
| `evidence_ledger` | list[LedgerEntry] | `provider`, `query`, `url`, `doc_date`, `date_trust`, `note` |
| `contamination` | str | `clean` \| `contaminated` \| `contaminated-confirmed` |
| `cutoff_mode` | str | `strict` \| `pragmatic` |
| `error` | str or null | per-question failure isolation |
| `schema_version` | int | currently 1 |

## 3. Research-provider contract

Providers are async functions named `_<name>_research`, resolved by name
from `bot.research`'s module namespace (the `globals().get(f"_{name}_research")`
dispatch in `bot/research.py`). Historical providers live in
`backtest/timetravel/providers.py` and are installed at harness startup by
`install_historical_providers()` — runtime `setattr` into `bot.research`,
zero `bot/` file edits, zero live-path changes.

Signature — identical to the live providers:

```python
async def _<name>_research(question: dict, cfg: BotConfig) -> str
```

- `question` keys: `question_text`, `resolution_criteria`, `fine_print`.
- Returns one markdown research section. Empty/whitespace output counts as
  provider failure: the compositor drops the section and degrades exactly
  like the live two-track salvage; total failure yields a flagged empty
  `ResearchResult`, never an exception.
- Historical providers read `ACTIVE_CONTEXT.get()`
  (`backtest/timetravel/asof.py`) for `as_of`, the evidence ledger, and the
  corpus store. They MUST raise `RuntimeError` when no context is set, or
  when the context's `cutoff_mode` forbids their `cutoff_class`.
- Each provider declares a `cutoff_class`: `strict` (time-locked by
  mechanism: AskNews historical, GDELT+Wayback, Exa `endCrawlDate`,
  markets-at-T, NYT/Guardian archives) or `pragmatic` (date-filtered live
  search — Serper/Tavily — screened per-doc by
  `backtest/timetravel/leakage.py:screen_docs`, results stamped
  `date_trust="low"`).
- Required ledger writes: one `LedgerEntry` per surfaced document, including
  documents the leak-screen later discards (discards are stamped too).
- The live Perplexity provider (`_search_llm_research`) has no historical
  mode and is excluded from backtests; every report states this
  comparability gap.

## 4. Bot seam inventory

The sidecar touches `bot/` at exactly these seams. Changing any of them
requires a matching harness update in the same PR — the smoke gate and
`tests/backtest/` integration tests enforce the contract. Line numbers are
a snapshot (2026-06-11, post `as_of` threading); names/signatures are the
contract, tests pin them, line numbers merely locate them.

| Seam | Location | Harness use |
|---|---|---|
| `BotConfig` fields (incl. `as_of: datetime \| None`) | `bot/config.py:34-54` | variant config patches; as-of threading |
| `config_hash(cfg)` — pops `as_of` before hashing | `bot/config.py:61-63` | live hash invariance with `as_of=None` |
| `CalibratedEnsembleBot._invoke_model(member_and_prompt)` | `bot/ensemble_bot.py:83-89` | wrapped by `CachedLLM` |
| `_days_remaining(close_iso, now=None)` | `bot/ensemble_bot.py:56-61` | as-of date math |
| `today=` prompt injections (3 sites) | `bot/ensemble_bot.py:129,150,170` | prompt masking with T |
| `run_binary/mc/numeric_ensemble(...)` result dataclasses | `bot/ensemble_bot.py` | adapter converts to `Forecast` |
| `run_research(question, cfg)` + `globals()` provider dispatch | `bot/research.py:84-90` | historical provider installation |
| `_prompt_args(question, cfg)` | `bot/research.py:34-40` | as-of search prompts |
| `percentiles_to_cdf(...)`, `CDF_SIZE = 201` | `bot/distributions.py` | numeric forecast conversion |
| `append_forecast(record, path)` / `read_forecasts(path)` | `bot/logging_store.py:12,20` | snapshot hook; calibration flywheel |
| `calibration_dir`, `log_path` config fields | `bot/config.py:48-49` | as-of-T calibration snapshots; log reads |
| `max_questions_per_run` cost-brake pattern | `bot/ensemble_bot.py:301-304` | precedent the runner's cost brake extends |

## 5. Calibration contract

The `CalibratedEnsembleAdapter` wraps the live `CalibratedEnsembleBot`.
Calibration (Platt scaling) applies to binary forecasts when
`allow_calibration=True` is passed at construction. The default is
`allow_calibration=False`, which acts as a sentinel guard: if a fitted
`binary.json` exists in `calibration_dir` and `allow_calibration=False`,
the adapter redirects `calibration_dir` to a temp location (no calibration
applied) and logs a warning. This prevents silent calibration application
in backtests where the Platt map was fitted on future data relative to T.

To deliberately use a calibration map in a backtest variant (e.g., to
measure calibration quality), explicitly pass `allow_calibration=True` and
point `calibration_dir` at a map whose training set predates T.

## 6. Scoring interface

A system is scoreable iff its artifact contains a final forecast in
platform submission format:

- **binary:** `binary_prob` in (0,1); the scorer clamps to [0.001, 0.999];
- **MC:** `mc_probs` over exactly the question's `options`, summing to ~1;
- **numeric/discrete:** `cdf` of `cdf_size` (default 201) ascending values
  in [0,1] satisfying the 2026 validity constraints (step >= 5e-05,
  per-bin mass <= 0.2).

Scores (`backtest/scoring/metaculus_scores.py`): `log_score`,
`baseline_score`, `peer_score`, `head_to_head`, `tournament_take`, `brier` —
golden-tested against the Metaculus Scores FAQ worked examples. The scorer
scores the **submitted (capped) forecast**: self-caps are a measured
tunable, not an invisible distortion. Out-of-bounds numeric resolutions are
scored as binary on boundary mass.

## 7. Cost accounting

Every `StageRecord` reports `tokens_in`, `tokens_out`, `cost_usd`,
`started_at`, `ended_at`. Systems that route all calls through `ctx.llm`
get these filled from the cache layer; systems with non-LLM paid calls
(e.g. metered search APIs) must fill them in their own stage records.
The runner manifest records `actual_cost_usd` (total spend for the run),
and the mid-run cost brake aborts cleanly when projected spend exceeds
the profile's `cost_cap_usd` — partial artifacts remain scoreable.
Reports aggregate cost/latency per stage; Pareto (score-vs-cost) analyses
assume these fields are honest.
