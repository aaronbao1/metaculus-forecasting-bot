# Retrocast Data Layer

Everything under `data/backtest/` is a **permanent local archive** —
gitignored, never auto-deleted. Run outputs land under the experiment
`--out` directory (CI uses `logs/backtest/`, also gitignored).

## Layout

```
data/backtest/
├── corpus.db        # SQLite metadata (schema below)
├── blobs/           # bulky series + raw snapshots, referenced by relative path
│   ├── <safe-qid>__<sha1-8>__<kind>.json   # one JSON file per (qid, kind) series;
│   │                                        #   kind = "cp" | "market_price";
│   │                                        #   <safe-qid> = qid with chars outside
│   │                                        #   [alnum -_.] replaced by "_";
│   │                                        #   <sha1-8> = first 8 hex chars of
│   │                                        #   SHA-1(qid) — collision-proof when
│   │                                        #   distinct qids have the same safe form
│   └── snapshots/
│       └── metaculus_<id>_<stamp>.json  # raw post-detail JSON written by the
│                                        #   forward snapshot accumulator
└── datasets/        # raw upstream downloads (halawi/, forecastbench-datasets/,
                     #  bulk market archives) — kept verbatim for provenance
```

The content-addressed **completion cache** does NOT live under
`data/backtest/`: it sits at `<out_dir>/llm_cache/` (the experiment `--out`
directory, e.g. `logs/backtest/agg-sweep/llm_cache/`) and is **shared across
run_ids** under that out directory — one `<sha256-key>.json` per completion,
keyed by (model, prompt, temperature, run_index) and carrying the completion
plus token/cost metadata. Post-LLM re-runs and `replay` hit it for $0.

## `corpus.db` schema (`schema_version` 1)

The authoritative DDL is `_SCHEMA` in `backtest/corpus/store.py`, reproduced
verbatim below; this section must be updated in the same PR as any store
schema change.

```sql
CREATE TABLE IF NOT EXISTS questions (
    qid TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    native_id TEXT NOT NULL,
    qtype TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    resolution_criteria TEXT NOT NULL DEFAULT '',
    fine_print TEXT NOT NULL DEFAULT '',
    open_time TEXT,
    scheduled_close_time TEXT,
    spot_scoring_time TEXT,
    actual_resolve_time TEXT,
    resolution TEXT,            -- JSON-encoded; SQL NULL means unresolved
    options TEXT NOT NULL DEFAULT '[]',
    scaling TEXT,
    question_weight REAL NOT NULL DEFAULT 1.0,
    provenance TEXT NOT NULL DEFAULT '{}',
    schema_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS series (
    qid TEXT NOT NULL,
    kind TEXT NOT NULL,         -- "cp" | "market_price"
    blob_path TEXT NOT NULL,    -- relative to store root
    n_points INTEGER NOT NULL,
    PRIMARY KEY (qid, kind)
);
CREATE TABLE IF NOT EXISTS contamination (
    qid TEXT NOT NULL,
    model_set_hash TEXT NOT NULL,
    stratum TEXT NOT NULL,      -- clean | contaminated | contaminated-confirmed
    PRIMARY KEY (qid, model_set_hash)
);
CREATE TABLE IF NOT EXISTS crosswalk (
    qid TEXT NOT NULL,
    market_qid TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    verified INTEGER NOT NULL DEFAULT 0,
    manual_override TEXT,       -- non-null always wins in market_for()
    PRIMARY KEY (qid, market_qid)
);
```

Semantics worth pinning:

- **Series live in blobs, not rows.** The `series` table holds one row per
  `(qid, kind)` pointing (via `blob_path`, relative to the store root) at
  `blobs/<safe-qid>__<sha1-8>__<kind>.json` — the full, sorted point list.
  `value_at(qid, kind, T)` is a step-function lookup: last point with
  `ts <= T`, else `None`. CP gap sentinels (stored as JSON `null`) are
  preserved in the series and skipped by `value_at`.

- **UTC normalization on write.** `put_series` normalizes all timestamps to
  UTC ISO-8601 before writing, so mixed-offset inputs ("+02:00", "Z",
  "-05:00") are stored and retrieved in correct chronological order. Naive
  timestamps are treated as UTC.

- **Withheld != missing.** The tier-aware Metaculus fetcher records fields
  the token cannot see as *withheld* (re-fetched when better access lands)
  — distinct from fields the source simply lacks.

- **Resolution-date validation runs at import** on every question (>=3.8%
  of even curated datasets have resolved-before-forecast-date defects);
  malformed records are quarantined with reasons, not silently dropped.

- **Contamination strata:** `clean` iff `open_time` > max model *release
  date* (not stated training cutoff — cutoff labels demonstrably lie)
  across the ensemble's models; the release-date table is data:
  `backtest/corpus/model_releases.py:RELEASE_DATES`. The memorization probe
  refines `contaminated` to `contaminated-confirmed`.

- **WAL mode + `check_same_thread=False`.** The store opens its SQLite
  connection with WAL journal mode and `check_same_thread=False`, so async
  fetchers can safely call the store from coroutines in the same process.
  Concurrent multi-process or multi-thread WRITES are not safe.

## Provenance and licensing per source

Every stored row carries `provenance` (source, fetch_time, license note).
Obligations that survive into derived work:

| Source | Access | License / terms | Provenance notes recorded |
|---|---|---|---|
| Metaculus API | bot token; tier-aware | Metaculus ToU — AI-eval use legitimized via the Data Needs Form (api-requests@metaculus.com) | tier at fetch; withheld fields; own-forecast questions fully visible |
| Halawi HF datasets (`YuehHanChen/forecasting[_raw]`) | HF download | dataset card license as published | dataset text stored **verbatim** (may drift from current Metaculus copies); import validation results |
| ForecastBench (`forecastingresearch/forecastbench-datasets`) | git clone | **CC BY-SA 4.0** — attribution required; *published* derivatives of the data must carry the same license | freeze values; cross-platform ids |
| Polymarket (Gamma + CLOB) | public API, no auth | public API terms; prices-history is empty below ~12h fidelity for resolved markets | fidelity used; fetch window |
| Kalshi | public REST | public API; ~3-month rolling live window, then `/historical/*`; fixed-point dollar fields post-2026-03; scalar markets skipped | settled timestamp; candlestick interval |
| Manifold | `/v0/bets` chain | API public; **bulk dumps (<=2024-07) non-commercial**; play-money market | redemption bets excluded; play-money caveat |
| pmxt archive | bulk download | CC BY 4.0 | snapshot granularity |
| jon-becker/prediction-market-analysis | bulk Parquet | repo license as published | which tables imported |

## Snapshot-on-sight policy

Market and CP history **decays**: the Kalshi live window rolls (~3 months),
Polymarket pruning is undocumented (its `/orderbook-history` endpoint
already died), and Metaculus visibility depends on access tier. Therefore:
anything fetched from any source is archived under `data/backtest/`
immediately and permanently at first sight; nothing is ever auto-deleted;
re-fetches add rows (keyed by fetch time) rather than overwrite. The
forward snapshot accumulator extends the same policy to every question the
live bot touches (`scripts/run_tournament.py` post-run hook,
`SNAPSHOT_AFTER_RUN` default true).

## `schema_version` evolution rules

1. **Additive-only within a version:** new optional fields/columns with
   defaults. Existing readers must keep working unmodified.
2. Any rename, removal, or semantic change **bumps the integer version** —
   in `backtest/core.py` dataclasses and in stored rows.
3. Readers accept the current and the previous version; older data is
   migrated by scripted, idempotent migrations in
   `backtest/corpus/store.py` that run on open.
4. Blobs embed their `schema_version` alongside payloads; the JSONL/Parquet
   layout never changes meaning silently.
5. `docs/backtesting/DATA.md` and `COMPATIBILITY.md` are updated in the
   same PR as any version bump (CI's seam tests make drift visible).
