# Retrocast — Backtesting the Forecasting Bot

Retrocast replays the full forecasting pipeline — or any single stage of it —
on historical questions as if "today" were a past date T, with enforceable,
auditable information cutoffs, and scores the way Metaculus scores bots.

**Doctrine (never forget this):** the backtest is a *relative comparison
instrument plus an upper-bound absolute estimate* — never ground truth for
live skill. Prospective evaluation (MiniBench, the bot testing area,
ForecastBench submission) remains the absolute yardstick. Headline numbers
come from the **clean contamination stratum** under **strict** cutoff mode.

Design spec: `docs/superpowers/specs/2026-06-11-backtesting-system-design.md`.
Companion docs: `COMPATIBILITY.md` (the contract systems plug into),
`DATA.md` (corpus store, blobs, provenance, licenses).

## Layout

```
backtest/            sidecar package (corpus / timetravel / systems / scoring / experiments / reports)
scripts/backtest.py  CLI: build-corpus | run | replay | compare | report
experiments/         experiment YAML specs (smoke.yaml is the CI gate)
data/backtest/       corpus.db + blobs + datasets               (gitignored, permanent)
logs/backtest/       run outputs (artifacts, scores, reports)
                     + llm_cache/ shared across run_ids         (gitignored)
```

## Install

```bash
pip install -e ".[dev]"
```

The harness reuses the live bot's `.env` keys. Required keys:
`OPENROUTER_API_KEY`, `ASKNEWS_CLIENT_ID`, `ASKNEWS_CLIENT_SECRET`,
`EXA_API_KEY`, `METACULUS_TOKEN`. Backtest-only: `SERPER_API_KEY`
(required for pragmatic mode). NYT and Guardian first-party archive
providers are spec'd but not yet implemented — see the deferred-items
list in the implementation plan (`docs/superpowers/specs/`).

## Building a corpus

All fetches snapshot-on-sight: everything retrieved is archived permanently
under `data/backtest/` (see `DATA.md`). Check the exact flags with
`python scripts/backtest.py build-corpus --help`.

**Metaculus (tier-aware; degraded until Data Needs Form approval):**

```bash
python scripts/backtest.py build-corpus --source metaculus \
    --tournaments aibq3 aibq4 fall-aib-2025 spring-aib-2026 summer-futureeval-2026 minibench
```

The fetcher stores what the token can see, records withheld fields
distinctly from missing ones, and upgrades rows when better access lands.
Submit the Metaculus Data Needs Form (api-requests@metaculus.com) — it is
both the access path and the Terms-of-Use legitimacy path.

**Halawi et al. datasets (pre-lockdown, 2015–2024):**

```bash
pip install -U huggingface_hub
hf download YuehHanChen/forecasting     --repo-type dataset --local-dir data/backtest/datasets/halawi
hf download YuehHanChen/forecasting_raw --repo-type dataset --local-dir data/backtest/datasets/halawi_raw
python scripts/backtest.py build-corpus --source halawi \
    --dataset-dir data/backtest/datasets/halawi
```

**ForecastBench datasets (nightly, CC BY-SA 4.0):**

```bash
git clone https://github.com/forecastingresearch/forecastbench-datasets \
    data/backtest/datasets/forecastbench-datasets
python scripts/backtest.py build-corpus --source forecastbench \
    --dataset-dir data/backtest/datasets/forecastbench-datasets
```

**Markets as questions + opponent price series** (closed/settled selection
happens inside the fetchers):

```bash
python scripts/backtest.py build-corpus --source polymarket
python scripts/backtest.py build-corpus --source kalshi
python scripts/backtest.py build-corpus --source manifold
```

**Forward snapshot accumulator (zero effort, highest leverage):** every
`scripts/run_tournament.py` run archives every question it touched into
`data/backtest/` (the bot token sees own-forecast questions at full fidelity
forever). On by default; disable with `SNAPSHOT_AFTER_RUN=false`. Note:
GitHub-Actions runners are ephemeral — the accumulator pays off where
`data/backtest/` persists (local runs), not in the scheduled CI run.

## Defining an experiment

```yaml
# experiments/agg-sweep.yaml
name: agg-sweep
corpus:                      # CorpusStore.query() filters
  platforms: [metaculus]
  qtypes: [binary]
  resolved_only: true
  stratum: clean             # headline stratum; omit to include contaminated
horizons: [close-7d, close-30d]   # also: close-365d, spot, geo-k
cutoff_mode: strict          # strict is default + headline; pragmatic is flagged
profile: standard            # smoke | standard | full  (see table below)
variants:                    # each entry patches BotConfig via adapter.variant()
  - {name: trimmed_gmo, patch: {aggregation_method: trimmed_gmo}}
  - {name: median,      patch: {aggregation_method: median}}
baselines: [chance, cp_echo] # floors every report contextualizes against
systems:                     # optional: registry-backed non-ensemble systems
  - {name: ts_naive, kind: timeseries, params: {engine: naive_drift}}
seed: 0
```

`systems:` entries resolve through `backtest/systems/registry.py` (kind
`timeseries` → `TimeSeriesSystem`, engines in
`backtest/submodels/engines.py`: `naive_drift` always available; `theta`/
`autoarima`/`autoets` via the `timeseries` extra; `chronos2`/`timesfm` via
`tsfm`). They spend no LLM budget and forecast numeric/discrete questions
that have a `kind="target"` series (see DATA.md) — build those with
`build-corpus --source fred --series-ids CPIAUCSL`, the CSV importer
(`backtest/corpus/series_csv.py`), and `synth-questions` (synthetic numeric
questions from a stored series; Retrocast ext. 4). The `timeseries` research
provider injects the same anchor as a dossier section; `asknews_rated`
relevance-filters the AskNews dossier (GABRIEL-style, `bot/rating.py`).

Horizon T anchors to `scheduled_close_time` (never actual resolve time —
that leaks "resolves about now"), clamped to `open_time`; `spot` uses the
question's `spot_scoring_time`.

## Budget profiles

| Profile | Max questions | Ensemble | Intended use |
|---|---|---|---|
| `smoke` | 30 | single member (`openai/gpt-5.4`, 1 run) | CI, wiring checks |
| `standard` | 300 | full configured ensemble | routine variant comparisons |
| `full` | unlimited | full ensemble, all strata/horizons | pre-decision deep runs |

Authoritative numbers (including per-profile `cost_cap_usd`) live in
`backtest/experiments/profiles.py`. The runner prints a cost estimate and
asks for confirmation before spending, and a hard cost brake aborts cleanly
(partial artifacts remain scoreable) when the projection exceeds the cap.

## Running

```bash
# run an experiment (resumable: re-invoking continues pending questions)
python scripts/backtest.py run --spec experiments/agg-sweep.yaml \
    --corpus data/backtest --out logs/backtest/agg-sweep

# replay from cache only — $0, fails on any cache miss
python scripts/backtest.py replay --spec experiments/agg-sweep.yaml \
    --corpus data/backtest --out logs/backtest/agg-sweep

# compare two variants with paired stats (bootstrap CIs, paired t)
python scripts/backtest.py compare logs/backtest/agg-sweep/<run_id> trimmed_gmo median

# render the report (leaderboard, strata, diagnostics, ledger summary,
# cost/latency, worst-10 post-mortems)
python scripts/backtest.py report logs/backtest/agg-sweep/<run_id>
```

`<run_id>` is deterministic (`{name}-{seed}-{config-fingerprint-hash}`), so
reruns of an unchanged spec resume rather than fork. Re-running with only
post-LLM changes (aggregation, caps, calibration) costs $0 thanks to the
content-addressed completion cache.

## CI smoke gate

`.github/workflows/backtest-smoke.yml` runs on every PR: backtest unit
tests, then the full pipeline on the committed 5-question mini corpus
(`tests/backtest/fixtures/mini_corpus/`) with the deterministic no-network
`StubSystem` (`scripts/backtest.py run --spec experiments/smoke.yaml --system stub`),
builds the report, and fails if the mean `baseline_score` drops more than
the tolerance below `tests/backtest/fixtures/smoke_baseline.json`.
After an intentional scoring change:

```bash
python scripts/check_regression.py --scores <run_dir>/scores.jsonl \
    --baseline tests/backtest/fixtures/smoke_baseline.json \
    --update-baseline --tolerance 1.0
```

and commit the updated baseline with the change that moved it.

## Failure taxonomy and optimization

`backtest failures <run-dir>` post-mortems a run's worst losses into a
Clio-style clustered taxonomy (`failures.json` + `failures.md`; replay-only
unless `--live`). Downstream: `scripts/playbook_update.py` turns facets into
candidate playbook bullets, and `scripts/optimize_prompt.py` evolves the
binary prompt against cached runs — promotion rules and the item-6
evolutionary-search plug point live in `OPTIMIZE.md`.

## Calibration flywheel

```python
from backtest.experiments.export_calibration import (
    artifacts_to_calibration_rows, resolutions_from_rows)
from scripts.fit_calibration import join_resolutions
from bot.calibration import MIN_FIT_N, fit_platt, save_platt

rows = artifacts_to_calibration_rows("logs/backtest/agg-sweep/<run_id>")
preds, outcomes = join_resolutions(rows, resolutions_from_rows(rows))
if len(preds) >= MIN_FIT_N:
    save_platt(fit_platt(preds, outcomes), "calibration-backtest/binary.json")
```

Fit into a *separate* calibration dir and point a variant at it
(`patch: {calibration_dir: calibration-backtest}`); promote to the live
`calibration/` dir only deliberately — backtest-fitted maps inherit the
backtest's biases.

## Comparability caveats (stated in every report)

- The live Perplexity research track (`search_llm`) has no historical mode
  and is **excluded** from backtests — backtests under-represent it.
- Market evidence differs in kind: live uses keyword `market_research()`;
  backtests use the crosswalk + price-at-T.
- Prompt-masked `today=T` is date arithmetic, not a contamination defense;
  trust the strata, and the clean stratum first.
