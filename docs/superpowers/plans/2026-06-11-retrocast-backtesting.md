# Retrocast Backtesting System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Retrocast sidecar backtesting system from the approved spec ([docs/superpowers/specs/2026-06-11-backtesting-system-design.md](../specs/2026-06-11-backtesting-system-design.md)): replay the forecasting pipeline (or any stage of it) on historical questions at past dates T with auditable information cutoffs, score it the way Metaculus scores bots, and compare arbitrary system variants with proper paired statistics.

**Architecture:** A new `backtest/` package (corpus → timetravel → systems → scoring/experiments/reports layers, one-way dependencies) that injects into the existing bot through its tested seams. The **only** changes to `bot/` happen in Task 18 (`as_of` threading through `BotConfig`, prompts, and `days_until`; `config_hash` unchanged for live configs). Historical research providers are runtime-registered into `bot.research`'s dispatch — no live code path changes; with `as_of=None` the live bot is bit-identical to today.

**Tech Stack:** Python 3.12, pytest + pytest-asyncio (existing conventions), stdlib sqlite3 for the corpus store, httpx (existing shared-client pattern from `bot/markets.py`), numpy/scipy (already dependencies), pyyaml (added in Task 29), forecasting-tools (existing dependency, lazy-imported).

**Task map:** Phase 1 = Tasks 1–7 (core types + scoring); Phase 2 = Tasks 8–17 (corpus); Phase 3 = Tasks 18–24 (time travel + the one bot-touching task); Phase 4 = Tasks 25–31 (systems, experiments, reports, CLI); Phase 5 = Tasks 32–36 (live snapshot hook, CI smoke gate, calibration flywheel, docs, housekeeping). Phases must land in order; tasks within a phase are sequential.

**Cross-cutting rules for implementers:**
- Tests never touch the network: monkeypatch httpx clients / module attributes (pattern: `tests/test_framework_adapter.py`).
- Only Task 18 may modify files under `bot/`. Everything else is additive sidecar.
- Commit after every task; messages `feat(backtest):` / `test(backtest):` ending with the Co-Authored-By line shown in each commit step.
- A user-side action runs in parallel with no code dependency: **submit the Metaculus Data Needs Form** (spec §3.2) — the corpus fetcher (Task 11) is tier-aware and upgrades records automatically when access lands.

---

## Deferred items (post-MVP backlog — deliberate cuts, recorded for honesty)

These spec items are NOT implemented by this plan. Each lists where it hooks in when picked up later:

| Deferred item | Spec § | Hook point |
|---|---|---|
| Retry-with-backoff on corpus fetchers (currently: cache + per-record isolation + quarantine, no retries) | §7 | wrap HTTP calls in `backtest/corpus/*.py` |
| Embedding-based crosswalk candidate generation (MVP ships lexical top-k; seam comment in code) | §3.3 | `backtest/corpus/crosswalk.py` |
| Manifold bet-chain pagination beyond 1,000 bets (MVP: single page; seam comment) | §3.2 | `backtest/corpus/manifold.py` |
| `nyt_archive`, `guardian`, `tavily_dated` research providers (env keys reserved in `.env.template`) | §4.3 | `backtest/timetravel/providers.py` |
| Raw single-model-no-research baseline (4th baseline) | §5 | `backtest/systems/baselines.py` |
| Expected-baseline-vs-CP resolution-free metric; ForecastBench-style interim truth for unresolved markets | §6.2 | `backtest/scoring/opponents.py` + runner scoring pass |
| Paleka consistency battery; RMS calibration error | §6.3 | `backtest/scoring/diagnostics.py` |
| Cluster-aware question weighting; multi-seed noise floor reruns | §6.4 | `backtest/scoring/stats.py` + `experiments/runner.py` |
| CP-at-T parity golden test vs forecasting-tools `get_cp_at_time` (semantics asserted, parity test deferred) | §8 | `tests/backtest/test_metaculus_fetch.py` |
| Canary leakage runs for every provider (MVP: serper path only) | §8 | `tests/backtest/` + leakage audit harness (spec §10 ext. 6) |
| Worst-10 post-mortems with full transcripts + failure-taxonomy tagging (MVP: worst-10 table only); HTML report | §6.6 | `backtest/reports/report.py` |
| `from_member_forecasts` beyond binary (MVP: binary-only, raises NotImplementedError otherwise) | §5 | `backtest/systems/ensemble_adapter.py` |

Known intentional fail-safes the implementer must resolve before trusting results: `backtest/corpus/model_releases.py` ships placeholder release dates (verify before the first stratified report — checklist item in Task 36); out-of-bounds continuous baseline uses the platform's 0.05 chance value (provenance comment in `metaculus_scores.py`).

---
## Phase 1 — Core types & scoring layer (Tasks 1–7)

**Scope:** spec §6.1–§6.4 (scoring, opponents, diagnostics, statistics) plus the core type contract. Everything in this phase is pure math over in-memory values: no network, no I/O beyond the repo, no async. All tests are plain synchronous pytest functions (repo convention: `asyncio_mode = "auto"`, plain functions, no test classes).

**Formula provenance.** The Metaculus Scores FAQ (`https://www.metaculus.com/help/scores-faq/`) is Cloudflare-gated to non-browser fetchers (HTTP 403), so the FAQ's formulas and worked examples were extracted from the identical published source in the Metaculus open-source repo (`github.com/Metaculus/metaculus`): `front_end/src/app/(main)/help/scores-faq/page.tsx`, `.../components/baseline_math.tsx`, `.../components/peer_math.tsx`. Bucket mapping and the out-of-bounds baseline convention were verified against the platform's actual scoring implementation: `scoring/score_math.py` and `utils/the_math/formulas.py` in the same repo. Every golden number in the tests below cites the FAQ section it comes from.

**Conventions implemented (contract + FAQ + platform source):**
- Binary/MC probabilities clamped to `[0.001, 0.999]` before any `ln` (FAQ: "Metaculus restricts binary predictions to be between 0.1% and 99.9%").
- Continuous (numeric) questions score the 201-point CDF: `pmf = diff(cdf)` over `n_bins = len(cdf) - 1` standardized-domain bins, `pdf = pmf × n_bins`, clamped to `[0.01, 35.0]` (FAQ: pdfs have a 0.01 uniform floor and are restricted "to be between 0.01 and ~35").
- Discrete questions score the pmf directly with floor `0.01 / n_bins` (FAQ: "the pmf is used in place of the pdf").
- Out-of-bounds resolutions are scored as binary on the boundary mass (`cdf[0]` below, `1 − cdf[-1]` above). For the baseline score the chance value of that implied binary is **0.05** — the uniform baseline's own out-of-bound mass — per the platform implementation (`scoring/score_math.py`, `evaluate_forecasts_baseline_*`: boundary buckets use `baseline = 0.05`), not the binary-baseline 0.5. This is the one place the FAQ prose ("scored like a binary question") underdetermines the formula; the platform source is authoritative.
- Continuous/discrete peer and head-to-head scores are divided by 2 (FAQ "Peer score math").
- Resolution→bucket mapping mirrors `utils/the_math/formulas.py:unscaled_location_to_bucket_index` exactly, including log scaling via `zero_point` and the "exactly range_min lands in bucket 1, exactly range_max in bucket n_bins" edge rules.

Run all commands from the repo root (`/Users/aaronbao/Developer/sideprojects/forecasting`).

---

### Task 1: Package scaffold + core types + artifact serialization

**Files:**
- Create: `backtest/__init__.py`
- Create: `backtest/core.py`
- Create: `tests/backtest/__init__.py`
- Modify: `pyproject.toml:17-18` (`[tool.setuptools]` packages)
- Test: `tests/backtest/test_core.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/__init__.py` (empty file), then create `tests/backtest/test_core.py`:

```python
"""Core type contract: QType, BacktestQuestion, Forecast, StageRecord,
LedgerEntry, ForecastArtifact, and the artifact <-> dict JSONL round-trip."""
import json
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from backtest.core import (
    BacktestQuestion,
    Forecast,
    ForecastArtifact,
    LedgerEntry,
    QType,
    StageRecord,
    artifact_from_dict,
    artifact_to_dict,
)


def _question(**kw):
    base = dict(
        qid="metaculus:31678",
        platform="metaculus",
        native_id="31678",
        qtype=QType.BINARY,
        title="Will X happen?",
    )
    base.update(kw)
    return BacktestQuestion(**base)


def test_qtype_values_are_platform_strings():
    assert QType.BINARY.value == "binary"
    assert QType.MULTIPLE_CHOICE.value == "multiple_choice"
    assert QType.NUMERIC.value == "numeric"
    assert QType.DISCRETE.value == "discrete"
    assert QType("binary") is QType.BINARY


def test_question_defaults_and_frozen():
    q = _question()
    assert q.description == ""
    assert q.resolution_criteria == ""
    assert q.fine_print == ""
    assert q.open_time is None and q.scheduled_close_time is None
    assert q.spot_scoring_time is None and q.actual_resolve_time is None
    assert q.resolution is None
    assert q.options == ()
    assert q.scaling is None
    assert q.question_weight == 1.0
    assert q.provenance == {}
    with pytest.raises(FrozenInstanceError):
        q.title = "nope"


def test_forecast_is_frozen_with_optional_fields():
    f = Forecast(qtype=QType.BINARY, binary_prob=0.6)
    assert f.mc_probs is None and f.cdf is None
    with pytest.raises(FrozenInstanceError):
        f.binary_prob = 0.7


def test_artifact_round_trip_full():
    art = ForecastArtifact(
        qid="metaculus:31678",
        horizon="close-7d",
        as_of=datetime(2025, 3, 1, 12, 30),
        system_name="ensemble",
        config_fingerprint="abc123",
        forecast=Forecast(
            qtype=QType.NUMERIC, cdf=tuple(i / 200 for i in range(201))
        ),
        stages=[
            StageRecord(
                stage="research",
                inputs={"question": "Will X happen?"},
                outputs={"report": "evidence summary"},
                model="openai/gpt-5.4",
                tokens_in=10,
                tokens_out=20,
                cost_usd=0.01,
                started_at="2025-03-01T12:00:00",
                ended_at="2025-03-01T12:01:00",
            )
        ],
        evidence_ledger=[
            LedgerEntry(
                provider="asknews",
                query="x event",
                url="https://example.com/a",
                doc_date="2025-02-27",
                date_trust="snapshot",
                note="archive hit",
            )
        ],
        contamination="clean",
        cutoff_mode="strict",
        error=None,
    )
    d = artifact_to_dict(art)
    wire = json.loads(json.dumps(d))  # must survive a JSONL round-trip
    back = artifact_from_dict(wire)
    assert back == art
    assert back.forecast.cdf == art.forecast.cdf  # tuple restored, not list


def test_artifact_round_trip_skipped_forecast():
    art = ForecastArtifact(
        qid="q",
        horizon="spot",
        as_of=datetime(2025, 1, 1),
        system_name="chance",
        config_fingerprint="00",
        forecast=None,
        error="skipped: no data",
    )
    back = artifact_from_dict(json.loads(json.dumps(artifact_to_dict(art))))
    assert back == art
    assert back.forecast is None
    assert back.error == "skipped: no data"


def test_artifact_round_trip_mc_forecast():
    art = ForecastArtifact(
        qid="metaculus:9",
        horizon="close-30d",
        as_of=datetime(2025, 6, 1, 0, 0, 1),
        system_name="s",
        config_fingerprint="f",
        forecast=Forecast(
            qtype=QType.MULTIPLE_CHOICE, mc_probs={"a": 0.7, "b": 0.3}
        ),
    )
    back = artifact_from_dict(json.loads(json.dumps(artifact_to_dict(art))))
    assert back == art
    assert back.forecast.mc_probs == {"a": 0.7, "b": 0.3}


def test_dict_uses_iso_strings_and_enum_values():
    art = ForecastArtifact(
        qid="q",
        horizon="spot",
        as_of=datetime(2025, 1, 2, 3, 4, 5),
        system_name="s",
        config_fingerprint="f",
        forecast=Forecast(qtype=QType.BINARY, binary_prob=0.6),
    )
    d = artifact_to_dict(art)
    assert d["as_of"] == "2025-01-02T03:04:05"
    assert d["forecast"]["qtype"] == "binary"
    assert d["schema_version"] == 1
    assert d["contamination"] == "contaminated"  # default per contract
    assert d["cutoff_mode"] == "strict"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest'` (collection error)

- [ ] **Step 3: Write minimal implementation**

Create `backtest/__init__.py`:

```python
"""Retrocast: sidecar backtesting harness for the forecasting bot."""
```

Create `backtest/core.py` (contract-exact types; stdlib only):

```python
"""Core backtest types (contract: stdlib-only; see docs/superpowers/specs/
2026-06-11-backtesting-system-design.md and docs/backtesting/COMPATIBILITY.md).

Artifacts serialize to JSONL via artifact_to_dict / artifact_from_dict:
datetimes as iso8601 strings, enums as their string values, tuples as lists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum


class QType(str, Enum):
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    DISCRETE = "discrete"


@dataclass(frozen=True)
class BacktestQuestion:
    qid: str                      # "<platform>:<native_id>", e.g. "metaculus:31678"
    platform: str                 # metaculus|polymarket|kalshi|manifold|halawi|forecastbench
    native_id: str
    qtype: QType
    title: str
    description: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    open_time: datetime | None = None
    scheduled_close_time: datetime | None = None
    spot_scoring_time: datetime | None = None
    actual_resolve_time: datetime | None = None
    resolution: object | None = None   # bool (binary) | option str (MC) | float (numeric) | None
    options: tuple[str, ...] = ()
    scaling: dict | None = None        # numeric/discrete: {"range_min","range_max","zero_point",
                                       #  "open_lower_bound","open_upper_bound","cdf_size","unit"}
    question_weight: float = 1.0
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Forecast:
    qtype: QType
    binary_prob: float | None = None          # (0,1)
    mc_probs: dict[str, float] | None = None  # option -> prob, sums to ~1
    cdf: tuple[float, ...] | None = None      # ascending values in [0,1]; len == cdf_size (201 default)


@dataclass
class StageRecord:
    stage: str                    # "research"|"member_llm"|"aggregation"|"calibration"|"probe"|...
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    started_at: str = ""          # iso8601
    ended_at: str = ""


@dataclass
class LedgerEntry:
    provider: str
    query: str
    url: str | None = None
    doc_date: str | None = None   # iso or None
    date_trust: str = "low"       # "snapshot" | "publisher" | "low"
    note: str = ""


@dataclass
class ForecastArtifact:
    qid: str
    horizon: str                  # "close-7d"|"close-30d"|"close-365d"|"spot"|"geo-k"
    as_of: datetime
    system_name: str
    config_fingerprint: str
    forecast: Forecast | None     # None if skipped/failed
    stages: list[StageRecord] = field(default_factory=list)
    evidence_ledger: list[LedgerEntry] = field(default_factory=list)
    contamination: str = "contaminated"   # "clean"|"contaminated"|"contaminated-confirmed"
    cutoff_mode: str = "strict"
    error: str | None = None
    schema_version: int = 1


def artifact_to_dict(a: ForecastArtifact) -> dict:
    """JSON-safe dict: datetimes -> iso strings, enums -> values, tuples -> lists."""
    d = asdict(a)
    d["as_of"] = a.as_of.isoformat()
    if a.forecast is not None:
        d["forecast"]["qtype"] = a.forecast.qtype.value
        if a.forecast.cdf is not None:
            d["forecast"]["cdf"] = list(a.forecast.cdf)
    return d


def artifact_from_dict(d: dict) -> ForecastArtifact:
    fc = d.get("forecast")
    forecast = None
    if fc is not None:
        forecast = Forecast(
            qtype=QType(fc["qtype"]),
            binary_prob=fc.get("binary_prob"),
            mc_probs=fc.get("mc_probs"),
            cdf=tuple(fc["cdf"]) if fc.get("cdf") is not None else None,
        )
    return ForecastArtifact(
        qid=d["qid"],
        horizon=d["horizon"],
        as_of=datetime.fromisoformat(d["as_of"]),
        system_name=d["system_name"],
        config_fingerprint=d["config_fingerprint"],
        forecast=forecast,
        stages=[StageRecord(**s) for s in d.get("stages", [])],
        evidence_ledger=[LedgerEntry(**e) for e in d.get("evidence_ledger", [])],
        contamination=d.get("contamination", "contaminated"),
        cutoff_mode=d.get("cutoff_mode", "strict"),
        error=d.get("error"),
        schema_version=d.get("schema_version", 1),
    )
```

In `pyproject.toml`, replace:

```toml
[tool.setuptools]
packages = ["bot"]
```

with:

```toml
[tool.setuptools.packages.find]
include = ["bot*", "backtest*"]
```

(auto-discovers `backtest` and its subpackages added in later tasks; `bot` discovery is unchanged).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_core.py -v`
Expected: PASS (7 tests). Also run `pytest tests/ -q` to confirm the pyproject change broke nothing.

- [ ] **Step 5: Commit**

```bash
git add backtest/__init__.py backtest/core.py tests/backtest/__init__.py tests/backtest/test_core.py pyproject.toml && git commit -m "feat(backtest): core types + ForecastArtifact JSONL round-trip" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Metaculus log score (all question types) + Brier

**Files:**
- Create: `backtest/scoring/__init__.py`
- Create: `backtest/scoring/metaculus_scores.py`
- Test: `tests/backtest/test_log_score.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_log_score.py`:

```python
"""Golden tests for the Metaculus log score and the Brier score.

Golden numbers come from the worked examples in the Metaculus Scores FAQ
(https://www.metaculus.com/help/scores-faq/, content verified against the
platform's open-source repo). The FAQ section is cited next to each number.
"""
import math

import pytest

from backtest.core import BacktestQuestion, Forecast, QType
from backtest.scoring.metaculus_scores import brier, log_score


def _binary_q(resolution):
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="Will it happen?", resolution=resolution)


def _mc_q(resolution, options):
    return BacktestQuestion(
        qid="metaculus:2", platform="metaculus", native_id="2",
        qtype=QType.MULTIPLE_CHOICE, title="Which one?",
        resolution=resolution, options=options)


def _numeric_q(resolution, *, qtype=QType.NUMERIC, range_min=0.0,
               range_max=100.0, zero_point=None,
               open_lower=False, open_upper=False, cdf_size=201):
    return BacktestQuestion(
        qid="metaculus:3", platform="metaculus", native_id="3", qtype=qtype,
        title="How many?", resolution=resolution,
        scaling={"range_min": range_min, "range_max": range_max,
                 "zero_point": zero_point, "open_lower_bound": open_lower,
                 "open_upper_bound": open_upper, "cdf_size": cdf_size,
                 "unit": "units"})


def build_cdf(*, n_points=201, below=0.0, above=0.0,
              spike_bucket=None, spike_mass=0.0):
    """CDF on the standardized domain. `below`/`above` are out-of-bound
    masses; `spike_mass` goes into inbound bin `spike_bucket` (1-indexed,
    platform bucket convention: bin b spans cdf[b-1]..cdf[b]); the remaining
    inbound mass is spread uniformly over the other bins."""
    n_bins = n_points - 1
    inbound = 1.0 - below - above
    if spike_bucket is None:
        masses = [inbound / n_bins] * n_bins
    else:
        base = (inbound - spike_mass) / (n_bins - 1)
        masses = [spike_mass if b == spike_bucket else base
                  for b in range(1, n_bins + 1)]
    cdf = [below]
    for m in masses:
        cdf.append(cdf[-1] + m)
    cdf[-1] = 1.0 - above
    return tuple(cdf)


def _bin(p):
    return Forecast(qtype=QType.BINARY, binary_prob=p)


# ------------------------------------------------------------- binary and MC

def test_binary_log_score_faq_extreme_probabilities_table():
    # FAQ "What is the log score?": predicting 99% / 99.9% Yes scores
    # -0.01 / -0.001 if the outcome is Yes and -4.6 / -6.9 if it is No.
    assert log_score(_bin(0.99), _binary_q(True)) == pytest.approx(-0.01, abs=0.005)
    assert log_score(_bin(0.99), _binary_q(False)) == pytest.approx(-4.6, abs=0.01)
    assert log_score(_bin(0.999), _binary_q(True)) == pytest.approx(-0.001, abs=0.0005)
    # 1 - 0.999 = 0.001 sits exactly on the clamp floor: ln(0.001) = -6.9078
    assert log_score(_bin(0.999), _binary_q(False)) == pytest.approx(-6.9, abs=0.01)


def test_binary_log_score_faq_die_roll_table():
    # FAQ "What is a proper scoring rule?" die-roll table: log scores of
    # predicting 5% / 17% / 30% on "die lands 6" are -3 / -1.77 / -1.2 when it
    # lands 6, and -0.05 / -0.19 / -0.37 otherwise.
    assert log_score(_bin(0.05), _binary_q(True)) == pytest.approx(-3.0, abs=0.005)
    assert log_score(_bin(0.17), _binary_q(True)) == pytest.approx(-1.77, abs=0.005)
    assert log_score(_bin(0.30), _binary_q(True)) == pytest.approx(-1.2, abs=0.005)
    assert log_score(_bin(0.05), _binary_q(False)) == pytest.approx(-0.05, abs=0.002)
    assert log_score(_bin(0.17), _binary_q(False)) == pytest.approx(-0.19, abs=0.005)
    assert log_score(_bin(0.30), _binary_q(False)) == pytest.approx(-0.37, abs=0.02)


def test_binary_log_score_clamps_to_0_001_0_999():
    # FAQ notable-values tables: "Metaculus restricts binary predictions to
    # be between 0.1% and 99.9%" — degenerate inputs must stay finite.
    assert log_score(_bin(1.0), _binary_q(True)) == pytest.approx(math.log(0.999))
    assert log_score(_bin(0.0), _binary_q(True)) == pytest.approx(math.log(0.001))
    assert log_score(_bin(1.0), _binary_q(False)) == pytest.approx(math.log(0.001))


def test_mc_log_score_die_roll_as_six_options():
    # FAQ "What is a proper scoring rule?": P(six)=17% -> ln(0.17) = -1.77.
    opts = ("one", "two", "three", "four", "five", "six")
    probs = {o: 0.166 for o in opts[:5]}
    probs["six"] = 0.17
    f = Forecast(qtype=QType.MULTIPLE_CHOICE, mc_probs=probs)
    assert log_score(f, _mc_q("six", opts)) == pytest.approx(-1.77, abs=0.005)


def test_mc_log_score_unknown_resolution_raises():
    f = Forecast(qtype=QType.MULTIPLE_CHOICE, mc_probs={"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError):
        log_score(f, _mc_q("c", ("a", "b")))


def test_unresolved_question_raises():
    with pytest.raises(ValueError):
        log_score(_bin(0.5), _binary_q(None))


# ------------------------------------------------------- continuous (numeric)

def test_continuous_log_score_uniform_forecast_is_zero():
    # FAQ "What is the log score for continuous questions?": ln(pdf(outcome));
    # the uniform pdf on closed bounds is 1.0 -> ln(1) = 0.
    f = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    assert log_score(f, _numeric_q(50.0)) == pytest.approx(0.0, abs=1e-9)


def test_continuous_log_score_known_pdf_at_cap():
    # mass 0.175 in one of 200 bins -> pdf = 0.175 * 200 = 35.0 (= the cap).
    # FAQ: "Metaculus restricts ... continuous pdfs to be between 0.01 and ~35".
    f = Forecast(qtype=QType.NUMERIC,
                 cdf=build_cdf(below=0.05, above=0.05,
                               spike_bucket=100, spike_mass=0.175))
    q = _numeric_q(49.75, open_lower=True, open_upper=True)
    assert log_score(f, q) == pytest.approx(math.log(35.0))


def test_continuous_log_score_pdf_cap_applies():
    # A bin holding mass 0.999 has raw pdf 199.8; it must be capped to 35.
    f = Forecast(qtype=QType.NUMERIC,
                 cdf=build_cdf(spike_bucket=50, spike_mass=0.999))
    assert log_score(f, _numeric_q(24.75)) == pytest.approx(math.log(35.0))


def test_continuous_log_score_pdf_floor_applies():
    # Same spike forecast scored far from the spike: raw pdf ~0.001,
    # floored to 0.01 (FAQ: 0.01 uniform floor prevents extreme log scores).
    f = Forecast(qtype=QType.NUMERIC,
                 cdf=build_cdf(spike_bucket=50, spike_mass=0.999))
    assert log_score(f, _numeric_q(74.75)) == pytest.approx(math.log(0.01))


def test_out_of_bounds_resolution_scored_as_binary_on_boundary_mass():
    # FAQ "What is the log score for continuous questions?": out-of-bounds
    # resolutions are scored "as in the question 'Will the value be below the
    # lower bound?'" — i.e. ln(boundary mass).
    f = Forecast(qtype=QType.NUMERIC, cdf=build_cdf(below=0.05, above=0.05))
    q_lo = _numeric_q(-3.0, open_lower=True, open_upper=True)
    q_hi = _numeric_q(123.0, open_lower=True, open_upper=True)
    assert log_score(f, q_lo) == pytest.approx(math.log(0.05))
    assert log_score(f, q_hi) == pytest.approx(math.log(0.05))


def test_log_scaled_question_zero_point_bucket_mapping():
    # Mirrors Metaculus utils/the_math/formulas.py log scaling: range 1..1000
    # with zero_point 0 puts sqrt(1000) = 31.62 at the standardized midpoint,
    # i.e. in bucket 100 of 200.
    f = Forecast(qtype=QType.NUMERIC,
                 cdf=build_cdf(spike_bucket=100, spike_mass=0.175))
    q = _numeric_q(31.6227766, range_min=1.0, range_max=1000.0, zero_point=0.0)
    assert log_score(f, q) == pytest.approx(math.log(35.0))


def test_boundary_value_bucket_edge_rules():
    # Platform rule: exactly range_min lands in bucket 1 (not "below");
    # exactly range_max lands in bucket n_bins (not "above").
    f = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    assert log_score(f, _numeric_q(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert log_score(f, _numeric_q(100.0)) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------- discrete

def test_discrete_log_score_uses_pmf_with_floor():
    # FAQ: "For discrete continuous questions, the pmf is used in place of the
    # pdf, and the minimum value assigned to any outcome cannot go below
    # 0.01 / number of possible inbound outcomes."
    uniform = Forecast(qtype=QType.DISCRETE, cdf=build_cdf(n_points=11))
    q = _numeric_q(55.0, qtype=QType.DISCRETE, cdf_size=11)
    assert log_score(uniform, q) == pytest.approx(math.log(0.1))
    spiky = Forecast(
        qtype=QType.DISCRETE,
        cdf=build_cdf(n_points=11, spike_bucket=1, spike_mass=0.9919))
    # non-spike bins hold 0.0009 < floor 0.01/10 = 0.001
    assert log_score(spiky, q) == pytest.approx(math.log(0.001))


# ---------------------------------------------------------------------- brier

def test_brier_binary():
    assert brier(_bin(0.8), _binary_q(True)) == pytest.approx(0.04)
    assert brier(_bin(0.8), _binary_q(False)) == pytest.approx(0.64)


def test_brier_mc_standard_multiclass():
    f = Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 0.7, "b": 0.2, "c": 0.1})
    q = _mc_q("a", ("a", "b", "c"))
    # (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.14
    assert brier(f, q) == pytest.approx(0.14)


def test_brier_numeric_not_implemented():
    f = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    with pytest.raises(NotImplementedError):
        brier(f, _numeric_q(50.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_log_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.scoring'` (collection error)

- [ ] **Step 3: Write minimal implementation**

Create `backtest/scoring/__init__.py`:

```python
"""Scoring layer: Metaculus score family, opponents, diagnostics, statistics."""
```

Create `backtest/scoring/metaculus_scores.py`:

```python
"""Metaculus score family.

Formulas verified against the Metaculus Scores FAQ
(https://www.metaculus.com/help/scores-faq/) and the platform's open-source
implementation (github.com/Metaculus/metaculus: scoring/score_math.py and
utils/the_math/formulas.py). Golden tests in tests/backtest/ encode the FAQ's
worked examples; the FAQ section is cited next to each golden number.

Conventions:
- Binary/MC probabilities are clamped to [0.001, 0.999] before any log
  (FAQ: "Metaculus restricts binary predictions to be between 0.1% and 99.9%").
- Numeric ("continuous") questions are scored from the 201-point CDF:
  pmf = diff(cdf) over n_bins = len(cdf) - 1 standardized-domain bins,
  pdf = pmf * n_bins, clamped to [0.01, 35.0]
  (FAQ: "all pdfs have a uniform distribution of height 0.01 added to them";
   "continuous pdfs to be between 0.01 and ~35").
- Discrete questions score the pmf directly with floor 0.01 / n_bins
  (FAQ: "the pmf is used in place of the pdf").
- Out-of-bounds resolutions are scored as binary on the boundary mass
  (mass below = cdf[0], mass above = 1 - cdf[-1]). For the baseline score the
  chance value of that implied binary is 0.05 — the uniform baseline's own
  out-of-bound mass — per the platform implementation
  (scoring/score_math.py: boundary buckets use baseline = 0.05).
- Continuous/discrete peer and head-to-head scores are divided by 2
  (FAQ "Peer score math": factor chosen empirically).
"""
from __future__ import annotations

import math

from backtest.core import BacktestQuestion, Forecast, QType

PROB_FLOOR = 0.001
PROB_CEIL = 0.999
PDF_FLOOR = 0.01
PDF_CAP = 35.0


def _clamp_prob(p: float) -> float:
    return min(max(float(p), PROB_FLOOR), PROB_CEIL)


def _require_resolution(q: BacktestQuestion) -> None:
    if q.resolution is None:
        raise ValueError(f"{q.qid}: question is unresolved; cannot score")


def _outcome_prob(forecast: Forecast, q: BacktestQuestion) -> float:
    """Clamped probability assigned to the realized outcome (binary/MC)."""
    if forecast.qtype is QType.BINARY:
        if forecast.binary_prob is None:
            raise ValueError("binary forecast missing binary_prob")
        p = forecast.binary_prob if q.resolution else 1.0 - forecast.binary_prob
        return _clamp_prob(p)
    if forecast.qtype is QType.MULTIPLE_CHOICE:
        if forecast.mc_probs is None:
            raise ValueError("multiple_choice forecast missing mc_probs")
        if q.resolution not in forecast.mc_probs:
            raise ValueError(
                f"resolution {q.resolution!r} not among forecast options")
        return _clamp_prob(forecast.mc_probs[q.resolution])
    raise ValueError(f"not a categorical question type: {forecast.qtype}")


def _scaling(q: BacktestQuestion) -> dict:
    if not q.scaling:
        raise ValueError(f"{q.qid}: numeric/discrete question missing scaling")
    return q.scaling


def _unscaled_location(x: float, q: BacktestQuestion) -> float:
    """Actual value -> standardized [0, 1] domain. Mirror of Metaculus
    utils/the_math/formulas.py:scaled_location_to_unscaled_location
    (log scaling when zero_point is set, else linear)."""
    s = _scaling(q)
    rmin, rmax = float(s["range_min"]), float(s["range_max"])
    zero_point = s.get("zero_point")
    if zero_point is not None:
        deriv_ratio = (rmax - zero_point) / (rmin - zero_point)
        return (
            math.log((x - rmin) * (deriv_ratio - 1.0) + (rmax - rmin))
            - math.log(rmax - rmin)
        ) / math.log(deriv_ratio)
    return (x - rmin) / (rmax - rmin)


def _resolution_bucket(x: float, q: BacktestQuestion, n_bins: int) -> int:
    """0 = below lower bound, 1..n_bins = inbound bins, n_bins + 1 = above
    upper. Mirror of Metaculus
    utils/the_math/formulas.py:unscaled_location_to_bucket_index."""
    s = _scaling(q)
    rmin, rmax = float(s["range_min"]), float(s["range_max"])
    if x < rmin:
        return 0
    if x > rmax:
        return n_bins + 1
    u = _unscaled_location(x, q)
    if u >= 1.0:
        return n_bins
    return max(int(u * n_bins + 1 - 1e-10), 1)


def _pdf_value(forecast: Forecast, q: BacktestQuestion) -> tuple[str, float]:
    """Score input for a numeric/discrete forecast at q.resolution.

    Returns ("binary", clamped boundary mass) when the resolution is out of
    bounds, ("pmf", floored pmf) for in-bounds discrete, and
    ("pdf", clamped pdf) for in-bounds numeric.
    """
    if forecast.cdf is None:
        raise ValueError("numeric/discrete forecast missing cdf")
    cdf = forecast.cdf
    n_bins = len(cdf) - 1
    if n_bins < 1:
        raise ValueError("cdf must have at least 2 points")
    bucket = _resolution_bucket(float(q.resolution), q, n_bins)
    if bucket == 0:
        return "binary", _clamp_prob(cdf[0])
    if bucket == n_bins + 1:
        return "binary", _clamp_prob(1.0 - cdf[-1])
    pmf = float(cdf[bucket] - cdf[bucket - 1])
    if forecast.qtype is QType.DISCRETE:
        return "pmf", max(pmf, PDF_FLOOR / n_bins)
    return "pdf", min(max(pmf * n_bins, PDF_FLOOR), PDF_CAP)


def log_score(forecast: Forecast, q: BacktestQuestion) -> float:
    """FAQ "What is the log score?": ln P(outcome) for binary/MC.
    FAQ "What is the log score for continuous questions?": ln pdf(outcome)
    for numeric, ln pmf(outcome) for discrete; out-of-bounds resolutions are
    scored as binary on the boundary mass."""
    _require_resolution(q)
    if forecast.qtype in (QType.BINARY, QType.MULTIPLE_CHOICE):
        return math.log(_outcome_prob(forecast, q))
    _, value = _pdf_value(forecast, q)
    return math.log(value)


def brier(forecast: Forecast, q: BacktestQuestion) -> float:
    """Brier score (not part of the Metaculus score family; reported for
    cross-literature comparability). Binary: (p_yes - outcome)^2, the Halawi/
    ForecastBench convention. MC: standard multiclass Brier,
    sum_k (p_k - 1[k == outcome])^2. Numeric/discrete: not defined here."""
    _require_resolution(q)
    if forecast.qtype is QType.BINARY:
        if forecast.binary_prob is None:
            raise ValueError("binary forecast missing binary_prob")
        outcome = 1.0 if q.resolution else 0.0
        return (float(forecast.binary_prob) - outcome) ** 2
    if forecast.qtype is QType.MULTIPLE_CHOICE:
        if forecast.mc_probs is None:
            raise ValueError("multiple_choice forecast missing mc_probs")
        return sum(
            (float(forecast.mc_probs.get(opt, 0.0))
             - (1.0 if opt == q.resolution else 0.0)) ** 2
            for opt in q.options
        )
    raise NotImplementedError(
        "Brier score is implemented for binary and multiple_choice only")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_log_score.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add backtest/scoring/__init__.py backtest/scoring/metaculus_scores.py tests/backtest/test_log_score.py && git commit -m "feat(backtest): Metaculus log score (binary/MC/numeric/discrete) + Brier, FAQ golden-tested" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Baseline score (all question types)

**Files:**
- Modify: `backtest/scoring/metaculus_scores.py` (append `baseline_score`)
- Test: `tests/backtest/test_baseline_score.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_baseline_score.py`:

```python
"""Golden tests for the Metaculus Baseline score, encoding the worked
examples and notable values from the Metaculus Scores FAQ
(https://www.metaculus.com/help/scores-faq/). FAQ table values are rounded
integers, hence the abs=0.6 tolerance on those assertions; exact-formula
assertions accompany them."""
import math

import pytest

from backtest.core import BacktestQuestion, Forecast, QType
from backtest.scoring.metaculus_scores import baseline_score


def _binary_q(resolution):
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="Will it happen?", resolution=resolution)


def _mc_q(resolution, options):
    return BacktestQuestion(
        qid="metaculus:2", platform="metaculus", native_id="2",
        qtype=QType.MULTIPLE_CHOICE, title="Which one?",
        resolution=resolution, options=options)


def _numeric_q(resolution, *, qtype=QType.NUMERIC, range_min=0.0,
               range_max=100.0, zero_point=None,
               open_lower=False, open_upper=False, cdf_size=201):
    return BacktestQuestion(
        qid="metaculus:3", platform="metaculus", native_id="3", qtype=qtype,
        title="How many?", resolution=resolution,
        scaling={"range_min": range_min, "range_max": range_max,
                 "zero_point": zero_point, "open_lower_bound": open_lower,
                 "open_upper_bound": open_upper, "cdf_size": cdf_size,
                 "unit": "units"})


def build_cdf(*, n_points=201, below=0.0, above=0.0,
              spike_bucket=None, spike_mass=0.0):
    """CDF on the standardized domain (see tests/backtest/test_log_score.py
    for the bucket convention)."""
    n_bins = n_points - 1
    inbound = 1.0 - below - above
    if spike_bucket is None:
        masses = [inbound / n_bins] * n_bins
    else:
        base = (inbound - spike_mass) / (n_bins - 1)
        masses = [spike_mass if b == spike_bucket else base
                  for b in range(1, n_bins + 1)]
    cdf = [below]
    for m in masses:
        cdf.append(cdf[-1] + m)
    cdf[-1] = 1.0 - above
    return tuple(cdf)


def _bin(p):
    return Forecast(qtype=QType.BINARY, binary_prob=p)


def test_binary_baseline_faq_predicting_extremes_table():
    # FAQ "Can I get better scores by predicting extreme values?":
    # prediction -> (score if Yes, score if No):
    # 70% -> (+48, -74); 80% -> (+68, -132); 90% -> (+85, -232); 99% -> (+99, -564)
    table = [(0.7, 48, -74), (0.8, 68, -132), (0.9, 85, -232), (0.99, 99, -564)]
    for p, yes, no in table:
        assert baseline_score(_bin(p), _binary_q(True)) == pytest.approx(yes, abs=0.6)
        assert baseline_score(_bin(p), _binary_q(False)) == pytest.approx(no, abs=0.6)


def test_binary_baseline_exact_formula():
    # FAQ "Baseline score math": 100 * (ln P(outcome) - ln 50%) / ln 2,
    # equivalently 100 * (log2(P) + 1).
    assert baseline_score(_bin(0.8), _binary_q(True)) == pytest.approx(
        100.0 * (math.log(0.8) - math.log(0.5)) / math.log(2.0))
    assert baseline_score(_bin(0.8), _binary_q(True)) == pytest.approx(
        100.0 * (math.log2(0.8) + 1.0))


def test_binary_baseline_chance_zero_and_notable_values():
    # FAQ "What is the Baseline score?": chance prediction scores 0; notable
    # values (binary): best +99.9 at 99.9%, worst -897 at 0.1%.
    assert baseline_score(_bin(0.5), _binary_q(True)) == pytest.approx(0.0)
    assert baseline_score(_bin(0.999), _binary_q(True)) == pytest.approx(99.9, abs=0.6)
    # clamp: a 0.0 prediction scores as 0.001
    assert baseline_score(_bin(0.0), _binary_q(True)) == pytest.approx(-897, abs=0.6)


def test_mc_baseline_exact_formula_and_uniform_zero():
    # FAQ "Baseline score math": 100 * (ln P(outcome) - ln(1/N)) / ln N.
    opts = ("a", "b", "c", "d")
    uniform = Forecast(qtype=QType.MULTIPLE_CHOICE,
                       mc_probs={o: 0.25 for o in opts})
    assert baseline_score(uniform, _mc_q("b", opts)) == pytest.approx(0.0)
    f = Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1})
    assert baseline_score(f, _mc_q("a", opts)) == pytest.approx(
        100.0 * (math.log(0.4) - math.log(0.25)) / math.log(4.0))


def test_mc_baseline_faq_notable_values_8_options():
    # FAQ "What is the Baseline score?" notable values (MC, 8 options):
    # best +99.9, worst -232.
    opts = tuple("abcdefgh")
    best = Forecast(
        qtype=QType.MULTIPLE_CHOICE,
        mc_probs={o: (0.999 if o == "a" else 0.001 / 7) for o in opts})
    worst = Forecast(
        qtype=QType.MULTIPLE_CHOICE,
        mc_probs={o: (0.001 if o == "a" else 0.999 / 7) for o in opts})
    assert baseline_score(best, _mc_q("a", opts)) == pytest.approx(99.9, abs=0.6)
    assert baseline_score(worst, _mc_q("a", opts)) == pytest.approx(-232, abs=0.6)


def test_continuous_baseline_uniform_is_zero_for_each_bound_config():
    # FAQ "Baseline score math": continuous baseline pdf is 1 / 0.95 / 0.9
    # for closed / one-open / two-open bounds — so the matching uniform
    # forecast scores exactly 0 in every configuration.
    cases = [
        (build_cdf(), dict()),
        (build_cdf(above=0.05), dict(open_upper=True)),
        (build_cdf(below=0.05, above=0.05),
         dict(open_lower=True, open_upper=True)),
    ]
    for cdf, bounds in cases:
        f = Forecast(qtype=QType.NUMERIC, cdf=cdf)
        assert baseline_score(f, _numeric_q(50.0, **bounds)) == pytest.approx(
            0.0, abs=1e-9)


def test_continuous_baseline_faq_best_and_worst_notable_values():
    # FAQ "What is the Baseline score?" notable values (continuous):
    # best +183 = 100*(ln 35 - ln 0.9)/2 (pdf cap, two open bounds);
    # worst -230 = 100*(ln 0.01 - ln 1)/2 (pdf floor, closed bounds).
    best = Forecast(qtype=QType.NUMERIC,
                    cdf=build_cdf(below=0.05, above=0.05,
                                  spike_bucket=100, spike_mass=0.175))
    q2 = _numeric_q(49.75, open_lower=True, open_upper=True)
    assert baseline_score(best, q2) == pytest.approx(183, abs=0.6)
    assert baseline_score(best, q2) == pytest.approx(
        100.0 * (math.log(35.0) - math.log(0.9)) / 2.0)
    worst = Forecast(qtype=QType.NUMERIC,
                     cdf=build_cdf(spike_bucket=50, spike_mass=0.999))
    assert baseline_score(worst, _numeric_q(74.75)) == pytest.approx(-230, abs=0.6)
    assert baseline_score(worst, _numeric_q(74.75)) == pytest.approx(
        100.0 * math.log(0.01) / 2.0)


def test_continuous_baseline_out_of_bounds_uses_boundary_mass_vs_0_05():
    # Platform implementation (Metaculus repo scoring/score_math.py,
    # evaluate_forecasts_baseline_*): out-of-bounds buckets score the
    # boundary mass against the uniform baseline's own out-of-bound mass
    # 0.05 — so the chance forecast scores 0 even out of bounds.
    q = _numeric_q(-3.0, open_lower=True, open_upper=True)
    chance = Forecast(qtype=QType.NUMERIC,
                      cdf=build_cdf(below=0.05, above=0.05))
    assert baseline_score(chance, q) == pytest.approx(0.0, abs=1e-9)
    heavier = Forecast(qtype=QType.NUMERIC,
                       cdf=build_cdf(below=0.10, above=0.05))
    assert baseline_score(heavier, q) == pytest.approx(
        100.0 * math.log(0.10 / 0.05) / 2.0)


def test_discrete_baseline_uniform_is_zero_and_floor_matches_continuous():
    # Server formula 100 * ln(pmf / ((1 - 0.05*open) / n_bins)) / 2 applies to
    # discrete questions unchanged; the FAQ floor 0.01/n_bins maps to the same
    # -230.26 worst case as continuous.
    q = _numeric_q(55.0, qtype=QType.DISCRETE, cdf_size=11)
    uniform = Forecast(qtype=QType.DISCRETE, cdf=build_cdf(n_points=11))
    assert baseline_score(uniform, q) == pytest.approx(0.0, abs=1e-9)
    spiky = Forecast(
        qtype=QType.DISCRETE,
        cdf=build_cdf(n_points=11, spike_bucket=1, spike_mass=0.9919))
    assert baseline_score(spiky, q) == pytest.approx(
        100.0 * (math.log(0.001) - math.log(0.1)) / 2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_baseline_score.py -v`
Expected: FAIL with `ImportError: cannot import name 'baseline_score' from 'backtest.scoring.metaculus_scores'` (collection error)

- [ ] **Step 3: Write minimal implementation**

Add the following function to the end of `backtest/scoring/metaculus_scores.py`:

```python
def baseline_score(forecast: Forecast, q: BacktestQuestion) -> float:
    """FAQ "Baseline score math":
    binary:     100 * (ln P(outcome) - ln 0.5) / ln 2
    MC:         100 * (ln P(outcome) - ln(1/N)) / ln N
    continuous: 100 * (ln pdf(outcome) - ln baseline) / 2, with baseline
                1 / 0.95 / 0.9 for closed / one-open / two-open bounds;
                discrete uses the pmf against the uniform pmf (identical
                formula after the pmf -> pdf change of units).
    Out-of-bounds resolutions score the boundary mass against the uniform
    baseline's own out-of-bound mass 0.05 (platform source:
    scoring/score_math.py, evaluate_forecasts_baseline_*)."""
    _require_resolution(q)
    if forecast.qtype is QType.BINARY:
        p = _outcome_prob(forecast, q)
        return 100.0 * (math.log(p) - math.log(0.5)) / math.log(2.0)
    if forecast.qtype is QType.MULTIPLE_CHOICE:
        n = len(q.options)
        if n < 2:
            raise ValueError("multiple_choice question needs >= 2 options")
        p = _outcome_prob(forecast, q)
        return 100.0 * (math.log(p) - math.log(1.0 / n)) / math.log(n)
    s = _scaling(q)
    open_count = (int(bool(s.get("open_lower_bound")))
                  + int(bool(s.get("open_upper_bound"))))
    kind, value = _pdf_value(forecast, q)
    if kind == "binary":
        return 100.0 * (math.log(value) - math.log(0.05)) / 2.0
    if kind == "pmf":  # discrete: pmf vs uniform pmf
        n_bins = len(forecast.cdf) - 1
        chance = (1.0 - 0.05 * open_count) / n_bins
        return 100.0 * (math.log(value) - math.log(chance)) / 2.0
    chance = 1.0 - 0.05 * open_count
    return 100.0 * (math.log(value) - math.log(chance)) / 2.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_baseline_score.py -v`
Expected: PASS (9 tests). Also run `pytest tests/backtest -q` to confirm the existing tests in `tests/backtest/test_log_score.py` still pass.

- [ ] **Step 5: Commit**

```bash
git add backtest/scoring/metaculus_scores.py tests/backtest/test_baseline_score.py && git commit -m "feat(backtest): Metaculus baseline score for all question types, FAQ golden-tested" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Peer score, head-to-head, tournament take + property tests

**Files:**
- Modify: `backtest/scoring/metaculus_scores.py` (append `peer_score`, `head_to_head`, `tournament_take`)
- Test: `tests/backtest/test_peer_and_tournament.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_peer_and_tournament.py`:

```python
"""Peer score, head-to-head, tournament take — plus the scoring-layer
property tests required by the spec (§8): peer zero-sum, proper-scoring
sanity via grid search, clamps applied. FAQ sections cited inline."""
import math

import pytest

from backtest.core import BacktestQuestion, Forecast, QType
from backtest.scoring.metaculus_scores import (
    baseline_score,
    head_to_head,
    log_score,
    peer_score,
    tournament_take,
)


def _binary_q(resolution):
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="Will it happen?", resolution=resolution)


def _mc_q(resolution, options):
    return BacktestQuestion(
        qid="metaculus:2", platform="metaculus", native_id="2",
        qtype=QType.MULTIPLE_CHOICE, title="Which one?",
        resolution=resolution, options=options)


def _numeric_q(resolution, *, range_min=0.0, range_max=100.0,
               open_lower=False, open_upper=False, cdf_size=201):
    return BacktestQuestion(
        qid="metaculus:3", platform="metaculus", native_id="3",
        qtype=QType.NUMERIC, title="How many?", resolution=resolution,
        scaling={"range_min": range_min, "range_max": range_max,
                 "zero_point": None, "open_lower_bound": open_lower,
                 "open_upper_bound": open_upper, "cdf_size": cdf_size,
                 "unit": "units"})


def build_cdf(*, n_points=201, below=0.0, above=0.0,
              spike_bucket=None, spike_mass=0.0):
    """CDF on the standardized domain (see tests/backtest/test_log_score.py
    for the bucket convention)."""
    n_bins = n_points - 1
    inbound = 1.0 - below - above
    if spike_bucket is None:
        masses = [inbound / n_bins] * n_bins
    else:
        base = (inbound - spike_mass) / (n_bins - 1)
        masses = [spike_mass if b == spike_bucket else base
                  for b in range(1, n_bins + 1)]
    cdf = [below]
    for m in masses:
        cdf.append(cdf[-1] + m)
    cdf[-1] = 1.0 - above
    return tuple(cdf)


def _bin(p):
    return Forecast(qtype=QType.BINARY, binary_prob=p)


# ----------------------------------------------------------------- peer score

def test_peer_binary_matches_geometric_mean_formula():
    # FAQ "Peer score math": peer = 100 * (ln p - ln GM(other predictions)).
    f = _bin(0.8)
    others = [_bin(0.5), _bin(0.6)]
    expected = 100.0 * (math.log(0.8) - (math.log(0.5) + math.log(0.6)) / 2.0)
    got = peer_score(f, others, _binary_q(True))
    assert got == pytest.approx(expected)
    assert got == pytest.approx(37.8843, abs=1e-3)


def test_peer_zero_sum_binary():
    # FAQ "What is the Peer score?": "on any given question, the sum of all
    # participants' Peer scores is always 0" (worked example: log scores
    # -1, 1, 2 give peer values -2.5, +0.5, +2, summing to 0).
    fs = [_bin(p) for p in (0.9, 0.6, 0.3, 0.45)]
    q = _binary_q(True)
    total = sum(peer_score(f, [o for o in fs if o is not f], q) for f in fs)
    assert total == pytest.approx(0.0, abs=1e-9)


def test_peer_zero_sum_mc():
    opts = ("a", "b", "c")
    fs = [
        Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 0.6, "b": 0.3, "c": 0.1}),
        Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 0.2, "b": 0.5, "c": 0.3}),
        Forecast(qtype=QType.MULTIPLE_CHOICE,
                 mc_probs={"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}),
    ]
    q = _mc_q("b", opts)
    total = sum(peer_score(f, [o for o in fs if o is not f], q) for f in fs)
    assert total == pytest.approx(0.0, abs=1e-9)


def test_peer_zero_sum_continuous():
    cdfs = [build_cdf(),
            build_cdf(spike_bucket=100, spike_mass=0.175),
            build_cdf(spike_bucket=120, spike_mass=0.10)]
    fs = [Forecast(qtype=QType.NUMERIC, cdf=c) for c in cdfs]
    q = _numeric_q(50.0)
    total = sum(peer_score(f, [o for o in fs if o is not f], q) for f in fs)
    assert total == pytest.approx(0.0, abs=1e-9)


def test_peer_continuous_is_halved():
    # FAQ "Peer score math": "Peer scores of continuous and discrete
    # questions are further divided by 2."
    sharp = Forecast(qtype=QType.NUMERIC,
                     cdf=build_cdf(spike_bucket=100, spike_mass=0.175))
    flat = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    q = _numeric_q(49.75)
    raw_gap = log_score(sharp, q) - log_score(flat, q)
    assert peer_score(sharp, [flat], q) == pytest.approx(100.0 * raw_gap / 2.0)
    # pdf 35 vs pdf 1: 100 * ln(35) / 2
    assert peer_score(sharp, [flat], q) == pytest.approx(
        100.0 * math.log(35.0) / 2.0)


def test_peer_with_no_others_is_zero():
    assert peer_score(_bin(0.8), [], _binary_q(True)) == 0.0


# --------------------------------------------------------------- head-to-head

def test_head_to_head_binary_log_ratio():
    # spec §6.2: head-to-head spot peer vs CP-at-T = 100 * ln(p_bot / p_CP)
    # on the realized side (the statistic in Metaculus Pros-vs-Bots analyses).
    q = _binary_q(True)
    assert head_to_head(_bin(0.7), _bin(0.5), q) == pytest.approx(
        100.0 * math.log(0.7 / 0.5))
    assert head_to_head(_bin(0.7), _bin(0.5), q) == pytest.approx(33.647, abs=1e-3)


def test_head_to_head_antisymmetric_and_zero_on_tie():
    q = _binary_q(False)
    a, b = _bin(0.7), _bin(0.4)
    assert head_to_head(a, b, q) == pytest.approx(-head_to_head(b, a, q))
    assert head_to_head(a, a, q) == pytest.approx(0.0)


def test_head_to_head_continuous_halved():
    sharp = Forecast(qtype=QType.NUMERIC,
                     cdf=build_cdf(spike_bucket=100, spike_mass=0.175))
    flat = Forecast(qtype=QType.NUMERIC, cdf=build_cdf())
    q = _numeric_q(49.75)
    assert head_to_head(sharp, flat, q) == pytest.approx(
        100.0 * math.log(35.0) / 2.0)


# ------------------------------------------------------------ tournament take

def test_tournament_take_formula():
    # FAQ "How are my tournament Score, Take, Prize, and Rank calculated?":
    # total = sum(peer score * question weight); take = max(total, 0)^2.
    assert tournament_take([20.0, -10.0, 5.0], [1.0, 1.0, 0.5]) == pytest.approx(156.25)
    assert tournament_take([-5.0, -1.0], [1.0, 1.0]) == 0.0
    assert tournament_take([], []) == 0.0


def test_tournament_take_length_mismatch_raises():
    with pytest.raises(ValueError):
        tournament_take([1.0], [1.0, 2.0])


# ------------------------------------------------------------ property tests

def test_truth_telling_maximizes_expected_log_score():
    # Proper-scoring sanity (spec §8): over a 1%..99% grid, expected log
    # score under true belief 0.7 is maximized exactly at 0.7.
    q_yes, q_no = _binary_q(True), _binary_q(False)
    belief = 0.7

    def expected_score(pred):
        f = _bin(pred)
        return belief * log_score(f, q_yes) + (1 - belief) * log_score(f, q_no)

    grid = [i / 100 for i in range(1, 100)]
    best = max(grid, key=expected_score)
    assert best == pytest.approx(belief)


def test_baseline_expected_value_matches_faq_extremes_table():
    # FAQ "Can I get better scores by predicting extreme values?" — belief
    # 80%: expected baseline scores +24 / +28 / +21 / -34 for predictions
    # 70% / 80% / 90% / 99% (the proper-score maximum sits at the belief).
    q_yes, q_no = _binary_q(True), _binary_q(False)
    for pred, expected in [(0.7, 24.0), (0.8, 28.0), (0.9, 21.0), (0.99, -34.0)]:
        f = _bin(pred)
        ev = 0.8 * baseline_score(f, q_yes) + 0.2 * baseline_score(f, q_no)
        assert ev == pytest.approx(expected, abs=0.6)


def test_clamps_keep_all_scores_finite():
    # Degenerate probabilities never produce infinities (clamp [0.001, 0.999]).
    q_yes = _binary_q(True)
    for p in (0.0, 1.0):
        assert math.isfinite(log_score(_bin(p), q_yes))
        assert math.isfinite(baseline_score(_bin(p), q_yes))
        assert math.isfinite(peer_score(_bin(p), [_bin(0.5)], q_yes))
        assert math.isfinite(head_to_head(_bin(p), _bin(0.5), q_yes))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_peer_and_tournament.py -v`
Expected: FAIL with `ImportError: cannot import name 'peer_score' from 'backtest.scoring.metaculus_scores'` (collection error)

- [ ] **Step 3: Write minimal implementation**

Add the following functions to the end of `backtest/scoring/metaculus_scores.py`:

```python
def peer_score(forecast: Forecast, others: list[Forecast],
               q: BacktestQuestion) -> float:
    """FAQ "Peer score math": 100 * (ln p - ln GM(others)) — equivalently the
    average log-score difference against every other forecast (the FAQ's
    worked zero-sum example: log scores -1, 1, 2 give peer values -2.5,
    +0.5, +2, summing to 0). Continuous/discrete scores are divided by 2.
    `others` excludes the scored forecast. Empty `others` scores 0."""
    if not others:
        return 0.0
    mine = log_score(forecast, q)
    mean_others = sum(log_score(o, q) for o in others) / len(others)
    score = 100.0 * (mine - mean_others)
    if forecast.qtype in (QType.NUMERIC, QType.DISCRETE):
        score /= 2.0
    return score


def head_to_head(forecast: Forecast, opponent: Forecast,
                 q: BacktestQuestion) -> float:
    """Spot head-to-head vs a single synthetic opponent:
    100 * ln(p / p_opponent) on the realized outcome (the statistic used in
    Metaculus's Pros-vs-Bots analyses); continuous/discrete divided by 2.
    Identical to a peer score with a single other participant."""
    return peer_score(forecast, [opponent], q)


def tournament_take(spot_peer_scores: list[float],
                    weights: list[float]) -> float:
    """FAQ "How are my tournament Score, Take, Prize, and Rank calculated?":
    total = sum(spot peer score * question weight); take = max(total, 0)^2."""
    if len(spot_peer_scores) != len(weights):
        raise ValueError("scores and weights must have equal length")
    total = sum(s * w for s, w in zip(spot_peer_scores, weights))
    return max(total, 0.0) ** 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_peer_and_tournament.py -v`
Expected: PASS (14 tests). Also run `pytest tests/backtest -q` to confirm the earlier scoring tests still pass.

- [ ] **Step 5: Commit**

```bash
git add backtest/scoring/metaculus_scores.py tests/backtest/test_peer_and_tournament.py && git commit -m "feat(backtest): peer score, head-to-head, tournament take + zero-sum and propriety property tests" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Synthetic opponent constructors

**Files:**
- Create: `backtest/scoring/opponents.py`
- Test: `tests/backtest/test_opponents.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_opponents.py`:

```python
"""cp_opponent / market_opponent constructors (spec §6.2). Pure constructors:
data lookup (CP history, market price at T) lives in the corpus layer."""
import math

import pytest

from backtest.core import BacktestQuestion, QType
from backtest.scoring.metaculus_scores import log_score
from backtest.scoring.opponents import cp_opponent, market_opponent


def test_cp_opponent_binary_scalar():
    f = cp_opponent(0.62)
    assert f.qtype is QType.BINARY
    assert f.binary_prob == pytest.approx(0.62)
    assert f.mc_probs is None and f.cdf is None


def test_cp_opponent_clamps_degenerate_probabilities():
    # constructors maintain the Forecast invariant binary_prob in (0, 1)
    assert cp_opponent(0.0).binary_prob == pytest.approx(0.001)
    assert cp_opponent(1.0).binary_prob == pytest.approx(0.999)


def test_cp_opponent_mc_dict_vector():
    f = cp_opponent({"a": 0.5, "b": 0.3, "c": 0.2})
    assert f.qtype is QType.MULTIPLE_CHOICE
    assert f.mc_probs == {"a": 0.5, "b": 0.3, "c": 0.2}


def test_cp_opponent_mc_bad_sum_raises():
    with pytest.raises(ValueError):
        cp_opponent({"a": 0.5, "b": 0.2})


def test_cp_opponent_numeric_cdf_vector():
    cdf = [i / 200 for i in range(201)]
    f = cp_opponent(cdf)
    assert f.qtype is QType.NUMERIC
    assert f.cdf == tuple(cdf)


def test_cp_opponent_rejects_invalid_values():
    with pytest.raises(TypeError):
        cp_opponent("0.6")
    with pytest.raises(TypeError):
        cp_opponent(True)          # bool is not a probability
    with pytest.raises(TypeError):
        cp_opponent(None)
    with pytest.raises(ValueError):
        cp_opponent(1.7)           # probability out of [0, 1]
    with pytest.raises(ValueError):
        cp_opponent([0.5, 0.2, 0.9])   # cdf not ascending
    with pytest.raises(ValueError):
        cp_opponent([0.5])         # cdf too short


def test_market_opponent_binary_scalar():
    f = market_opponent(0.35)
    assert f.qtype is QType.BINARY
    assert f.binary_prob == pytest.approx(0.35)


def test_market_opponent_raises_on_non_binary_values():
    # contract: market price is binary-only — vectors/dicts must raise
    with pytest.raises(TypeError):
        market_opponent({"a": 0.5, "b": 0.5})
    with pytest.raises(TypeError):
        market_opponent([0.1, 0.9])
    with pytest.raises(TypeError):
        market_opponent(True)
    with pytest.raises(ValueError):
        market_opponent(1.5)


def test_market_opponent_clamps_settled_prices():
    assert market_opponent(0.0).binary_prob == pytest.approx(0.001)
    assert market_opponent(1.0).binary_prob == pytest.approx(0.999)


def test_opponents_are_scoreable():
    q = BacktestQuestion(
        qid="polymarket:x", platform="polymarket", native_id="x",
        qtype=QType.BINARY, title="Will it?", resolution=True)
    assert log_score(market_opponent(0.35), q) == pytest.approx(math.log(0.35))
    assert log_score(cp_opponent(0.62), q) == pytest.approx(math.log(0.62))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_opponents.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.scoring.opponents'` (collection error)

- [ ] **Step 3: Write minimal implementation**

Create `backtest/scoring/opponents.py`:

```python
"""Synthetic opponent constructors (spec §6.2): CP-at-T and
market-price-at-T. Pure constructors — looking the values up at time T
(CP history intervals, market candles) is the corpus layer's job.

cp_opponent accepts the three shapes a community prediction takes:
a scalar probability (binary), an option->probability dict (MC), or a CDF
vector (numeric/discrete). market_opponent is binary-only by contract: a
market price is a single Yes-probability, and MC/numeric questions have no
single-price equivalent — anything but a scalar raises.
"""
from __future__ import annotations

from numbers import Real

from backtest.core import Forecast, QType

_PROB_FLOOR = 0.001
_PROB_CEIL = 0.999


def _clamped_binary(p: float) -> Forecast:
    return Forecast(
        qtype=QType.BINARY,
        binary_prob=min(max(float(p), _PROB_FLOOR), _PROB_CEIL),
    )


def cp_opponent(value) -> Forecast:
    """Community-prediction-at-T opponent.

    - float in [0, 1]            -> binary forecast
    - dict option -> prob (~1)   -> multiple_choice forecast
    - sequence of floats         -> numeric forecast (ascending CDF in [0, 1])
    """
    if isinstance(value, bool):
        raise TypeError("cp value must be a probability, dict, or cdf; got bool")
    if isinstance(value, Real):
        p = float(value)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"cp probability out of [0, 1]: {p}")
        return _clamped_binary(p)
    if isinstance(value, dict):
        probs = {str(k): float(v) for k, v in value.items()}
        total = sum(probs.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"cp mc probabilities sum to {total}, expected ~1")
        return Forecast(qtype=QType.MULTIPLE_CHOICE, mc_probs=probs)
    if isinstance(value, (list, tuple)):
        cdf = tuple(float(v) for v in value)
        if len(cdf) < 2:
            raise ValueError("cp cdf needs at least 2 points")
        if cdf[0] < 0.0 or cdf[-1] > 1.0 or any(
            b < a for a, b in zip(cdf, cdf[1:])
        ):
            raise ValueError("cp cdf must be ascending within [0, 1]")
        return Forecast(qtype=QType.NUMERIC, cdf=cdf)
    raise TypeError(f"unsupported cp value type: {type(value).__name__}")


def market_opponent(price: float) -> Forecast:
    """Market-price-at-T opponent. Binary only (contract): raises TypeError
    for vectors/dicts, ValueError for prices outside [0, 1]."""
    if isinstance(price, bool) or not isinstance(price, Real):
        raise TypeError(
            f"market price must be a number, got {type(price).__name__}")
    p = float(price)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"market price out of [0, 1]: {p}")
    return _clamped_binary(p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_opponents.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add backtest/scoring/opponents.py tests/backtest/test_opponents.py && git commit -m "feat(backtest): cp_opponent and market_opponent synthetic-opponent constructors" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Diagnostics — calibration curve, Murphy decomposition, discrimination

**Files:**
- Create: `backtest/scoring/diagnostics.py`
- Test: `tests/backtest/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_diagnostics.py`:

```python
"""Diagnostics (spec §6.3): quintile calibration curves with binomial
consistency bands, Murphy decomposition, discrimination separation.
`pairs` is list[(binary prob, outcome bool)]."""
import math

import pytest

from backtest.scoring.diagnostics import (
    calibration_curve,
    discrimination_separation,
    murphy_decomposition,
)


def test_calibration_curve_quintiles_perfectly_calibrated():
    # Five forecast levels, 20 each, hit rates exactly equal to the level:
    # every bin's frequency must equal its mean forecast and sit inside the
    # binomial consistency band.
    pairs = []
    for p, k in [(0.1, 2), (0.3, 6), (0.5, 10), (0.7, 14), (0.9, 18)]:
        pairs += [(p, True)] * k + [(p, False)] * (20 - k)
    curve = calibration_curve(pairs, n_bins=5)
    assert len(curve) == 5
    assert sum(b["n"] for b in curve) == 100
    for b in curve:
        assert set(b) == {"p_mean", "freq", "n", "lo", "hi"}
        assert b["freq"] == pytest.approx(b["p_mean"])
        assert b["lo"] <= b["freq"] <= b["hi"]
    p_means = [b["p_mean"] for b in curve]
    assert p_means == sorted(p_means)  # quantile bins ascend


def test_calibration_curve_flags_overconfidence():
    # 50 forecasts of 0.9 that all resolve No: every bin's observed frequency
    # (0.0) must fall below the lower consistency band
    # (binom.ppf(0.05, 10, 0.9)/10 = 0.7).
    pairs = [(0.9, False)] * 50
    curve = calibration_curve(pairs, n_bins=5)
    assert all(b["freq"] < b["lo"] for b in curve)


def test_calibration_curve_empty_input():
    assert calibration_curve([]) == []


def test_murphy_decomposition_identity():
    # Murphy (1973): brier = uncertainty - resolution + reliability, exact
    # when grouping by unique forecast value.
    pairs = [(0.2, False), (0.2, True), (0.7, True), (0.7, True),
             (0.7, False), (0.5, True)]
    out = murphy_decomposition(pairs)
    assert set(out) == {"brier", "uncertainty", "resolution", "reliability"}
    assert out["brier"] == pytest.approx(
        out["uncertainty"] - out["resolution"] + out["reliability"], abs=1e-12)


def test_murphy_perfect_and_constant_forecasters():
    perfect = murphy_decomposition([(1.0, True), (0.0, False)])
    assert perfect["brier"] == pytest.approx(0.0)
    assert perfect["reliability"] == pytest.approx(0.0)
    assert perfect["resolution"] == pytest.approx(perfect["uncertainty"])
    constant = murphy_decomposition([(0.5, True), (0.5, False)])
    assert constant["resolution"] == pytest.approx(0.0)
    assert constant["reliability"] == pytest.approx(0.0)
    assert constant["brier"] == pytest.approx(constant["uncertainty"])


def test_murphy_empty_raises():
    with pytest.raises(ValueError):
        murphy_decomposition([])


def test_discrimination_separation_golden():
    # spec §6.3: mean forecast on Yes-resolvers minus mean on No-resolvers
    # (the Pros-vs-bots 44pp-vs-26pp statistic).
    pairs = [(0.8, True), (0.6, True), (0.3, False), (0.1, False)]
    assert discrimination_separation(pairs) == pytest.approx(0.5)


def test_discrimination_separation_degenerate_is_nan():
    assert math.isnan(discrimination_separation([(0.5, True)]))
    assert math.isnan(discrimination_separation([]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.scoring.diagnostics'` (collection error)

- [ ] **Step 3: Write minimal implementation**

Create `backtest/scoring/diagnostics.py`:

```python
"""Forecast diagnostics (spec §6.3). All functions take
pairs = list[(binary prob, outcome bool)]."""
from __future__ import annotations

from scipy.stats import binom

Pair = tuple[float, bool]


def calibration_curve(pairs: list[Pair], n_bins: int = 5) -> list[dict]:
    """Quantile-binned (default quintile) calibration curve with consistency
    bands from binomial quantiles (Broecker & Smith): under perfect
    calibration the bin frequency lies within
    [binom.ppf(0.05, n, p_mean), binom.ppf(0.95, n, p_mean)] / n
    with ~90% probability. Returns one dict per non-empty bin:
    {"p_mean", "freq", "n", "lo", "hi"}."""
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda t: t[0])
    n_total = len(ordered)
    edges = [round(i * n_total / n_bins) for i in range(n_bins + 1)]
    out: list[dict] = []
    for i in range(n_bins):
        chunk = ordered[edges[i]:edges[i + 1]]
        if not chunk:
            continue
        n = len(chunk)
        p_mean = sum(p for p, _ in chunk) / n
        freq = sum(1.0 for _, o in chunk if o) / n
        lo = float(binom.ppf(0.05, n, p_mean)) / n
        hi = float(binom.ppf(0.95, n, p_mean)) / n
        out.append({"p_mean": p_mean, "freq": freq, "n": n, "lo": lo, "hi": hi})
    return out


def murphy_decomposition(pairs: list[Pair]) -> dict:
    """Murphy (1973) decomposition of the Brier score, grouped by unique
    forecast value: brier = uncertainty - resolution + reliability (exact
    identity under this grouping). Reliability ~ recalibration headroom;
    resolution ~ research/decisiveness investment signal (spec §6.3)."""
    if not pairs:
        raise ValueError("murphy_decomposition needs at least one pair")
    n = len(pairs)
    obar = sum(1.0 for _, o in pairs if o) / n
    groups: dict[float, list[bool]] = {}
    for p, o in pairs:
        groups.setdefault(float(p), []).append(o)
    reliability = 0.0
    resolution = 0.0
    for p, outcomes in groups.items():
        nk = len(outcomes)
        ok = sum(1.0 for o in outcomes if o) / nk
        reliability += nk * (p - ok) ** 2
        resolution += nk * (ok - obar) ** 2
    reliability /= n
    resolution /= n
    uncertainty = obar * (1.0 - obar)
    brier_score = sum((p - (1.0 if o else 0.0)) ** 2 for p, o in pairs) / n
    return {
        "brier": brier_score,
        "uncertainty": uncertainty,
        "resolution": resolution,
        "reliability": reliability,
    }


def discrimination_separation(pairs: list[Pair]) -> float:
    """Mean forecast on Yes-resolvers minus mean forecast on No-resolvers.
    NaN when either side is empty."""
    yes = [p for p, o in pairs if o]
    no = [p for p, o in pairs if not o]
    if not yes or not no:
        return float("nan")
    return sum(yes) / len(yes) - sum(no) / len(no)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_diagnostics.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add backtest/scoring/diagnostics.py tests/backtest/test_diagnostics.py && git commit -m "feat(backtest): calibration curve with binomial consistency bands, Murphy decomposition, discrimination separation" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Statistics — paired bootstrap/t, detectable effect, difficulty adjustment

**Files:**
- Create: `backtest/scoring/stats.py`
- Test: `tests/backtest/test_stats.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_stats.py`:

```python
"""Paired statistics and difficulty adjustment (spec §6.4)."""
import pytest
from scipy import stats as scipy_stats

from backtest.scoring.stats import (
    detectable_effect,
    difficulty_adjust,
    paired_bootstrap,
    paired_t,
)


def test_paired_bootstrap_deterministic_and_brackets_mean():
    diffs = [1.0, 1.2, 0.8, 1.1, 0.9, 1.05, 0.95, 1.15, 0.85, 1.0]
    a = paired_bootstrap(diffs, n=2000, seed=0)
    b = paired_bootstrap(diffs, n=2000, seed=0)
    assert a == b                                   # same seed, same result
    assert set(a) == {"mean", "lo", "hi"}
    assert a["mean"] == pytest.approx(sum(diffs) / len(diffs))
    assert a["lo"] <= a["mean"] <= a["hi"]
    assert a["lo"] > 0                              # clearly positive effect


def test_paired_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        paired_bootstrap([])


def test_paired_t_matches_scipy_one_sample():
    diffs = [0.3, -0.1, 0.4, 0.2, 0.0, 0.5, -0.2, 0.1]
    out = paired_t(diffs)
    ref = scipy_stats.ttest_1samp(diffs, 0.0)
    assert set(out) == {"mean", "t", "p"}
    assert out["mean"] == pytest.approx(sum(diffs) / len(diffs))
    assert out["t"] == pytest.approx(float(ref.statistic))
    assert out["p"] == pytest.approx(float(ref.pvalue))


def test_paired_t_needs_two_points():
    with pytest.raises(ValueError):
        paired_t([1.0])


def test_detectable_effect_spec_power_example():
    # spec §6.4 power honesty: "~400 paired questions for a 0.01 Brier gap
    # at 2 sigma (delta ~ 0.1)" -> 2 * sd / sqrt(n).
    assert detectable_effect(400, 0.1) == pytest.approx(0.01)
    assert detectable_effect(100, 0.2) == pytest.approx(0.04)
    with pytest.raises(ValueError):
        detectable_effect(0, 0.1)


def test_difficulty_adjust_recovers_known_additive_structure():
    # b_ij = alpha_i + gamma_j with mean-zero gamma and no noise: adjusted
    # scores must equal alpha_i exactly (contract: exact recovery test).
    alpha = {"A": 1.0, "B": 3.0}
    gamma = {"q1": -2.0, "q2": 0.0, "q3": 2.0}
    scores = {(s, q): alpha[s] + gamma[q] for s in alpha for q in gamma}
    adjusted = difficulty_adjust(scores)
    assert set(adjusted) == set(scores)
    for (s, q), v in adjusted.items():
        assert v == pytest.approx(alpha[s], abs=1e-9)


def test_difficulty_adjust_preserves_paired_differences():
    # Subtracting a per-question effect never changes within-question
    # system differences.
    scores = {("A", "q1"): 5.0, ("B", "q1"): 1.0,
              ("A", "q2"): -3.0, ("B", "q2"): 0.5}
    adjusted = difficulty_adjust(scores)
    for q in ("q1", "q2"):
        assert adjusted[("A", q)] - adjusted[("B", q)] == pytest.approx(
            scores[("A", q)] - scores[("B", q)])


def test_difficulty_adjust_empty():
    assert difficulty_adjust({}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.scoring.stats'` (collection error)

- [ ] **Step 3: Write minimal implementation**

Create `backtest/scoring/stats.py`:

```python
"""Paired statistics and difficulty adjustment (spec §6.4).

All variant comparisons in the harness are paired on identical
(question, horizon) sets; these helpers consume the per-question paired
score differences."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as scipy_stats


def paired_bootstrap(diffs: list[float], n: int = 10_000, seed: int = 0) -> dict:
    """Percentile bootstrap 95% CI for the mean paired difference.
    Returns {"mean", "lo", "hi"}; deterministic for a given seed."""
    arr = np.asarray(diffs, dtype=float)
    if arr.size == 0:
        raise ValueError("paired_bootstrap needs at least one difference")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi)}


def paired_t(diffs: list[float]) -> dict:
    """Two-sided one-sample t-test of mean(diffs) == 0.
    Returns {"mean", "t", "p"}."""
    arr = np.asarray(diffs, dtype=float)
    if arr.size < 2:
        raise ValueError("paired_t needs at least two differences")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        t = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        p = 0.0 if mean != 0 else 1.0
    else:
        t = mean / (sd / math.sqrt(arr.size))
        p = 2.0 * float(scipy_stats.t.sf(abs(t), arr.size - 1))
    return {"mean": mean, "t": t, "p": p}


def detectable_effect(n: int, sd: float) -> float:
    """Minimal detectable mean paired difference at ~2 sigma: 2 * sd / sqrt(n).
    Spec §6.4 power example: n=400, sd=0.1 -> 0.01. Reports must refuse to
    bold a winner whose mean difference is inside this band."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 2.0 * float(sd) / math.sqrt(n)


def difficulty_adjust(scores: dict) -> dict:
    """Two-way fixed effects b_ij = alpha_i + gamma_j + eps (system i,
    question j), solved with np.linalg.lstsq (minimum-norm solution for the
    rank-deficient dummy design). Question effects gamma are centered to mean
    zero — the centered gamma is invariant to the alpha/gamma level ambiguity
    — and the adjusted score is b~ = b - gamma_hat (spec §6.4, ForecastBench
    updated difficulty-adjustment methodology).

    `scores` maps (system, qid) -> score; returns the same keys."""
    if not scores:
        return {}
    systems = sorted({s for s, _ in scores})
    questions = sorted({qid for _, qid in scores})
    s_index = {s: i for i, s in enumerate(systems)}
    q_index = {qid: i for i, qid in enumerate(questions)}
    keys = list(scores)
    design = np.zeros((len(keys), len(systems) + len(questions)))
    y = np.zeros(len(keys))
    for row, (s, qid) in enumerate(keys):
        design[row, s_index[s]] = 1.0
        design[row, len(systems) + q_index[qid]] = 1.0
        y[row] = float(scores[(s, qid)])
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    gamma = theta[len(systems):]
    gamma = gamma - gamma.mean()
    return {
        (s, qid): float(scores[(s, qid)] - gamma[q_index[qid]])
        for s, qid in keys
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_stats.py -v`
Expected: PASS (8 tests). Then run the full suite to close out the phase: `pytest tests/ -q` — all green, no regressions in `tests/test_*.py`.

- [ ] **Step 5: Commit**

```bash
git add backtest/scoring/stats.py tests/backtest/test_stats.py && git commit -m "feat(backtest): paired bootstrap/t-test, detectable-effect reporting, two-way fixed-effects difficulty adjustment" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
## Phase 2 — Corpus layer (Tasks 8–17)

Builds `backtest/corpus/`: the SQLite-backed `CorpusStore` (tables `questions`/`series`/`contamination`/`crosswalk`, big JSON under `root/blobs/`), the model-release contamination table, six source fetchers/importers, the question↔market crosswalk, and the forward snapshot accumulator.

**Preconditions:** `backtest/core.py` (with `BacktestQuestion`, `QType`) and `backtest/__init__.py` already exist (Phase 1 deliverables). `pyproject.toml` sets `asyncio_mode = "auto"`, so async tests are plain `async def test_*` functions — no decorators (convention of `tests/test_research.py`).

**Phase-wide rules:**
- Tests NEVER touch the network. Every fetcher module exposes a module-level `async def _get_json(url, params=None, headers=None, client=None)` mirroring the client-injection pattern of `bot/markets.py:_get_json`; tests monkeypatch the module attribute (exactly how `tests/test_markets.py` monkeypatches `bot.markets._get_json`).
- All ingestion is per-record isolated: a malformed record appends to `FetchReport.errors` and never aborts the run (spec §7).
- "Withheld" (API tier hid a populated field — Metaculus post-lockdown) is recorded in `provenance["withheld"]` and counted in `FetchReport.withheld`, distinct from "missing" (source never had the field).

### Task 8: CorpusStore core — schema, FetchReport, upsert/get round-trip

**Files:**
- Create: `backtest/corpus/__init__.py`
- Create: `backtest/corpus/store.py`
- Create: `tests/backtest/__init__.py` (only if Phase 1 did not already create it)
- Test: `tests/backtest/test_corpus_store.py`

- [ ] **Step 1: Create package scaffolding**
```bash
mkdir -p backtest/corpus tests/backtest
touch backtest/corpus/__init__.py tests/backtest/__init__.py
```
- [ ] **Step 2: Write the failing test**

Create `tests/backtest/test_corpus_store.py`:
```python
from dataclasses import replace
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def _numeric_question():
    return BacktestQuestion(
        qid="metaculus:31678", platform="metaculus", native_id="31678",
        qtype=QType.NUMERIC, title="How many widgets by 2027?",
        description="Background.", resolution_criteria="Counted by X.",
        fine_print="Edge cases.", open_time=_utc(2026, 1, 1),
        scheduled_close_time=_utc(2026, 12, 31),
        spot_scoring_time=_utc(2026, 12, 1),
        actual_resolve_time=_utc(2027, 1, 5), resolution=41.5,
        options=(),
        scaling={"range_min": 0, "range_max": 100, "zero_point": None,
                 "open_lower_bound": False, "open_upper_bound": True,
                 "cdf_size": 201, "unit": "widgets"},
        question_weight=0.5,
        provenance={"source": "metaculus", "withheld": ["aggregations"]})


def test_fetch_report_defaults():
    r = FetchReport()
    assert r.fetched == 0 and r.withheld == {} and r.errors == []


def test_init_creates_db_and_blobs_dir(tmp_path):
    CorpusStore(tmp_path / "corpus")
    assert (tmp_path / "corpus" / "corpus.db").exists()
    assert (tmp_path / "corpus" / "blobs").is_dir()


def test_upsert_get_round_trip(tmp_path):
    store = CorpusStore(tmp_path)
    q = _numeric_question()
    store.upsert_question(q)
    got = store.get_question("metaculus:31678")
    assert got == q          # frozen dataclass equality covers every field


def test_get_missing_returns_none(tmp_path):
    assert CorpusStore(tmp_path).get_question("metaculus:0") is None


def test_upsert_replaces_existing_row(tmp_path):
    store = CorpusStore(tmp_path)
    q = _numeric_question()
    store.upsert_question(q)
    store.upsert_question(replace(q, title="Updated title"))
    assert store.get_question(q.qid).title == "Updated title"


def test_binary_bool_and_none_resolution_round_trip(tmp_path):
    store = CorpusStore(tmp_path)
    base = _numeric_question()
    yes = replace(base, qid="metaculus:1", qtype=QType.BINARY,
                  resolution=True, scaling=None)
    unresolved = replace(base, qid="metaculus:2", resolution=None)
    store.upsert_question(yes)
    store.upsert_question(unresolved)
    assert store.get_question("metaculus:1").resolution is True
    assert store.get_question("metaculus:2").resolution is None
```
- [ ] **Step 3: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.store'` (collection error)
- [ ] **Step 4: Write minimal implementation**

Create `backtest/corpus/store.py`:
```python
"""Corpus store: SQLite metadata + JSON blobs for bulky series.

Layout: root/corpus.db + root/blobs/. Tables: questions, series,
contamination, crosswalk. Big payloads (CP/price history, raw snapshots)
live as JSON files under blobs/ and are referenced by relative path from
the series table (spec §3.1).

Provenance convention: "withheld" (the API tier hid a populated field —
Metaculus post-lockdown reality) is recorded in provenance["withheld"],
DISTINCT from "missing" (the source never had the field, stored as ''/None
with no withheld marker).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backtest.core import BacktestQuestion, QType

SCHEMA_VERSION = 1

_SCHEMA = """
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
"""


@dataclass
class FetchReport:
    fetched: int = 0
    withheld: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _from_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _safe(qid: str) -> str:
    """qid -> filesystem-safe blob filename fragment."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in qid)


class CorpusStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.root / "corpus.db")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def upsert_question(self, q: BacktestQuestion) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO questions
               (qid, platform, native_id, qtype, title, description,
                resolution_criteria, fine_print, open_time,
                scheduled_close_time, spot_scoring_time,
                actual_resolve_time, resolution, options, scaling,
                question_weight, provenance, schema_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (q.qid, q.platform, q.native_id, q.qtype.value, q.title,
             q.description, q.resolution_criteria, q.fine_print,
             _iso(q.open_time), _iso(q.scheduled_close_time),
             _iso(q.spot_scoring_time), _iso(q.actual_resolve_time),
             None if q.resolution is None else json.dumps(q.resolution),
             json.dumps(list(q.options)),
             None if q.scaling is None else json.dumps(q.scaling),
             q.question_weight, json.dumps(q.provenance), SCHEMA_VERSION))
        self.conn.commit()

    def get_question(self, qid: str) -> BacktestQuestion | None:
        row = self.conn.execute(
            "SELECT * FROM questions WHERE qid = ?", (qid,)).fetchone()
        return self._row_to_question(row) if row else None

    @staticmethod
    def _row_to_question(row: sqlite3.Row) -> BacktestQuestion:
        return BacktestQuestion(
            qid=row["qid"], platform=row["platform"],
            native_id=row["native_id"], qtype=QType(row["qtype"]),
            title=row["title"], description=row["description"],
            resolution_criteria=row["resolution_criteria"],
            fine_print=row["fine_print"],
            open_time=_from_iso(row["open_time"]),
            scheduled_close_time=_from_iso(row["scheduled_close_time"]),
            spot_scoring_time=_from_iso(row["spot_scoring_time"]),
            actual_resolve_time=_from_iso(row["actual_resolve_time"]),
            resolution=(None if row["resolution"] is None
                        else json.loads(row["resolution"])),
            options=tuple(json.loads(row["options"])),
            scaling=(None if row["scaling"] is None
                     else json.loads(row["scaling"])),
            question_weight=row["question_weight"],
            provenance=json.loads(row["provenance"]))
```
- [ ] **Step 5: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_store.py -v`
Expected: PASS (6 tests)
- [ ] **Step 6: Commit**
```bash
git add backtest/corpus/__init__.py backtest/corpus/store.py tests/backtest/__init__.py tests/backtest/test_corpus_store.py && git commit -m "feat(backtest): CorpusStore SQLite core — 4-table schema, FetchReport, question round-trip" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 9: CorpusStore — series blobs, value_at, contamination, query

**Files:**
- Modify: `backtest/corpus/store.py` (append methods to `CorpusStore`)
- Test: `tests/backtest/test_corpus_store.py` (append)

- [ ] **Step 1: Write the failing tests for series + contamination**

Append to `tests/backtest/test_corpus_store.py`:
```python
def test_put_series_and_value_at_step_lookup(tmp_path):
    store = CorpusStore(tmp_path)
    store.upsert_question(_numeric_question())
    store.put_series("metaculus:31678", "cp", [
        ("2026-03-01T00:00:00+00:00", 0.40),
        ("2026-01-01T00:00:00+00:00", 0.25),     # unsorted on purpose
        ("2026-05-01T00:00:00+00:00", 0.55),
    ])
    # step function: last point with ts <= t
    assert store.value_at("metaculus:31678", "cp", _utc(2025, 12, 1)) is None
    assert store.value_at("metaculus:31678", "cp", _utc(2026, 1, 1)) == 0.25
    assert store.value_at("metaculus:31678", "cp", _utc(2026, 4, 15)) == 0.40
    assert store.value_at("metaculus:31678", "cp", _utc(2027, 1, 1)) == 0.55
    # absent kind / absent qid
    assert store.value_at("metaculus:31678", "market_price",
                          _utc(2026, 4, 15)) is None
    assert store.value_at("metaculus:0", "cp", _utc(2026, 4, 15)) is None
    # put_series REPLACES the series
    store.put_series("metaculus:31678", "cp",
                     [("2026-06-01T00:00:00+00:00", 0.9)])
    assert store.value_at("metaculus:31678", "cp", _utc(2026, 4, 15)) is None
    # blob lives under blobs/
    assert list((tmp_path / "blobs").glob("*__cp.json"))


def test_value_at_naive_query_against_aware_series(tmp_path):
    store = CorpusStore(tmp_path)
    store.put_series("metaculus:31678", "cp",
                     [("2026-01-01T00:00:00+00:00", 0.25)])
    assert store.value_at("metaculus:31678", "cp",
                          datetime(2026, 2, 1)) == 0.25


def test_contamination_set_get(tmp_path):
    store = CorpusStore(tmp_path)
    assert store.get_contamination("metaculus:1", "h1") is None
    store.set_contamination("metaculus:1", "h1", "clean")
    store.set_contamination("metaculus:1", "h2", "contaminated")
    assert store.get_contamination("metaculus:1", "h1") == "clean"
    assert store.get_contamination("metaculus:1", "h2") == "contaminated"
    store.set_contamination("metaculus:1", "h1", "contaminated-confirmed")
    assert (store.get_contamination("metaculus:1", "h1")
            == "contaminated-confirmed")
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_store.py -v`
Expected: FAIL with `AttributeError: 'CorpusStore' object has no attribute 'put_series'` (and `set_contamination`)
- [ ] **Step 3: Write minimal implementation**

Append these methods to `class CorpusStore` in `backtest/corpus/store.py`:
```python
    def put_series(self, qid: str, kind: str,
                   points: list[tuple[str, object]]) -> None:
        """Replace the (qid, kind) series. points: (iso_timestamp, value).
        Sorted on write; mixed naive/aware timestamps within ONE series are
        the fetcher's responsibility to avoid (each fetcher is internally
        consistent)."""
        pts = sorted(points, key=lambda p: p[0])
        path = self.blobs / f"{_safe(qid)}__{kind}.json"
        path.write_text(json.dumps(pts))
        self.conn.execute(
            "INSERT OR REPLACE INTO series (qid, kind, blob_path, n_points)"
            " VALUES (?,?,?,?)",
            (qid, kind, str(path.relative_to(self.root)), len(pts)))
        self.conn.commit()

    def value_at(self, qid: str, kind: str, t: datetime) -> object | None:
        """Step-function lookup: last point with ts <= t, else None."""
        row = self.conn.execute(
            "SELECT blob_path FROM series WHERE qid = ? AND kind = ?",
            (qid, kind)).fetchone()
        if row is None:
            return None
        pts = json.loads((self.root / row["blob_path"]).read_text())
        best = None
        for ts, value in pts:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None and t.tzinfo is not None:
                dt = dt.replace(tzinfo=timezone.utc)
            elif dt.tzinfo is not None and t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if dt <= t:
                best = value
            else:
                break
        return best

    def set_contamination(self, qid: str, model_set_hash: str,
                          stratum: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO contamination"
            " (qid, model_set_hash, stratum) VALUES (?,?,?)",
            (qid, model_set_hash, stratum))
        self.conn.commit()

    def get_contamination(self, qid: str,
                          model_set_hash: str) -> str | None:
        row = self.conn.execute(
            "SELECT stratum FROM contamination"
            " WHERE qid = ? AND model_set_hash = ?",
            (qid, model_set_hash)).fetchone()
        return row["stratum"] if row else None
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_store.py -v`
Expected: PASS (9 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/corpus/store.py tests/backtest/test_corpus_store.py && git commit -m "feat(backtest): corpus series blobs + value_at step lookup + contamination strata" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
- [ ] **Step 6: Write the failing test for query filters**

Append to `tests/backtest/test_corpus_store.py`:
```python
def _seed_query_corpus(store):
    base = _numeric_question()
    store.upsert_question(replace(
        base, qid="metaculus:1", qtype=QType.BINARY, resolution=True,
        scaling=None, open_time=_utc(2026, 5, 1)))
    store.upsert_question(replace(
        base, qid="metaculus:2", resolution=None))           # unresolved
    store.upsert_question(replace(
        base, qid="polymarket:a", platform="polymarket",
        qtype=QType.BINARY, resolution=False, scaling=None,
        open_time=_utc(2026, 1, 10)))


def test_query_filters(tmp_path):
    store = CorpusStore(tmp_path)
    _seed_query_corpus(store)
    assert [q.qid for q in store.query()] == ["metaculus:1", "polymarket:a"]
    assert [q.qid for q in store.query(resolved_only=False)] == \
        ["metaculus:1", "metaculus:2", "polymarket:a"]
    assert [q.qid for q in store.query(platforms=["polymarket"])] == \
        ["polymarket:a"]
    assert [q.qid for q in store.query(qtypes=[QType.BINARY],
                                       open_after=_utc(2026, 3, 1))] == \
        ["metaculus:1"]
    assert [q.qid for q in store.query(qtypes=["numeric"],
                                       resolved_only=False)] == \
        ["metaculus:2"]
    assert store.query(close_before=_utc(2025, 1, 1)) == []
    assert len(store.query(limit=1)) == 1


def test_query_stratum_requires_hash_and_joins(tmp_path):
    import pytest
    store = CorpusStore(tmp_path)
    _seed_query_corpus(store)
    store.set_contamination("metaculus:1", "abc123", "clean")
    store.set_contamination("polymarket:a", "abc123", "contaminated")
    with pytest.raises(ValueError):
        store.query(stratum="clean")
    assert [q.qid for q in store.query(stratum="clean",
                                       model_set_hash="abc123")] == \
        ["metaculus:1"]
    assert [q.qid for q in store.query(stratum="contaminated",
                                       model_set_hash="abc123")] == \
        ["polymarket:a"]
```
- [ ] **Step 7: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_store.py -v`
Expected: FAIL with `AttributeError: 'CorpusStore' object has no attribute 'query'`
- [ ] **Step 8: Write minimal implementation**

Append this method to `class CorpusStore` in `backtest/corpus/store.py`:
```python
    def query(self, *, platforms=None, qtypes=None, resolved_only=True,
              open_after=None, close_before=None, stratum=None,
              model_set_hash=None, limit=None) -> list[BacktestQuestion]:
        """Date filters compare ISO-8601 strings lexicographically — fine at
        date granularity for the formats the fetchers write."""
        sql = "SELECT q.* FROM questions q"
        where: list[str] = []
        args: list = []
        if stratum is not None:
            if model_set_hash is None:
                raise ValueError("stratum filter requires model_set_hash")
            sql += (" JOIN contamination c ON c.qid = q.qid"
                    " AND c.model_set_hash = ? AND c.stratum = ?")
            args += [model_set_hash, stratum]
        if platforms:
            where.append(
                f"q.platform IN ({','.join('?' * len(platforms))})")
            args += list(platforms)
        if qtypes:
            vals = [t.value if isinstance(t, QType) else t for t in qtypes]
            where.append(f"q.qtype IN ({','.join('?' * len(vals))})")
            args += vals
        if resolved_only:
            where.append("q.resolution IS NOT NULL")
        if open_after is not None:
            where.append("q.open_time > ?")
            args.append(open_after.isoformat())
        if close_before is not None:
            where.append("q.scheduled_close_time < ?")
            args.append(close_before.isoformat())
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY q.qid"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        return [self._row_to_question(r)
                for r in self.conn.execute(sql, args).fetchall()]
```
- [ ] **Step 9: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_store.py -v`
Expected: PASS (11 tests)
- [ ] **Step 10: Commit**
```bash
git add backtest/corpus/store.py tests/backtest/test_corpus_store.py && git commit -m "feat(backtest): CorpusStore.query with platform/type/date/stratum filters" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 10: Model release dates, model_set_hash, stratify

**Files:**
- Create: `backtest/corpus/model_releases.py`
- Test: `tests/backtest/test_model_releases.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_model_releases.py`:
```python
from dataclasses import replace
from datetime import datetime, timezone

import backtest.corpus.model_releases as mr
from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore

DATES = {
    "openai/gpt-5.4": "2026-02-10",
    "anthropic/claude-opus-4-6": "2026-01-20",
    "google/gemini-3.1-pro-preview": "2026-03-05",
}

MODEL_SET = list(DATES)


def _q(qid, open_time):
    return BacktestQuestion(
        qid=qid, platform="metaculus", native_id=qid.split(":")[1],
        qtype=QType.BINARY, title="T?", open_time=open_time,
        resolution=True)


def test_release_table_covers_current_ensemble_slugs():
    # the three slugs of bot/config.py:_default_ensemble()
    for slug in MODEL_SET:
        assert slug in mr.RELEASE_DATES
        datetime.fromisoformat(mr.RELEASE_DATES[slug])   # valid iso date


def test_model_set_hash_deterministic_and_order_independent(monkeypatch):
    monkeypatch.setattr(mr, "RELEASE_DATES", dict(DATES))
    h1 = mr.model_set_hash(MODEL_SET)
    h2 = mr.model_set_hash(list(reversed(MODEL_SET)))
    assert h1 == h2
    assert len(h1) == 12 and all(c in "0123456789abcdef" for c in h1)
    # hash changes when a release date changes
    monkeypatch.setattr(mr, "RELEASE_DATES",
                        dict(DATES, **{"openai/gpt-5.4": "2026-04-01"}))
    assert mr.model_set_hash(MODEL_SET) != h1
    # unknown slugs are total: deterministic hash, no exception
    monkeypatch.setattr(mr, "RELEASE_DATES", dict(DATES))
    h_u = mr.model_set_hash(MODEL_SET + ["vendor/unlisted-model"])
    assert h_u == mr.model_set_hash(MODEL_SET + ["vendor/unlisted-model"])
    assert len(h_u) == 12


def test_stratify_clean_vs_contaminated(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "RELEASE_DATES", dict(DATES))
    store = CorpusStore(tmp_path)
    h = mr.model_set_hash(MODEL_SET)
    # opened AFTER the latest release date (2026-03-05) -> clean
    clean_q = _q("metaculus:1",
                 datetime(2026, 4, 1, tzinfo=timezone.utc))
    assert mr.stratify(store, clean_q, MODEL_SET) == "clean"
    assert store.get_contamination("metaculus:1", h) == "clean"
    # opened BEFORE the latest release date -> contaminated
    dirty_q = _q("metaculus:2",
                 datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert mr.stratify(store, dirty_q, MODEL_SET) == "contaminated"
    assert store.get_contamination("metaculus:2", h) == "contaminated"
    # unknown open_time -> conservative: contaminated
    unknown_q = replace(_q("metaculus:3", None))
    assert mr.stratify(store, unknown_q, MODEL_SET) == "contaminated"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_model_releases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.model_releases'`
- [ ] **Step 3: Write minimal implementation**

Create `backtest/corpus/model_releases.py`:
```python
"""Model release dates — the contamination boundary (spec §3.4).

clean iff question.open_time > max RELEASE DATE across the ensemble's
models. Release date, NOT stated training cutoff: cutoff labels
demonstrably lie; release dates are the trustworthy bound.

!!! PLACEHOLDER DATES — IMPLEMENTER MUST VERIFY BEFORE USE !!!
The dates below are set to 2026-06-11 (the design date) as the MAXIMALLY
CONSERVATIVE placeholder: with these values NOTHING in the corpus can be
labeled "clean", so a forgotten verification fails safe (no false clean
stratum) rather than silently contaminating the headline metric. Before the
first real stratified report, replace each with the model's verified public
release date (OpenRouter model page / provider announcement) and re-run
stratify over the corpus. The slugs mirror bot/config.py:_default_ensemble().
"""
from __future__ import annotations

import hashlib

from backtest.core import BacktestQuestion
from backtest.corpus.store import CorpusStore

RELEASE_DATES: dict[str, str] = {
    # OpenRouter slug -> iso release date. PLACEHOLDERS — see module docstring.
    "openai/gpt-5.4": "2026-06-11",
    "anthropic/claude-opus-4-6": "2026-06-11",
    "google/gemini-3.1-pro-preview": "2026-06-11",
}


def model_set_hash(model_set: list[str]) -> str:
    """sha256 of the sorted release dates (contract formula). Two model sets
    with identical release-date multisets intentionally share strata.
    Total: unknown slugs hash as "unknown" — never raises."""
    dates = sorted(RELEASE_DATES.get(m, "unknown") for m in model_set)
    return hashlib.sha256("|".join(dates).encode()).hexdigest()[:12]


def stratify(store: CorpusStore, q: BacktestQuestion,
             model_set: list[str]) -> str:
    """Write and return the contamination stratum for q under model_set.
    The memorization probe (backtest/timetravel/probe.py) may later upgrade
    "contaminated" to "contaminated-confirmed" via set_contamination."""
    h = model_set_hash(model_set)
    latest = max(RELEASE_DATES[m] for m in model_set)
    if q.open_time is not None and q.open_time.date().isoformat() > latest:
        stratum = "clean"
    else:
        stratum = "contaminated"   # unknown open_time fails safe
    store.set_contamination(q.qid, h, stratum)
    return stratum
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_model_releases.py -v`
Expected: PASS (3 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/corpus/model_releases.py tests/backtest/test_model_releases.py && git commit -m "feat(backtest): model release-date table, model_set_hash, contamination stratify" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 11: Tier-aware Metaculus fetcher

**Files:**
- Create: `backtest/corpus/metaculus.py`
- Test: `tests/backtest/test_corpus_metaculus.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_corpus_metaculus.py`:
```python
from datetime import datetime, timezone

import pytest

import backtest.corpus.metaculus as metaculus_mod
from backtest.corpus.metaculus import fetch_metaculus
from backtest.corpus.store import CorpusStore

LIST_PAGE = {
    "next": None,
    "results": [{"id": 31678, "title": "Will X happen before 2027?"},
                {"id": 31999, "title": "Locked-down question?"}],
}

# Full-visibility detail (own-forecast question: token sees everything)
DETAIL_FULL = {
    "id": 31678,
    "title": "Will X happen before 2027?",
    "status": "resolved",
    "question": {
        "type": "binary",
        "title": "Will X happen before 2027?",
        "description": "Background paragraph.",
        "resolution_criteria": "Resolves YES if X occurs before 2027-01-01.",
        "fine_print": "Standard fine print.",
        "open_time": "2026-04-01T00:00:00Z",
        "scheduled_close_time": "2026-06-01T00:00:00Z",
        "spot_scoring_time": "2026-05-15T00:00:00Z",
        "actual_resolve_time": "2026-06-02T12:00:00Z",
        "resolution": "yes",
        "question_weight": 0.75,
        "aggregations": {"recency_weighted": {"history": [
            {"start_time": 1775000000.0, "end_time": 1775500000.0,
             "centers": [0.40], "means": [0.41]},
            {"start_time": 1775500000.0, "end_time": None,
             "centers": [0.55]},
        ]}},
    },
}

# Degraded detail (post-lockdown plain-token tier: nulls are WITHHELD)
DETAIL_DEGRADED = {
    "id": 31999,
    "title": "Locked-down question?",
    "status": "resolved",
    "question": {
        "type": "binary",
        "description": None,
        "resolution_criteria": "",
        "open_time": "2026-01-01T00:00:00Z",
        "scheduled_close_time": "2026-03-01T00:00:00Z",
        "actual_resolve_time": "2026-03-02T00:00:00Z",
        "resolution": None,
        "aggregations": {"recency_weighted": {"history": None}},
    },
}

# Multiple-choice detail: per-option CP values keyed by option name
DETAIL_MC = {
    "id": 32500,
    "title": "Who wins the 2026 race?",
    "status": "resolved",
    "question": {
        "type": "multiple_choice",
        "title": "Who wins the 2026 race?",
        "description": "Background.",
        "resolution_criteria": "Resolves to the winner.",
        "open_time": "2026-01-01T00:00:00Z",
        "scheduled_close_time": "2026-06-01T00:00:00Z",
        "actual_resolve_time": "2026-06-02T00:00:00Z",
        "resolution": "Alice",
        "options": ["Alice", "Bob"],
        "aggregations": {"recency_weighted": {"history": [
            {"start_time": 1775000000.0, "end_time": None,
             "centers": [0.3, 0.7]},
        ]}},
    },
}

# Resolved BEFORE open -> quarantined (resolution-date validation, spec §3.4)
DETAIL_BAD_DATES = {
    "id": 31700,
    "title": "Resolved before it opened?",
    "status": "resolved",
    "question": {
        "type": "binary",
        "description": "Background.",
        "open_time": "2026-04-01T00:00:00Z",
        "scheduled_close_time": "2026-06-01T00:00:00Z",
        "actual_resolve_time": "2026-03-01T00:00:00Z",   # before open
        "resolution": "yes",
    },
}


@pytest.fixture
def no_throttle(monkeypatch):
    calls = []

    async def fake_throttle():
        calls.append(1)

    monkeypatch.setattr(metaculus_mod, "_throttle", fake_throttle)
    return calls


@pytest.fixture
def fake_api(monkeypatch):
    seen = {"headers": [], "params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        seen["headers"].append(headers)
        seen["params"].append(params)
        if url.endswith("/posts/"):
            return LIST_PAGE
        if url.endswith("/posts/31678/"):
            return DETAIL_FULL
        if url.endswith("/posts/31999/"):
            return DETAIL_DEGRADED
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    return seen


async def test_fetch_ingests_full_and_degraded(tmp_path, monkeypatch,
                                               fake_api, no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store, tournaments=["minibench"])
    assert report.fetched == 2
    assert report.withheld == {"description": 1, "resolution": 1,
                               "aggregations": 1}
    assert fake_api["headers"][0] == {"Authorization": "Token test-token"}
    assert fake_api["params"][0]["tournaments"] == "minibench"
    assert fake_api["params"][0]["statuses"] == "resolved"

    full = store.get_question("metaculus:31678")
    assert full.resolution is True            # "yes" -> bool
    assert full.question_weight == 0.75
    assert full.provenance["withheld"] == []
    assert full.open_time == datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert full.spot_scoring_time == datetime(2026, 5, 15,
                                              tzinfo=timezone.utc)

    degraded = store.get_question("metaculus:31999")
    assert degraded.resolution is None        # withheld, NOT resolved-no
    assert degraded.description == ""
    assert set(degraded.provenance["withheld"]) == \
        {"description", "resolution", "aggregations"}
    assert no_throttle, "self-throttle must run between detail GETs"


async def test_cp_history_flattened_to_step_points(tmp_path, monkeypatch,
                                                   fake_api, no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    store = CorpusStore(tmp_path)
    await fetch_metaculus(store)
    # interval entry start_time <= t < end_time, flattened to
    # (start_time, value) step points; "centers" preferred over "means"
    t_mid = datetime.fromtimestamp(1775200000.0, tz=timezone.utc)
    assert store.value_at("metaculus:31678", "cp", t_mid) == 0.40
    t_late = datetime.fromtimestamp(1775600000.0, tz=timezone.utc)
    assert store.value_at("metaculus:31678", "cp", t_late) == 0.55
    t_early = datetime.fromtimestamp(1774000000.0, tz=timezone.utc)
    assert store.value_at("metaculus:31678", "cp", t_early) is None


async def test_mc_cp_values_stored_as_option_dict(tmp_path, monkeypatch,
                                                  no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/"):
            return {"next": None,
                    "results": [{"id": 32500, "title": "Who wins?"}]}
        if url.endswith("/posts/32500/"):
            return DETAIL_MC
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store)
    assert report.fetched == 1
    # MC CP entries are stored as option -> probability dicts
    t = datetime.fromtimestamp(1775500000.0, tz=timezone.utc)
    assert store.value_at("metaculus:32500", "cp", t) == \
        {"Alice": 0.3, "Bob": 0.7}


async def test_resolve_before_open_is_quarantined(tmp_path, monkeypatch,
                                                  no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/"):
            return {"next": None,
                    "results": [{"id": 31700, "title": "Bad dates?"}]}
        if url.endswith("/posts/31700/"):
            return DETAIL_BAD_DATES
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store)
    assert report.fetched == 0
    assert any("quarantined metaculus:31700" in e for e in report.errors)
    assert store.get_question("metaculus:31700") is None


async def test_missing_token_raises(tmp_path, monkeypatch, fake_api):
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        await fetch_metaculus(CorpusStore(tmp_path))


async def test_detail_failure_is_isolated(tmp_path, monkeypatch,
                                          no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    async def flaky_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/"):
            return LIST_PAGE
        if url.endswith("/posts/31678/"):
            return DETAIL_FULL
        raise RuntimeError("503 from Metaculus")

    monkeypatch.setattr(metaculus_mod, "_get_json", flaky_get)
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store)
    assert report.fetched == 1
    assert any("31999" in e for e in report.errors)
    assert store.get_question("metaculus:31678") is not None


async def test_limit_caps_detail_fetches(tmp_path, monkeypatch, fake_api,
                                         no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    report = await fetch_metaculus(CorpusStore(tmp_path), limit=1)
    assert report.fetched == 1
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_metaculus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.metaculus'`
- [ ] **Step 3: Write minimal implementation**

Create `backtest/corpus/metaculus.py`:
```python
"""Tier-aware Metaculus fetcher (spec §3.2 source 1).

Post-March-2026 lockdown: a plain token sees IDs/titles/types/timestamps
but description/resolution/aggregations come back null — EXCEPT questions
this bot ever forecast, which stay fully visible. This fetcher stores what
is visible, records withheld fields in provenance["withheld"] and
FetchReport.withheld, and simply re-fetching after the Bot Benchmarking
Access Tier / Data Needs Form approval lands upgrades rows in place
(upsert_question replaces).

Enumeration: GET /api/posts/ with statuses/forecast_type/tournaments
params, limit<=100 offset pagination. Detail: GET /api/posts/{id}/.
Auth: "Authorization: Token <METACULUS_TOKEN>". Self-throttle ~3.5s +
jitter between requests (forecasting-tools convention).

CP history: aggregations.recency_weighted.history interval entries
(start_time <= t < end_time) are flattened to (start_time, value) step
points — CorpusStore.value_at's last-point-<=-t lookup then reproduces
CP-at-T (matches forecasting-tools get_cp_at_time semantics).
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport

logger = logging.getLogger(__name__)

API_BASE = "https://www.metaculus.com/api"
PAGE_SIZE = 100
THROTTLE_S = 3.5
HTTP_TIMEOUT_S = 30.0


def _auth_headers() -> dict:
    token = os.environ.get("METACULUS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("METACULUS_TOKEN is not set")
    return {"Authorization": f"Token {token}"}


async def _throttle() -> None:
    await asyncio.sleep(THROTTLE_S + random.uniform(0.0, 0.5))


async def _get_json(url: str, params: dict | None = None,
                    headers: dict | None = None, client=None):
    import httpx
    if client is not None:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as one_shot:
        resp = await one_shot.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _parse_time(v) -> datetime | None:
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)
    return datetime.fromisoformat(str(v))


def _cp_points(history: list[dict],
               options: tuple = ()) -> list[tuple[str, object]]:
    """Flatten recency_weighted.history intervals to step points.
    MC questions (len(values) == len(options)) store an option->prob dict."""
    pts: list[tuple[str, object]] = []
    for entry in history or []:
        start = entry.get("start_time")
        values = entry.get("centers") or entry.get("means")
        if start is None or not values:
            continue
        if len(values) == 1:
            value = values[0]
        elif options and len(values) == len(options):
            value = dict(zip(options, values))
        else:
            value = list(values)
        pts.append((_parse_time(start).isoformat(), value))
    return pts


def _ingest_detail(store: CorpusStore, post: dict,
                   report: FetchReport) -> str | None:
    """Map one /api/posts/{id}/ payload into the store. Returns qid."""
    q = post.get("question") or {}
    if not q.get("type"):
        report.errors.append(f"post {post.get('id')}: no question payload")
        return None
    qid = f"metaculus:{post['id']}"
    open_time = _parse_time(q.get("open_time"))
    actual_resolve_time = _parse_time(q.get("actual_resolve_time"))
    # resolution-date validation at import (spec §3.4)
    if (open_time and actual_resolve_time
            and actual_resolve_time < open_time):
        report.errors.append(
            f"quarantined {qid}: resolve_time < open_time")
        return None
    withheld: list[str] = []

    description = q.get("description")
    if description is None:          # withheld by tier, not missing
        withheld.append("description")
        description = ""

    resolution = q.get("resolution")
    is_resolved = (post.get("status") == "resolved"
                   or actual_resolve_time is not None)
    if resolution is None and is_resolved:
        withheld.append("resolution")
    qtype = QType(q["type"])
    if qtype is QType.BINARY and isinstance(resolution, str):
        # "annulled"/"ambiguous" stay None (unscoreable, not withheld)
        resolution = {"yes": True, "no": False}.get(resolution.lower())
    elif qtype in (QType.NUMERIC, QType.DISCRETE) \
            and resolution is not None:
        try:
            resolution = float(resolution)
        except (TypeError, ValueError):
            pass    # "above_upper_bound"/"below_lower_bound" stay strings

    history = ((q.get("aggregations") or {})
               .get("recency_weighted") or {}).get("history")
    if history is None:
        withheld.append("aggregations")

    for f in withheld:
        report.withheld[f] = report.withheld.get(f, 0) + 1

    store.upsert_question(BacktestQuestion(
        qid=qid, platform="metaculus", native_id=str(post["id"]),
        qtype=qtype, title=post.get("title") or q.get("title") or "",
        description=description,
        resolution_criteria=q.get("resolution_criteria") or "",
        fine_print=q.get("fine_print") or "",
        open_time=open_time,
        scheduled_close_time=_parse_time(q.get("scheduled_close_time")),
        spot_scoring_time=_parse_time(q.get("spot_scoring_time")),
        actual_resolve_time=actual_resolve_time,
        resolution=resolution,
        options=tuple(q.get("options") or ()),
        scaling=q.get("scaling"),
        question_weight=float(q.get("question_weight") or 1.0),
        provenance={"source": "metaculus",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "withheld": withheld}))
    if history:
        pts = _cp_points(history, tuple(q.get("options") or ()))
        if pts:
            store.put_series(qid, "cp", pts)
    report.fetched += 1
    return qid


async def fetch_metaculus(store: CorpusStore, *, tournaments=None,
                          statuses="resolved",
                          limit=None) -> FetchReport:
    headers = _auth_headers()
    report = FetchReport()
    params: dict = {"statuses": statuses, "limit": PAGE_SIZE, "offset": 0,
                    "include_descriptions": "true"}
    if tournaments:
        params["tournaments"] = ",".join(tournaments)

    post_ids: list[int] = []
    while True:
        page = await _get_json(f"{API_BASE}/posts/", params=params,
                               headers=headers)
        for post in page.get("results") or []:
            post_ids.append(post["id"])
            if limit is not None and len(post_ids) >= limit:
                break
        if limit is not None and len(post_ids) >= limit:
            break
        if not page.get("next"):
            break
        params = dict(params, offset=params["offset"] + PAGE_SIZE)
        await _throttle()

    for post_id in post_ids:
        await _throttle()
        try:
            detail = await _get_json(f"{API_BASE}/posts/{post_id}/",
                                     headers=headers)
            _ingest_detail(store, detail, report)
        except Exception as e:     # per-question isolation (spec §7)
            logger.warning("metaculus post %s failed: %s", post_id, e)
            report.errors.append(f"post {post_id}: {e}")
    return report
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_metaculus.py -v`
Expected: PASS (7 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/corpus/metaculus.py tests/backtest/test_corpus_metaculus.py && git commit -m "feat(backtest): tier-aware Metaculus fetcher — withheld tracking, CP step series, throttled pagination" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 12: Halawi dataset importer (local files only)

**Files:**
- Create: `backtest/corpus/halawi.py`
- Test: `tests/backtest/test_corpus_halawi.py`

Importers never touch the network — they read a previously downloaded local directory. Document for the corpus builder (this is a manual step done once, NOT part of the test suite):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download YuehHanChen/forecasting --repo-type dataset --local-dir data/backtest/raw/halawi
# the raw 48k-question variant, if wanted later:
# huggingface-cli download YuehHanChen/forecasting_raw --repo-type dataset --local-dir data/backtest/raw/halawi_raw
```

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_corpus_halawi.py`:
```python
import json
from datetime import datetime

from backtest.corpus.halawi import import_halawi
from backtest.corpus.store import CorpusStore

RECORDS = [
    {"id": "q-101", "question": "Will the 2024 X summit occur?",
     "background": "Verbatim dataset background.",
     "resolution_criteria": "Resolves YES if the summit occurs in 2024.",
     "date_begin": "2024-01-05", "date_close": "2024-06-30",
     "date_resolve_at": "2024-07-02", "resolution": 1.0,
     "source": "metaculus"},
    # resolved BEFORE open -> quarantined (resolution-date validation,
    # spec §3.4: >=3.8% of even the curated set had this defect)
    {"id": "q-102", "question": "Bad dates question?",
     "background": "", "date_begin": "2024-05-01",
     "date_close": "2024-06-01", "date_resolve_at": "2024-04-01",
     "resolution": 0.0, "source": "cset"},
    # no usable resolution -> skipped with error
    {"id": "q-103", "question": "No resolution recorded?",
     "date_begin": "2024-01-01", "date_close": "2024-02-01"},
]


def test_import_halawi_jsonl(tmp_path):
    dataset_dir = tmp_path / "halawi"
    dataset_dir.mkdir()
    with open(dataset_dir / "train.jsonl", "w") as f:
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")
    store = CorpusStore(tmp_path / "corpus")
    report = import_halawi(store, dataset_dir)
    assert report.fetched == 1
    assert len(report.errors) == 2     # bad dates + missing resolution
    q = store.get_question("halawi:q-101")
    assert q.platform == "halawi"
    assert q.qtype.value == "binary"
    assert q.resolution is True
    # dataset text stored verbatim (spec §3.2: text may drift from the
    # current Metaculus copy — provenance carries the origin platform)
    assert q.description == "Verbatim dataset background."
    assert q.open_time == datetime.fromisoformat("2024-01-05")
    assert q.actual_resolve_time == datetime.fromisoformat("2024-07-02")
    assert q.provenance["source"] == "halawi"
    assert q.provenance["origin_platform"] == "metaculus"


def test_import_halawi_json_array(tmp_path):
    dataset_dir = tmp_path / "halawi"
    dataset_dir.mkdir()
    (dataset_dir / "test.json").write_text(json.dumps([RECORDS[0]]))
    store = CorpusStore(tmp_path / "corpus")
    report = import_halawi(store, dataset_dir)
    assert report.fetched == 1
    assert report.errors == []


def test_import_halawi_zero_resolution_is_false(tmp_path):
    dataset_dir = tmp_path / "halawi"
    dataset_dir.mkdir()
    rec = dict(RECORDS[0], id="q-110", resolution=0.0)
    (dataset_dir / "one.json").write_text(json.dumps(rec))
    store = CorpusStore(tmp_path / "corpus")
    import_halawi(store, dataset_dir)
    assert store.get_question("halawi:q-110").resolution is False
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_halawi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.halawi'`
- [ ] **Step 3: Write minimal implementation**

Create `backtest/corpus/halawi.py`:
```python
"""Halawi et al. 2024 (arXiv 2402.18563) dataset importer.

Reads LOCAL files only — never the network. Download first (manual step):

    pip install -U "huggingface_hub[cli]"
    huggingface-cli download YuehHanChen/forecasting \
        --repo-type dataset --local-dir data/backtest/raw/halawi

VERIFY THE RECORD SHAPE against the downloaded files before a real import:
this importer targets the curated binary split and tolerates the common
field aliases via _first(); unknown layouts surface as per-record errors,
never crashes. Dataset text is stored VERBATIM with provenance (text may
drift from current platform copies — spec §3.2).

Resolution-date validation runs on every record (resolve < open =>
quarantined): >=3.8% of even the curated set had resolved-before-forecast
defects (spec §3.4).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport


def _first(rec: dict, *names, default=None):
    for n in names:
        if rec.get(n) not in (None, ""):
            return rec[n]
    return default


def _parse_dt(v) -> datetime | None:
    if v in (None, ""):
        return None
    return datetime.fromisoformat(str(v))


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(round(v))
    if isinstance(v, str) and v.strip().lower() in ("yes", "no"):
        return v.strip().lower() == "yes"
    return None


def _iter_records(dataset_dir: Path):
    for path in sorted(Path(dataset_dir).rglob("*.json*")):
        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
            yield from (data if isinstance(data, list) else [data])


def import_halawi(store: CorpusStore, dataset_dir) -> FetchReport:
    report = FetchReport()
    for rec in _iter_records(dataset_dir):
        try:
            title = str(_first(rec, "question", "title", default=""))
            rid = str(_first(rec, "id", "question_id")
                      or hashlib.sha256(title.encode()).hexdigest()[:12])
            resolution = _to_bool(_first(rec, "resolution", "answer"))
            if resolution is None:
                report.errors.append(f"{rid}: no usable resolution")
                continue
            open_time = _parse_dt(_first(
                rec, "date_begin", "begin_date", "open_time"))
            close_time = _parse_dt(_first(
                rec, "date_close", "close_date", "close_time"))
            resolve_time = _parse_dt(_first(
                rec, "date_resolve_at", "resolve_date", "resolution_date"))
            if (open_time and resolve_time
                    and _aware(resolve_time) < _aware(open_time)):
                report.errors.append(
                    f"{rid}: resolved before open — quarantined")
                continue
            store.upsert_question(BacktestQuestion(
                qid=f"halawi:{rid}", platform="halawi", native_id=rid,
                qtype=QType.BINARY, title=title,
                description=str(_first(
                    rec, "background", "description", default="")),
                resolution_criteria=str(_first(
                    rec, "resolution_criteria", default="")),
                open_time=open_time, scheduled_close_time=close_time,
                actual_resolve_time=resolve_time, resolution=resolution,
                provenance={
                    "source": "halawi",
                    "origin_platform": str(_first(
                        rec, "source", "platform", default="unknown")),
                    "license_note": ("research dataset (arXiv 2402.18563);"
                                     " text stored verbatim"),
                    "imported_at":
                        datetime.now(timezone.utc).isoformat()}))
            report.fetched += 1
        except Exception as e:      # per-record isolation
            report.errors.append(f"record skipped: {e}")
    return report
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_halawi.py -v`
Expected: PASS (3 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/corpus/halawi.py tests/backtest/test_corpus_halawi.py && git commit -m "feat(backtest): Halawi dataset importer — local files, verbatim text, resolution-date quarantine" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 13: ForecastBench datasets importer

**Files:**
- Create: `backtest/corpus/forecastbench.py`
- Test: `tests/backtest/test_corpus_forecastbench.py`

Local-only importer. Document for the corpus builder (manual, once):

```bash
git clone https://github.com/forecastingresearch/forecastbench-datasets data/backtest/raw/forecastbench-datasets
```

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_corpus_forecastbench.py`:
```python
import json
from datetime import datetime, timezone

from backtest.corpus.forecastbench import import_forecastbench
from backtest.corpus.store import CorpusStore

QUESTIONS_FILE = {
    "forecast_due_date": "2024-07-21",
    "questions": [
        {"id": "11111", "source": "metaculus",
         "question": "Will Y happen by 2025?",
         "background": "FB background.",
         "resolution_criteria": "Resolves YES if Y.",
         "market_info_open_datetime": "2024-01-01T00:00:00+00:00",
         "market_info_close_datetime": "2024-12-31T00:00:00+00:00",
         "freeze_datetime": "2024-07-21T00:00:00+00:00",
         "freeze_datetime_value": "0.62",
         "combination_of": "N/A",
         "resolution_date": "2025-01-02"},
        # combination questions are derived — skipped
        {"id": ["11111", "22222"], "source": "metaculus",
         "question": "combo", "combination_of": [{"id": "11111"}]},
        # no resolution row -> imported unresolved
        {"id": "33333", "source": "manifold",
         "question": "Open one?", "combination_of": "N/A",
         "market_info_open_datetime": "2024-02-01T00:00:00+00:00",
         "market_info_close_datetime": "2026-12-31T00:00:00+00:00",
         "freeze_datetime": "2024-07-21T00:00:00+00:00",
         "freeze_datetime_value": None},
    ],
}

RESOLUTIONS_FILE = {
    "resolutions": [
        {"id": "11111", "source": "metaculus",
         "resolution_date": "2025-01-02", "resolved_to": 1},
    ],
}


def _write_tree(tmp_path):
    root = tmp_path / "forecastbench-datasets"
    qdir = root / "datasets" / "question_sets"
    rdir = root / "datasets" / "resolution_sets"
    qdir.mkdir(parents=True)
    rdir.mkdir(parents=True)
    (qdir / "2024-07-21-llm.json").write_text(json.dumps(QUESTIONS_FILE))
    (rdir / "2024-07-21_resolution_set.json").write_text(
        json.dumps(RESOLUTIONS_FILE))
    return root


def test_import_forecastbench(tmp_path):
    root = _write_tree(tmp_path)
    store = CorpusStore(tmp_path / "corpus")
    report = import_forecastbench(store, root)
    assert report.fetched == 2            # combo skipped
    q = store.get_question("forecastbench:metaculus:11111")
    assert q.qtype.value == "binary"
    assert q.resolution is True           # joined from resolution_sets
    assert q.title == "Will Y happen by 2025?"
    assert q.provenance["source"] == "forecastbench"
    assert q.provenance["origin_platform"] == "metaculus"
    assert q.provenance["license"] == "CC BY-SA 4.0"
    # freeze value becomes a one-point cp series (crowd value at freeze)
    assert store.value_at(
        "forecastbench:metaculus:11111", "cp",
        datetime(2024, 8, 1, tzinfo=timezone.utc)) == 0.62
    open_q = store.get_question("forecastbench:manifold:33333")
    assert open_q.resolution is None
    assert store.value_at(
        "forecastbench:manifold:33333", "cp",
        datetime(2024, 8, 1, tzinfo=timezone.utc)) is None


def test_import_forecastbench_dedupes_across_nightly_sets(tmp_path):
    root = _write_tree(tmp_path)
    qdir = root / "datasets" / "question_sets"
    (qdir / "2024-07-28-llm.json").write_text(json.dumps(QUESTIONS_FILE))
    store = CorpusStore(tmp_path / "corpus")
    report = import_forecastbench(store, root)
    assert report.fetched == 2            # unique qids, not rows
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_forecastbench.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.forecastbench'`
- [ ] **Step 3: Write minimal implementation**

Create `backtest/corpus/forecastbench.py`:
```python
"""ForecastBench datasets importer (arXiv 2409.19839, CC BY-SA 4.0).

Reads a LOCAL clone only — never the network. Download first (manual):

    git clone https://github.com/forecastingresearch/forecastbench-datasets \
        data/backtest/raw/forecastbench-datasets

Targets the nightly JSON layout: question_sets/*.json holding
{"questions": [...]} and resolution_sets/*.json holding
{"resolutions": [...]} anywhere under the given directory (rglob — the
repo nests them under datasets/). Questions and resolutions join on
(source, id). Combination questions (combination_of != "N/A") are derived
and skipped. freeze_datetime_value is the platform crowd value at freeze —
stored as a one-point "cp" series so CP-echo opponents work on
ForecastBench questions too.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport


def _parse_dt(v) -> datetime | None:
    if v in (None, ""):
        return None
    return datetime.fromisoformat(str(v))


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def import_forecastbench(store: CorpusStore, datasets_dir) -> FetchReport:
    report = FetchReport()
    datasets_dir = Path(datasets_dir)

    resolutions: dict[tuple[str, str], object] = {}
    for path in sorted(datasets_dir.rglob("resolution_sets/*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            report.errors.append(f"{path.name}: {e}")
            continue
        for r in data.get("resolutions") or []:
            resolutions[(str(r.get("source")), str(r.get("id")))] = \
                r.get("resolved_to")

    seen: set[str] = set()
    for path in sorted(datasets_dir.rglob("question_sets/*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            report.errors.append(f"{path.name}: {e}")
            continue
        for rec in data.get("questions") or []:
            try:
                if rec.get("combination_of") not in (None, "N/A", [], ""):
                    continue            # derived combo question
                source = str(rec.get("source"))
                rid = str(rec.get("id"))
                qid = f"forecastbench:{source}:{rid}"
                resolved_to = resolutions.get((source, rid))
                resolution = None
                if resolved_to in (0, 1, 0.0, 1.0):
                    resolution = bool(round(float(resolved_to)))
                open_time = _parse_dt(
                    rec.get("market_info_open_datetime"))
                close_time = _parse_dt(
                    rec.get("market_info_close_datetime"))
                resolve_time = _parse_dt(rec.get("resolution_date"))
                if (open_time and resolve_time
                        and _aware(resolve_time) < _aware(open_time)):
                    report.errors.append(
                        f"{qid}: resolved before open — quarantined")
                    continue
                store.upsert_question(BacktestQuestion(
                    qid=qid, platform="forecastbench",
                    native_id=f"{source}:{rid}", qtype=QType.BINARY,
                    title=str(rec.get("question") or ""),
                    description=str(rec.get("background") or ""),
                    resolution_criteria=str(
                        rec.get("resolution_criteria") or ""),
                    open_time=open_time,
                    scheduled_close_time=close_time,
                    actual_resolve_time=resolve_time,
                    resolution=resolution,
                    provenance={
                        "source": "forecastbench",
                        "origin_platform": source,
                        "license": "CC BY-SA 4.0",
                        "question_set": path.name,
                        "imported_at":
                            datetime.now(timezone.utc).isoformat()}))
                freeze_t = rec.get("freeze_datetime")
                freeze_v = rec.get("freeze_datetime_value")
                if freeze_t and freeze_v not in (None, ""):
                    try:
                        store.put_series(
                            qid, "cp", [(str(freeze_t), float(freeze_v))])
                    except (TypeError, ValueError):
                        pass            # non-numeric freeze values
                if qid not in seen:
                    seen.add(qid)
                    report.fetched += 1
            except Exception as e:      # per-record isolation
                report.errors.append(
                    f"{path.name}:{rec.get('id')}: {e}")
    return report
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_forecastbench.py -v`
Expected: PASS (2 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/corpus/forecastbench.py tests/backtest/test_corpus_forecastbench.py && git commit -m "feat(backtest): ForecastBench importer — question/resolution join, freeze-value cp series, combo skip" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 14: Polymarket fetcher (Gamma enumeration + CLOB price history)

**Files:**
- Create: `backtest/corpus/polymarket.py`
- Test: `tests/backtest/test_corpus_polymarket.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_corpus_polymarket.py`:
```python
from datetime import datetime, timezone

import backtest.corpus.polymarket as polymarket_mod
from backtest.corpus.polymarket import fetch_polymarket
from backtest.corpus.store import CorpusStore

# Gamma /markets?closed=true returns a bare array; outcomes/outcomePrices/
# clobTokenIds are JSON-encoded STRINGS (same quirk bot/markets.py handles)
GAMMA_PAGE = [
    {"id": "516710", "question": "Will Z win the 2026 election?",
     "conditionId": "0xabc", "slug": "will-z-win",
     "description": "Resolves to Yes if Z wins.",
     "outcomes": '["No", "Yes"]',              # reversed on purpose:
     "outcomePrices": '["0", "1"]',            # Yes settled at index 1
     "clobTokenIds": '["1111", "2222"]',
     "startDate": "2026-01-01T00:00:00Z",
     "endDate": "2026-03-01T00:00:00Z",
     "closed": True, "volumeNum": 250000.0},
    # not a Yes/No binary -> skipped
    {"id": "516711", "question": "Up or down market?",
     "outcomes": '["Up", "Down"]', "outcomePrices": '["0.6", "0.4"]',
     "clobTokenIds": '["3333", "4444"]',
     "startDate": "2026-01-01T00:00:00Z",
     "endDate": "2026-03-01T00:00:00Z",
     "closed": True, "volumeNum": 10.0},
    # settled BEFORE open -> quarantined (resolution-date validation,
    # spec §3.4)
    {"id": "516712", "question": "Settled before it opened?",
     "outcomes": '["No", "Yes"]', "outcomePrices": '["0", "1"]',
     "clobTokenIds": '["5555", "6666"]',
     "startDate": "2026-02-01T00:00:00Z",
     "endDate": "2026-03-01T00:00:00Z",
     "closedTime": "2026-01-15T00:00:00Z",     # before startDate
     "closed": True, "volumeNum": 50.0},
]

PRICES_HISTORY = {"history": [
    {"t": 1767225600, "p": 0.30},      # 2026-01-01T00:00:00Z
    {"t": 1769904000, "p": 0.65},      # 2026-02-01T00:00:00Z
]}


async def test_fetch_polymarket_ingests_and_snapshots_prices(
        tmp_path, monkeypatch):
    seen = {"gamma_params": [], "clob_params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        if "gamma-api" in url:
            seen["gamma_params"].append(params)
            return GAMMA_PAGE if params.get("offset", 0) == 0 else []
        if "prices-history" in url:
            seen["clob_params"].append(params)
            return PRICES_HISTORY
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(polymarket_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_polymarket(store)
    assert report.fetched == 1        # Up/Down skipped, 516712 quarantined
    assert any("quarantined polymarket:516712" in e for e in report.errors)
    assert store.get_question("polymarket:516712") is None
    assert seen["gamma_params"][0]["closed"] == "true"

    q = store.get_question("polymarket:516710")
    assert q.resolution is True                # Yes price "1" at index 1
    assert q.qtype.value == "binary"
    assert q.title == "Will Z win the 2026 election?"
    assert q.scheduled_close_time == datetime(2026, 3, 1,
                                              tzinfo=timezone.utc)
    assert q.provenance["condition_id"] == "0xabc"

    # price history requested for the YES token, chosen via the outcomes
    # field (index 1 here), at fidelity >= 720 (resolved markets return
    # empty below ~12h granularity)
    assert seen["clob_params"][0]["market"] == "2222"
    assert seen["clob_params"][0]["fidelity"] == 720
    t = datetime(2026, 1, 15, tzinfo=timezone.utc)
    assert store.value_at("polymarket:516710", "market_price", t) == 0.30
    t2 = datetime(2026, 2, 15, tzinfo=timezone.utc)
    assert store.value_at("polymarket:516710", "market_price", t2) == 0.65


async def test_fetch_polymarket_passes_filters_and_limit(
        tmp_path, monkeypatch):
    seen = {"params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        if "gamma-api" in url:
            seen["params"].append(params)
            return GAMMA_PAGE
        return {"history": []}

    monkeypatch.setattr(polymarket_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_polymarket(store, start_date_min="2026-01-01",
                                    limit=1)
    assert report.fetched == 1                 # limit stops after first
    assert seen["params"][0]["start_date_min"] == "2026-01-01"


async def test_fetch_polymarket_isolates_bad_market(tmp_path, monkeypatch):
    bad = [{"id": "X", "question": "broken", "outcomes": "not json",
            "outcomePrices": "[]", "clobTokenIds": "[]"}] + GAMMA_PAGE[:1]

    async def fake_get(url, params=None, headers=None, client=None):
        if "gamma-api" in url:
            return bad if params.get("offset", 0) == 0 else []
        return PRICES_HISTORY

    monkeypatch.setattr(polymarket_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_polymarket(store)
    assert report.fetched == 1
    assert any("X" in e for e in report.errors)
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_polymarket.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.polymarket'`
- [ ] **Step 3: Write minimal implementation**

Create `backtest/corpus/polymarket.py`:
```python
"""Polymarket fetcher: Gamma /markets enumeration + CLOB prices-history.

No auth needed. Gamma quirk (verified live in bot/markets.py): outcomes /
outcomePrices / clobTokenIds are JSON-encoded STRINGS. Resolved markets
return EMPTY price history below fidelity~720 (12h) — 12h-1d granularity
is the design assumption for price-at-T (spec §3.2).

Snapshot-on-sight: market history is decaying (Polymarket pruning is
undocumented; /orderbook-history already died) — everything fetched is
stored permanently via put_series.

Resolution inference: a settled Yes/No market's outcomePrices collapse to
0/1; >=0.99 on the Yes outcome -> True, <=0.01 -> False, anything between
stays None (closed-but-not-settled; unscoreable).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport

logger = logging.getLogger(__name__)

GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES = "https://clob.polymarket.com/prices-history"
FIDELITY_MIN = 720          # minutes; resolved markets empty below ~12h
PAGE_SIZE = 100
HTTP_TIMEOUT_S = 30.0


async def _get_json(url: str, params: dict | None = None,
                    headers: dict | None = None, client=None):
    import httpx
    if client is not None:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as one_shot:
        resp = await one_shot.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _parse_dt(v) -> datetime | None:
    if v in (None, ""):
        return None
    return datetime.fromisoformat(str(v))


async def _ingest_market(store: CorpusStore, m: dict,
                         report: FetchReport) -> bool:
    outcomes = [str(o).lower() for o in json.loads(m["outcomes"])]
    if sorted(outcomes) != ["no", "yes"]:
        return False                       # only Yes/No binaries
    yes_idx = outcomes.index("yes")        # map index via outcomes field
    prices = [float(p) for p in json.loads(m["outcomePrices"])]
    clob_ids = json.loads(m.get("clobTokenIds") or "[]")

    yes_price = prices[yes_idx]
    if yes_price >= 0.99:
        resolution: bool | None = True
    elif yes_price <= 0.01:
        resolution = False
    else:
        resolution = None                  # closed but not settled 0/1

    qid = f"polymarket:{m['id']}"
    open_time = _parse_dt(m.get("startDate"))
    resolve_time = _parse_dt(m.get("closedTime"))   # Gamma settle timestamp
    # resolution-date validation at import (spec §3.4)
    if open_time and resolve_time and resolve_time < open_time:
        report.errors.append(
            f"quarantined {qid}: resolve_time < open_time")
        return False
    store.upsert_question(BacktestQuestion(
        qid=qid, platform="polymarket", native_id=str(m["id"]),
        qtype=QType.BINARY, title=m.get("question") or "",
        description=m.get("description") or "",
        open_time=open_time,
        scheduled_close_time=_parse_dt(m.get("endDate")),
        resolution=resolution,
        provenance={
            "source": "polymarket",
            "condition_id": m.get("conditionId"),
            "slug": m.get("slug"),
            "note": "resolution inferred from settled outcomePrices",
            "fetched_at": datetime.now(timezone.utc).isoformat()}))

    if len(clob_ids) > yes_idx:            # snapshot-on-sight
        hist = await _get_json(CLOB_PRICES, params={
            "market": clob_ids[yes_idx], "fidelity": FIDELITY_MIN})
        pts = [(datetime.fromtimestamp(h["t"], tz=timezone.utc).isoformat(),
                float(h["p"]))
               for h in (hist.get("history") or [])]
        if pts:
            store.put_series(qid, "market_price", pts)
    report.fetched += 1
    return True


async def fetch_polymarket(store: CorpusStore, *, closed=True,
                           start_date_min=None,
                           limit=None) -> FetchReport:
    report = FetchReport()
    offset = 0
    while True:
        params: dict = {"closed": str(bool(closed)).lower(),
                        "limit": PAGE_SIZE, "offset": offset}
        if start_date_min is not None:
            params["start_date_min"] = start_date_min
        try:
            page = await _get_json(GAMMA_MARKETS, params=params)
        except Exception as e:
            report.errors.append(f"gamma page offset={offset}: {e}")
            break
        for m in page or []:
            if limit is not None and report.fetched >= limit:
                return report
            try:
                await _ingest_market(store, m, report)
            except Exception as e:         # per-market isolation
                logger.warning("polymarket %s failed: %s", m.get("id"), e)
                report.errors.append(f"market {m.get('id')}: {e}")
        if not page or len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return report
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_polymarket.py -v`
Expected: PASS (3 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/corpus/polymarket.py tests/backtest/test_corpus_polymarket.py && git commit -m "feat(backtest): Polymarket Gamma+CLOB fetcher — snapshot-on-sight price history at fidelity 720" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 15: Kalshi fetcher (settled markets + daily candlesticks)

**Files:**
- Create: `backtest/corpus/kalshi.py`
- Test: `tests/backtest/test_corpus_kalshi.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_corpus_kalshi.py`:
```python
from datetime import datetime, timezone

import backtest.corpus.kalshi as kalshi_mod
from backtest.corpus.kalshi import fetch_kalshi
from backtest.corpus.store import CorpusStore

MARKETS_PAGE_1 = {
    "cursor": "abc",
    "markets": [{
        "ticker": "FED-26JUN-T3.00",
        "event_ticker": "FED-26JUN",
        "title": "Fed funds above 3.00 after June 2026 meeting?",
        "rules_primary":
            "Resolves Yes if the target range floor exceeds 3.00.",
        "open_time": "2026-01-02T00:00:00Z",
        "close_time": "2026-06-18T00:00:00Z",
        "result": "no",
        "status": "settled",
    }, {
        # settled BEFORE open -> quarantined (resolution-date validation,
        # spec §3.4)
        "ticker": "BAD-26JAN-T1",
        "event_ticker": "BAD-26JAN",
        "title": "Settled before it opened?",
        "open_time": "2026-02-01T00:00:00Z",
        "close_time": "2026-03-01T00:00:00Z",
        "settled_time": "2026-01-15T00:00:00Z",    # before open_time
        "result": "yes",
        "status": "settled",
    }],
}
MARKETS_PAGE_2 = {"cursor": "", "markets": []}

CANDLES = {"candlesticks": [
    # prices in CENTS; price.close may be null on no-trade days ->
    # fall back to yes_bid/yes_ask mid
    {"end_period_ts": 1767312000,                  # 2026-01-02
     "price": {"close": 42},
     "yes_bid": {"close": 41}, "yes_ask": {"close": 43}},
    {"end_period_ts": 1769904000,                  # 2026-02-01
     "price": {"close": None},
     "yes_bid": {"close": 30}, "yes_ask": {"close": 34}},
]}


async def test_fetch_kalshi_ingests_settled_market(tmp_path, monkeypatch):
    seen = {"urls": [], "params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        seen["urls"].append(url)
        seen["params"].append(params)
        if url.endswith("/markets"):
            return (MARKETS_PAGE_1 if not params.get("cursor")
                    else MARKETS_PAGE_2)
        if "/candlesticks" in url:
            return CANDLES
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(kalshi_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_kalshi(store)
    assert report.fetched == 1                 # BAD-26JAN-T1 quarantined
    assert any("quarantined kalshi:BAD-26JAN-T1" in e
               for e in report.errors)
    assert seen["params"][0]["status"] == "settled"

    q = store.get_question("kalshi:FED-26JUN-T3.00")
    assert q.resolution is False               # result: "no"
    assert q.qtype.value == "binary"
    assert q.resolution_criteria.startswith("Resolves Yes")
    assert q.scheduled_close_time == datetime(2026, 6, 18,
                                              tzinfo=timezone.utc)

    candle_url = next(u for u in seen["urls"] if "/candlesticks" in u)
    # series ticker derived from event ticker prefix
    assert ("/series/FED/markets/FED-26JUN-T3.00/candlesticks"
            in candle_url)
    cp = seen["params"][seen["urls"].index(candle_url)]
    assert cp["period_interval"] == 1440
    assert cp["include_latest_before_start"] == "true"

    t = datetime(2026, 1, 15, tzinfo=timezone.utc)
    assert store.value_at("kalshi:FED-26JUN-T3.00",
                          "market_price", t) == 0.42
    t2 = datetime(2026, 2, 15, tzinfo=timezone.utc)
    assert store.value_at("kalshi:FED-26JUN-T3.00",
                          "market_price", t2) == 0.32   # bid/ask mid


async def test_fetch_kalshi_passes_min_settled_ts_and_limit(
        tmp_path, monkeypatch):
    seen = {"params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/markets"):
            seen["params"].append(params)
            return MARKETS_PAGE_1
        return {"candlesticks": []}

    monkeypatch.setattr(kalshi_mod, "_get_json", fake_get)
    report = await fetch_kalshi(CorpusStore(tmp_path),
                                min_settled_ts=1767225600, limit=1)
    assert report.fetched == 1
    assert seen["params"][0]["min_settled_ts"] == 1767225600


async def test_fetch_kalshi_isolates_candle_failure(tmp_path, monkeypatch):
    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/markets"):
            return (MARKETS_PAGE_1 if not params.get("cursor")
                    else MARKETS_PAGE_2)
        raise RuntimeError("candles 404 — outside rolling window")

    monkeypatch.setattr(kalshi_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_kalshi(store)
    # question still ingested; only the series failed
    assert report.fetched == 1
    assert store.get_question("kalshi:FED-26JUN-T3.00") is not None
    assert any("candlesticks" in e for e in report.errors)
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_kalshi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.kalshi'`
- [ ] **Step 3: Write minimal implementation**

Create `backtest/corpus/kalshi.py`:
```python
"""Kalshi fetcher: settled markets + daily candlesticks.

GET /markets?status=settled (cursor pagination), then per market
GET /series/{series}/markets/{ticker}/candlesticks with
period_interval=1440 (daily) and include_latest_before_start=true.
Candlestick prices are in CENTS (1-99) -> /100 to probability;
price.close may be null on no-trade days -> yes_bid/yes_ask mid fallback.

ROLLING WINDOW WARNING: the live candlesticks API keeps only ~3 months of
history (GET /historical/cutoff reports the boundary); older series must
come from the /historical/* archive endpoints — an explicit post-MVP
fallback seam. Snapshot-on-sight (spec §3.2) makes every successful fetch
permanent locally, which is the primary mitigation.

The markets payload does not carry series_ticker; it is derived from the
event ticker prefix (event "FED-26JUN" -> series "FED") — verify against
live payloads when the archive fallback is built.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
PAGE_SIZE = 200
HTTP_TIMEOUT_S = 30.0


async def _get_json(url: str, params: dict | None = None,
                    headers: dict | None = None, client=None):
    import httpx
    if client is not None:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as one_shot:
        resp = await one_shot.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _parse_dt(v) -> datetime | None:
    if v in (None, ""):
        return None
    return datetime.fromisoformat(str(v))


def _epoch(dt: datetime | None) -> int | None:
    return int(dt.timestamp()) if dt else None


def _candle_prob(c: dict) -> float | None:
    price = (c.get("price") or {}).get("close")
    if price is not None:
        return float(price) / 100.0
    bid = (c.get("yes_bid") or {}).get("close")
    ask = (c.get("yes_ask") or {}).get("close")
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0 / 100.0
    return None


async def _ingest_market(store: CorpusStore, m: dict,
                         report: FetchReport) -> None:
    ticker = m["ticker"]
    qid = f"kalshi:{ticker}"
    result = (m.get("result") or "").lower()
    resolution = {"yes": True, "no": False}.get(result)
    open_time = _parse_dt(m.get("open_time"))
    close_time = _parse_dt(m.get("close_time"))
    settled_time = _parse_dt(m.get("settled_time"))
    # resolution-date validation at import (spec §3.4)
    if open_time and settled_time and settled_time < open_time:
        report.errors.append(
            f"quarantined {qid}: resolve_time < open_time")
        return
    store.upsert_question(BacktestQuestion(
        qid=qid, platform="kalshi", native_id=ticker,
        qtype=QType.BINARY,
        title=m.get("title") or "",
        resolution_criteria=m.get("rules_primary") or "",
        open_time=open_time, scheduled_close_time=close_time,
        resolution=resolution,
        provenance={
            "source": "kalshi",
            "event_ticker": m.get("event_ticker"),
            "fetched_at": datetime.now(timezone.utc).isoformat()}))
    report.fetched += 1

    series_ticker = (m.get("series_ticker")
                     or str(m.get("event_ticker") or ticker).split("-")[0])
    try:
        candles = await _get_json(
            f"{KALSHI_BASE}/series/{series_ticker}/markets/{ticker}"
            "/candlesticks",
            params={"start_ts": _epoch(open_time) or 0,
                    "end_ts": _epoch(close_time) or 0,
                    "period_interval": 1440,
                    "include_latest_before_start": "true"})
    except Exception as e:    # series fetch fails; question already stored
        logger.warning("kalshi candlesticks %s failed: %s", ticker, e)
        report.errors.append(f"candlesticks {ticker}: {e}")
        return
    pts = []
    for c in candles.get("candlesticks") or []:
        prob = _candle_prob(c)
        ts = c.get("end_period_ts")
        if prob is None or ts is None:
            continue
        pts.append((datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    prob))
    if pts:
        store.put_series(qid, "market_price", pts)   # snapshot-on-sight


async def fetch_kalshi(store: CorpusStore, *, status="settled",
                       min_settled_ts=None, limit=None) -> FetchReport:
    report = FetchReport()
    cursor = ""
    while True:
        params: dict = {"status": status, "limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        if min_settled_ts is not None:
            params["min_settled_ts"] = min_settled_ts
        try:
            page = await _get_json(f"{KALSHI_BASE}/markets", params=params)
        except Exception as e:
            report.errors.append(f"markets page: {e}")
            break
        for m in page.get("markets") or []:
            if limit is not None and report.fetched >= limit:
                return report
            try:
                await _ingest_market(store, m, report)
            except Exception as e:     # per-market isolation
                logger.warning("kalshi %s failed: %s", m.get("ticker"), e)
                report.errors.append(f"market {m.get('ticker')}: {e}")
        cursor = page.get("cursor") or ""
        if not cursor:
            break
    return report
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_kalshi.py -v`
Expected: PASS (3 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/corpus/kalshi.py tests/backtest/test_corpus_kalshi.py && git commit -m "feat(backtest): Kalshi settled-markets fetcher — daily candlesticks, cents->prob, rolling-window seam" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 16: Manifold fetcher (bet-chain price reconstruction)

**Files:**
- Create: `backtest/corpus/manifold.py`
- Test: `tests/backtest/test_corpus_manifold.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_corpus_manifold.py`:
```python
from datetime import datetime, timezone

import backtest.corpus.manifold as manifold_mod
from backtest.corpus.manifold import fetch_manifold
from backtest.corpus.store import CorpusStore

# /v0/markets returns a bare array of LiteMarkets; times are epoch MS
MARKETS_PAGE = [
    {"id": "abc123", "question": "Will W resign in 2026?",
     "outcomeType": "BINARY", "isResolved": True, "resolution": "YES",
     "createdTime": 1767225600000,             # 2026-01-01
     "closeTime": 1772323200000,               # 2026-03-01
     "url": "https://manifold.markets/u/will-w-resign",
     "volume": 1234.5},
    # non-binary -> skipped
    {"id": "mc999", "question": "Which team wins?",
     "outcomeType": "MULTIPLE_CHOICE", "isResolved": True,
     "resolution": "a1", "createdTime": 1767225600000,
     "closeTime": 1772323200000},
    # resolved MKT/CANCEL -> not scoreable as binary -> skipped
    {"id": "cancel1", "question": "Cancelled?", "outcomeType": "BINARY",
     "isResolved": True, "resolution": "CANCEL",
     "createdTime": 1767225600000, "closeTime": 1772323200000},
    # resolved BEFORE creation -> quarantined (resolution-date validation,
    # spec §3.4)
    {"id": "bad1", "question": "Resolved before created?",
     "outcomeType": "BINARY", "isResolved": True, "resolution": "YES",
     "createdTime": 1769904000000,             # 2026-02-01
     "resolutionTime": 1767225600000,          # 2026-01-01, before created
     "closeTime": 1772323200000},
]

# /v0/bets returns newest-first; redemption bets carry no price signal
BETS = [
    {"createdTime": 1769904000000,             # 2026-02-01
     "probBefore": 0.35, "probAfter": 0.62, "isRedemption": False},
    {"createdTime": 1768608000000,             # 2026-01-17
     "probBefore": 0.50, "probAfter": 0.50, "isRedemption": True},
    {"createdTime": 1767916800000,             # 2026-01-09
     "probBefore": 0.40, "probAfter": 0.35, "isRedemption": False},
]


async def test_fetch_manifold_bet_chain_step_series(tmp_path, monkeypatch):
    seen = {"bet_params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/v0/markets"):
            return MARKETS_PAGE if not params.get("before") else []
        if url.endswith("/v0/bets"):
            seen["bet_params"].append(params)
            return BETS
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(manifold_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_manifold(store)
    assert report.fetched == 1     # MC + CANCEL skipped, bad1 quarantined
    assert any("quarantined manifold:bad1" in e for e in report.errors)
    assert seen["bet_params"][0]["contractId"] == "abc123"

    q = store.get_question("manifold:abc123")
    assert q.resolution is True
    assert q.qtype.value == "binary"
    assert q.scheduled_close_time == datetime(2026, 3, 1,
                                              tzinfo=timezone.utc)
    assert q.provenance["play_money"] is True  # caveat recorded

    # step function: market creation seeds earliest probBefore, then each
    # non-redemption bet contributes (createdTime, probAfter)
    at = lambda *a: store.value_at("manifold:abc123", "market_price",
                                   datetime(*a, tzinfo=timezone.utc))
    assert at(2026, 1, 5) == 0.40              # before first bet
    assert at(2026, 1, 10) == 0.35             # after first bet
    assert at(2026, 1, 20) == 0.35             # redemption bet EXCLUDED
    assert at(2026, 2, 2) == 0.62              # after last bet


async def test_fetch_manifold_limit_and_bets_failure_isolated(
        tmp_path, monkeypatch):
    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/v0/markets"):
            return MARKETS_PAGE if not params.get("before") else []
        raise RuntimeError("bets endpoint down")

    monkeypatch.setattr(manifold_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_manifold(store, limit=1)
    # question stored even though its bet series failed
    assert report.fetched == 1
    assert store.get_question("manifold:abc123") is not None
    assert any("bets" in e for e in report.errors)
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_manifold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.manifold'`
- [ ] **Step 3: Write minimal implementation**

Create `backtest/corpus/manifold.py`:
```python
"""Manifold fetcher: /v0/markets enumeration + /v0/bets chain.

Price-at-T is reconstructed EXACTLY from the bet chain: each
non-redemption bet's probAfter becomes a step point at its createdTime,
seeded with the earliest bet's probBefore at market creation. Redemption
bets are EXCLUDED — they move no price (spec §3.2). Times are epoch ms.

Play-money caveat recorded in provenance (provenance["play_money"]) so
reports can stratify market evidence quality. MC markets need
answerId-filtered bets — out of scope here (binary only), explicit seam.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport

logger = logging.getLogger(__name__)

MANIFOLD_BASE = "https://api.manifold.markets/v0"
PAGE_SIZE = 500
BETS_LIMIT = 1000     # single page; >1000-bet markets are a seam (beforeTime)
HTTP_TIMEOUT_S = 30.0


async def _get_json(url: str, params: dict | None = None,
                    headers: dict | None = None, client=None):
    import httpx
    if client is not None:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as one_shot:
        resp = await one_shot.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _ms(v) -> datetime | None:
    if v is None:
        return None
    return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)


async def _ingest_market(store: CorpusStore, m: dict,
                         report: FetchReport) -> None:
    if m.get("outcomeType") != "BINARY" or not m.get("isResolved"):
        return
    resolution = {"YES": True, "NO": False}.get(
        str(m.get("resolution", "")).upper())
    if resolution is None:
        return                              # MKT / CANCEL: not scoreable
    qid = f"manifold:{m['id']}"
    open_time = _ms(m.get("createdTime"))
    resolve_time = _ms(m.get("resolutionTime"))
    # resolution-date validation at import (spec §3.4)
    if open_time and resolve_time and resolve_time < open_time:
        report.errors.append(
            f"quarantined {qid}: resolve_time < open_time")
        return
    store.upsert_question(BacktestQuestion(
        qid=qid, platform="manifold", native_id=str(m["id"]),
        qtype=QType.BINARY, title=m.get("question") or "",
        open_time=open_time,
        scheduled_close_time=_ms(m.get("closeTime")),
        resolution=resolution,
        provenance={
            "source": "manifold", "play_money": True,
            "url": m.get("url"),
            "fetched_at": datetime.now(timezone.utc).isoformat()}))
    report.fetched += 1

    try:
        bets = await _get_json(f"{MANIFOLD_BASE}/bets", params={
            "contractId": m["id"], "limit": BETS_LIMIT})
    except Exception as e:    # question already stored; series failed
        logger.warning("manifold bets %s failed: %s", m.get("id"), e)
        report.errors.append(f"bets {m.get('id')}: {e}")
        return
    chain = sorted(
        (b for b in (bets or [])
         if not b.get("isRedemption")
         and b.get("probAfter") is not None),
        key=lambda b: b["createdTime"])
    if not chain:
        return
    pts = [(_ms(m.get("createdTime")).isoformat(),
            float(chain[0]["probBefore"]))]
    pts += [(_ms(b["createdTime"]).isoformat(), float(b["probAfter"]))
            for b in chain]
    store.put_series(qid, "market_price", pts)   # snapshot-on-sight


async def fetch_manifold(store: CorpusStore, *, limit=None) -> FetchReport:
    report = FetchReport()
    before = None
    while True:
        params: dict = {"limit": PAGE_SIZE}
        if before:
            params["before"] = before
        try:
            page = await _get_json(f"{MANIFOLD_BASE}/markets",
                                   params=params)
        except Exception as e:
            report.errors.append(f"markets page: {e}")
            break
        if not page:
            break
        for m in page:
            if limit is not None and report.fetched >= limit:
                return report
            try:
                await _ingest_market(store, m, report)
            except Exception as e:     # per-market isolation
                logger.warning("manifold %s failed: %s", m.get("id"), e)
                report.errors.append(f"market {m.get('id')}: {e}")
        if len(page) < PAGE_SIZE:
            break
        before = page[-1]["id"]
    return report
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_manifold.py -v`
Expected: PASS (2 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/corpus/manifold.py tests/backtest/test_corpus_manifold.py && git commit -m "feat(backtest): Manifold fetcher — bet-chain step series, redemption bets excluded" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 17: Crosswalk (lexical top-k + LLM gate) and forward snapshot accumulator

**Files:**
- Create: `backtest/corpus/crosswalk.py`
- Create: `backtest/corpus/snapshot.py`
- Test: `tests/backtest/test_corpus_crosswalk.py`
- Test: `tests/backtest/test_corpus_snapshot.py`

- [ ] **Step 1: Write the failing crosswalk test**

Create `tests/backtest/test_corpus_crosswalk.py`:
```python
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.crosswalk import build_crosswalk, market_for
from backtest.corpus.store import CorpusStore


def _q(qid, platform, title, criteria, close=None):
    return BacktestQuestion(
        qid=qid, platform=platform, native_id=qid.split(":", 1)[1],
        qtype=QType.BINARY, title=title, resolution_criteria=criteria,
        scheduled_close_time=close, resolution=True)


def _seed(store):
    store.upsert_question(_q(
        "metaculus:1", "metaculus",
        "Will the Fed cut rates at the June 2026 meeting?",
        "Resolves YES if the FOMC lowers the target range in June 2026.",
        close=datetime(2026, 6, 18, tzinfo=timezone.utc)))
    store.upsert_question(_q(
        "polymarket:9", "polymarket",
        "Fed rate cut at June 2026 meeting?",
        "Resolves Yes if the FOMC cuts rates at its June 2026 meeting.",
        close=datetime(2026, 6, 18, tzinfo=timezone.utc)))
    store.upsert_question(_q(
        "kalshi:Z", "kalshi",
        "Will it rain in Seattle tomorrow?", "NWS gauge reading."))


async def test_build_crosswalk_lexical_then_llm_gate(tmp_path):
    store = CorpusStore(tmp_path)
    _seed(store)
    prompts = []

    async def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "YES" if "Fed rate cut" in prompt else "NO"

    n = await build_crosswalk(store, llm, top_k=2)
    assert n == 1
    assert market_for(store, "metaculus:1") == "polymarket:9"
    assert market_for(store, "kalshi:Z") is None   # market side: no row
    # the gate saw resolution criteria AND dates of BOTH questions
    assert "FOMC lowers the target range" in prompts[0]
    assert "FOMC cuts rates" in prompts[0]
    assert "2026-06-18" in prompts[0]
    # lexical ranking put the near-duplicate first: one call was enough
    assert len(prompts) == 1


async def test_build_crosswalk_caches_verdicts(tmp_path):
    store = CorpusStore(tmp_path)
    _seed(store)

    async def llm(prompt: str) -> str:
        return "YES" if "Fed rate cut" in prompt else "NO"

    await build_crosswalk(store, llm, top_k=2)

    async def exploding_llm(prompt: str) -> str:
        raise AssertionError("LLM must not be called on cached pairs")

    n2 = await build_crosswalk(store, exploding_llm, top_k=2)
    assert n2 == 0
    assert market_for(store, "metaculus:1") == "polymarket:9"


async def test_rejected_candidates_are_cached_too(tmp_path):
    store = CorpusStore(tmp_path)
    _seed(store)

    async def llm_no(prompt: str) -> str:
        return "NO — dates differ."

    assert await build_crosswalk(store, llm_no, top_k=2) == 0
    assert market_for(store, "metaculus:1") is None
    calls = []

    async def llm_count(prompt: str) -> str:
        calls.append(prompt)
        return "NO"

    await build_crosswalk(store, llm_count, top_k=2)
    assert calls == []                  # NO verdicts cached as well


def test_manual_override_wins(tmp_path):
    store = CorpusStore(tmp_path)
    store.conn.execute(
        "INSERT INTO crosswalk"
        " (qid, market_qid, score, verified, manual_override)"
        " VALUES (?,?,?,?,?)",
        ("metaculus:1", "polymarket:9", 0.9, 1, "kalshi:OVERRIDE"))
    store.conn.commit()
    assert market_for(store, "metaculus:1") == "kalshi:OVERRIDE"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_crosswalk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.crosswalk'`
- [ ] **Step 3: Write minimal crosswalk implementation**

Create `backtest/corpus/crosswalk.py`:
```python
"""Question <-> market crosswalk (spec §3.3).

Candidate generation is LEXICAL: difflib title ratio + token Jaccard over
title+resolution_criteria, top_k candidates per question.
EMBEDDING SEAM: replace _lexical_score/_candidates with an embedding
index (top-20 NN recall >99.9% of true equivalents, arXiv 2601.01706)
without touching the LLM gate, the table schema, or market_for() — kept
lexical in the MVP to avoid heavy dependencies.

Verification: an injected async LLM callable gates each candidate on
resolution criteria AND dates — date mismatch is the dominant false
positive (arXiv 2512.02436). Verdicts (YES and NO) are cached in the
crosswalk table; re-runs never re-ask. A non-null manual_override column
always wins in market_for().

Note for reports: live bot market evidence is keyword search
(bot/markets.py market_research()); backtest market evidence is this
crosswalk — different in kind, stated in every report's comparability
section.
"""
from __future__ import annotations

import difflib
import re

from backtest.core import BacktestQuestion
from backtest.corpus.store import CorpusStore

MARKET_PLATFORMS = ("polymarket", "kalshi", "manifold")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_score(a: BacktestQuestion, b: BacktestQuestion) -> float:
    ratio = difflib.SequenceMatcher(
        None, a.title.lower(), b.title.lower()).ratio()
    ta = _tokens(a.title + " " + a.resolution_criteria)
    tb = _tokens(b.title + " " + b.resolution_criteria)
    union = ta | tb
    jacc = len(ta & tb) / len(union) if union else 0.0
    return 0.5 * ratio + 0.5 * jacc


def _fmt(q: BacktestQuestion) -> str:
    close = (q.scheduled_close_time.date().isoformat()
             if q.scheduled_close_time else "unknown")
    return (f"TITLE: {q.title}\n"
            f"RESOLUTION CRITERIA: {q.resolution_criteria}\n"
            f"CLOSE DATE: {close}")


def _verify_prompt(q: BacktestQuestion, m: BacktestQuestion) -> str:
    return (
        "You are verifying whether a forecasting question and a prediction"
        " market are equivalent. Answer YES only if BOTH hold:\n"
        "1. The resolution criteria describe the same underlying event"
        " with the same resolution conditions.\n"
        "2. The dates match: same resolution/close timeframe. A date"
        " mismatch means NO even when the topic is identical.\n\n"
        f"QUESTION:\n{_fmt(q)}\n\nMARKET:\n{_fmt(m)}\n\n"
        "Answer with YES or NO on the first line.")


async def build_crosswalk(store: CorpusStore, llm, *,
                          top_k: int = 50) -> int:
    """llm: async (prompt: str) -> str. Returns count of NEW verified
    matches. Verdict cache: any (qid, market_qid) row is final."""
    questions = store.query(platforms=["metaculus"], resolved_only=False)
    markets: list[BacktestQuestion] = []
    for p in MARKET_PLATFORMS:
        markets.extend(store.query(platforms=[p], resolved_only=False))
    cached = {(r["qid"], r["market_qid"]) for r in store.conn.execute(
        "SELECT qid, market_qid FROM crosswalk").fetchall()}
    matched = {r["qid"] for r in store.conn.execute(
        "SELECT qid FROM crosswalk WHERE verified = 1").fetchall()}

    new_matches = 0
    for q in questions:
        if q.qid in matched:
            continue                       # already has a verified market
        candidates = sorted(
            ((_lexical_score(q, m), m) for m in markets),
            key=lambda x: x[0], reverse=True)[:top_k]
        for score, m in candidates:
            if (q.qid, m.qid) in cached:
                continue                   # cached NO verdict is final
            answer = await llm(_verify_prompt(q, m))
            verified = answer.strip().upper().startswith("YES")
            store.conn.execute(
                "INSERT OR REPLACE INTO crosswalk"
                " (qid, market_qid, score, verified, manual_override)"
                " VALUES (?,?,?,?,NULL)",
                (q.qid, m.qid, score, int(verified)))
            store.conn.commit()
            cached.add((q.qid, m.qid))
            if verified:
                new_matches += 1
                break                      # one verified market suffices
    return new_matches


def market_for(store: CorpusStore, qid: str) -> str | None:
    """Best market qid for a question. manual_override wins."""
    row = store.conn.execute(
        "SELECT manual_override FROM crosswalk"
        " WHERE qid = ? AND manual_override IS NOT NULL", (qid,)).fetchone()
    if row:
        return row["manual_override"]
    row = store.conn.execute(
        "SELECT market_qid FROM crosswalk"
        " WHERE qid = ? AND verified = 1"
        " ORDER BY score DESC LIMIT 1", (qid,)).fetchone()
    return row["market_qid"] if row else None
```
- [ ] **Step 4: Run crosswalk test to verify it passes**
Run: `pytest tests/backtest/test_corpus_crosswalk.py -v`
Expected: PASS (4 tests)
- [ ] **Step 5: Commit crosswalk**
```bash
git add backtest/corpus/crosswalk.py tests/backtest/test_corpus_crosswalk.py && git commit -m "feat(backtest): question-market crosswalk — lexical top-k, LLM gate on criteria AND dates, cached verdicts, manual override" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
- [ ] **Step 6: Write the failing snapshot test**

Create `tests/backtest/test_corpus_snapshot.py`:
```python
import json

import backtest.corpus.metaculus as metaculus_mod
from backtest.corpus.snapshot import snapshot_questions
from backtest.corpus.store import CorpusStore

SNAP_DETAIL = {
    "id": 31678,
    "title": "Will X happen before 2027?",
    "status": "resolved",
    "question": {
        "type": "binary",
        "description": "Background paragraph.",
        "resolution_criteria": "Resolves YES if X occurs before 2027.",
        "open_time": "2026-04-01T00:00:00Z",
        "scheduled_close_time": "2026-06-01T00:00:00Z",
        "actual_resolve_time": "2026-06-02T12:00:00Z",
        "resolution": "yes",
        "aggregations": {"recency_weighted": {"history": [
            {"start_time": 1775000000.0, "end_time": None,
             "centers": [0.55]},
        ]}},
    },
}


def test_snapshot_archives_raw_json_and_ingests(tmp_path, monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "tok")

    async def fake_throttle():
        pass

    async def fake_get(url, params=None, headers=None, client=None):
        assert headers == {"Authorization": "Token tok"}
        assert url.endswith("/posts/31678/")
        return SNAP_DETAIL

    monkeypatch.setattr(metaculus_mod, "_throttle", fake_throttle)
    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = snapshot_questions(store, [31678])
    assert report.fetched == 1
    # ingested through the shared metaculus mapping
    q = store.get_question("metaculus:31678")
    assert q is not None and q.resolution is True
    assert q.provenance["snapshot"] is True
    # raw payload archived at full fidelity, permanently
    blobs = list((tmp_path / "blobs" / "snapshots")
                 .glob("metaculus_31678_*.json"))
    assert len(blobs) == 1
    assert json.loads(blobs[0].read_text())["id"] == 31678


def test_snapshot_isolates_per_post_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "tok")

    async def fake_throttle():
        pass

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/31678/"):
            return SNAP_DETAIL
        raise RuntimeError("403")

    monkeypatch.setattr(metaculus_mod, "_throttle", fake_throttle)
    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = snapshot_questions(store, [31678, 99999])
    assert report.fetched == 1
    assert any("99999" in e for e in report.errors)
```
- [ ] **Step 7: Run test to verify it fails**
Run: `pytest tests/backtest/test_corpus_snapshot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.corpus.snapshot'`
- [ ] **Step 8: Write minimal snapshot implementation**

Create `backtest/corpus/snapshot.py`:
```python
"""Forward snapshot accumulator (spec §3.2 item 4 / extension 1).

Every question this bot's token ever forecast stays fully visible to it
forever — so archiving every live-touched question at full fidelity turns
the live bot into a corpus-building machine and decays the post-lockdown
corpus problem over months.

Reuses backtest/corpus/metaculus.py internals (auth, throttle, _get_json,
_ingest_detail) via module attributes — monkeypatching metaculus's
attributes in tests covers this module too. Stores the RAW detail JSON
under blobs/snapshots/ in addition to the normalized store row.

Sync entry point (contract): wraps asyncio.run, so the Phase-5 hook at the
end of scripts/run_tournament.py can call it from synchronous teardown.
Do not call from inside a running event loop.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone

from backtest.corpus import metaculus as metaculus_mod
from backtest.corpus.store import CorpusStore, FetchReport


def snapshot_questions(store: CorpusStore,
                       post_ids: list[int]) -> FetchReport:
    return asyncio.run(_snapshot_async(store, post_ids))


async def _snapshot_async(store: CorpusStore,
                          post_ids: list[int]) -> FetchReport:
    headers = metaculus_mod._auth_headers()
    report = FetchReport()
    snap_dir = store.blobs / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for post_id in post_ids:
        await metaculus_mod._throttle()
        try:
            detail = await metaculus_mod._get_json(
                f"{metaculus_mod.API_BASE}/posts/{post_id}/",
                headers=headers)
        except Exception as e:      # per-question isolation
            report.errors.append(f"post {post_id}: {e}")
            continue
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (snap_dir / f"metaculus_{post_id}_{stamp}.json").write_text(
            json.dumps(detail))
        qid = metaculus_mod._ingest_detail(store, detail, report)
        if qid:
            q = store.get_question(qid)
            store.upsert_question(replace(
                q, provenance=dict(q.provenance, snapshot=True)))
    return report
```
- [ ] **Step 9: Run test to verify it passes**
Run: `pytest tests/backtest/test_corpus_snapshot.py -v`
Expected: PASS (2 tests)
- [ ] **Step 10: Run the full Phase 2 suite**
Run: `pytest tests/backtest/ -v`
Expected: PASS (all Phase 2 tests; no network access anywhere)
- [ ] **Step 11: Commit snapshot**
```bash
git add backtest/corpus/snapshot.py tests/backtest/test_corpus_snapshot.py && git commit -m "feat(backtest): forward snapshot accumulator — raw-JSON archive + ingest of live-touched questions" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
## Phase 3 — Time travel & bot as_of threading (Tasks 18–24)

**Prerequisites:** Phases 1–2 complete: `backtest/core.py` (BacktestQuestion, Forecast, StageRecord, LedgerEntry, QType), `backtest/corpus/store.py` (CorpusStore with `value_at`), and `backtest/corpus/crosswalk.py` (`market_for`) exist and their tests pass.

**Conventions (binding for every task below):** `asyncio_mode = "auto"` is set in `pyproject.toml`, so async tests are plain `async def` functions — no decorators. Tests never touch the network: every HTTP/SDK/LLM call is monkeypatched at a module-attribute seam, exactly like `tests/test_research.py` patches `bot.research._asknews_research` and `tests/test_framework_adapter.py` patches `CalibratedEnsembleBot._invoke_model`. The AskNews SDK may not be installed in the test environment, so all SDK imports are lazy (inside functions), mirroring `bot/research.py`.

### Task 18: BotConfig.as_of + threading through prompts and day math (the ONLY bot/ changes)

This task is the entire bot/ surface of the backtesting project. Everything after it is an additive sidecar. With `as_of=None` every output must be bit-identical to current behavior; `config_hash` must ignore `as_of` entirely so live hashes are unchanged.

**Files:**
- Modify: `bot/config.py` (imports; `BotConfig` field; `config_hash`)
- Modify: `bot/ensemble_bot.py:56-61` (`_days_remaining`), `bot/ensemble_bot.py:125-130,144-150,162-170` (three prompt-format call sites)
- Modify: `bot/research.py:34-40` (`_prompt_args`) and its caller `_search_llm_research`
- Test: `tests/backtest/test_asof_threading.py`

- [ ] **Step 1: Ensure the test package directory exists**
```bash
mkdir -p tests/backtest && touch tests/backtest/__init__.py
```
- [ ] **Step 2: Write the failing test**

Create `tests/backtest/test_asof_threading.py`:
```python
"""as_of threading: with as_of=None the bot's prompt strings and day math are
bit-identical to pre-change behavior; with as_of set, 'today' and
days-remaining move to T. config_hash must ignore as_of entirely (live
hashes unchanged; backtest artifacts share the live hash)."""
from datetime import date, datetime, timezone

import pytest

from bot.config import BotConfig, EnsembleMember, config_hash
from bot.ensemble_bot import CalibratedEnsembleBot, _days_remaining
from bot.prompts import BINARY_PROMPT
from bot.research import _prompt_args

QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F")
AS_OF = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)


def test_config_hash_ignores_as_of():
    assert config_hash(BotConfig()) == config_hash(BotConfig(as_of=None))
    assert config_hash(BotConfig(as_of=AS_OF)) == config_hash(BotConfig())


def test_config_hash_still_sensitive_to_real_fields():
    assert config_hash(BotConfig(aggregation_method="mean")) != \
        config_hash(BotConfig())


def test_days_remaining_accepts_now():
    assert _days_remaining("2024-06-25T12:00:00Z", now=AS_OF) == \
        pytest.approx(10.0)
    # close before now clamps to zero regardless of the clock used
    assert _days_remaining("2020-01-01T00:00:00Z", now=AS_OF) == 0.0


def test_days_remaining_default_now_is_wall_clock():
    assert _days_remaining("2020-01-01T00:00:00Z") == 0.0


def test_prompt_args_today_follows_as_of():
    assert _prompt_args(QUESTION, BotConfig(as_of=AS_OF))["today"] == \
        "2024-06-15"
    assert _prompt_args(QUESTION, BotConfig(as_of=None))["today"] == \
        date.today().isoformat()


def _cfg(tmp_path, as_of, **kw):
    return BotConfig(ensemble=[EnsembleMember("fake/model", runs=3)],
                     as_of=as_of, calibration_dir=str(tmp_path), **kw)


def _capture_invoke(monkeypatch, completion):
    captured = []

    async def fake_invoke(self, member_and_prompt):
        captured.append(member_and_prompt[1])
        return completion

    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", fake_invoke,
                        raising=True)
    return captured


async def test_binary_prompt_identical_with_as_of_none(monkeypatch, tmp_path):
    """Regression: as_of=None renders BINARY_PROMPT exactly as the pre-change
    code did (today=date.today(), wall-clock day math)."""
    captured = _capture_invoke(monkeypatch, "reasoning\nProbability: 60%")
    bot = CalibratedEnsembleBot(bot_config=_cfg(tmp_path, None))
    await bot.run_binary_ensemble(
        question_text="Q?", background_info="B", resolution_criteria="R",
        fine_print="F", research="research text",
        close_iso="2020-01-01T00:00:00Z")
    expected = BINARY_PROMPT.format(
        question_text="Q?", background_info="B", resolution_criteria="R",
        fine_print="F", research="research text",
        today=date.today().isoformat(),
        close_date="2020-01-01", days_remaining=0.0)
    assert captured[0] == expected


async def test_binary_prompt_uses_as_of_when_set(monkeypatch, tmp_path):
    captured = _capture_invoke(monkeypatch, "reasoning\nProbability: 60%")
    bot = CalibratedEnsembleBot(bot_config=_cfg(tmp_path, AS_OF))
    await bot.run_binary_ensemble(
        question_text="Q?", background_info="B", resolution_criteria="R",
        fine_print="F", research="research text",
        close_iso="2024-06-25T12:00:00Z")
    assert "Today's date: 2024-06-15" in captured[0]
    assert "(10 days remaining)" in captured[0]
    assert date.today().isoformat() not in captured[0]


async def test_mc_prompt_uses_as_of_when_set(monkeypatch, tmp_path):
    captured = _capture_invoke(monkeypatch, "thinking\nA: 60%\nB: 40%")
    bot = CalibratedEnsembleBot(bot_config=_cfg(tmp_path, AS_OF))
    await bot.run_mc_ensemble(
        question_text="Q?", background_info="B", resolution_criteria="R",
        fine_print="F", research="research text",
        close_iso="2024-06-25T12:00:00Z", options=("A", "B"))
    assert "Today's date: 2024-06-15" in captured[0]
    assert "(10 days remaining)" in captured[0]


async def test_numeric_prompt_uses_as_of_when_set(monkeypatch, tmp_path):
    completion = ("analysis\n"
                  "Percentile 2.5: 10\nPercentile 10: 20\nPercentile 25: 30\n"
                  "Percentile 50: 40\nPercentile 75: 50\nPercentile 90: 60\n"
                  "Percentile 97.5: 70")
    captured = _capture_invoke(monkeypatch, completion)
    bot = CalibratedEnsembleBot(bot_config=_cfg(tmp_path, AS_OF))
    await bot.run_numeric_ensemble(
        question_text="Q?", background_info="B", resolution_criteria="R",
        fine_print="F", research="research text",
        close_iso="2024-06-25T12:00:00Z", units="kg",
        lower_bound_message="lo", upper_bound_message="hi")
    assert "Today's date: 2024-06-15" in captured[0]
    assert "(10 days remaining)" in captured[0]
```
- [ ] **Step 3: Run test to verify it fails**
Run: `pytest tests/backtest/test_asof_threading.py -v`
Expected: FAIL — `TypeError: BotConfig.__init__() got an unexpected keyword argument 'as_of'` (config tests), `TypeError: _days_remaining() got an unexpected keyword argument 'now'`, `TypeError: _prompt_args() takes 1 positional argument but 2 were given`
- [ ] **Step 4: Implement — `bot/config.py`**

Add to the import block (after `import os`):
```python
from datetime import datetime
```
At the end of the `BotConfig` field list (directly after the `log_path: str = "logs/forecasts.jsonl"` line, before the `submit_predictions` property):
```python
    # Backtesting (Retrocast): pretend "today" is this instant everywhere a
    # date or day-count enters a prompt. None = live behavior, bit-identical
    # to pre-backtest code. Excluded from config_hash so live hashes are
    # unchanged and backtest variants fingerprint identically to live.
    as_of: datetime | None = None
```
Replace the whole `config_hash` function:
```python
def config_hash(cfg: BotConfig) -> str:
    d = asdict(cfg)
    d.pop("as_of", None)     # backtest-only knob; must not perturb live hashes
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
```
- [ ] **Step 5: Implement — `bot/ensemble_bot.py`**

Replace `_days_remaining` (lines 56–61) with:
```python
def _days_remaining(close_iso: str, now: datetime | None = None) -> float:
    try:
        close = datetime.fromisoformat(str(close_iso).replace("Z", "+00:00"))
        return max(days_until(close, now=now), 0.0)
    except Exception:
        return 0.0
```
In `run_binary_ensemble`, replace the prompt construction (lines 125–130) with:
```python
        prompt = BINARY_PROMPT.format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research,
            today=(self.cfg.as_of.date() if self.cfg.as_of
                   else date.today()).isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso, now=self.cfg.as_of))
```
In `run_mc_ensemble`, replace the prompt construction (lines 144–150) with:
```python
        prompt = MC_PROMPT.format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research,
            today=(self.cfg.as_of.date() if self.cfg.as_of
                   else date.today()).isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso, now=self.cfg.as_of),
            options=", ".join(options))
```
In `run_numeric_ensemble`, replace the prompt construction (lines 162–170) with:
```python
        prompt = NUMERIC_PROMPT.format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research,
            today=(self.cfg.as_of.date() if self.cfg.as_of
                   else date.today()).isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso, now=self.cfg.as_of),
            units=units or "(see question)",
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message)
```
- [ ] **Step 6: Implement — `bot/research.py`**

Replace `_prompt_args` (lines 34–40) with:
```python
def _prompt_args(question: dict, cfg: BotConfig) -> dict:
    return dict(
        question_text=question["question_text"],
        resolution_criteria=question.get("resolution_criteria", ""),
        fine_print=question.get("fine_print", ""),
        today=(cfg.as_of.date() if cfg.as_of else date.today()).isoformat(),
    )
```
In `_search_llm_research`, replace the line `args = _prompt_args(question)` with:
```python
    args = _prompt_args(question, cfg)
```
- [ ] **Step 7: Run test to verify it passes**
Run: `pytest tests/backtest/test_asof_threading.py -v`
Expected: PASS (all 9 tests)
- [ ] **Step 8: Run the full existing suite — regression gate**
Run: `pytest tests/ -v`
Expected: PASS — every pre-existing test unchanged (with `as_of=None` all branches reduce to current behavior)
- [ ] **Step 9: Commit**
```bash
git add bot/config.py bot/ensemble_bot.py bot/research.py tests/backtest/test_asof_threading.py && git commit -m "feat(backtest): thread BotConfig.as_of through prompt dates and day math (live behavior unchanged)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 19: AsOfContext, ACTIVE_CONTEXT and horizon_to_asof

**Files:**
- Create: `backtest/timetravel/__init__.py` (empty)
- Create: `backtest/timetravel/asof.py`
- Test: `tests/backtest/test_asof.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_asof.py`:
```python
from datetime import datetime, timedelta, timezone

import pytest

from backtest.core import BacktestQuestion, LedgerEntry, QType
from backtest.timetravel.asof import (ACTIVE_CONTEXT, AsOfContext,
                                      horizon_to_asof)

OPEN = datetime(2024, 1, 1, tzinfo=timezone.utc)
CLOSE = datetime(2024, 4, 11, tzinfo=timezone.utc)   # span exactly 101 days


def _q(**kw):
    base = dict(qid="metaculus:1", platform="metaculus", native_id="1",
                qtype=QType.BINARY, title="Q?")
    base.update(kw)
    return BacktestQuestion(**base)


def test_close_horizons_subtract_from_scheduled_close():
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE)
    assert horizon_to_asof(q, "close-7d") == CLOSE - timedelta(days=7)
    assert horizon_to_asof(q, "close-30d") == CLOSE - timedelta(days=30)


def test_close_horizon_landing_before_open_is_skipped():
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE)
    assert horizon_to_asof(q, "close-365d") is None


def test_close_horizon_without_close_time_is_skipped():
    assert horizon_to_asof(_q(open_time=OPEN), "close-7d") is None


def test_spot_uses_spot_scoring_time():
    spot = datetime(2024, 3, 1, tzinfo=timezone.utc)
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE, spot_scoring_time=spot)
    assert horizon_to_asof(q, "spot") == spot
    assert horizon_to_asof(_q(open_time=OPEN), "spot") is None


def test_geo_k_matches_halawi_on_known_interval():
    # span 101 days -> offsets (101-1)^(k/5) = 100^(k/5) days from open
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE)
    t5 = horizon_to_asof(q, "geo-5")
    assert t5 == OPEN + timedelta(days=100.0)          # == close - 1 day
    assert t5 == CLOSE - timedelta(days=1)
    t1 = horizon_to_asof(q, "geo-1")
    expect = OPEN + timedelta(days=100.0 ** 0.2)       # ~2.5119 days
    assert abs((t1 - expect).total_seconds()) < 1.0
    ts = [horizon_to_asof(q, f"geo-{k}") for k in range(1, 6)]
    assert ts == sorted(ts)                            # strictly increasing
    assert all(OPEN < t < CLOSE for t in ts)           # inside [open, close)


def test_geo_k_out_of_range_or_unknown_horizon_raises():
    q = _q(open_time=OPEN, scheduled_close_time=CLOSE)
    with pytest.raises(ValueError):
        horizon_to_asof(q, "geo-6")
    with pytest.raises(ValueError):
        horizon_to_asof(q, "geo-x")
    with pytest.raises(ValueError):
        horizon_to_asof(q, "close-tomorrow")


def test_geo_k_degenerate_interval_is_skipped():
    q = _q(open_time=OPEN, scheduled_close_time=OPEN + timedelta(hours=12))
    assert horizon_to_asof(q, "geo-3") is None


def test_context_defaults_record_and_contextvar():
    assert ACTIVE_CONTEXT.get() is None
    ctx = AsOfContext(as_of=OPEN)
    assert ctx.cutoff_mode == "strict"
    assert ctx.ledger == [] and ctx.llm is None and ctx.store is None
    ctx.record(LedgerEntry(provider="p", query="q"))
    assert len(ctx.ledger) == 1 and ctx.ledger[0].provider == "p"
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        assert ACTIVE_CONTEXT.get() is ctx
    finally:
        ACTIVE_CONTEXT.reset(token)
    assert ACTIVE_CONTEXT.get() is None
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_asof.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.timetravel'`
- [ ] **Step 3: Write minimal implementation**
```bash
touch backtest/timetravel/__init__.py
```
Create `backtest/timetravel/asof.py`:
```python
"""As-of context and horizon semantics. T is always anchored to
scheduled_close_time / spot_scoring_time — never actual_resolve_time
(anchoring to resolution leaks 'this resolves about now'; Halawi's protocol
deliberately avoids it). Every provider records what it surfaced into the
context's evidence ledger, which makes leakage auditable per question."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from backtest.core import BacktestQuestion, LedgerEntry

if TYPE_CHECKING:
    from backtest.corpus.store import CorpusStore
    from backtest.timetravel.llm_cache import CachedLLM


@dataclass
class AsOfContext:
    as_of: datetime
    cutoff_mode: str = "strict"          # "strict" | "pragmatic"
    ledger: list[LedgerEntry] = field(default_factory=list)
    llm: "CachedLLM | None" = None
    store: "CorpusStore | None" = None

    def record(self, e: LedgerEntry) -> None:
        self.ledger.append(e)


# Set by the adapter per question; providers read it at call time. Default
# None means "not inside the backtest harness" — historical providers refuse
# to run.
ACTIVE_CONTEXT: ContextVar[AsOfContext | None] = ContextVar(
    "ACTIVE_CONTEXT", default=None)

_CLOSE_DELTAS = {"close-7d": 7, "close-30d": 30, "close-365d": 365}
_GEO_N = 5


def horizon_to_asof(q: BacktestQuestion, horizon: str) -> datetime | None:
    """Map a horizon name to T for this question; None = skip this horizon.

    close-Nd: scheduled_close_time - N days; skipped (None) when that lands
    before open_time. spot: spot_scoring_time. geo-k (k=1.._GEO_N): Halawi
    geometric retrieval dates, T_k = open + (span_days - 1)^(k/n) days.
    """
    if horizon in _CLOSE_DELTAS:
        if q.scheduled_close_time is None:
            return None
        t = q.scheduled_close_time - timedelta(days=_CLOSE_DELTAS[horizon])
        if q.open_time is not None and t < q.open_time:
            return None              # horizon predates the question; skip
        return t
    if horizon == "spot":
        return q.spot_scoring_time
    if horizon.startswith("geo-"):
        try:
            k = int(horizon.split("-", 1)[1])
        except ValueError:
            raise ValueError(f"unknown horizon {horizon!r}")
        if not 1 <= k <= _GEO_N:
            raise ValueError(f"geo-k supports k=1..{_GEO_N}, got {horizon!r}")
        if q.open_time is None or q.scheduled_close_time is None:
            return None
        span_days = ((q.scheduled_close_time - q.open_time).total_seconds()
                     / 86400.0)
        if span_days <= 1.0:
            return None              # Halawi offset undefined; skip
        return q.open_time + timedelta(days=(span_days - 1.0) ** (k / _GEO_N))
    raise ValueError(f"unknown horizon {horizon!r}")
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_asof.py -v`
Expected: PASS (all 8 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/timetravel/__init__.py backtest/timetravel/asof.py tests/backtest/test_asof.py && git commit -m "feat(backtest): AsOfContext, ACTIVE_CONTEXT and horizon_to_asof (close ladder, spot, Halawi geo-k)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 20: CachedLLM content-addressed completion cache

**Files:**
- Create: `backtest/timetravel/llm_cache.py`
- Test: `tests/backtest/test_llm_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_llm_cache.py`:
```python
"""CachedLLM: content-addressed on (model, temperature, run_index,
prompt sha). Live calls go through forecasting_tools.GeneralLlm with the
openrouter/ prefix (monkeypatched here — never the network); live=False is
replay-only and raises CacheMiss on any uncached call."""
import hashlib
import json

import pytest

from backtest.timetravel.llm_cache import CachedLLM, CacheMiss, cache_key


class FakeGeneralLlm:
    constructed = []

    def __init__(self, model, temperature, timeout=None):
        FakeGeneralLlm.constructed.append(
            dict(model=model, temperature=temperature))

    async def invoke(self, prompt):
        return f"completion for: {prompt}"


@pytest.fixture
def llm(tmp_path, monkeypatch):
    FakeGeneralLlm.constructed = []
    monkeypatch.setattr("forecasting_tools.GeneralLlm", FakeGeneralLlm)
    return CachedLLM(tmp_path / "cache")


def test_key_derivation_is_content_addressed():
    prompt_sha = hashlib.sha256(b"P").hexdigest()
    expected = hashlib.sha256(f"m|0.5|2|{prompt_sha}".encode()).hexdigest()
    assert cache_key("m", "P", 0.5, 2) == expected
    assert cache_key("m", "P", 0.5, 3) != expected      # run_index in key
    assert cache_key("m", "P", 0.7, 2) != expected      # temperature in key
    assert cache_key("m2", "P", 0.5, 2) != expected     # model in key
    assert cache_key("m", "Q", 0.5, 2) != expected      # prompt in key


async def test_live_call_writes_cache_file_with_full_record(llm, tmp_path):
    out = await llm.invoke("openai/gpt-5.4-mini", "P", 0.0, run_index=1)
    assert out == "completion for: P"
    assert FakeGeneralLlm.constructed == [
        dict(model="openrouter/openai/gpt-5.4-mini", temperature=0.0)]
    key = cache_key("openai/gpt-5.4-mini", "P", 0.0, 1)
    rec = json.loads((tmp_path / "cache" / f"{key}.json").read_text())
    assert rec == {"model": "openai/gpt-5.4-mini", "temperature": 0.0,
                   "run_index": 1,
                   "prompt_sha": hashlib.sha256(b"P").hexdigest(),
                   "completion": "completion for: P",
                   "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}


async def test_second_call_hits_cache_not_the_llm(llm):
    await llm.invoke("m", "P", 0.0)
    n = len(FakeGeneralLlm.constructed)
    assert await llm.invoke("m", "P", 0.0) == "completion for: P"
    assert len(FakeGeneralLlm.constructed) == n         # no new live call


async def test_distinct_run_index_is_a_distinct_completion(llm):
    await llm.invoke("m", "P", 0.0, run_index=0)
    await llm.invoke("m", "P", 0.0, run_index=1)
    assert len(FakeGeneralLlm.constructed) == 2


async def test_replay_mode_raises_cache_miss(tmp_path):
    replay = CachedLLM(tmp_path / "cache", live=False)
    with pytest.raises(CacheMiss):
        await replay.invoke("m", "P", 0.0)


async def test_replay_mode_serves_cached_completions(llm, tmp_path):
    await llm.invoke("m", "P", 0.0)
    replay = CachedLLM(tmp_path / "cache", live=False)
    assert await replay.invoke("m", "P", 0.0) == "completion for: P"


async def test_tokens_and_cost_recorded_when_response_carries_them(
        tmp_path, monkeypatch):
    class Usage:
        prompt_tokens = 11
        completion_tokens = 7

    class Resp(str):
        usage = Usage()
        cost = 0.0042

    class RichLlm:
        def __init__(self, model, temperature, timeout=None):
            pass

        async def invoke(self, prompt):
            return Resp("rich completion")

    monkeypatch.setattr("forecasting_tools.GeneralLlm", RichLlm)
    llm = CachedLLM(tmp_path / "cache")
    assert await llm.invoke("m", "P", 0.0) == "rich completion"
    key = cache_key("m", "P", 0.0, 0)
    rec = json.loads((tmp_path / "cache" / f"{key}.json").read_text())
    assert rec["tokens_in"] == 11 and rec["tokens_out"] == 7
    assert rec["cost_usd"] == pytest.approx(0.0042)
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_llm_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.timetravel.llm_cache'`
- [ ] **Step 3: Write minimal implementation**

Create `backtest/timetravel/llm_cache.py`:
```python
"""Content-addressed completion cache around the ensemble's single LLM seam.
Key = sha256(f"{model}|{temperature}|{run_index}|{sha256(prompt)}"), one JSON
file per completion. live=False is replay-only (CacheMiss on any uncached
call) — that is what makes post-LLM reruns (aggregation, caps, calibration)
cost $0, and lets CI replay a frozen corpus with no keys at all."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CacheMiss(Exception):
    """Raised in live=False (replay) mode when a completion is not cached."""


def cache_key(model: str, prompt: str, temperature: float,
              run_index: int = 0) -> str:
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    return hashlib.sha256(
        f"{model}|{temperature}|{run_index}|{prompt_sha}".encode()).hexdigest()


class CachedLLM:

    def __init__(self, cache_dir: Path, live: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.live = live

    async def invoke(self, model: str, prompt: str, temperature: float,
                     run_index: int = 0) -> str:
        key = cache_key(model, prompt, temperature, run_index)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["completion"]
        if not self.live:
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            raise CacheMiss(f"model={model} temperature={temperature} "
                            f"run_index={run_index} "
                            f"prompt_sha={prompt_sha[:12]}")
        completion, tokens_in, tokens_out, cost_usd = await self._call_live(
            model, prompt, temperature)
        record = {"model": model, "temperature": temperature,
                  "run_index": run_index,
                  "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(),
                  "completion": completion, "tokens_in": tokens_in,
                  "tokens_out": tokens_out, "cost_usd": cost_usd}
        tmp = path.with_suffix(".json.tmp")     # atomic: never a torn cache file
        tmp.write_text(json.dumps(record))
        tmp.replace(path)
        return completion

    async def _call_live(self, model: str, prompt: str, temperature: float):
        """Single live seam (monkeypatch target in tests). Token/cost figures
        are recorded when the response object carries them, else zeros."""
        from forecasting_tools import GeneralLlm
        llm = GeneralLlm(model=f"openrouter/{model}", temperature=temperature,
                         timeout=900)
        resp = await llm.invoke(prompt)
        completion = resp if isinstance(resp, str) else str(resp)
        usage = getattr(resp, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        cost_usd = float(getattr(resp, "cost", 0.0) or 0.0)
        return completion, tokens_in, tokens_out, cost_usd
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_llm_cache.py -v`
Expected: PASS (all 7 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/timetravel/llm_cache.py tests/backtest/test_llm_cache.py && git commit -m "feat(backtest): content-addressed CachedLLM completion cache with replay-only mode" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 21: Historical providers I — markets-at-T and AskNews historical

**Files:**
- Create: `backtest/timetravel/providers.py`
- Test: `tests/backtest/test_providers_historical.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_providers_historical.py`:
```python
"""markets_historical (crosswalk + store.value_at price-at-T; leak-free by
construction) and asknews_historical (SDK historical=True end_timestamp=T).
Both read ACTIVE_CONTEXT and refuse to run without it. The AskNews SDK may
not be installed: the provider imports it lazily through the
_get_asknews_sdk module attribute, which tests monkeypatch."""
from datetime import datetime, timezone

import pytest

import backtest.timetravel.providers as providers
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot.config import BotConfig

T = datetime(2024, 6, 15, tzinfo=timezone.utc)
PUB = datetime(2024, 6, 1, tzinfo=timezone.utc)
QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F",
                qid="metaculus:31678")
CFG = BotConfig()


@pytest.fixture
def ctx():
    c = AsOfContext(as_of=T)
    token = ACTIVE_CONTEXT.set(c)
    yield c
    ACTIVE_CONTEXT.reset(token)


class FakeStore:
    def __init__(self, value=0.42):
        self.value = value
        self.calls = []

    def value_at(self, qid, kind, t):
        self.calls.append((qid, kind, t))
        return self.value


async def test_markets_requires_context():
    assert ACTIVE_CONTEXT.get() is None
    with pytest.raises(RuntimeError, match="no AsOfContext"):
        await providers._markets_historical_research(QUESTION, CFG)


async def test_markets_requires_store_and_qid(ctx):
    with pytest.raises(RuntimeError, match="store"):
        await providers._markets_historical_research(QUESTION, CFG)
    ctx.store = FakeStore()
    with pytest.raises(RuntimeError, match="qid"):
        await providers._markets_historical_research(
            dict(question_text="Q?"), CFG)


async def test_markets_formats_price_at_T_and_ledgers_snapshot(
        ctx, monkeypatch):
    ctx.store = FakeStore(value=0.42)
    import backtest.corpus.crosswalk as crosswalk
    monkeypatch.setattr(crosswalk, "market_for",
                        lambda store, qid: "polymarket:abc")
    out = await providers._markets_historical_research(QUESTION, CFG)
    assert "42%" in out
    assert "polymarket:abc" in out
    assert "2024-06-15" in out
    assert ctx.store.calls == [("polymarket:abc", "market_price", T)]
    assert len(ctx.ledger) == 1
    e = ctx.ledger[0]
    assert e.provider == "markets_historical"
    assert e.query == "metaculus:31678"
    assert e.date_trust == "snapshot"


async def test_markets_empty_when_no_crosswalk_match(ctx, monkeypatch):
    ctx.store = FakeStore()
    import backtest.corpus.crosswalk as crosswalk
    monkeypatch.setattr(crosswalk, "market_for", lambda store, qid: None)
    assert await providers._markets_historical_research(QUESTION, CFG) == ""
    assert ctx.ledger == []


async def test_markets_empty_when_no_price_at_T(ctx, monkeypatch):
    ctx.store = FakeStore(value=None)
    import backtest.corpus.crosswalk as crosswalk
    monkeypatch.setattr(crosswalk, "market_for",
                        lambda store, qid: "polymarket:abc")
    assert await providers._markets_historical_research(QUESTION, CFG) == ""
    assert ctx.ledger == []


class FakeAskNewsSDK:
    last = None

    def __init__(self, **kwargs):
        FakeAskNewsSDK.last = self
        self.kwargs = kwargs
        self.news = self
        self.search_kwargs = None

    async def search_news(self, **kwargs):
        self.search_kwargs = kwargs

        class Resp:
            as_dicts = [
                {"eng_title": "Title A", "summary": "Sum A",
                 "article_url": "https://a.example/1", "pub_date": PUB},
                {"eng_title": "Title B", "summary": "Sum B",
                 "article_url": "https://b.example/2", "pub_date": PUB},
            ]
        return Resp()


async def test_asknews_requires_context(monkeypatch):
    monkeypatch.setenv("ASKNEWS_API_KEY", "k")
    with pytest.raises(RuntimeError, match="no AsOfContext"):
        await providers._asknews_historical_research(QUESTION, CFG)


async def test_asknews_requires_credentials(ctx, monkeypatch):
    for var in ("ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        await providers._asknews_historical_research(QUESTION, CFG)


async def test_asknews_historical_end_timestamp_and_snapshot_ledger(
        ctx, monkeypatch):
    monkeypatch.setenv("ASKNEWS_API_KEY", "k")
    monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
    monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
    monkeypatch.setattr(providers, "_get_asknews_sdk",
                        lambda: FakeAskNewsSDK)
    out = await providers._asknews_historical_research(QUESTION, CFG)
    sk = FakeAskNewsSDK.last.search_kwargs
    assert sk["historical"] is True
    assert sk["end_timestamp"] == int(T.timestamp())
    assert sk["query"] == "Q?"
    assert "Title A" in out and "Sum A" in out and "Title B" in out
    assert len(ctx.ledger) == 2
    assert {e.provider for e in ctx.ledger} == {"asknews_historical"}
    assert all(e.date_trust == "snapshot" for e in ctx.ledger)
    assert ctx.ledger[0].url == "https://a.example/1"
    assert ctx.ledger[0].doc_date == PUB.isoformat()


async def test_asknews_oauth_credentials_pass_through(ctx, monkeypatch):
    monkeypatch.delenv("ASKNEWS_API_KEY", raising=False)
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "cid")
    monkeypatch.setenv("ASKNEWS_SECRET", "sec")
    monkeypatch.setattr(providers, "_get_asknews_sdk",
                        lambda: FakeAskNewsSDK)
    await providers._asknews_historical_research(QUESTION, CFG)
    assert FakeAskNewsSDK.last.kwargs["client_id"] == "cid"
    assert FakeAskNewsSDK.last.kwargs["client_secret"] == "sec"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_providers_historical.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.timetravel.providers'`
- [ ] **Step 3: Write minimal implementation**

Create `backtest/timetravel/providers.py`:
```python
"""Historical research providers. Same signature as the live providers in
bot/research.py — async fn(question: dict, cfg) -> str — so the existing
globals()-dispatch resolves them by config name once
install_historical_providers() registers them into bot.research's namespace.
Each provider reads ACTIVE_CONTEXT for as_of/ledger/store/llm, writes an
evidence-ledger entry for everything it surfaces, and raises RuntimeError
when no context is set or the cutoff mode forbids it (bot/research.py's
never-raise contract converts that into a skipped provider)."""
from __future__ import annotations

import logging
import os
from datetime import datetime

from backtest.core import LedgerEntry
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext

logger = logging.getLogger(__name__)

ASKNEWS_N_ARTICLES = 10


def _require_context(provider: str) -> AsOfContext:
    ctx = ACTIVE_CONTEXT.get()
    if ctx is None:
        raise RuntimeError(
            f"{provider}: no AsOfContext set (ACTIVE_CONTEXT is None); "
            "historical providers only run inside the backtest harness")
    return ctx


def _fmt_price(p: float) -> str:
    """Never render a non-degenerate price as a literal 0%/100% (same rule
    as bot/markets.py — a false certainty anchor is the failure mode)."""
    pct = p * 100.0
    if 0.0 <= pct < 0.1:
        return "<0.1%"
    if 99.9 < pct <= 100.0:
        return ">99.9%"
    return f"{pct:.1f}%" if pct < 1 or pct > 99 else f"{pct:.0f}%"


async def _markets_historical_research(question: dict, cfg) -> str:
    """Crosswalk-matched market price at T. Leak-free by construction: the
    price series was snapshotted on sight; value_at is a step-function lookup
    of the last point <= T."""
    ctx = _require_context("markets_historical")
    if ctx.store is None:
        raise RuntimeError("markets_historical: AsOfContext.store is not set")
    qid = question.get("qid")
    if not qid:
        raise RuntimeError("markets_historical: question dict has no 'qid' "
                           "(the backtest adapter must pass it)")
    from backtest.corpus.crosswalk import market_for
    market_qid = market_for(ctx.store, qid)
    if market_qid is None:
        return ""               # no equivalent market; compositor skips section
    price = ctx.store.value_at(market_qid, "market_price", ctx.as_of)
    if price is None:
        return ""               # no price history at/before T
    ctx.record(LedgerEntry(
        provider="markets_historical", query=qid, url=None,
        doc_date=ctx.as_of.isoformat(), date_trust="snapshot",
        note=f"price-at-T from {market_qid} (leak-free by construction)"))
    return (f"Crosswalk-matched prediction market {market_qid} priced "
            f"P(Yes)={_fmt_price(float(price))} as of "
            f"{ctx.as_of.date().isoformat()}. Treat the price as one anchor "
            "among several, never the answer: have explicit reasons to "
            "concur or deviate.")


def _get_asknews_sdk():
    """Lazy import seam: the SDK may not be installed in test/CI
    environments; tests monkeypatch this module attribute."""
    from asknews_sdk import AsyncAskNewsSDK
    return AsyncAskNewsSDK


def _field(item, name, default=""):
    """AskNews SDK returns dicts or attribute objects depending on
    return_type/version — accept both."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


async def _asknews_historical_research(question: dict, cfg) -> str:
    """AskNews archive search with historical=True and end_timestamp=T.
    Ingestion-time snapshots (leak-low by architecture, unaudited) — allowed
    in strict mode; maximizes parity with the live asknews provider."""
    ctx = _require_context("asknews_historical")
    # same credential contract as bot/research._asknews_research
    has_oauth = bool(os.getenv("ASKNEWS_CLIENT_ID")
                     and os.getenv("ASKNEWS_SECRET"))
    if not (has_oauth or os.getenv("ASKNEWS_API_KEY")):
        raise RuntimeError("AskNews credentials not configured")
    sdk_cls = _get_asknews_sdk()
    if has_oauth:
        sdk = sdk_cls(client_id=os.getenv("ASKNEWS_CLIENT_ID"),
                      client_secret=os.getenv("ASKNEWS_SECRET"),
                      scopes={"news"})
    else:
        sdk = sdk_cls(api_key=os.getenv("ASKNEWS_API_KEY"), scopes={"news"})
    resp = await sdk.news.search_news(
        query=question["question_text"], n_articles=ASKNEWS_N_ARTICLES,
        return_type="dicts", historical=True,
        end_timestamp=int(ctx.as_of.timestamp()))
    articles = getattr(resp, "as_dicts", None) or []
    parts = []
    for a in articles:
        title = _field(a, "eng_title") or _field(a, "title")
        summary = _field(a, "summary")
        url = str(_field(a, "article_url") or "")
        pub = _field(a, "pub_date", None)
        pub_iso = (pub.isoformat() if isinstance(pub, datetime)
                   else (str(pub) if pub else None))
        ctx.record(LedgerEntry(
            provider="asknews_historical", query=question["question_text"],
            url=url or None, doc_date=pub_iso, date_trust="snapshot",
            note="AskNews ingestion-time snapshot (historical archive)"))
        parts.append(f"[{pub_iso or 'undated'}] {title} — {summary} ({url})")
    if not parts:
        return ""
    return ("Historical news snapshots (AskNews archive, ingestion-time, "
            f"end_timestamp={ctx.as_of.isoformat()}):\n"
            + "\n".join(f"- {p}" for p in parts))
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_providers_historical.py -v`
Expected: PASS (all 10 tests)
- [ ] **Step 5: Commit**
```bash
git add backtest/timetravel/providers.py tests/backtest/test_providers_historical.py && git commit -m "feat(backtest): markets-at-T and AskNews historical research providers" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 22: Historical providers II — GDELT+Wayback and Exa endCrawlDate

**Files:**
- Modify: `backtest/timetravel/providers.py` (imports; append HTTP helpers + two providers)
- Test: `tests/backtest/test_providers_wayback_exa.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_providers_wayback_exa.py`:
```python
"""gdelt_wayback: GDELT DOC 2.0 ArtList URL enumeration in 3-month windows
ending at T, then Wayback CDX lookup for the latest capture <= T fetched in
raw id_ mode — the strongest snapshot guarantee. exa_historical: POST
api.exa.ai/search with endCrawlDate=T; publisher date trust, semi-trusted.
All HTTP goes through providers._http_get_json/_http_get_text/_http_post_json
module seams, monkeypatched here — never the network."""
from datetime import datetime, timedelta, timezone

import pytest

import backtest.timetravel.providers as providers
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot.config import BotConfig

T = datetime(2024, 6, 15, tzinfo=timezone.utc)
QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F")
CFG = BotConfig()
RAW_URL = ("https://web.archive.org/web/20240601120000id_/"
           "https://news.example/story")


@pytest.fixture
def ctx():
    c = AsOfContext(as_of=T)
    token = ACTIVE_CONTEXT.set(c)
    yield c
    ACTIVE_CONTEXT.reset(token)


async def test_gdelt_requires_context():
    with pytest.raises(RuntimeError, match="no AsOfContext"):
        await providers._gdelt_wayback_research(QUESTION, CFG)


async def test_gdelt_wayback_snapshot_flow(ctx, monkeypatch):
    calls = []

    async def fake_get_json(url, params):
        calls.append((url, dict(params)))
        if url == providers.GDELT_DOC_URL:
            if params["enddatetime"] == providers._gdelt_ts(T):
                return {"articles": [{"url": "https://news.example/story",
                                      "title": "Story"}]}
            return {"articles": []}            # older window: nothing
        if url == providers.WAYBACK_CDX_URL:
            return [["urlkey", "timestamp", "original", "mimetype",
                     "statuscode", "digest", "length"],
                    ["x", "20240601120000", "https://news.example/story",
                     "text/html", "200", "D", "123"]]
        raise AssertionError(f"unexpected URL {url}")

    async def fake_get_text(url):
        assert url == RAW_URL
        return "<html><script>x()</script><body><p>Pre-T content.</p></body></html>"

    monkeypatch.setattr(providers, "_http_get_json", fake_get_json)
    monkeypatch.setattr(providers, "_http_get_text", fake_get_text)
    out = await providers._gdelt_wayback_research(QUESTION, CFG)
    assert "Pre-T content." in out
    assert "x()" not in out                    # script stripped
    gdelt = [c for c in calls if c[0] == providers.GDELT_DOC_URL]
    assert len(gdelt) == providers.GDELT_WINDOWS
    assert gdelt[0][1]["mode"] == "ArtList"
    assert gdelt[0][1]["enddatetime"] == providers._gdelt_ts(T)
    assert gdelt[0][1]["startdatetime"] == \
        providers._gdelt_ts(T - timedelta(days=90))
    assert gdelt[1][1]["enddatetime"] == \
        providers._gdelt_ts(T - timedelta(days=90))
    cdx = [c for c in calls if c[0] == providers.WAYBACK_CDX_URL]
    assert len(cdx) == 1
    assert cdx[0][1]["to"] == providers._gdelt_ts(T)
    assert cdx[0][1]["url"] == "https://news.example/story"
    assert len(ctx.ledger) == 1
    e = ctx.ledger[0]
    assert e.provider == "gdelt_wayback"
    assert e.date_trust == "snapshot"
    assert e.url == RAW_URL
    assert e.doc_date == "2024-06-01T12:00:00"


async def test_gdelt_drops_articles_without_pre_T_snapshot(ctx, monkeypatch):
    fetched = []

    async def fake_get_json(url, params):
        if url == providers.GDELT_DOC_URL:
            if params["enddatetime"] == providers._gdelt_ts(T):
                return {"articles": [{"url": "https://news.example/story",
                                      "title": "Story"}]}
            return {"articles": []}
        if url == providers.WAYBACK_CDX_URL:
            return [["urlkey", "timestamp", "original", "mimetype",
                     "statuscode", "digest", "length"]]   # header only
        raise AssertionError(url)

    async def fake_get_text(url):
        fetched.append(url)
        return "should never be fetched"

    monkeypatch.setattr(providers, "_http_get_json", fake_get_json)
    monkeypatch.setattr(providers, "_http_get_text", fake_get_text)
    assert await providers._gdelt_wayback_research(QUESTION, CFG) == ""
    assert fetched == []                       # not provably pre-T: no fetch
    assert ctx.ledger == []


async def test_exa_requires_context_and_key(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "k")
    with pytest.raises(RuntimeError, match="no AsOfContext"):
        await providers._exa_historical_research(QUESTION, CFG)


async def test_exa_missing_key_raises(ctx, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EXA_API_KEY"):
        await providers._exa_historical_research(QUESTION, CFG)


async def test_exa_end_crawl_date_and_publisher_ledger(ctx, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    captured = {}

    async def fake_post(url, payload, headers):
        captured.update(url=url, payload=payload, headers=headers)
        return {"results": [
            {"url": "https://doc.example/a", "title": "Doc A",
             "publishedDate": "2024-05-01", "text": "alpha text"},
            {"url": "https://doc.example/b", "title": "Doc B",
             "publishedDate": None, "text": "beta text"},
        ]}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    out = await providers._exa_historical_research(QUESTION, CFG)
    assert captured["url"] == providers.EXA_SEARCH_URL
    assert captured["payload"]["endCrawlDate"] == T.isoformat()
    assert captured["payload"]["query"] == "Q?"
    assert captured["headers"]["x-api-key"] == "exa-key"
    assert "Doc A" in out and "alpha text" in out and "Doc B" in out
    assert len(ctx.ledger) == 2
    assert all(e.provider == "exa_historical" for e in ctx.ledger)
    assert all(e.date_trust == "publisher" for e in ctx.ledger)
    assert all("semi-trusted" in e.note for e in ctx.ledger)
    assert ctx.ledger[0].doc_date == "2024-05-01"
    assert ctx.ledger[1].doc_date is None
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_providers_wayback_exa.py -v`
Expected: FAIL — `AttributeError: module 'backtest.timetravel.providers' has no attribute '_gdelt_wayback_research'` (and `_http_get_json` / `_exa_historical_research`)
- [ ] **Step 3: Write minimal implementation**

In `backtest/timetravel/providers.py`, replace the import block at the top of the file with:
```python
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta

from backtest.core import LedgerEntry
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
```
Append at the end of `backtest/timetravel/providers.py`:
```python
# --- shared HTTP seams (monkeypatch targets in tests; never hit in CI) -----

HTTP_TIMEOUT_S = 30.0


async def _http_get_json(url: str, params: dict):
    import httpx
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def _http_get_text(url: str) -> str:
    import httpx
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def _http_post_json(url: str, payload: dict, headers: dict):
    import httpx
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


# --- GDELT DOC 2.0 + Wayback (DIY RetroSearch; strongest snapshot trust) ----

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
WAYBACK_CDX_URL = "http://web.archive.org/cdx/search/cdx"
GDELT_MAX_DOCS = 6
GDELT_WINDOWS = 2            # 3-month ArtList windows stepping back from T
GDELT_DOC_CHARS = 1500


def _gdelt_ts(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


async def _wayback_snapshot_before(url: str, t: datetime):
    """Latest Wayback capture at or before t -> (raw_id_url, iso_ts) or None.
    CDX JSON output: first row is the header; limit=-1 returns the last
    matching capture."""
    rows = await _http_get_json(WAYBACK_CDX_URL, params={
        "url": url, "output": "json", "to": _gdelt_ts(t),
        "filter": "statuscode:200", "limit": "-1"})
    if not rows or len(rows) < 2:
        return None
    ts, original = rows[-1][1], rows[-1][2]
    raw = f"https://web.archive.org/web/{ts}id_/{original}"
    return raw, datetime.strptime(ts, "%Y%m%d%H%M%S").isoformat()


async def _gdelt_wayback_research(question: dict, cfg) -> str:
    """GDELT ArtList URL enumeration (3-month windows ending at T — ArtList
    only searches a trailing window, so we page windows backwards), then for
    each URL the latest Wayback capture <= T fetched in raw id_ mode. An
    article with no pre-T capture is dropped: not provably pre-T."""
    ctx = _require_context("gdelt_wayback")
    articles = []
    for w in range(GDELT_WINDOWS):
        end = ctx.as_of - timedelta(days=90 * w)
        start = end - timedelta(days=90)
        try:
            data = await _http_get_json(GDELT_DOC_URL, params={
                "query": question["question_text"], "mode": "ArtList",
                "format": "json", "maxrecords": GDELT_MAX_DOCS,
                "sort": "hybridrel",
                "startdatetime": _gdelt_ts(start),
                "enddatetime": _gdelt_ts(end)})
        except Exception as e:
            logger.warning("GDELT window %d failed: %s", w, e)
            continue
        articles.extend((data or {}).get("articles") or [])
        if len(articles) >= GDELT_MAX_DOCS:
            break
    parts = []
    for art in articles[:GDELT_MAX_DOCS]:
        url = art.get("url")
        if not url:
            continue
        try:
            snap = await _wayback_snapshot_before(url, ctx.as_of)
        except Exception as e:
            logger.warning("Wayback CDX lookup failed for %s: %s", url, e)
            continue
        if snap is None:
            continue
        raw_url, snap_iso = snap
        try:
            text = _strip_html(await _http_get_text(raw_url))[:GDELT_DOC_CHARS]
        except Exception as e:
            logger.warning("Wayback fetch failed for %s: %s", raw_url, e)
            continue
        ctx.record(LedgerEntry(
            provider="gdelt_wayback", query=question["question_text"],
            url=raw_url, doc_date=snap_iso, date_trust="snapshot",
            note=f"Wayback capture <= {ctx.as_of.isoformat()} of {url}"))
        parts.append(f"[snapshot {snap_iso}] {art.get('title', url)}\n{text}")
    if not parts:
        return ""
    return ("Archived web snapshots (GDELT URL enumeration + Wayback "
            "captures taken on or before T):\n\n" + "\n\n".join(parts))


# --- Exa endCrawlDate (strict-allowed, semi-trusted) ------------------------

EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_NUM_RESULTS = 6
EXA_DOC_CHARS = 1000


async def _exa_historical_research(question: dict, cfg) -> str:
    """Exa search filtered to pages crawled on or before T. Allowed in strict
    mode but semi-trusted: crawl date >> publication metadata, unaudited —
    every ledger entry carries the flag."""
    ctx = _require_context("exa_historical")
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY not configured")
    data = await _http_post_json(
        EXA_SEARCH_URL,
        payload={"query": question["question_text"],
                 "endCrawlDate": ctx.as_of.isoformat(),
                 "numResults": EXA_NUM_RESULTS,
                 "contents": {"text": {"maxCharacters": EXA_DOC_CHARS}}},
        headers={"x-api-key": api_key, "Content-Type": "application/json"})
    parts = []
    for r in (data or {}).get("results") or []:
        url = r.get("url")
        published = r.get("publishedDate")
        ctx.record(LedgerEntry(
            provider="exa_historical", query=question["question_text"],
            url=url, doc_date=published, date_trust="publisher",
            note="semi-trusted: endCrawlDate filter; crawl date >> "
                 "publication metadata, unaudited"))
        parts.append(f"[{published or 'undated'}] {r.get('title', url)} — "
                     f"{(r.get('text') or '')[:EXA_DOC_CHARS]} ({url})")
    if not parts:
        return ""
    return (f"Web results crawled on or before {ctx.as_of.date().isoformat()} "
            "(Exa endCrawlDate):\n" + "\n".join(f"- {p}" for p in parts))
```
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_providers_wayback_exa.py tests/backtest/test_providers_historical.py -v`
Expected: PASS (all 6 new tests; the provider tests in `tests/backtest/test_providers_historical.py` still pass)
- [ ] **Step 5: Commit**
```bash
git add backtest/timetravel/providers.py tests/backtest/test_providers_wayback_exa.py && git commit -m "feat(backtest): GDELT+Wayback and Exa endCrawlDate historical providers" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 23: Leak screen, pragmatic serper_dated, provider installation + cutoff-mode enforcement

**Files:**
- Create: `backtest/timetravel/leakage.py`
- Modify: `backtest/timetravel/providers.py` (append `_serper_dated_research`, `HISTORICAL_PROVIDERS`, `install_historical_providers`)
- Test: `tests/backtest/test_leakage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_leakage.py`:
```python
"""screen_docs: an LLM judge flags documents revealing post-as_of
information; flagged docs are dropped and every discard is stamped into the
evidence ledger. serper_dated: date-filtered live search (tbs cdr params),
pragmatic-only — raises in strict mode, and bot/research.py's never-raise
contract converts that into a skipped provider. install_historical_providers
registers all five providers into bot.research's namespace."""
from datetime import datetime, timezone

import pytest

import backtest.timetravel.providers as providers
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from backtest.timetravel.leakage import LEAK_JUDGE_MODEL, screen_docs
from bot.config import BotConfig

T = datetime(2024, 6, 15, tzinfo=timezone.utc)
QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F")
DOCS = [
    {"title": "Pre-T report", "url": "https://ok.example/1",
     "date": "2024-05-20", "text": "Surveys in May 2024 show ..."},
    {"title": "Post-T result", "url": "https://leak.example/2",
     "date": "2024-07-01", "text": "The outcome, announced July 1 ..."},
]


class FakeJudge:
    def __init__(self, completion):
        self.completion = completion
        self.calls = []

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append(dict(model=model, prompt=prompt,
                               temperature=temperature, run_index=run_index))
        return self.completion


@pytest.fixture
def ctx():
    c = AsOfContext(as_of=T, cutoff_mode="pragmatic")
    token = ACTIVE_CONTEXT.set(c)
    yield c
    ACTIVE_CONTEXT.reset(token)


@pytest.fixture
def installed():
    providers.install_historical_providers()
    yield
    import bot.research
    for fn in providers.HISTORICAL_PROVIDERS:
        if hasattr(bot.research, fn.__name__):
            delattr(bot.research, fn.__name__)


async def test_screen_docs_drops_flagged_and_ledgers_discards(ctx):
    judge = FakeJudge("1: CLEAN\n2: LEAK")
    survivors = await screen_docs(DOCS, T, judge)
    assert survivors == [DOCS[0]]
    call = judge.calls[0]
    assert call["model"] == LEAK_JUDGE_MODEL
    assert call["temperature"] == 0.0 and call["run_index"] == 0
    assert "Pre-T report" in call["prompt"]
    assert len(ctx.ledger) == 1
    e = ctx.ledger[0]
    assert e.provider == "leak_screen"
    assert e.url == "https://leak.example/2"
    assert e.doc_date == "2024-07-01"
    assert e.date_trust == "low"
    assert "discarded" in e.note


async def test_screen_docs_unparseable_verdict_drops_conservatively(ctx):
    survivors = await screen_docs(DOCS, T, FakeJudge("no idea, sorry"))
    assert survivors == []
    assert len(ctx.ledger) == 2


async def test_screen_docs_empty_input_skips_the_llm():
    judge = FakeJudge("unused")
    assert await screen_docs([], T, judge) == []
    assert judge.calls == []


async def test_screen_docs_without_context_still_screens():
    assert ACTIVE_CONTEXT.get() is None
    survivors = await screen_docs(DOCS, T, FakeJudge("1: CLEAN\n2: LEAK"))
    assert survivors == [DOCS[0]]


async def test_serper_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    ctx = AsOfContext(as_of=T, cutoff_mode="strict")
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        with pytest.raises(RuntimeError, match="pragmatic-only"):
            await providers._serper_dated_research(QUESTION, BotConfig())
    finally:
        ACTIVE_CONTEXT.reset(token)


async def test_serper_pragmatic_screens_and_ledgers(ctx, monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    captured = {}

    async def fake_post(url, payload, headers):
        captured.update(url=url, payload=payload, headers=headers)
        return {"organic": [
            {"title": "Pre-T report", "link": "https://ok.example/1",
             "date": "2024-05-20", "snippet": "Surveys in May ..."},
            {"title": "Post-T result", "link": "https://leak.example/2",
             "date": "2024-07-01", "snippet": "The outcome, announced ..."},
        ]}

    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    ctx.llm = FakeJudge("1: CLEAN\n2: LEAK")
    out = await providers._serper_dated_research(QUESTION, BotConfig())
    assert captured["url"] == providers.SERPER_URL
    assert captured["headers"]["X-API-KEY"] == "serper-key"
    assert "cd_max:6/15/2024" in captured["payload"]["tbs"]
    assert captured["payload"]["tbs"].startswith("cdr:1,")
    assert "Pre-T report" in out
    assert "leak.example" not in out
    survivors = [e for e in ctx.ledger if e.provider == "serper_dated"]
    discards = [e for e in ctx.ledger if e.provider == "leak_screen"]
    assert len(survivors) == 1 and survivors[0].date_trust == "low"
    assert len(discards) == 1 and discards[0].url == "https://leak.example/2"


async def test_serper_requires_llm_for_the_screen(ctx, monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    ctx.llm = None
    with pytest.raises(RuntimeError, match="llm"):
        await providers._serper_dated_research(QUESTION, BotConfig())


def test_install_registers_all_five_into_bot_research(installed):
    import bot.research
    for fn in providers.HISTORICAL_PROVIDERS:
        assert getattr(bot.research, fn.__name__) is fn
    assert bot.research._serper_dated_research is \
        providers._serper_dated_research


async def test_strict_mode_serper_skipped_by_run_research_never_raise(
        installed, monkeypatch):
    """Cutoff-mode enforcement end to end: under a strict context the serper
    provider raises; run_research's never-raise contract logs and skips it,
    keeping the surviving providers' sections."""
    import bot.research as research_mod
    from bot.research import run_research

    async def stub(q, cfg):
        return "stub section"

    monkeypatch.setattr(research_mod, "_asknews_research", stub)
    monkeypatch.setenv("SERPER_API_KEY", "k")
    ctx = AsOfContext(as_of=T, cutoff_mode="strict")
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        out = await run_research(QUESTION, BotConfig(
            research_providers=("serper_dated", "asknews")))
    finally:
        ACTIVE_CONTEXT.reset(token)
    assert out.providers == ("asknews",)
    assert "stub section" in out.text
    assert "serper" not in out.provider
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_leakage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.timetravel.leakage'`
- [ ] **Step 3: Write minimal implementation — `backtest/timetravel/leakage.py`**

Create `backtest/timetravel/leakage.py`:
```python
"""Per-document LLM leak screen. A document that reveals information dated
after as_of is dropped, and every discard is stamped into the evidence
ledger via ACTIVE_CONTEXT — leakage is auditable, never silent. Unparseable
verdicts drop the document (conservative default: a doc we cannot clear is
treated as leaked)."""
from __future__ import annotations

import logging
import re
from datetime import datetime

from backtest.core import LedgerEntry
from backtest.timetravel.asof import ACTIVE_CONTEXT
from backtest.timetravel.llm_cache import CachedLLM

logger = logging.getLogger(__name__)

LEAK_JUDGE_MODEL = "openai/gpt-5.4-mini"     # cheap judge; cached + temp 0
_DOC_CHARS = 1200

_JUDGE_PROMPT = """You are auditing research documents for a forecast made on {as_of}.
For each numbered document below, answer: does the text reveal any
information from after {as_of} (events, outcomes, announcements, or
references that postdate it)?

{docs}

Answer with exactly one line per document, in order, formatted as:
<number>: LEAK
or
<number>: CLEAN
No other text."""


def _render_doc(i: int, doc: dict) -> str:
    date_s = doc.get("date") or "undated"
    text = (doc.get("text") or doc.get("snippet") or "")[:_DOC_CHARS]
    return f"--- Document {i} (dated {date_s}) ---\n{doc.get('title', '')}\n{text}"


async def screen_docs(docs: list[dict], as_of: datetime,
                      llm: CachedLLM) -> list[dict]:
    """Returns the surviving docs; stamps every discard into the active
    context's ledger (when a context is set)."""
    if not docs:
        return []
    rendered = "\n\n".join(_render_doc(i + 1, d) for i, d in enumerate(docs))
    prompt = _JUDGE_PROMPT.format(as_of=as_of.isoformat(), docs=rendered)
    completion = await llm.invoke(LEAK_JUDGE_MODEL, prompt, temperature=0.0,
                                  run_index=0)
    verdicts = {}
    for line in completion.splitlines():
        m = re.match(r"\s*(\d+)\s*:\s*(LEAK|CLEAN)\s*$", line.strip(), re.I)
        if m:
            verdicts[int(m.group(1))] = m.group(2).upper()
    ctx = ACTIVE_CONTEXT.get()
    survivors = []
    for i, doc in enumerate(docs, start=1):
        if verdicts.get(i, "LEAK") == "CLEAN":   # unparseable -> drop
            survivors.append(doc)
            continue
        logger.info("leak screen dropped doc %d: %s", i, doc.get("url"))
        if ctx is not None:
            ctx.record(LedgerEntry(
                provider="leak_screen", query=doc.get("title", ""),
                url=doc.get("url"), doc_date=doc.get("date"),
                date_trust="low",
                note=("discarded: judged to reveal information after "
                      f"{as_of.isoformat()}")))
    return survivors
```
- [ ] **Step 4: Write minimal implementation — serper provider + installation**

Append at the end of `backtest/timetravel/providers.py`:
```python
# --- serper_dated (PRAGMATIC ONLY) + provider installation ------------------

SERPER_URL = "https://google.serper.dev/search"
SERPER_NUM = 10


async def _serper_dated_research(question: dict, cfg) -> str:
    """Date-filtered live Google search (Serper, tbs cdr params). Live search
    date filters measurably leak (71-81% of questions affected), so this
    provider is pragmatic-only, every result is leak-screened per document,
    and survivors are stamped date_trust=low."""
    ctx = _require_context("serper_dated")
    if ctx.cutoff_mode != "pragmatic":
        raise RuntimeError(
            "serper_dated is pragmatic-only: live search date filters leak "
            f"(cutoff_mode={ctx.cutoff_mode!r})")
    if ctx.llm is None:
        raise RuntimeError(
            "serper_dated: AsOfContext.llm is required for the leak screen")
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not configured")
    t = ctx.as_of
    tbs = f"cdr:1,cd_min:1/1/1990,cd_max:{t.month}/{t.day}/{t.year}"
    data = await _http_post_json(
        SERPER_URL,
        payload={"q": question["question_text"], "num": SERPER_NUM,
                 "tbs": tbs},
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"})
    docs = [{"title": r.get("title", ""), "url": r.get("link"),
             "date": r.get("date"), "snippet": r.get("snippet", "")}
            for r in (data or {}).get("organic") or []]
    if not docs:
        return ""
    from backtest.timetravel.leakage import screen_docs
    survivors = await screen_docs(docs, ctx.as_of, ctx.llm)
    parts = []
    for d in survivors:
        ctx.record(LedgerEntry(
            provider="serper_dated", query=question["question_text"],
            url=d["url"], doc_date=d.get("date"), date_trust="low",
            note="pragmatic-mode dated search result (survived leak screen)"))
        parts.append(f"[{d.get('date') or 'undated'}] {d['title']} — "
                     f"{d['snippet']} ({d['url']})")
    if not parts:
        return ""
    return ("Date-filtered web search results (PRAGMATIC mode; "
            "date_trust=low; leak-screened):\n"
            + "\n".join(f"- {p}" for p in parts))


HISTORICAL_PROVIDERS = (
    _asknews_historical_research,
    _gdelt_wayback_research,
    _exa_historical_research,
    _markets_historical_research,
    _serper_dated_research,
)


def install_historical_providers() -> None:
    """Register the historical providers into bot.research's module namespace
    so its existing globals()-dispatch (bot/research.py run_research) resolves
    them by config name, e.g. research_providers=("asknews_historical", ...).
    Additive runtime registration — no bot/ file edits, no live-path change."""
    import bot.research
    for fn in HISTORICAL_PROVIDERS:
        setattr(bot.research, fn.__name__, fn)
```
- [ ] **Step 5: Run test to verify it passes**
Run: `pytest tests/backtest/test_leakage.py -v`
Expected: PASS (all 10 tests)
- [ ] **Step 6: Run the provider suite to confirm no regressions**
Run: `pytest tests/backtest/ tests/test_research.py -v`
Expected: PASS (installation is additive; live `bot/research.py` provider tests unchanged)
- [ ] **Step 7: Commit**
```bash
git add backtest/timetravel/leakage.py backtest/timetravel/providers.py tests/backtest/test_leakage.py && git commit -m "feat(backtest): leak screen, pragmatic serper_dated provider, install_historical_providers + cutoff-mode enforcement" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 24: Memorization probe + canary leakage test

**Files:**
- Create: `backtest/timetravel/probe.py`
- Test: `tests/backtest/test_probe.py`
- Test: `tests/backtest/test_canary.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_probe.py`:
```python
"""memorization_probe (Halawi App. A.2 analogue): zero-research direct
outcome prompt per model, temperature=0 run_index=0 through CachedLLM (so
re-probing is free); confident-correct => contaminated-confirmed for that
model."""
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.timetravel.llm_cache import CachedLLM
from backtest.timetravel.probe import memorization_probe


def _q(qtype=QType.BINARY, resolution=True, options=()):
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1", qtype=qtype,
        title="Did X happen by 2024-06-01?", resolution_criteria="Resolves "
        "Yes if X happened before 2024-06-01.", resolution=resolution,
        options=options,
        actual_resolve_time=datetime(2024, 6, 1, tzinfo=timezone.utc))


class ScriptedLLM:
    def __init__(self, completions):
        self.completions = completions       # model -> completion text
        self.calls = []

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append(dict(model=model, prompt=prompt,
                               temperature=temperature, run_index=run_index))
        return self.completions[model]


async def test_binary_confident_correct_flags_contamination():
    llm = ScriptedLLM({"model-a": "I recall this.\nProbability: 97%",
                       "model-b": "Unsure.\nProbability: 60%"})
    out = await memorization_probe(_q(resolution=True),
                                   ["model-a", "model-b"], llm)
    assert out == {"model-a": True, "model-b": False}
    assert all(c["temperature"] == 0.0 and c["run_index"] == 0
               for c in llm.calls)
    assert {c["model"] for c in llm.calls} == {"model-a", "model-b"}


async def test_binary_confident_correct_on_no_resolution():
    llm = ScriptedLLM({"m": "Probability: 3%"})
    assert await memorization_probe(_q(resolution=False), ["m"], llm) == \
        {"m": True}
    # confident but WRONG is not contamination evidence
    llm2 = ScriptedLLM({"m": "Probability: 97%"})
    assert await memorization_probe(_q(resolution=False), ["m"], llm2) == \
        {"m": False}


async def test_unparseable_completion_is_not_confident():
    llm = ScriptedLLM({"m": "I have no idea."})
    assert await memorization_probe(_q(), ["m"], llm) == {"m": False}


async def test_mc_probe_matches_winning_option():
    q = _q(qtype=QType.MULTIPLE_CHOICE, resolution="Option B",
           options=("Option A", "Option B"))
    llm = ScriptedLLM({"hit": "From memory:\nOption B",
                       "miss": "From memory:\nOption A"})
    out = await memorization_probe(q, ["hit", "miss"], llm)
    assert out == {"hit": True, "miss": False}


async def test_unresolved_or_numeric_questions_never_flag():
    llm = ScriptedLLM({})
    assert await memorization_probe(_q(resolution=None), ["m"], llm) == \
        {"m": False}
    assert await memorization_probe(
        _q(qtype=QType.NUMERIC, resolution=42.0), ["m"], llm) == {"m": False}
    assert llm.calls == []                   # no spend on unprobeable types


async def test_probe_is_cached_second_run_makes_no_live_calls(
        tmp_path, monkeypatch):
    live_calls = []

    async def fake_call_live(self, model, prompt, temperature):
        live_calls.append(model)
        return "Probability: 97%", 0, 0, 0.0

    monkeypatch.setattr(CachedLLM, "_call_live", fake_call_live)
    llm = CachedLLM(tmp_path / "cache")
    q = _q(resolution=True)
    first = await memorization_probe(q, ["m1"], llm)
    second = await memorization_probe(q, ["m1"], llm)
    assert first == second == {"m1": True}
    assert live_calls == ["m1"]              # second probe fully cached
```

Create `tests/backtest/test_canary.py`:
```python
"""Canary leakage test: plant a document whose content exists only after T
through the pragmatic dated-search provider; the leak screen must drop it
from the research text and the evidence ledger must record the discard."""
from datetime import datetime, timezone

import backtest.timetravel.providers as providers
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot.config import BotConfig

T = datetime(2024, 6, 15, tzinfo=timezone.utc)


class ScriptedJudge:
    def __init__(self, completion):
        self.completion = completion

    async def invoke(self, model, prompt, temperature, run_index=0):
        return self.completion


async def test_planted_post_T_doc_is_dropped_and_ledgered(monkeypatch):
    canary = {"title": "Final results announced",
              "link": "https://canary.example/results",
              "date": "2024-07-01",
              "snippet": "The final results, announced July 1 2024, were ..."}
    clean = {"title": "Polling ahead of the vote",
             "link": "https://ok.example/polls",
             "date": "2024-05-20",
             "snippet": "Surveys conducted in May 2024 show ..."}

    async def fake_post(url, payload, headers):
        return {"organic": [clean, canary]}

    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setattr(providers, "_http_post_json", fake_post)
    ctx = AsOfContext(as_of=T, cutoff_mode="pragmatic",
                      llm=ScriptedJudge("1: CLEAN\n2: LEAK"))
    token = ACTIVE_CONTEXT.set(ctx)
    try:
        out = await providers._serper_dated_research(
            dict(question_text="Who won the vote?"), BotConfig())
    finally:
        ACTIVE_CONTEXT.reset(token)
    assert "canary.example" not in out       # planted doc never reaches research
    assert "ok.example" in out
    discards = [e for e in ctx.ledger if e.provider == "leak_screen"]
    assert len(discards) == 1
    assert discards[0].url == "https://canary.example/results"
    assert discards[0].doc_date == "2024-07-01"
    assert discards[0].date_trust == "low"
```
- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/backtest/test_probe.py tests/backtest/test_canary.py -v`
Expected: `tests/backtest/test_probe.py` FAILS with `ModuleNotFoundError: No module named 'backtest.timetravel.probe'`; `tests/backtest/test_canary.py` PASSES already (it exercises `backtest/timetravel/providers.py`'s serper path as a standing canary — keep it, it is this phase's leakage regression gate)
- [ ] **Step 3: Write minimal implementation**

Create `backtest/timetravel/probe.py`:
```python
"""Memorization probe (Halawi App. A.2 analogue): ask each model for the
outcome directly, zero research. A confident-correct answer upgrades the
question's stratum to contaminated-confirmed for that model. Deterministic
(temperature=0, run_index=0) and routed through CachedLLM, so re-probing a
corpus is free after the first pass. Numeric/discrete questions have no
crisp confident-correct criterion and are never flagged by the probe."""
from __future__ import annotations

import logging

from backtest.core import BacktestQuestion, QType
from backtest.timetravel.llm_cache import CachedLLM

logger = logging.getLogger(__name__)

CONFIDENT = 0.95

_BINARY_PROBE_PROMPT = """Answer from memory only — do not reason about what might happen; report what
you already know happened. The question below has already resolved.

QUESTION
{title}

RESOLUTION CRITERIA
{resolution_criteria}

If you know how this resolved, state it. The very last line of your answer
must be exactly:
Probability: ZZ%
where ZZ is the probability that it resolved Yes."""

_MC_PROBE_PROMPT = """Answer from memory only — do not reason about what might happen; report what
you already know happened. The question below has already resolved.

QUESTION
{title}

RESOLUTION CRITERIA
{resolution_criteria}

The options were: {options}

If you know how this resolved, the very last line of your answer must be
exactly the winning option's name, verbatim, and nothing else."""


def _binary_confident_correct(completion: str, resolution: bool) -> bool:
    from bot.parsing import ParseError, parse_binary
    try:
        p = parse_binary(completion)
    except ParseError:
        return False
    return p >= CONFIDENT if resolution else p <= 1.0 - CONFIDENT


def _mc_confident_correct(completion: str, resolution: str) -> bool:
    lines = completion.strip().splitlines()
    last = lines[-1].strip().lower() if lines else ""
    return last == str(resolution).strip().lower()


async def memorization_probe(q: BacktestQuestion, models: list[str],
                             llm: CachedLLM) -> dict[str, bool]:
    """True = contaminated-confirmed for that model on this question."""
    if q.resolution is None:
        return {m: False for m in models}
    if q.qtype == QType.BINARY:
        prompt = _BINARY_PROBE_PROMPT.format(
            title=q.title, resolution_criteria=q.resolution_criteria)
        resolved = bool(q.resolution)

        def check(c: str) -> bool:
            return _binary_confident_correct(c, resolved)
    elif q.qtype == QType.MULTIPLE_CHOICE:
        prompt = _MC_PROBE_PROMPT.format(
            title=q.title, resolution_criteria=q.resolution_criteria,
            options=", ".join(q.options))
        winning = str(q.resolution)

        def check(c: str) -> bool:
            return _mc_confident_correct(c, winning)
    else:
        return {m: False for m in models}    # numeric/discrete: not probeable
    out: dict[str, bool] = {}
    for m in models:
        try:
            completion = await llm.invoke(m, prompt, temperature=0.0,
                                          run_index=0)
        except Exception as e:
            logger.warning("memorization probe failed for %s: %s", m, e)
            out[m] = False
            continue
        out[m] = check(completion)
    return out
```
- [ ] **Step 4: Run tests to verify they pass**
Run: `pytest tests/backtest/test_probe.py tests/backtest/test_canary.py -v`
Expected: PASS (all 7 tests)
- [ ] **Step 5: Run the whole Phase 3 suite plus the bot regression suite**
Run: `pytest tests/ -v`
Expected: PASS — full repo green (Phase 1–3 backtest tests + all pre-existing bot tests)
- [ ] **Step 6: Commit**
```bash
git add backtest/timetravel/probe.py tests/backtest/test_probe.py tests/backtest/test_canary.py && git commit -m "feat(backtest): memorization probe (cached, temp-0) + canary leakage test" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
## Phase 4 — Systems, experiments, reports, CLI (Tasks 25–31)

This phase builds the `ForecastSystem` protocol, the `CalibratedEnsembleAdapter` (the real bot under time travel), the floor baselines, the YAML experiment layer with budget profiles, the resumable runner with cost gate + scoring pass, the markdown report builder, and the `scripts/backtest.py` CLI.

**Prerequisites (files created by earlier phases; these tasks import them):**
- `backtest/core.py` — `QType`, `BacktestQuestion`, `Forecast`, `StageRecord`, `LedgerEntry`, `ForecastArtifact`, `artifact_to_dict`, `artifact_from_dict`
- `backtest/scoring/metaculus_scores.py` (`log_score`, `baseline_score`, `head_to_head`, `brier`), `backtest/scoring/opponents.py` (`cp_opponent`, `market_opponent`), `backtest/scoring/diagnostics.py`, `backtest/scoring/stats.py` (`paired_bootstrap`, `paired_t`)
- `backtest/corpus/store.py` (`CorpusStore`), `backtest/corpus/crosswalk.py` (`market_for`), `backtest/corpus/model_releases.py` (`model_set_hash` — made total over unknown slugs, `RELEASE_DATES.get(m, "unknown")`, by the parallel phase-2 fix)
- `backtest/timetravel/asof.py` (`AsOfContext`, `ACTIVE_CONTEXT`, `horizon_to_asof`), `backtest/timetravel/llm_cache.py` (`CachedLLM`)
- `bot/config.py` with the `as_of: datetime | None = None` field threaded through `bot/ensemble_bot.py` and `bot/research.py`

**Conventions in force:** pytest `asyncio_mode = "auto"` (plain `async def test_*`, no decorators — matches `tests/test_research.py`); tests never touch the network (research and LLM seams are stubbed/monkeypatched); commit messages `feat(backtest):` / `test(backtest):` with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.

**Design decisions encoded below (resolved ambiguities):**
1. Systems cannot know the horizon label (it is not in `AsOfContext`), so every system writes `horizon=""` into its artifact and `backtest/experiments/runner.py` stamps `artifact.horizon` after each call (`ForecastArtifact` is mutable).
2. `CachedLLM.invoke()` returns only `str`, so `member_llm` StageRecords carry `cost_usd=0`/`tokens=0`; the report sums whatever StageRecords contain (works for any system that fills them).
3. `build_report(run_dir)` takes no store, but diagnostics and worst-10 titles need resolutions — the runner records `corpus_root` in `manifest.json`; the report opens the store from it and degrades gracefully when absent.
4. The smoke profile's single-member single-run override would violate `min_members=3`, so the runner sets `min_members=1` whenever it applies `ensemble_override`.
5. `run_index` is the per-member fan-out position (`0..runs-1`).
6. The cost estimate counts only LLM variants (`spec.variants`); baselines are free.
7. The cost brake aborts via `RuntimeError`; "resume pending only" = skip statuses `done`/`skipped`, retry `error`.
8. Baselines record `contamination="clean"` (no LLM involved).
9. `QType.DISCRETE` routes to the numeric handler; `cdf_size != 201` raises `NotImplementedError` (becomes an error artifact) for MVP.

---

### Task 25: ForecastSystem protocol (`backtest/systems/protocol.py`)

**Files:**
- Create: `backtest/systems/__init__.py`
- Create: `backtest/systems/protocol.py`
- Create (only if missing): `tests/backtest/__init__.py` (empty)
- Test: `tests/backtest/test_systems_protocol.py`

- [ ] **Step 1: Write the failing test**

`tests/backtest/test_systems_protocol.py`:

```python
"""ForecastSystem protocol conformance — pure typing, no I/O."""
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, Forecast, ForecastArtifact, QType
from backtest.systems.protocol import ForecastSystem


class _Dummy:
    name = "dummy"

    def config_fingerprint(self) -> str:
        return "dummy-v1"

    async def forecast(self, q, ctx) -> ForecastArtifact:
        return ForecastArtifact(
            qid=q.qid, horizon="", as_of=datetime.now(timezone.utc),
            system_name=self.name, config_fingerprint="dummy-v1",
            forecast=Forecast(qtype=QType.BINARY, binary_prob=0.5))


def test_conforming_instance_passes_runtime_check():
    assert isinstance(_Dummy(), ForecastSystem)


def test_nonconforming_instance_fails_runtime_check():
    class Nope:
        pass

    assert not isinstance(Nope(), ForecastSystem)


def test_protocol_module_reexports_core_types():
    from backtest.systems import protocol
    assert protocol.BacktestQuestion is BacktestQuestion
    assert protocol.Forecast is Forecast
    assert protocol.ForecastArtifact is ForecastArtifact
    assert protocol.QType is QType
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_systems_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.systems'`

- [ ] **Step 3: Write minimal implementation**

`backtest/systems/__init__.py`:

```python
```

(empty file)

`backtest/systems/protocol.py`:

```python
"""The compatibility contract every system under test implements (spec §5).

Semantics (mirrored in docs/backtesting/COMPATIBILITY.md):
- honor ctx.as_of — no wall-clock reads;
- route every model call through ctx.llm (cache/probe instrumentation);
- no unledgered network research;
- return a ForecastArtifact with per-stage inputs/outputs populated;
- NEVER raise from forecast() — failures return an artifact with error set.

Core types are re-exported so future systems can depend on this one module.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           LedgerEntry, QType, StageRecord)
from backtest.timetravel.asof import AsOfContext

__all__ = ["ForecastSystem", "BacktestQuestion", "Forecast",
           "ForecastArtifact", "LedgerEntry", "QType", "StageRecord",
           "AsOfContext"]


@runtime_checkable
class ForecastSystem(Protocol):
    name: str

    def config_fingerprint(self) -> str:
        ...

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        ...
```

If `tests/backtest/__init__.py` does not exist yet, create it empty.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_systems_protocol.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backtest/systems/__init__.py backtest/systems/protocol.py tests/backtest/test_systems_protocol.py tests/backtest/__init__.py && git commit -m "feat(backtest): ForecastSystem protocol with runtime conformance check" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 26: CalibratedEnsembleAdapter (`backtest/systems/ensemble_adapter.py`)

The real bot under time travel: `dataclasses.replace`'d `BotConfig` with `as_of=ctx.as_of`, all LLM calls routed through `ctx.llm` with an explicit `run_index` (the stock `CalibratedEnsembleBot._fan_out` at `bot/ensemble_bot.py:91-106` does **not** pass a run index, so `_fan_out` is overridden to enumerate `(member, run_idx)` pairs), `ACTIVE_CONTEXT` set around the whole call, full stage provenance, never raises.

**Files:**
- Create: `backtest/systems/ensemble_adapter.py`
- Test: `tests/backtest/test_ensemble_adapter.py`

- [ ] **Step 1: Write the failing tests (binary path, run_index routing, stages, error isolation, contamination)**

`tests/backtest/test_ensemble_adapter.py`:

```python
"""CalibratedEnsembleAdapter tests — no network, no forecasting-tools LLMs:
bot.research.run_research is monkeypatched and every model call goes through
a stub ctx.llm."""
from datetime import datetime, timezone

import pytest

import bot.research as research_mod
from backtest.core import BacktestQuestion, QType
from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot.config import BotConfig, EnsembleMember
from bot.research import ResearchResult

AS_OF = datetime(2024, 5, 15, 12, 0, tzinfo=timezone.utc)

BINARY_REPLY = "Reasoning...\nProbability: 64%"


class StubLLM:
    def __init__(self, reply=BINARY_REPLY, fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = []

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append(dict(model=model, prompt=prompt,
                               temperature=temperature, run_index=run_index))
        if self.fail:
            raise RuntimeError("boom")
        return self.reply


class FakeStore:
    def __init__(self, result="clean"):
        self.result = result
        self.calls = []

    def get_contamination(self, qid, model_set_hash):
        self.calls.append((qid, model_set_hash))
        return self.result


def _cfg(tmp_path, **kw):
    return BotConfig(
        ensemble=[EnsembleMember("fake/model-a", runs=2, temperature=0.3),
                  EnsembleMember("fake/model-b", runs=2, temperature=0.7)],
        calibration_dir=str(tmp_path / "calib"),
        log_path=str(tmp_path / "forecasts.jsonl"),
        **kw)


def _binary_q():
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="Will X happen?",
        description="Background.", resolution_criteria="Resolves YES if X.",
        fine_print="None.",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc))


@pytest.fixture
def stub_research(monkeypatch):
    seen = {}

    async def fake_run_research(question, cfg_):
        seen["ctx"] = ACTIVE_CONTEXT.get()
        seen["as_of"] = cfg_.as_of
        seen["question"] = question
        return ResearchResult(text="stub research", provider="asknews",
                              providers=("asknews",))

    monkeypatch.setattr(research_mod, "run_research", fake_run_research)
    return seen


async def test_binary_forecast_routes_through_ctx_llm(tmp_path, stub_research):
    llm = StubLLM()
    ctx = AsOfContext(as_of=AS_OF, llm=llm)
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_binary_q(), ctx)

    assert art.error is None
    assert art.qid == "metaculus:1"
    assert art.as_of == AS_OF
    assert art.cutoff_mode == "strict"
    assert art.system_name == "calibrated_ensemble"
    assert art.config_fingerprint == adapter.config_fingerprint()
    assert art.forecast.qtype is QType.BINARY
    # identity Platt (empty calibration dir) + caps [0.02, 0.98] -> 0.64
    assert art.forecast.binary_prob == pytest.approx(0.64, abs=1e-6)
    # 2 members x 2 runs routed through ctx.llm with per-member run_index
    assert len(llm.calls) == 4
    by_model = {}
    for c in llm.calls:
        by_model.setdefault(c["model"], []).append(c["run_index"])
    assert sorted(by_model["fake/model-a"]) == [0, 1]
    assert sorted(by_model["fake/model-b"]) == [0, 1]
    temps = {c["model"]: c["temperature"] for c in llm.calls}
    assert temps == {"fake/model-a": 0.3, "fake/model-b": 0.7}
    # as_of threaded into the prompt as `today`, research text injected
    assert "2024-05-15" in llm.calls[0]["prompt"]
    assert "stub research" in llm.calls[0]["prompt"]


async def test_active_context_set_during_research_and_reset_after(
        tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM())
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    await adapter.forecast(_binary_q(), ctx)
    assert stub_research["ctx"] is ctx
    assert stub_research["as_of"] == AS_OF
    assert ACTIVE_CONTEXT.get() is None


async def test_stage_records_cover_pipeline(tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM())
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_binary_q(), ctx)
    names = [s.stage for s in art.stages]
    assert names.count("research") == 1
    assert names.count("member_llm") == 4
    assert names.count("aggregation") == 1
    assert names.count("calibration") == 1
    research = next(s for s in art.stages if s.stage == "research")
    assert research.outputs["text"] == "stub research"
    assert research.outputs["provider"] == "asknews"
    member = next(s for s in art.stages if s.stage == "member_llm")
    assert member.model in ("fake/model-a", "fake/model-b")
    assert "prompt" in member.inputs
    assert member.outputs["completion"] == BINARY_REPLY
    agg = next(s for s in art.stages if s.stage == "aggregation")
    assert agg.inputs["method"] == "trimmed_gmo"
    assert agg.outputs["aggregate_raw"] == pytest.approx(0.64, abs=1e-6)
    cal = next(s for s in art.stages if s.stage == "calibration")
    assert cal.outputs["final"] == pytest.approx(0.64, abs=1e-6)


async def test_failure_returns_error_artifact_never_raises(
        tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM(fail=True))
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_binary_q(), ctx)
    assert art.forecast is None
    assert "ensemble members survived" in art.error
    failed = [s for s in art.stages if s.stage == "member_llm"]
    assert len(failed) == 4
    assert all("boom" in s.outputs["error"] for s in failed)
    assert ACTIVE_CONTEXT.get() is None


async def test_contamination_from_store_else_contaminated(
        tmp_path, stub_research):
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    # no store -> conservative default
    art = await adapter.forecast(
        _binary_q(), AsOfContext(as_of=AS_OF, llm=StubLLM()))
    assert art.contamination == "contaminated"
    # store present -> looked up by (qid, model_set_hash)
    fake = FakeStore("clean")
    art2 = await adapter.forecast(
        _binary_q(), AsOfContext(as_of=AS_OF, llm=StubLLM(), store=fake))
    assert art2.contamination == "clean"
    qid, mhash = fake.calls[0]
    assert qid == "metaculus:1"
    assert len(mhash) == 12


def test_variant_patches_config():
    v = CalibratedEnsembleAdapter.variant(aggregation_method="mean",
                                          cap_hi=0.95)
    assert v.cfg.aggregation_method == "mean"
    assert v.cfg.cap_hi == 0.95
    assert (v.config_fingerprint()
            != CalibratedEnsembleAdapter().config_fingerprint())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backtest/test_ensemble_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.systems.ensemble_adapter'`

- [ ] **Step 3: Write the implementation (binary path)**

`backtest/systems/ensemble_adapter.py`:

```python
"""CalibratedEnsembleAdapter: the live CalibratedEnsembleBot run under an
AsOfContext — research via bot.research (historical providers are installed
into its namespace by the harness), every LLM call routed through ctx.llm
(CachedLLM), full stage provenance recorded into the ForecastArtifact.

forecast() NEVER raises: any failure returns an artifact with error set and
forecast=None. The horizon label is unknown at this level — artifacts carry
horizon="" and the experiment runner stamps it.

Cost note: CachedLLM.invoke() returns only the completion string, so
member_llm StageRecords carry cost_usd=0 / tokens=0 for now; report cost
totals sum whatever StageRecords contain.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timezone

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord)
from backtest.corpus.model_releases import model_set_hash
from backtest.timetravel.asof import ACTIVE_CONTEXT, AsOfContext
from bot import research as bot_research
from bot.config import BotConfig, config_hash
from bot.ensemble_bot import CalibratedEnsembleBot

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_set_hash(cfg: BotConfig) -> str:
    """Delegate to the canonical corpus implementation so the adapter and
    the corpus stratifier always agree on the hash (model_set_hash is total
    over unknown slugs — RELEASE_DATES.get(m, "unknown"))."""
    return model_set_hash([m.model for m in cfg.ensemble])


def _question_dict(q: BacktestQuestion) -> dict:
    return dict(question_text=q.title,
                resolution_criteria=q.resolution_criteria,
                fine_print=q.fine_print,
                qid=q.qid)


def _common_fields(q: BacktestQuestion) -> dict:
    return dict(
        question_text=q.title,
        background_info=q.description,
        resolution_criteria=q.resolution_criteria,
        fine_print=q.fine_print,
        close_iso=(q.scheduled_close_time.isoformat()
                   if q.scheduled_close_time else ""))


class _BacktestEnsembleBot(CalibratedEnsembleBot):
    """Subclass whose only LLM seam is ctx.llm.

    The stock _fan_out (bot/ensemble_bot.py) does not pass a run index, so
    it is overridden here to enumerate (member, run_idx) pairs — the
    completion-cache key needs run_index to keep same-prompt same-member
    runs distinct."""

    def __init__(self, bot_config: BotConfig, ctx: AsOfContext,
                 stages: list[StageRecord]):
        super().__init__(bot_config=bot_config)
        self._ctx = ctx
        self._stages = stages

    async def _invoke_model(self, member_and_prompt) -> str:
        member, prompt = member_and_prompt
        return await self._invoke_member(member, prompt, 0)

    async def _invoke_member(self, member, prompt: str, run_idx: int) -> str:
        if self._ctx.llm is None:
            raise RuntimeError("AsOfContext.llm is not set")
        started = _now_iso()
        try:
            text = await self._ctx.llm.invoke(
                member.model, prompt, member.temperature, run_index=run_idx)
        except Exception as e:
            self._stages.append(StageRecord(
                stage="member_llm", model=member.model,
                inputs={"prompt": prompt, "temperature": member.temperature,
                        "run_index": run_idx},
                outputs={"error": f"{type(e).__name__}: {e}"},
                started_at=started, ended_at=_now_iso()))
            raise
        self._stages.append(StageRecord(
            stage="member_llm", model=member.model,
            inputs={"prompt": prompt, "temperature": member.temperature,
                    "run_index": run_idx},
            outputs={"completion": text},
            started_at=started, ended_at=_now_iso()))
        return text

    async def _fan_out(self, prompt: str):
        jobs, models = [], []
        for member in self.cfg.ensemble:
            for run_idx in range(member.runs):
                jobs.append(self._invoke_member(member, prompt, run_idx))
                models.append(member.model)
        results = await asyncio.gather(*jobs, return_exceptions=True)
        texts, kept = [], []
        for r, m in zip(results, models):
            if isinstance(r, BaseException):
                logger.warning("member %s failed: %s", m, r)
            else:
                texts.append(r)
                kept.append(m)
        return texts, kept


async def _forecast_binary(bot: _BacktestEnsembleBot, q: BacktestQuestion,
                           research_text: str,
                           stages: list[StageRecord]) -> Forecast:
    r = await bot.run_binary_ensemble(research=research_text,
                                      **_common_fields(q))
    stages.append(StageRecord(
        stage="aggregation",
        inputs={"member_probs": list(r.member_probs),
                "member_models": list(r.member_models),
                "method": bot.cfg.aggregation_method},
        outputs={"aggregate_raw": r.aggregate_raw, "spread": r.spread}))
    stages.append(StageRecord(
        stage="calibration",
        inputs={"aggregate_raw": r.aggregate_raw,
                "cap_lo": bot.cfg.cap_lo, "cap_hi": bot.cfg.cap_hi},
        outputs={"final": r.final}))
    return Forecast(qtype=q.qtype, binary_prob=r.final)


_HANDLERS = {
    QType.BINARY: _forecast_binary,
}


class CalibratedEnsembleAdapter:
    """ForecastSystem implementation wrapping the live ensemble bot."""

    def __init__(self, cfg: BotConfig | None = None):
        self.cfg = cfg or BotConfig()
        self.name = "calibrated_ensemble"

    @classmethod
    def variant(cls, base: BotConfig | None = None,
                **patch) -> "CalibratedEnsembleAdapter":
        return cls(replace(base or BotConfig(), **patch))

    def config_fingerprint(self) -> str:
        # config_hash pops as_of before hashing, so the fingerprint is
        # stable across horizons of the same configuration.
        return config_hash(self.cfg)

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        stages: list[StageRecord] = []
        token = ACTIVE_CONTEXT.set(ctx)
        try:
            run_cfg = replace(self.cfg, as_of=ctx.as_of)
            bot = _BacktestEnsembleBot(run_cfg, ctx, stages)
            started = _now_iso()
            res = await bot_research.run_research(_question_dict(q), run_cfg)
            stages.append(StageRecord(
                stage="research",
                inputs={"providers": list(run_cfg.research_providers)},
                outputs={"text": res.text, "provider": res.provider},
                started_at=started, ended_at=_now_iso()))
            fc = await self._dispatch(bot, q, res.text, stages)
            return self._artifact(q, ctx, fc, stages)
        except Exception as e:
            logger.warning("forecast failed for %s: %s", q.qid, e)
            return self._artifact(q, ctx, None, stages,
                                  error=f"{type(e).__name__}: {e}")
        finally:
            ACTIVE_CONTEXT.reset(token)

    async def _dispatch(self, bot: _BacktestEnsembleBot, q: BacktestQuestion,
                        research_text: str,
                        stages: list[StageRecord]) -> Forecast:
        handler = _HANDLERS.get(q.qtype)
        if handler is None:
            raise NotImplementedError(f"unsupported qtype: {q.qtype!r}")
        return await handler(bot, q, research_text, stages)

    def _artifact(self, q: BacktestQuestion, ctx: AsOfContext,
                  fc: Forecast | None, stages: list[StageRecord],
                  error: str | None = None) -> ForecastArtifact:
        return ForecastArtifact(
            qid=q.qid, horizon="", as_of=ctx.as_of,
            system_name=self.name,
            config_fingerprint=self.config_fingerprint(),
            forecast=fc, stages=stages,
            evidence_ledger=list(ctx.ledger),
            contamination=self._contamination(q, ctx),
            cutoff_mode=ctx.cutoff_mode, error=error)

    def _contamination(self, q: BacktestQuestion, ctx: AsOfContext) -> str:
        if ctx.store is None:
            return "contaminated"
        try:
            return (ctx.store.get_contamination(
                q.qid, _model_set_hash(self.cfg)) or "contaminated")
        except Exception as e:
            logger.warning("contamination lookup failed for %s: %s",
                           q.qid, e)
            return "contaminated"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/backtest/test_ensemble_adapter.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add backtest/systems/ensemble_adapter.py tests/backtest/test_ensemble_adapter.py && git commit -m "feat(backtest): CalibratedEnsembleAdapter binary path — ctx.llm routing with run_index, ACTIVE_CONTEXT scoping, stage provenance, never-raise" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Write the failing tests for MC + numeric dispatch**

Append to `tests/backtest/test_ensemble_adapter.py`:

```python
MC_REPLY = "Analysis...\nRed: 70%\nBlue: 30%"
NUMERIC_REPLY = (
    "Percentile 2.5: 10\nPercentile 10: 20\nPercentile 25: 30\n"
    "Percentile 50: 40\nPercentile 75: 50\nPercentile 90: 60\n"
    "Percentile 97.5: 70")


def _mc_q():
    return BacktestQuestion(
        qid="metaculus:2", platform="metaculus", native_id="2",
        qtype=QType.MULTIPLE_CHOICE, title="Which colour?",
        options=("Red", "Blue"),
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc))


def _numeric_q(**scaling_overrides):
    scaling = {"range_min": 0.0, "range_max": 100.0, "zero_point": None,
               "open_lower_bound": False, "open_upper_bound": True,
               "cdf_size": 201, "unit": "widgets"}
    scaling.update(scaling_overrides)
    return BacktestQuestion(
        qid="metaculus:3", platform="metaculus", native_id="3",
        qtype=QType.NUMERIC, title="How many widgets?",
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc),
        scaling=scaling)


async def test_mc_dispatch(tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM(reply=MC_REPLY))
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_mc_q(), ctx)
    assert art.error is None
    assert art.forecast.qtype is QType.MULTIPLE_CHOICE
    assert art.forecast.mc_probs["Red"] == pytest.approx(0.7, abs=1e-6)
    assert sum(art.forecast.mc_probs.values()) == pytest.approx(1.0)
    names = [s.stage for s in art.stages]
    assert names.count("aggregation") == 1
    assert names.count("calibration") == 1


async def test_numeric_dispatch_builds_201_point_cdf(tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM(reply=NUMERIC_REPLY))
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_numeric_q(), ctx)
    assert art.error is None
    cdf = art.forecast.cdf
    assert isinstance(cdf, tuple)
    assert len(cdf) == 201
    assert cdf[0] == pytest.approx(0.0, abs=1e-9)      # closed lower bound
    assert cdf[-1] == pytest.approx(0.999, abs=1e-9)   # open upper bound
    assert all(b - a >= 5e-05 - 1e-9 for a, b in zip(cdf, cdf[1:]))


async def test_numeric_nonstandard_cdf_size_is_error_artifact(
        tmp_path, stub_research):
    ctx = AsOfContext(as_of=AS_OF, llm=StubLLM(reply=NUMERIC_REPLY))
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.forecast(_numeric_q(cdf_size=101), ctx)
    assert art.forecast is None
    assert "NotImplementedError" in art.error
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `pytest tests/backtest/test_ensemble_adapter.py -v`
Expected: FAIL — `test_mc_dispatch` and `test_numeric_dispatch_builds_201_point_cdf` fail with `AssertionError` because `art.error` is `"NotImplementedError: unsupported qtype: ..."` and `art.forecast` is `None` (no handler registered yet). The 6 earlier tests and `test_numeric_nonstandard_cdf_size_is_error_artifact` may pass.

- [ ] **Step 8: Extend the implementation with MC + numeric handlers**

In `backtest/systems/ensemble_adapter.py`, add to the imports block:

```python
from bot.distributions import percentiles_to_cdf
```

Add these two functions immediately after `_forecast_binary`:

```python
async def _forecast_mc(bot: _BacktestEnsembleBot, q: BacktestQuestion,
                       research_text: str,
                       stages: list[StageRecord]) -> Forecast:
    r = await bot.run_mc_ensemble(research=research_text,
                                  options=list(q.options),
                                  **_common_fields(q))
    stages.append(StageRecord(
        stage="aggregation",
        inputs={"member_dicts": list(r.member_dicts),
                "member_models": list(r.member_models),
                "mc_floor": bot.cfg.mc_floor},
        outputs={"final": dict(r.final)}))
    stages.append(StageRecord(
        stage="calibration",
        inputs={"note": "MC floor+renormalize happens in aggregation; "
                        "no Platt layer for MC"},
        outputs={"final": dict(r.final)}))
    return Forecast(qtype=q.qtype, mc_probs=dict(r.final))


async def _forecast_numeric(bot: _BacktestEnsembleBot, q: BacktestQuestion,
                            research_text: str,
                            stages: list[StageRecord]) -> Forecast:
    scaling = q.scaling or {}
    if "range_min" not in scaling or "range_max" not in scaling:
        raise ValueError(f"numeric question {q.qid} lacks scaling bounds")
    if int(scaling.get("cdf_size") or 201) != 201:
        raise NotImplementedError(
            "cdf_size != 201 (discrete grids) unsupported in MVP")
    lo, hi = scaling["range_min"], scaling["range_max"]
    open_lo = bool(scaling.get("open_lower_bound"))
    open_hi = bool(scaling.get("open_upper_bound"))
    lo_msg = (f"The value is likely not lower than {lo}." if open_lo
              else f"The outcome can not be lower than {lo}.")
    hi_msg = (f"The value is likely not higher than {hi}." if open_hi
              else f"The outcome can not be higher than {hi}.")
    r = await bot.run_numeric_ensemble(
        research=research_text, units=scaling.get("unit") or "",
        lower_bound_message=lo_msg, upper_bound_message=hi_msg,
        **_common_fields(q))
    stages.append(StageRecord(
        stage="aggregation",
        inputs={"member_percentile_dicts": list(r.member_percentile_dicts),
                "member_models": list(r.member_models)},
        outputs={"median_percentiles": dict(r.median_percentiles)}))
    # NOTE: percentiles_to_cdf builds a linear-grid PCHIP CDF; log-scaled
    # questions (zero_point set) are scored on that approximation in MVP.
    cdf = percentiles_to_cdf(r.median_percentiles, lo, hi, open_lo, open_hi)
    cdf_t = tuple(float(v) for v in cdf)
    stages.append(StageRecord(
        stage="calibration",
        inputs={"median_percentiles": dict(r.median_percentiles),
                "scaling": dict(scaling)},
        outputs={"cdf_len": len(cdf_t), "cdf_first": cdf_t[0],
                 "cdf_last": cdf_t[-1]}))
    return Forecast(qtype=q.qtype, cdf=cdf_t)
```

Replace the `_HANDLERS` assignment with:

```python
_HANDLERS = {
    QType.BINARY: _forecast_binary,
    QType.MULTIPLE_CHOICE: _forecast_mc,
    QType.NUMERIC: _forecast_numeric,
    QType.DISCRETE: _forecast_numeric,   # 201-grid only; else error artifact
}
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/backtest/test_ensemble_adapter.py -v`
Expected: PASS (10 passed)

- [ ] **Step 10: Commit**

```bash
git add backtest/systems/ensemble_adapter.py tests/backtest/test_ensemble_adapter.py && git commit -m "feat(backtest): MC + numeric dispatch in CalibratedEnsembleAdapter (percentiles_to_cdf, 201-grid guard)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 27: Stage entry points — `from_research_text` / `from_member_forecasts`

Backtest *part* of the system: identical frozen research into two aggregation variants isolates that stage causally (spec §5.2). `from_member_forecasts` is **binary-only for MVP** (a `list[float]` of member probabilities) and raises `NotImplementedError` for other types.

**Files:**
- Modify: `backtest/systems/ensemble_adapter.py` (add two methods + imports)
- Test: `tests/backtest/test_ensemble_adapter_stages.py`

- [ ] **Step 1: Write the failing tests**

`tests/backtest/test_ensemble_adapter_stages.py`:

```python
"""Stage entry points: from_research_text (skip research) and
from_member_forecasts (aggregation+calibration only, no LLM)."""
from datetime import datetime, timezone

import pytest

import bot.research as research_mod
from backtest.core import BacktestQuestion, QType
from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
from backtest.timetravel.asof import AsOfContext
from bot.config import BotConfig, EnsembleMember

AS_OF = datetime(2024, 5, 15, 12, 0, tzinfo=timezone.utc)


class StubLLM:
    def __init__(self, reply="Reasoning...\nProbability: 64%"):
        self.reply = reply
        self.calls = []

    async def invoke(self, model, prompt, temperature, run_index=0):
        self.calls.append(dict(model=model, prompt=prompt,
                               temperature=temperature, run_index=run_index))
        return self.reply


def _cfg(tmp_path, **kw):
    return BotConfig(
        ensemble=[EnsembleMember("fake/model-a", runs=2, temperature=0.3),
                  EnsembleMember("fake/model-b", runs=2, temperature=0.7)],
        calibration_dir=str(tmp_path / "calib"),
        log_path=str(tmp_path / "forecasts.jsonl"),
        **kw)


def _binary_q():
    return BacktestQuestion(
        qid="metaculus:1", platform="metaculus", native_id="1",
        qtype=QType.BINARY, title="Will X happen?",
        description="Background.", resolution_criteria="Resolves YES if X.",
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc))


def _mc_q():
    return BacktestQuestion(
        qid="metaculus:2", platform="metaculus", native_id="2",
        qtype=QType.MULTIPLE_CHOICE, title="Which colour?",
        options=("Red", "Blue"),
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc))


async def test_from_research_text_skips_run_research(tmp_path, monkeypatch):
    async def explode(question, cfg_):
        raise AssertionError("run_research must not be called")

    monkeypatch.setattr(research_mod, "run_research", explode)
    llm = StubLLM()
    ctx = AsOfContext(as_of=AS_OF, llm=llm)
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    art = await adapter.from_research_text(_binary_q(), ctx,
                                           "FROZEN RESEARCH")
    assert art.error is None
    assert art.forecast.binary_prob == pytest.approx(0.64, abs=1e-6)
    research = next(s for s in art.stages if s.stage == "research")
    assert research.outputs["provider"] == "injected"
    assert research.outputs["text"] == "FROZEN RESEARCH"
    # the injected text actually reaches the member prompts
    assert "FROZEN RESEARCH" in llm.calls[0]["prompt"]


async def test_from_member_forecasts_aggregation_and_calibration_only(
        tmp_path):
    adapter = CalibratedEnsembleAdapter(
        _cfg(tmp_path, aggregation_method="mean"))
    ctx = AsOfContext(as_of=AS_OF)            # no llm: must not be needed
    art = await adapter.from_member_forecasts(_binary_q(), ctx,
                                              [0.6, 0.7, 0.8])
    assert art.error is None
    assert art.forecast.binary_prob == pytest.approx(0.7, abs=1e-9)
    assert [s.stage for s in art.stages] == ["aggregation", "calibration"]
    agg = art.stages[0]
    assert agg.inputs["member_probs"] == [0.6, 0.7, 0.8]
    assert agg.inputs["method"] == "mean"


async def test_from_member_forecasts_applies_caps(tmp_path):
    adapter = CalibratedEnsembleAdapter(
        _cfg(tmp_path, aggregation_method="mean"))
    ctx = AsOfContext(as_of=AS_OF)
    art = await adapter.from_member_forecasts(_binary_q(), ctx,
                                              [0.999, 0.999, 0.999])
    assert art.forecast.binary_prob == pytest.approx(0.98)   # cap_hi


async def test_from_member_forecasts_non_binary_raises(tmp_path):
    adapter = CalibratedEnsembleAdapter(_cfg(tmp_path))
    ctx = AsOfContext(as_of=AS_OF)
    with pytest.raises(NotImplementedError):
        await adapter.from_member_forecasts(_mc_q(), ctx, [{"Red": 1.0}])


async def test_from_member_forecasts_runtime_error_is_isolated(tmp_path):
    adapter = CalibratedEnsembleAdapter(
        _cfg(tmp_path, aggregation_method="mean"))
    ctx = AsOfContext(as_of=AS_OF)
    art = await adapter.from_member_forecasts(_binary_q(), ctx, [])
    assert art.forecast is None
    assert "no forecasts" in art.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backtest/test_ensemble_adapter_stages.py -v`
Expected: FAIL with `AttributeError: 'CalibratedEnsembleAdapter' object has no attribute 'from_research_text'`

- [ ] **Step 3: Write the implementation**

In `backtest/systems/ensemble_adapter.py`, add to the imports block:

```python
from pathlib import Path

from bot.aggregation import aggregate_binary
from bot.calibration import calibrate_binary, load_platt
```

Add these two methods to the `CalibratedEnsembleAdapter` class, after `forecast()`:

```python
    async def from_research_text(self, q: BacktestQuestion, ctx: AsOfContext,
                                 research_text: str) -> ForecastArtifact:
        """Stage entry point: skip run_research entirely and forecast from
        the given (frozen) research text — isolates downstream stages."""
        stages: list[StageRecord] = []
        token = ACTIVE_CONTEXT.set(ctx)
        try:
            run_cfg = replace(self.cfg, as_of=ctx.as_of)
            bot = _BacktestEnsembleBot(run_cfg, ctx, stages)
            stages.append(StageRecord(
                stage="research",
                inputs={"injected": True},
                outputs={"text": research_text, "provider": "injected"},
                started_at=_now_iso(), ended_at=_now_iso()))
            fc = await self._dispatch(bot, q, research_text, stages)
            return self._artifact(q, ctx, fc, stages)
        except Exception as e:
            logger.warning("from_research_text failed for %s: %s", q.qid, e)
            return self._artifact(q, ctx, None, stages,
                                  error=f"{type(e).__name__}: {e}")
        finally:
            ACTIVE_CONTEXT.reset(token)

    async def from_member_forecasts(self, q: BacktestQuestion,
                                    ctx: AsOfContext,
                                    member_outputs: list) -> ForecastArtifact:
        """Stage entry point: bypass research and the LLM entirely; run
        aggregation + calibration on pre-computed member outputs.

        BINARY-ONLY for MVP — member_outputs is a list[float] of member
        probabilities. Other question types raise NotImplementedError
        (a programming error, not a runtime failure)."""
        if q.qtype is not QType.BINARY:
            raise NotImplementedError(
                "from_member_forecasts is binary-only for MVP")
        stages: list[StageRecord] = []
        try:
            run_cfg = replace(self.cfg, as_of=ctx.as_of)
            probs = [float(p) for p in member_outputs]
            started = _now_iso()
            raw = aggregate_binary(probs, run_cfg.aggregation_method)
            stages.append(StageRecord(
                stage="aggregation",
                inputs={"member_probs": probs,
                        "method": run_cfg.aggregation_method},
                outputs={"aggregate_raw": raw},
                started_at=started, ended_at=_now_iso()))
            platt = load_platt(Path(run_cfg.calibration_dir) / "binary.json")
            final = calibrate_binary(raw, platt,
                                     run_cfg.cap_lo, run_cfg.cap_hi)
            stages.append(StageRecord(
                stage="calibration",
                inputs={"aggregate_raw": raw, "cap_lo": run_cfg.cap_lo,
                        "cap_hi": run_cfg.cap_hi},
                outputs={"final": final}))
            return self._artifact(q, ctx,
                                  Forecast(qtype=QType.BINARY,
                                           binary_prob=final), stages)
        except Exception as e:
            logger.warning("from_member_forecasts failed for %s: %s",
                           q.qid, e)
            return self._artifact(q, ctx, None, stages,
                                  error=f"{type(e).__name__}: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/backtest/test_ensemble_adapter_stages.py tests/backtest/test_ensemble_adapter.py -v`
Expected: PASS (15 passed — the 10 adapter tests must stay green)

- [ ] **Step 5: Commit**

```bash
git add backtest/systems/ensemble_adapter.py tests/backtest/test_ensemble_adapter_stages.py && git commit -m "feat(backtest): stage entry points from_research_text / from_member_forecasts (binary-only MVP)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 28: Baseline systems (`backtest/systems/baselines.py`)

Floors every report contextualizes variants against: chance (0.5 / uniform / uniform CDF), CP echo, market echo. Echo baselines read `store.value_at(...)` at `ctx.as_of` and **skip** (artifact with `forecast=None`, `error="no data: ..."`) when data is absent. No LLM is involved, so baseline artifacts carry `contamination="clean"`.

**Files:**
- Create: `backtest/systems/baselines.py`
- Test: `tests/backtest/test_baselines.py`

- [ ] **Step 1: Write the failing tests**

`tests/backtest/test_baselines.py`:

```python
"""Baseline systems — real CorpusStore in tmp_path, no network."""
from datetime import datetime, timezone

import pytest

import backtest.systems.baselines as baselines_mod
from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore
from backtest.systems.baselines import (ChanceBaseline, CPEchoBaseline,
                                        MarketEchoBaseline)
from backtest.systems.protocol import ForecastSystem
from backtest.timetravel.asof import AsOfContext

AS_OF = datetime(2024, 5, 25, tzinfo=timezone.utc)


def _q(qtype=QType.BINARY, **kw):
    defaults = dict(qid="metaculus:1", platform="metaculus", native_id="1",
                    qtype=qtype, title="Q?",
                    open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    scheduled_close_time=datetime(2024, 6, 1,
                                                  tzinfo=timezone.utc))
    defaults.update(kw)
    return BacktestQuestion(**defaults)


@pytest.fixture
def store(tmp_path):
    return CorpusStore(tmp_path / "corpus")


def test_baselines_satisfy_protocol(store):
    assert isinstance(ChanceBaseline(), ForecastSystem)
    assert isinstance(CPEchoBaseline(store), ForecastSystem)
    assert isinstance(MarketEchoBaseline(store), ForecastSystem)


async def test_chance_binary_mc_numeric():
    ctx = AsOfContext(as_of=AS_OF)
    b = ChanceBaseline()
    binary = await b.forecast(_q(), ctx)
    assert binary.forecast.binary_prob == 0.5
    assert binary.contamination == "clean"
    assert binary.system_name == "chance"

    mc = await b.forecast(_q(qtype=QType.MULTIPLE_CHOICE,
                             options=("A", "B", "C", "D")), ctx)
    assert mc.forecast.mc_probs == {o: 0.25 for o in ("A", "B", "C", "D")}

    num = await b.forecast(
        _q(qtype=QType.NUMERIC,
           scaling={"range_min": 0.0, "range_max": 10.0,
                    "open_lower_bound": False, "open_upper_bound": True,
                    "cdf_size": 201}), ctx)
    cdf = num.forecast.cdf
    assert len(cdf) == 201
    assert cdf[0] == pytest.approx(0.0)
    assert cdf[-1] == pytest.approx(0.999)
    assert all(b2 - a2 >= 5e-05 - 1e-9 for a2, b2 in zip(cdf, cdf[1:]))


async def test_cp_echo_reads_store_at_asof(store):
    store.upsert_question(_q(resolution=True))
    store.put_series("metaculus:1", "cp",
                     [("2024-05-01T00:00:00+00:00", 0.66),
                      ("2024-05-20T00:00:00+00:00", 0.72)])
    art = await CPEchoBaseline(store).forecast(
        _q(), AsOfContext(as_of=AS_OF))
    assert art.error is None
    assert art.forecast.binary_prob == pytest.approx(0.72)


async def test_cp_echo_skips_without_data(store):
    store.upsert_question(_q(resolution=True))
    store.put_series("metaculus:1", "cp",
                     [("2024-05-01T00:00:00+00:00", 0.66)])
    # as_of before the first CP point -> no data -> skip, never raise
    art = await CPEchoBaseline(store).forecast(
        _q(), AsOfContext(as_of=datetime(2024, 4, 1, tzinfo=timezone.utc)))
    assert art.forecast is None
    assert "no data" in art.error


async def test_market_echo_uses_crosswalk(store, monkeypatch):
    store.upsert_question(_q(resolution=True))
    store.upsert_question(_q(qid="polymarket:xyz", platform="polymarket",
                             native_id="xyz"))
    store.put_series("polymarket:xyz", "market_price",
                     [("2024-05-01T00:00:00+00:00", 0.41)])
    monkeypatch.setattr(baselines_mod, "market_for",
                        lambda s, qid: "polymarket:xyz")
    art = await MarketEchoBaseline(store).forecast(
        _q(), AsOfContext(as_of=AS_OF))
    assert art.error is None
    assert art.forecast.binary_prob == pytest.approx(0.41)


async def test_market_echo_skips_without_crosswalk(store, monkeypatch):
    store.upsert_question(_q(resolution=True))
    monkeypatch.setattr(baselines_mod, "market_for", lambda s, qid: None)
    art = await MarketEchoBaseline(store).forecast(
        _q(), AsOfContext(as_of=AS_OF))
    assert art.forecast is None
    assert "no data" in art.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backtest/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.systems.baselines'`

- [ ] **Step 3: Write the implementation**

`backtest/systems/baselines.py`:

```python
"""Floor systems every report contextualizes variants against (spec §5.3):
chance (0.5 / uniform / uniform CDF), CP echo, market echo.

No LLM, no research — artifacts carry empty stages and contamination="clean".
Echo baselines read the corpus series at ctx.as_of and skip (forecast=None,
error="no data: ...") when nothing is available; they never raise.
"""
from __future__ import annotations

import logging

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType)
from backtest.corpus.crosswalk import market_for
from backtest.scoring.opponents import cp_opponent, market_opponent
from backtest.timetravel.asof import AsOfContext

logger = logging.getLogger(__name__)


def _artifact(q: BacktestQuestion, ctx: AsOfContext, name: str,
              fingerprint: str, fc: Forecast | None,
              error: str | None = None) -> ForecastArtifact:
    return ForecastArtifact(
        qid=q.qid, horizon="", as_of=ctx.as_of, system_name=name,
        config_fingerprint=fingerprint, forecast=fc,
        contamination="clean", cutoff_mode=ctx.cutoff_mode, error=error)


class ChanceBaseline:
    name = "chance"

    def config_fingerprint(self) -> str:
        return "chance-v1"

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        if q.qtype is QType.BINARY:
            fc = Forecast(qtype=QType.BINARY, binary_prob=0.5)
        elif q.qtype is QType.MULTIPLE_CHOICE:
            n = len(q.options)
            if n == 0:
                return _artifact(q, ctx, self.name, "chance-v1", None,
                                 error="no data: MC question has no options")
            fc = Forecast(qtype=q.qtype,
                          mc_probs={o: 1.0 / n for o in q.options})
        else:
            scaling = q.scaling or {}
            open_lo = bool(scaling.get("open_lower_bound"))
            open_hi = bool(scaling.get("open_upper_bound"))
            size = int(scaling.get("cdf_size") or 201)
            first = 0.001 if open_lo else 0.0
            last = 0.999 if open_hi else 1.0
            step = (last - first) / (size - 1)
            fc = Forecast(qtype=q.qtype,
                          cdf=tuple(first + i * step for i in range(size)))
        return _artifact(q, ctx, self.name, "chance-v1", fc)


class CPEchoBaseline:
    name = "cp_echo"

    def __init__(self, store):
        self.store = store

    def config_fingerprint(self) -> str:
        return "cp-echo-v1"

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        try:
            v = self.store.value_at(q.qid, "cp", ctx.as_of)
            if v is None:
                return _artifact(q, ctx, self.name, "cp-echo-v1", None,
                                 error="no data: no CP at as_of")
            fc = cp_opponent(v)
        except (ValueError, TypeError) as e:
            logger.warning("cp value unusable for %s: %s", q.qid, e)
            return _artifact(q, ctx, self.name, "cp-echo-v1", None,
                             error=f"no data: cp value unusable: {e}")
        except Exception as e:
            logger.warning("cp lookup failed for %s: %s", q.qid, e)
            return _artifact(q, ctx, self.name, "cp-echo-v1", None,
                             error=f"no data: cp lookup failed: {e}")
        return _artifact(q, ctx, self.name, "cp-echo-v1", fc)


class MarketEchoBaseline:
    name = "market_echo"

    def __init__(self, store):
        self.store = store

    def config_fingerprint(self) -> str:
        return "market-echo-v1"

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        try:
            mqid = market_for(self.store, q.qid)
            price = (self.store.value_at(mqid, "market_price", ctx.as_of)
                     if mqid else None)
        except Exception as e:
            logger.warning("market lookup failed for %s: %s", q.qid, e)
            return _artifact(q, ctx, self.name, "market-echo-v1", None,
                             error=f"no data: market lookup failed: {e}")
        if price is None:
            return _artifact(
                q, ctx, self.name, "market-echo-v1", None,
                error="no data: no crosswalked market price at as_of")
        return _artifact(q, ctx, self.name, "market-echo-v1",
                         market_opponent(float(price)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/backtest/test_baselines.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add backtest/systems/baselines.py tests/backtest/test_baselines.py && git commit -m "feat(backtest): chance, CP-echo and market-echo baseline systems" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 29: Experiment spec + budget profiles (`backtest/experiments/`)

**Files:**
- Modify: `pyproject.toml:6-12` (add `pyyaml` dependency)
- Create: `backtest/experiments/__init__.py`
- Create: `backtest/experiments/spec.py`
- Create: `backtest/experiments/profiles.py`
- Test: `tests/backtest/test_experiment_spec.py`

- [ ] **Step 1: Add the pyyaml dependency (tooling: pip + setuptools — this repo has only `pyproject.toml`, no uv/poetry lock)**

`pyyaml` is currently importable only as a transitive dependency of `forecasting-tools` — it is NOT in `pyproject.toml` `[project] dependencies`, so depend on it explicitly. Replace the dependencies array in `pyproject.toml` with:

```toml
dependencies = [
    "forecasting-tools>=0.2.40",
    "numpy>=1.26",
    "scipy>=1.11",
    "python-dotenv>=1.0",
    "httpx>=0.27",
    "pyyaml>=6.0",
]
```

Then install and verify:

```bash
.venv/bin/python -m pip install -e ".[dev]" && .venv/bin/python -c "import yaml; print(yaml.__version__)"
```

Expected: prints a `6.x` version.

- [ ] **Step 2: Write the failing tests**

`tests/backtest/test_experiment_spec.py`:

```python
"""ExperimentSpec.from_yaml + budget profiles."""
import pytest

from backtest.experiments.profiles import PROFILES
from backtest.experiments.spec import ExperimentSpec

CONTRACT_YAML = """\
name: agg-sweep
corpus: {platforms: [metaculus], qtypes: [binary], resolved_only: true, stratum: clean}
horizons: [close-7d, close-30d]
cutoff_mode: strict
profile: standard
variants:
  - {name: trimmed_gmo, patch: {aggregation_method: trimmed_gmo}}
baselines: [chance, cp_echo]
seed: 0
"""


def test_from_yaml_parses_contract_example(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(CONTRACT_YAML)
    spec = ExperimentSpec.from_yaml(p)
    assert spec.name == "agg-sweep"
    assert spec.corpus["platforms"] == ["metaculus"]
    assert spec.corpus["stratum"] == "clean"
    assert spec.horizons == ["close-7d", "close-30d"]
    assert spec.cutoff_mode == "strict"
    assert spec.profile == "standard"
    assert spec.variants == [{"name": "trimmed_gmo",
                              "patch": {"aggregation_method": "trimmed_gmo"}}]
    assert spec.baselines == ["chance", "cp_echo"]
    assert spec.seed == 0


def test_from_yaml_defaults(tmp_path):
    p = tmp_path / "min.yaml"
    p.write_text("name: tiny\n")
    spec = ExperimentSpec.from_yaml(p)
    assert spec.cutoff_mode == "strict"
    assert spec.profile == "smoke"
    assert spec.horizons == ["close-7d"]
    assert spec.seed == 0
    assert spec.variants == []
    assert spec.baselines == []


def test_from_yaml_rejects_unknown_keys_and_missing_name(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nbudget: 99\n")
    with pytest.raises(ValueError, match="unknown spec keys"):
        ExperimentSpec.from_yaml(p)
    p2 = tmp_path / "noname.yaml"
    p2.write_text("profile: smoke\n")
    with pytest.raises(ValueError, match="name"):
        ExperimentSpec.from_yaml(p2)


def test_from_yaml_rejects_unknown_profile_and_bad_variant(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("name: x\nprofile: mega\n")
    with pytest.raises(ValueError, match="profile"):
        ExperimentSpec.from_yaml(p)
    p2 = tmp_path / "v.yaml"
    p2.write_text("name: x\nvariants: [{patch: {}}]\n")
    with pytest.raises(ValueError, match="variant"):
        ExperimentSpec.from_yaml(p2)


def test_profiles_shapes():
    assert set(PROFILES) == {"smoke", "standard", "full"}
    smoke = PROFILES["smoke"]
    assert smoke.max_questions == 30
    assert len(smoke.ensemble_override) == 1
    assert smoke.ensemble_override[0].runs == 1
    assert PROFILES["standard"].max_questions == 300
    assert PROFILES["standard"].ensemble_override is None
    assert PROFILES["full"].max_questions is None
    for p in PROFILES.values():
        assert p.cost_cap_usd > 0
        assert 0 < p.est_cost_per_question_usd < p.cost_cap_usd
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/backtest/test_experiment_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.experiments'`

- [ ] **Step 4: Write the implementation**

`backtest/experiments/__init__.py`:

```python
```

(empty file)

`backtest/experiments/profiles.py`:

```python
"""Budget profiles (spec §6.5).

Cost derivation: calls_per_question ≈ (Σ member.runs) ensemble calls +
~2-3 research calls. The standard ensemble is 3 models × 3 runs = 9 LLM
calls/question at roughly $0.02/call blended (mid-2026 OpenRouter rates,
long forecasting prompts), plus ~$0.05 of research calls → ≈ $0.25 per
question per horizon per variant. The smoke profile overrides the ensemble
to a single member with a single run → ~1 ensemble call + research ≈ $0.02
per question. These are deliberately rough planning numbers: the runner
prints the projection for a human gate and enforces cost_cap_usd as a hard
brake.

NOTE: the smoke ensemble_override (1 member × 1 run) cannot satisfy the
live default min_members=3 — the runner sets min_members=1 whenever it
applies an override.
"""
from __future__ import annotations

from dataclasses import dataclass

from bot.config import EnsembleMember


@dataclass(frozen=True)
class Profile:
    max_questions: int | None
    cost_cap_usd: float
    est_cost_per_question_usd: float
    ensemble_override: list[EnsembleMember] | None = None


PROFILES = {
    # CI-friendly: 30 q × 1 variant × $0.02 ≈ $0.60 projected; cap $5.
    "smoke": Profile(max_questions=30,
                     ensemble_override=[EnsembleMember("openai/gpt-5.4",
                                                       runs=1)],
                     cost_cap_usd=5.0,
                     est_cost_per_question_usd=0.02),
    # 300 q × $0.25 ≈ $75 per variant-horizon; cap $150.
    "standard": Profile(max_questions=300,
                        cost_cap_usd=150.0,
                        est_cost_per_question_usd=0.25),
    # Whole corpus, all strata; cap $500 forces deliberate spend.
    "full": Profile(max_questions=None,
                    cost_cap_usd=500.0,
                    est_cost_per_question_usd=0.25),
}
```

`backtest/experiments/spec.py`:

```python
"""Declarative experiment spec (spec §6.5): corpus filter × horizons ×
variant grid × budget profile × cutoff mode, loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentSpec:
    name: str
    corpus: dict = field(default_factory=dict)
    horizons: list = field(default_factory=lambda: ["close-7d"])
    cutoff_mode: str = "strict"
    profile: str = "smoke"
    variants: list = field(default_factory=list)
    baselines: list = field(default_factory=list)
    seed: int = 0

    @classmethod
    def from_yaml(cls, path) -> "ExperimentSpec":
        from backtest.experiments.profiles import PROFILES
        d = yaml.safe_load(Path(path).read_text())
        if not isinstance(d, dict) or "name" not in d:
            raise ValueError(
                f"experiment spec {path} must be a mapping with a 'name'")
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown spec keys: {sorted(unknown)}")
        if d.get("profile", "smoke") not in PROFILES:
            raise ValueError(f"unknown profile: {d.get('profile')!r} "
                             f"(choose from {sorted(PROFILES)})")
        for v in d.get("variants") or []:
            if not isinstance(v, dict) or "name" not in v:
                raise ValueError(
                    f"each variant needs at least a 'name': {v!r}")
        return cls(**d)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/backtest/test_experiment_spec.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml backtest/experiments/__init__.py backtest/experiments/spec.py backtest/experiments/profiles.py tests/backtest/test_experiment_spec.py && git commit -m "feat(backtest): experiment spec loader and budget profiles (explicit pyyaml dependency)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 30: Experiment runner (`backtest/experiments/runner.py`)

Deterministic `run_id`, printed cost estimate with confirmation gate (`yes=True` skips) and hard cost brake, manifest-based resume, per-(qid, horizon, variant) failure isolation, `artifacts.jsonl` via `artifact_to_dict`, scoring pass into `scores.jsonl`, paired bootstrap stats per variant pair into `stats.json`.

**Files:**
- Create: `backtest/experiments/runner.py`
- Test: `tests/backtest/test_runner.py`

- [ ] **Step 1: Write the failing tests**

`tests/backtest/test_runner.py`:

```python
"""Runner tests: full offline runs over a real CorpusStore in tmp_path with
a stub system. No network; CachedLLM is constructed but never invoked."""
import json
from datetime import datetime, timezone

import pytest

import backtest.experiments.runner as runner_mod
from backtest.core import BacktestQuestion, Forecast, ForecastArtifact, QType
from backtest.corpus.store import CorpusStore
from backtest.experiments.profiles import PROFILES, Profile
from backtest.experiments.runner import RunResult, run_experiment
from backtest.experiments.spec import ExperimentSpec


class StubSystem:
    name = "stub"

    def __init__(self):
        self.calls = 0

    def config_fingerprint(self) -> str:
        return "stub-v1"

    async def forecast(self, q, ctx):
        self.calls += 1
        return ForecastArtifact(
            qid=q.qid, horizon="", as_of=ctx.as_of, system_name=self.name,
            config_fingerprint="stub-v1",
            forecast=Forecast(qtype=QType.BINARY, binary_prob=0.8),
            cutoff_mode=ctx.cutoff_mode)


def _q(i: int, resolution: bool) -> BacktestQuestion:
    return BacktestQuestion(
        qid=f"metaculus:{i}", platform="metaculus", native_id=str(i),
        qtype=QType.BINARY, title=f"Question {i}?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
        resolution=resolution)


@pytest.fixture
def store(tmp_path):
    s = CorpusStore(tmp_path / "corpus")
    s.upsert_question(_q(1, True))
    s.upsert_question(_q(2, False))
    s.put_series("metaculus:1", "cp",
                 [("2024-05-01T00:00:00+00:00", 0.66)])
    return s


def _spec(**kw):
    base = dict(name="t", corpus={"qtypes": ["binary"]},
                horizons=["close-7d"], cutoff_mode="strict",
                profile="smoke",
                variants=[{"name": "stub", "patch": {}}],
                baselines=[], seed=0)
    base.update(kw)
    return ExperimentSpec(**base)


@pytest.fixture
def stub_systems(monkeypatch):
    stub = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)
    return stub


async def test_full_run_writes_manifest_artifacts_scores(
        tmp_path, store, stub_systems):
    result = await run_experiment(_spec(), store, tmp_path / "runs",
                                  live_llm=False, yes=True)
    assert isinstance(result, RunResult)
    run_dir = result.run_dir

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert len(manifest["statuses"]) == 2     # 2 q × 1 horizon × 1 system
    assert set(manifest["statuses"].values()) == {"done"}

    artifacts = [json.loads(l) for l in
                 (run_dir / "artifacts.jsonl").read_text().splitlines()]
    assert len(artifacts) == 2
    assert all(a["horizon"] == "close-7d" for a in artifacts)

    rows = [json.loads(l) for l in
            (run_dir / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert set(rows[0]) == {"qid", "horizon", "variant", "log_score",
                            "baseline_score", "brier", "h2h_cp",
                            "h2h_market", "stratum", "resolution"}
    by_qid = {r["qid"]: r for r in rows}
    assert by_qid["metaculus:1"]["h2h_cp"] is not None   # CP series present
    assert by_qid["metaculus:2"]["h2h_cp"] is None
    assert isinstance(by_qid["metaculus:1"]["baseline_score"], float)
    assert (run_dir / "stats.json").exists()


async def test_run_id_is_deterministic(tmp_path, store, stub_systems):
    r1 = await run_experiment(_spec(), store, tmp_path / "a",
                              live_llm=False, yes=True)
    r2 = await run_experiment(_spec(), store, tmp_path / "b",
                              live_llm=False, yes=True)
    assert r1.run_id == r2.run_id


async def test_resume_skips_completed(tmp_path, store, stub_systems,
                                      monkeypatch):
    await run_experiment(_spec(), store, tmp_path / "runs",
                         live_llm=False, yes=True)
    stub2 = StubSystem()
    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": stub2})
    result = await run_experiment(_spec(), store, tmp_path / "runs",
                                  live_llm=False, yes=True)
    assert stub2.calls == 0
    assert set(result.statuses.values()) == {"done"}


async def test_unreachable_horizon_is_skipped(tmp_path, store, stub_systems):
    # close − 365d is before open_time -> horizon_to_asof returns None
    result = await run_experiment(_spec(horizons=["close-365d"]), store,
                                  tmp_path / "runs", live_llm=False, yes=True)
    assert set(result.statuses.values()) == {"skipped"}
    assert stub_systems.calls == 0


async def test_system_crash_is_isolated(tmp_path, store, monkeypatch):
    class Crashy(StubSystem):
        async def forecast(self, q, ctx):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(runner_mod, "_build_systems",
                        lambda spec, store: {"stub": Crashy()})
    monkeypatch.setattr(runner_mod, "market_for", lambda store, qid: None)
    result = await run_experiment(_spec(), store, tmp_path / "runs",
                                  live_llm=False, yes=True)
    assert set(result.statuses.values()) == {"error"}
    artifacts = [json.loads(l) for l in
                 (result.run_dir / "artifacts.jsonl").read_text().splitlines()]
    assert all("kaboom" in a["error"] for a in artifacts)
    assert not (result.run_dir / "scores.jsonl").read_text().strip()


async def test_cost_brake_aborts(tmp_path, store, stub_systems, monkeypatch):
    monkeypatch.setitem(PROFILES, "smoke",
                        Profile(max_questions=2, cost_cap_usd=0.01,
                                est_cost_per_question_usd=1.0))
    with pytest.raises(RuntimeError, match="cost brake"):
        await run_experiment(_spec(), store, tmp_path / "runs",
                             live_llm=False, yes=True)


async def test_confirmation_gate_aborts_on_no(tmp_path, store, stub_systems,
                                              monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    result = await run_experiment(_spec(), store, tmp_path / "runs",
                                  live_llm=False, yes=False)
    assert result.statuses == {}
    assert not (result.run_dir / "artifacts.jsonl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backtest/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.experiments.runner'`

- [ ] **Step 3: Write the implementation**

`backtest/experiments/runner.py`:

```python
"""Experiment runner (spec §6.5): cost gate → manifest-resumable
per-(question, horizon, variant) loop with failure isolation →
artifacts.jsonl → scoring pass (scores.jsonl) → paired stats (stats.json).

Resume semantics: statuses "done"/"skipped" are not re-run; "error" is
retried on the next invocation. The scoring pass is idempotent — it rebuilds
scores.jsonl from the latest artifact per (qid, horizon, variant) each run.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from backtest.core import (ForecastArtifact, artifact_from_dict,
                           artifact_to_dict)
from backtest.corpus.crosswalk import market_for
from backtest.experiments.profiles import PROFILES
from backtest.experiments.spec import ExperimentSpec
from backtest.scoring import stats as stats_mod
from backtest.scoring.metaculus_scores import (baseline_score, brier,
                                               head_to_head, log_score)
from backtest.scoring.opponents import cp_opponent, market_opponent
from backtest.systems.baselines import (ChanceBaseline, CPEchoBaseline,
                                        MarketEchoBaseline)
from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
from backtest.timetravel.asof import AsOfContext, horizon_to_asof
from backtest.timetravel.llm_cache import CachedLLM
from backtest.timetravel.providers import install_historical_providers
from bot.config import BotConfig

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    statuses: dict = field(default_factory=dict)
    projected_cost_usd: float = 0.0


_BASELINE_FACTORIES = {
    "chance": lambda store: ChanceBaseline(),
    "cp_echo": lambda store: CPEchoBaseline(store),
    "market_echo": lambda store: MarketEchoBaseline(store),
}


def _build_systems(spec: ExperimentSpec, store) -> dict:
    """name -> ForecastSystem. Module-level so tests can monkeypatch it."""
    install_historical_providers()
    # The live BotConfig default providers (incl. the Perplexity track)
    # would do leaky live research — backtests run on the historical set.
    # Variants may still patch research_providers via their spec patch.
    profile = PROFILES[spec.profile]
    base = BotConfig(research_providers=(
        "asknews_historical", "gdelt_wayback",
        "exa_historical", "markets_historical"))
    if profile.ensemble_override is not None:
        # 1-member smoke override cannot satisfy the live min_members=3
        base = replace(base, ensemble=list(profile.ensemble_override),
                       min_members=1)
    systems: dict = {}
    for v in spec.variants:
        system = CalibratedEnsembleAdapter.variant(base=base,
                                                   **v.get("patch", {}))
        system.name = v["name"]
        systems[v["name"]] = system
    for b in spec.baselines:
        factory = _BASELINE_FACTORIES.get(b)
        if factory is None:
            raise ValueError(f"unknown baseline: {b!r} "
                             f"(choose from {sorted(_BASELINE_FACTORIES)})")
        systems[b] = factory(store)
    return systems


def _run_id(spec: ExperimentSpec, systems: dict) -> str:
    fps = "|".join(sorted(s.config_fingerprint() for s in systems.values()))
    return (f"{spec.name}-{spec.seed}-"
            f"{hashlib.sha256(fps.encode()).hexdigest()[:8]}")


def _select_questions(spec: ExperimentSpec, store, profile) -> list:
    corpus = dict(spec.corpus)
    kwargs = {"resolved_only": corpus.get("resolved_only", True),
              "limit": profile.max_questions}
    for key in ("platforms", "qtypes", "open_after", "close_before",
                "stratum", "model_set_hash"):
        if key in corpus:
            kwargs[key] = corpus[key]
    # CorpusStore.query raises for stratum-without-hash; the canonical
    # spec YAML says `stratum: clean` with no hash -> default to the
    # live ensemble's model set.
    if "stratum" in corpus and "model_set_hash" not in corpus:
        from backtest.corpus.model_releases import model_set_hash
        kwargs["model_set_hash"] = model_set_hash(
            [m.model for m in BotConfig().ensemble])
    return sorted(store.query(**kwargs), key=lambda q: q.qid)


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2))


async def run_experiment(spec: ExperimentSpec, store, out_dir: Path, *,
                         live_llm: bool = True,
                         yes: bool = False,
                         systems: list | None = None) -> RunResult:
    profile = PROFILES[spec.profile]
    questions = _select_questions(spec, store, profile)
    systems_map = (_build_systems(spec, store) if systems is None
                   else {s.name: s for s in systems})
    run_id = _run_id(spec, systems_map)
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Cost estimate + hard brake + human gate (baselines spend nothing,
    # so only spec.variants count).
    projected = (len(questions) * len(spec.horizons) * len(spec.variants)
                 * profile.est_cost_per_question_usd)
    print(f"[retrocast] run {run_id}: {len(questions)} questions x "
          f"{len(spec.horizons)} horizons x {len(spec.variants)} LLM "
          f"variants (+{len(spec.baselines)} free baselines) "
          f"~= ${projected:.2f} (cap ${profile.cost_cap_usd:.2f})")
    if projected > profile.cost_cap_usd:
        raise RuntimeError(
            f"cost brake: projected ${projected:.2f} exceeds "
            f"'{spec.profile}' cap ${profile.cost_cap_usd:.2f}")
    if not yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("[retrocast] aborted before spend")
            return RunResult(run_id=run_id, run_dir=run_dir, statuses={},
                             projected_cost_usd=projected)

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"run_id": run_id, "spec_name": spec.name,
                    "corpus_root": str(getattr(store, "root", "")),
                    "statuses": {}}
    statuses: dict = manifest["statuses"]

    # Cache lives at out_dir level, shared across run_ids: keys are
    # content-addressed (model/prompt/temperature/run_index), so sharing is
    # safe and post-LLM-only config changes (new run_id) replay at $0.
    llm = CachedLLM(Path(out_dir) / "llm_cache", live=live_llm)
    artifacts_path = run_dir / "artifacts.jsonl"

    for q in questions:
        for horizon in spec.horizons:
            as_of = horizon_to_asof(q, horizon)
            for name, system in systems_map.items():
                key = f"{q.qid}|{horizon}|{name}"
                if statuses.get(key) in ("done", "skipped"):
                    continue
                if as_of is None:
                    statuses[key] = "skipped"
                    _write_manifest(manifest_path, manifest)
                    continue
                ctx = AsOfContext(as_of=as_of, cutoff_mode=spec.cutoff_mode,
                                  llm=llm, store=store)
                try:
                    artifact = await system.forecast(q, ctx)
                except Exception as e:
                    # systems must not raise, but a buggy one must not
                    # kill the run either
                    logger.exception("system %s crashed on %s", name, key)
                    artifact = ForecastArtifact(
                        qid=q.qid, horizon=horizon, as_of=as_of,
                        system_name=name,
                        config_fingerprint=system.config_fingerprint(),
                        forecast=None, cutoff_mode=spec.cutoff_mode,
                        error=f"{type(e).__name__}: {e}")
                artifact.horizon = horizon   # systems can't know the label
                with artifacts_path.open("a") as f:
                    f.write(json.dumps(artifact_to_dict(artifact)) + "\n")
                statuses[key] = ("done" if artifact.error is None
                                 else "error")
                _write_manifest(manifest_path, manifest)

    _score_run(run_dir, store, seed=spec.seed)
    return RunResult(run_id=run_id, run_dir=run_dir,
                     statuses=dict(statuses), projected_cost_usd=projected)


def _score_run(run_dir: Path, store, seed: int = 0) -> None:
    artifacts_path = run_dir / "artifacts.jsonl"
    if not artifacts_path.exists():
        return
    latest: dict = {}
    with artifacts_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            a = artifact_from_dict(json.loads(line))
            latest[(a.qid, a.horizon, a.system_name)] = a

    rows = []
    for (qid, horizon, variant), a in sorted(latest.items()):
        if a.forecast is None:
            continue
        q = store.get_question(qid)
        if q is None or q.resolution is None:
            continue
        try:
            row = {"qid": qid, "horizon": horizon, "variant": variant,
                   "log_score": log_score(a.forecast, q),
                   "baseline_score": baseline_score(a.forecast, q),
                   "brier": None, "h2h_cp": None, "h2h_market": None,
                   "stratum": a.contamination,
                   "resolution": q.resolution}
        except Exception as e:
            logger.warning("scoring failed for %s/%s/%s: %s",
                           qid, horizon, variant, e)
            continue
        try:
            row["brier"] = brier(a.forecast, q)
        except NotImplementedError:
            pass
        try:
            cp = store.value_at(qid, "cp", a.as_of)
            if cp is not None:
                row["h2h_cp"] = head_to_head(a.forecast, cp_opponent(cp), q)
        except Exception as e:
            logger.warning("h2h_cp failed for %s: %s", qid, e)
        try:
            mqid = market_for(store, qid)
            price = (store.value_at(mqid, "market_price", a.as_of)
                     if mqid else None)
            if price is not None:
                row["h2h_market"] = head_to_head(
                    a.forecast, market_opponent(float(price)), q)
        except Exception as e:
            logger.warning("h2h_market failed for %s: %s", qid, e)
        rows.append(row)

    with (run_dir / "scores.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    _paired_stats(run_dir, rows, seed)


def _paired_stats(run_dir: Path, rows: list, seed: int) -> None:
    """Paired bootstrap on baseline_score diffs over the common
    (qid, horizon) set, for every variant pair."""
    by_variant: dict = {}
    for r in rows:
        by_variant.setdefault(r["variant"], {})[
            (r["qid"], r["horizon"])] = r["baseline_score"]
    out = {}
    variants = sorted(by_variant)
    for i, a in enumerate(variants):
        for b in variants[i + 1:]:
            common = sorted(set(by_variant[a]) & set(by_variant[b]))
            diffs = [by_variant[a][k] - by_variant[b][k] for k in common]
            if not diffs:
                continue
            out[f"{a}__vs__{b}"] = {
                "n": len(diffs),
                "baseline_score_diff": stats_mod.paired_bootstrap(
                    diffs, seed=seed),
            }
    (run_dir / "stats.json").write_text(json.dumps(out, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/backtest/test_runner.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add backtest/experiments/runner.py tests/backtest/test_runner.py && git commit -m "feat(backtest): resumable experiment runner — cost gate + brake, manifest resume, scoring pass, paired stats" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 31: Report builder + CLI (`backtest/reports/report.py`, `scripts/backtest.py`)

**Files:**
- Create: `backtest/reports/__init__.py`
- Create: `backtest/reports/report.py`
- Create: `scripts/backtest.py`
- Test: `tests/backtest/test_report.py`
- Test: `tests/backtest/test_cli.py`

Note: `backtest/systems/stub.py` (`StubSystem`, referenced by `run --system stub`) ships with the CI-gate task (Phase 5) in `tests/backtest` and `backtest/systems/stub.py`; the CLI import is lazy (inside the `if`), so Phase 4 tests do not need it to exist.

- [ ] **Step 1: Write the failing report tests**

`tests/backtest/test_report.py`:

```python
"""build_report renders every required section from a fabricated run dir
(no corpus store: corpus_root empty -> diagnostics degrade gracefully)."""
import json
from datetime import datetime, timezone

from backtest.core import (Forecast, ForecastArtifact, LedgerEntry, QType,
                           StageRecord, artifact_to_dict)
from backtest.reports.report import build_report


def _write_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "demo-0-abc12345", "corpus_root": "",
         "statuses": {"metaculus:1|close-7d|a": "done"}}))
    rows = [
        {"qid": "metaculus:1", "horizon": "close-7d", "variant": "a",
         "log_score": -0.2, "baseline_score": 20.0, "brier": 0.04,
         "h2h_cp": 5.0, "h2h_market": None, "stratum": "clean"},
        {"qid": "metaculus:2", "horizon": "close-7d", "variant": "a",
         "log_score": -0.4, "baseline_score": 10.0, "brier": 0.09,
         "h2h_cp": 3.0, "h2h_market": None, "stratum": "contaminated"},
        {"qid": "metaculus:1", "horizon": "close-7d", "variant": "chance",
         "log_score": -0.69, "baseline_score": 0.0, "brier": 0.25,
         "h2h_cp": None, "h2h_market": None, "stratum": "clean"},
        {"qid": "metaculus:2", "horizon": "close-7d", "variant": "chance",
         "log_score": -0.69, "baseline_score": 0.0, "brier": 0.25,
         "h2h_cp": None, "h2h_market": None, "stratum": "contaminated"},
    ]
    with (run_dir / "scores.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    art = ForecastArtifact(
        qid="metaculus:1", horizon="close-7d",
        as_of=datetime(2024, 5, 25, tzinfo=timezone.utc),
        system_name="a", config_fingerprint="fp",
        forecast=Forecast(qtype=QType.BINARY, binary_prob=0.8),
        stages=[StageRecord(stage="member_llm", model="fake/model-a",
                            cost_usd=0.12, tokens_in=100, tokens_out=50)],
        evidence_ledger=[
            LedgerEntry(provider="asknews", query="q",
                        date_trust="snapshot"),
            LedgerEntry(provider="serper", query="q", date_trust="low")])
    with (run_dir / "artifacts.jsonl").open("w") as f:
        f.write(json.dumps(artifact_to_dict(art)) + "\n")
    return run_dir


def test_build_report_renders_all_sections(tmp_path):
    run_dir = _write_run_dir(tmp_path)
    path = build_report(run_dir)
    assert path == run_dir / "report.md"
    text = path.read_text()
    assert "## Leaderboard" in text
    # ranked by mean baseline score: 'a' (15.0) above 'chance' (0.0)
    assert text.index("| a |") < text.index("| chance |")
    assert "[" in text                       # bootstrap CI rendered for h2h
    assert "## Mean baseline score by contamination stratum" in text
    assert "contaminated" in text
    assert "## Diagnostics" in text          # degrades without a store
    assert "## Evidence-ledger leakage summary" in text
    assert "| snapshot | 1 |" in text
    assert "| low | 1 |" in text
    assert "## Cost" in text
    assert "0.1200" in text
    assert "## Worst 10 by log score" in text
    assert "metaculus:1" in text


def test_build_report_tolerates_empty_run_dir(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    path = build_report(run_dir)
    assert path.exists()
    assert "Leaderboard" in path.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backtest/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.reports'`

- [ ] **Step 3: Write the report implementation**

`backtest/reports/__init__.py`:

```python
```

(empty file)

`backtest/reports/report.py`:

```python
"""Per-run markdown report (spec §6.6): leaderboard vs baselines, scores
stratified by contamination, diagnostics, evidence-ledger leakage summary,
cost totals from StageRecords, worst-10 post-mortem table.

Diagnostics and worst-10 titles need question resolutions; the runner
records corpus_root in manifest.json — when it is missing or stale the
report degrades gracefully instead of failing.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from backtest.core import artifact_from_dict
from backtest.scoring.diagnostics import (calibration_curve,
                                          discrimination_separation,
                                          murphy_decomposition)
from backtest.scoring.stats import paired_bootstrap


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    scores = _read_jsonl(run_dir / "scores.jsonl")
    artifacts = [artifact_from_dict(d)
                 for d in _read_jsonl(run_dir / "artifacts.jsonl")]
    manifest = {}
    mpath = run_dir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())

    store = None
    root = manifest.get("corpus_root") or ""
    if root and Path(root).exists():
        from backtest.corpus.store import CorpusStore
        store = CorpusStore(Path(root))

    lines = [f"# Retrocast report — {manifest.get('run_id', run_dir.name)}",
             ""]
    status_counts = Counter(manifest.get("statuses", {}).values())
    if status_counts:
        lines += [f"Statuses: {dict(sorted(status_counts.items()))}", ""]

    by_variant = defaultdict(list)
    for r in scores:
        by_variant[r["variant"]].append(r)

    # ---- Leaderboard (ranked by mean baseline score) ----
    lines += ["## Leaderboard", "",
              "| variant | n | mean baseline | mean log | "
              "h2h vs CP (mean [95% CI]) |",
              "|---|---|---|---|---|"]
    ranked = sorted(
        by_variant.items(),
        key=lambda kv: -(_mean([r["baseline_score"] for r in kv[1]])
                         if kv[1] else float("-inf")))
    for variant, rows in ranked:
        h2h = [r["h2h_cp"] for r in rows if r["h2h_cp"] is not None]
        if h2h:
            bs = paired_bootstrap(h2h)
            h2h_cell = (f"{bs['mean']:+.2f} [{bs['lo']:+.2f}, "
                        f"{bs['hi']:+.2f}] (n={len(h2h)})")
        else:
            h2h_cell = "—"
        lines.append(
            f"| {variant} | {len(rows)} | "
            f"{_fmt(_mean([r['baseline_score'] for r in rows]))} | "
            f"{_fmt(_mean([r['log_score'] for r in rows]))} | {h2h_cell} |")

    # ---- Contamination strata ----
    lines += ["", "## Mean baseline score by contamination stratum", "",
              "| variant | stratum | n | mean baseline |", "|---|---|---|---|"]
    strata = defaultdict(list)
    for r in scores:
        strata[(r["variant"], r["stratum"])].append(r["baseline_score"])
    for (variant, stratum), vals in sorted(strata.items()):
        lines.append(f"| {variant} | {stratum} | {len(vals)} | "
                     f"{_fmt(_mean(vals))} |")

    # ---- Diagnostics (binary) ----
    lines += ["", "## Diagnostics (binary)", ""]
    if store is None:
        lines.append("_corpus unavailable (manifest corpus_root missing) — "
                     "diagnostics skipped_")
    else:
        for variant in sorted(by_variant):
            pairs = []
            for a in artifacts:
                if (a.system_name == variant and a.forecast is not None
                        and a.forecast.binary_prob is not None):
                    q = store.get_question(a.qid)
                    if q is not None and isinstance(q.resolution, bool):
                        pairs.append((a.forecast.binary_prob, q.resolution))
            if len(pairs) < 2:
                lines.append(f"- **{variant}**: not enough binary pairs "
                             f"(n={len(pairs)})")
                continue
            lines.append(
                f"- **{variant}** (n={len(pairs)}): "
                f"murphy={murphy_decomposition(pairs)}, "
                f"discrimination={discrimination_separation(pairs)}, "
                f"calibration_curve={calibration_curve(pairs, n_bins=5)}")

    # ---- Evidence-ledger leakage summary ----
    lines += ["", "## Evidence-ledger leakage summary", ""]
    trust = Counter(e.date_trust
                    for a in artifacts for e in a.evidence_ledger)
    if trust:
        lines += ["| date_trust | entries |", "|---|---|"]
        for k, v in sorted(trust.items()):
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("_no ledger entries recorded_")

    # ---- Cost (from StageRecords) ----
    lines += ["", "## Cost (from StageRecords)", ""]
    if artifacts:
        cost_by: dict = defaultdict(float)
        tok_by: dict = defaultdict(int)
        for a in artifacts:
            for s in a.stages:
                cost_by[a.system_name] += s.cost_usd
                tok_by[a.system_name] += s.tokens_in + s.tokens_out
        lines += ["| variant | cost (USD) | tokens |", "|---|---|---|"]
        for name in sorted({a.system_name for a in artifacts}):
            lines.append(f"| {name} | {cost_by[name]:.4f} | "
                         f"{tok_by[name]} |")
    else:
        lines.append("_no artifacts_")

    # ---- Worst-10 post-mortem table ----
    lines += ["", "## Worst 10 by log score", "",
              "| qid | title | variant | horizon | log score |",
              "|---|---|---|---|---|"]
    scored = [r for r in scores if r.get("log_score") is not None]
    for r in sorted(scored, key=lambda r: r["log_score"])[:10]:
        title = ""
        if store is not None:
            q = store.get_question(r["qid"])
            title = (q.title if q is not None else "")[:60]
        lines.append(f"| {r['qid']} | {title} | {r['variant']} | "
                     f"{r['horizon']} | {r['log_score']:.3f} |")

    out = run_dir / "report.md"
    out.write_text("\n".join(lines) + "\n")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/backtest/test_report.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add backtest/reports/__init__.py backtest/reports/report.py tests/backtest/test_report.py && git commit -m "feat(backtest): markdown run report — leaderboard, strata, diagnostics, ledger, cost, worst-10" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Write the failing CLI tests**

`tests/backtest/test_cli.py`:

```python
"""CLI smoke tests: --help, a tiny offline replay run + report, compare.
No network — the tiny run uses only the chance baseline (no LLM)."""
import json
from datetime import datetime, timezone

import pytest

from scripts.backtest import main


def test_top_level_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("build-corpus", "run", "replay", "compare", "report"):
        assert cmd in out


def test_replay_then_report_tiny_run(tmp_path):
    from backtest.core import BacktestQuestion, QType
    from backtest.corpus.store import CorpusStore
    root = tmp_path / "corpus"
    store = CorpusStore(root)
    store.upsert_question(BacktestQuestion(
        qid="metaculus:9", platform="metaculus", native_id="9",
        qtype=QType.BINARY, title="Tiny?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
        resolution=True))
    spec = tmp_path / "exp.yaml"
    spec.write_text(
        "name: cli-smoke\n"
        "corpus: {qtypes: [binary]}\n"
        "horizons: [close-7d]\n"
        "cutoff_mode: strict\n"
        "profile: smoke\n"
        "variants: []\n"
        "baselines: [chance]\n"
        "seed: 0\n")
    out = tmp_path / "runs"
    assert main(["replay", "--spec", str(spec), "--corpus", str(root),
                 "--out", str(out), "--yes"]) == 0
    run_dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "manifest.json").exists()
    assert (run_dirs[0] / "scores.jsonl").exists()

    assert main(["report", str(run_dirs[0])]) == 0
    text = (run_dirs[0] / "report.md").read_text()
    assert "Leaderboard" in text
    assert "chance" in text


def test_compare_outputs_paired_stats(tmp_path, capsys):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    rows = []
    for qid, a, b in (("m:1", 10.0, 5.0), ("m:2", 20.0, 12.0)):
        for variant, score in (("a", a), ("b", b)):
            rows.append({"qid": qid, "horizon": "close-7d",
                         "variant": variant, "log_score": -0.3,
                         "baseline_score": score, "brier": None,
                         "h2h_cp": None, "h2h_market": None,
                         "stratum": "clean"})
    with (run_dir / "scores.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert main(["compare", str(run_dir), "a", "b"]) == 0
    out = capsys.readouterr().out
    assert '"n": 2' in out
    assert '"mean"' in out


def test_compare_unknown_variant_fails_cleanly(tmp_path, capsys):
    run_dir = tmp_path / "r2"
    run_dir.mkdir()
    (run_dir / "scores.jsonl").write_text("")
    assert main(["compare", str(run_dir), "a", "b"]) == 1
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `pytest tests/backtest/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.backtest'`

- [ ] **Step 8: Write the CLI implementation**

`scripts/backtest.py`:

```python
"""Retrocast CLI: build-corpus | run | replay | compare | report.

replay == run with live_llm=False: every completion must come from the
run's cache (CachedLLM raises CacheMiss otherwise) — $0 re-runs for
post-LLM changes. Heavy imports live inside subcommand functions so
`--help` stays fast and dependency-light.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_build_corpus(args) -> int:
    import asyncio

    from backtest.corpus.store import CorpusStore
    store = CorpusStore(Path(args.corpus))
    if args.source in ("halawi", "forecastbench") and not args.dataset_dir:
        print("--dataset-dir is required for halawi/forecastbench")
        return 2
    if args.source == "metaculus":
        from backtest.corpus.metaculus import fetch_metaculus
        report = asyncio.run(fetch_metaculus(
            store, tournaments=args.tournaments or None, limit=args.limit))
    elif args.source == "polymarket":
        from backtest.corpus.polymarket import fetch_polymarket
        report = asyncio.run(fetch_polymarket(store, limit=args.limit))
    elif args.source == "kalshi":
        from backtest.corpus.kalshi import fetch_kalshi
        report = asyncio.run(fetch_kalshi(store, limit=args.limit))
    elif args.source == "manifold":
        from backtest.corpus.manifold import fetch_manifold
        report = asyncio.run(fetch_manifold(store, limit=args.limit))
    elif args.source == "halawi":
        from backtest.corpus.halawi import import_halawi
        report = import_halawi(store, Path(args.dataset_dir))
    else:  # forecastbench
        from backtest.corpus.forecastbench import import_forecastbench
        report = import_forecastbench(store, Path(args.dataset_dir))
    print(f"fetched={report.fetched} withheld={report.withheld} "
          f"errors={len(report.errors)}")
    return 0


def _cmd_run(args, live_llm: bool) -> int:
    import asyncio

    from backtest.corpus.store import CorpusStore
    from backtest.experiments.runner import run_experiment
    from backtest.experiments.spec import ExperimentSpec
    spec = ExperimentSpec.from_yaml(args.spec)
    store = CorpusStore(Path(args.corpus))
    systems = None
    if getattr(args, "system", None) == "stub":
        # lazy import: StubSystem ships with the CI-gate task (Phase 5)
        from backtest.systems.stub import StubSystem
        systems = [StubSystem()]
    result = asyncio.run(run_experiment(spec, store, Path(args.out),
                                        live_llm=live_llm, yes=args.yes,
                                        systems=systems))
    done = sum(1 for s in result.statuses.values() if s == "done")
    print(f"[retrocast] {result.run_id}: {done}/{len(result.statuses)} "
          f"done -> {result.run_dir}")
    return 0


def cmd_compare(args) -> int:
    from backtest.scoring.stats import paired_bootstrap, paired_t
    scores_path = Path(args.run_dir) / "scores.jsonl"
    rows = []
    if scores_path.exists():
        rows = [json.loads(line)
                for line in scores_path.read_text().splitlines()
                if line.strip()]
    by: dict = {}
    for r in rows:
        by.setdefault(r["variant"], {})[
            (r["qid"], r["horizon"])] = r["baseline_score"]
    if args.variant_a not in by or args.variant_b not in by:
        print(f"variants not found in scores: "
              f"{args.variant_a!r}, {args.variant_b!r} "
              f"(have: {sorted(by)})")
        return 1
    common = sorted(set(by[args.variant_a]) & set(by[args.variant_b]))
    diffs = [by[args.variant_a][k] - by[args.variant_b][k] for k in common]
    if not diffs:
        print("no common (qid, horizon) rows between the two variants")
        return 1
    print(json.dumps({
        "n": len(diffs),
        "baseline_score_diff": paired_bootstrap(diffs),
        "paired_t": paired_t(diffs),
    }, indent=2))
    return 0


def cmd_report(args) -> int:
    from backtest.reports.report import build_report
    path = build_report(Path(args.run_dir))
    print(path)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="backtest",
        description="Retrocast: build corpora, run/replay backtest "
                    "experiments, compare variants, render reports.")
    sub = p.add_subparsers(dest="command", required=True)

    bc = sub.add_parser("build-corpus",
                        help="fetch/import questions into the corpus store")
    bc.add_argument("--source", required=True,
                    choices=["metaculus", "polymarket", "kalshi", "manifold",
                             "halawi", "forecastbench"])
    bc.add_argument("--corpus", default="data/backtest",
                    help="corpus root (default data/backtest)")
    bc.add_argument("--limit", type=int, default=None)
    bc.add_argument("--tournaments", nargs="*", default=None)
    bc.add_argument("--dataset-dir", default=None,
                    help="local dataset dir (halawi/forecastbench)")
    bc.set_defaults(func=cmd_build_corpus)

    for name, live, help_text in (
            ("run", True, "run an experiment (live LLM, cached)"),
            ("replay", False,
             "re-run from the completion cache only (no live LLM)")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--spec", required=True, help="experiment YAML")
        sp.add_argument("--corpus", default="data/backtest",
                        help="corpus root (default data/backtest)")
        sp.add_argument("--out", default="data/backtest/runs",
                        help="runs output dir")
        sp.add_argument("--yes", action="store_true",
                        help="skip the cost confirmation gate")
        if name == "run":
            sp.add_argument("--system", choices=["stub"], default=None,
                            help="override systems under test; 'stub' runs "
                                 "the deterministic no-network StubSystem "
                                 "(CI smoke gate)")
        sp.set_defaults(func=lambda a, _live=live: _cmd_run(a, _live))

    cp = sub.add_parser("compare",
                        help="paired stats between two variants of one run")
    cp.add_argument("run_dir")
    cp.add_argument("variant_a")
    cp.add_argument("variant_b")
    cp.set_defaults(func=cmd_compare)

    rp = sub.add_parser("report",
                        help="render the markdown report for a run dir")
    rp.add_argument("run_dir")
    rp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/backtest/test_cli.py -v`
Expected: PASS (4 passed)

- [ ] **Step 10: Run the whole Phase-4 suite plus the live-bot regression set**

Run: `pytest tests/backtest/ tests/test_ensemble_bot.py tests/test_research.py tests/test_framework_adapter.py -v`
Expected: PASS — all Phase-4 tests green and zero regressions in the live bot's tests.

- [ ] **Step 11: Commit**

```bash
git add scripts/backtest.py tests/backtest/test_cli.py && git commit -m "feat(backtest): scripts/backtest.py CLI — build-corpus/run/replay/compare/report" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
## Phase 5 — Live integration, CI gate, flywheel, docs (Tasks 32–36)

Prerequisites: Phases 1–4 are implemented — `backtest/core.py`, `backtest/corpus/store.py` + `backtest/corpus/snapshot.py`, `backtest/timetravel/asof.py`, `backtest/systems/protocol.py`, `backtest/experiments/{spec,runner,profiles}.py`, `backtest/reports/report.py`, and the `scripts/backtest.py` CLI all exist. All commands run from the repo root `/Users/aaronbao/Developer/sideprojects/forecasting`. Tests use the repo's pytest-asyncio `asyncio_mode = "auto"` convention (plain `async def` test functions, no marker), `monkeypatch` fixtures, and never touch the network.

### Task 32: Forward snapshot accumulator hook in `scripts/run_tournament.py`

Every live tournament run archives every question it touched into the backtest corpus (spec §10 item 1: the token sees own-forecast questions at full fidelity forever, so the live bot becomes a corpus factory). The hook must NEVER fail the tournament run, and is controlled by `SNAPSHOT_AFTER_RUN` (default `true`).

**Files:**
- Create: `tests/backtest/test_snapshot_hook.py`
- Modify: `scripts/run_tournament.py` (full rewrite shown below; current file is 32 lines)

- [ ] **Step 1: Ensure the test package directory exists**
Run: `mkdir -p tests/backtest && touch tests/backtest/__init__.py`
(Idempotent — the directory was created in earlier phases; this guards a fresh checkout.)

- [ ] **Step 2: Write the failing test**

Create `tests/backtest/test_snapshot_hook.py`:

```python
"""Post-run snapshot accumulator hook: archives this run's questions into the
backtest corpus. The hook must NEVER fail a tournament run."""
import logging
from pathlib import Path

import scripts.run_tournament as rt


class _FakeStore:
    def __init__(self, root):
        self.root = root


class _FakeReport:
    fetched = 2
    withheld: dict = {}
    errors: list = []


def _install_fakes(monkeypatch, calls, fail=False):
    """Patch the corpus seams the hook imports at call time."""
    import backtest.corpus.snapshot as snapshot_mod
    import backtest.corpus.store as store_mod

    def fake_snapshot(store, post_ids):
        if fail:
            raise RuntimeError("metaculus down")
        calls.append((store, post_ids))
        return _FakeReport()

    monkeypatch.setattr(snapshot_mod, "snapshot_questions", fake_snapshot)
    monkeypatch.setattr(store_mod, "CorpusStore", _FakeStore)


def test_collects_deduped_sorted_int_ids_into_default_store(monkeypatch):
    monkeypatch.delenv("SNAPSHOT_AFTER_RUN", raising=False)
    calls = []
    _install_fakes(monkeypatch, calls)
    records = [{"question_id": 5, "type": "binary"},
               {"question_id": 3, "type": "binary"},
               {"question_id": 5, "type": "multiple_choice"},
               {"question_id": "7", "type": "binary"},   # string ids coerced
               {"question_id": None}]
    rt.post_run_snapshot(records)
    assert len(calls) == 1
    store, ids = calls[0]
    assert ids == [3, 5, 7]
    assert store.root == Path("data/backtest")


def test_env_flag_false_disables_snapshot(monkeypatch):
    monkeypatch.setenv("SNAPSHOT_AFTER_RUN", "false")
    calls = []
    _install_fakes(monkeypatch, calls)
    rt.post_run_snapshot([{"question_id": 1}])
    assert calls == []


def test_env_flag_defaults_to_true(monkeypatch):
    monkeypatch.delenv("SNAPSHOT_AFTER_RUN", raising=False)
    calls = []
    _install_fakes(monkeypatch, calls)
    rt.post_run_snapshot([{"question_id": 1}])
    assert len(calls) == 1


def test_empty_records_do_not_call_snapshot(monkeypatch):
    monkeypatch.delenv("SNAPSHOT_AFTER_RUN", raising=False)
    calls = []
    _install_fakes(monkeypatch, calls)
    rt.post_run_snapshot([])
    rt.post_run_snapshot([{"question_id": None}])
    assert calls == []


def test_snapshot_failure_is_swallowed_and_logged(monkeypatch, caplog):
    monkeypatch.delenv("SNAPSHOT_AFTER_RUN", raising=False)
    _install_fakes(monkeypatch, calls=[], fail=True)
    with caplog.at_level(logging.ERROR):
        rt.post_run_snapshot([{"question_id": 1}])   # must not raise
    assert "snapshot failed" in caplog.text
```

- [ ] **Step 3: Run test to verify it fails**
Run: `pytest tests/backtest/test_snapshot_hook.py -v`
Expected: FAIL — all 5 tests with `AttributeError: module 'scripts.run_tournament' has no attribute 'post_run_snapshot'`

- [ ] **Step 4: Write minimal implementation**

Replace `scripts/run_tournament.py` in full:

```python
"""Gate E: forecast the live seasonal FutureEval tournament + MiniBench.
Refuses to publish unless SUBMIT_PREDICTION=true (defense in depth: the
flag also gates publish_reports_to_metaculus inside build_framework_bot).

Post-run, the forward snapshot accumulator (backtest spec §10 item 1)
archives every question this run touched into the backtest corpus at
data/backtest/ — the Metaculus token retains full visibility into
own-forecast questions forever, so every live run grows the corpus.
Guarded: a snapshot failure can NEVER fail the tournament run.
Disable with SNAPSHOT_AFTER_RUN=false (default: true). Note: on ephemeral
CI runners the archive is discarded with the container — the accumulator
pays off on machines with a persistent data/backtest/."""
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from bot.config import DEFAULT_CONFIG
from bot.ensemble_bot import build_framework_bot
from bot.logging_store import read_forecasts

logger = logging.getLogger(__name__)


def post_run_snapshot(records: list[dict],
                      store_root: str = "data/backtest") -> None:
    """Archive every question id appearing in `records` into the backtest
    corpus. NEVER raises — log-and-continue only (the tournament result is
    already in; the snapshot is a bonus, not a dependency)."""
    if os.getenv("SNAPSHOT_AFTER_RUN", "true").strip().lower() != "true":
        logger.info("post-run snapshot disabled (SNAPSHOT_AFTER_RUN != true)")
        return
    try:
        # Imported at call time: the live bot must keep working even if the
        # backtest sidecar is broken or absent.
        from backtest.corpus.snapshot import snapshot_questions
        from backtest.corpus.store import CorpusStore

        ids = sorted({int(r["question_id"]) for r in records
                      if r.get("question_id") is not None})
        if not ids:
            logger.info("post-run snapshot: no question ids in this run")
            return
        store = CorpusStore(Path(store_root))
        report = snapshot_questions(store, ids)
        logger.info("post-run snapshot: %d archived, %d errors",
                    report.fetched, len(report.errors))
    except Exception:
        logger.exception("post-run snapshot failed (tournament run unaffected)")


async def main():
    from forecasting_tools import MetaculusApi
    n_before = len(read_forecasts(DEFAULT_CONFIG.log_path))
    bot = build_framework_bot(DEFAULT_CONFIG)
    for tournament in (MetaculusApi.CURRENT_AI_COMPETITION_ID,
                       MetaculusApi.CURRENT_MINIBENCH_ID):
        reports = await bot.forecast_on_tournament(
            tournament, return_exceptions=True)
        errors = [r for r in reports if isinstance(r, Exception)]
        print(f"{tournament}: {len(reports) - len(errors)} ok, "
              f"{len(errors)} failed")
        for e in errors:
            print(f"  ERROR: {e}")
    # Only the records appended by THIS run (the log is append-only).
    post_run_snapshot(read_forecasts(DEFAULT_CONFIG.log_path)[n_before:])


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: Run test to verify it passes**
Run: `pytest tests/backtest/test_snapshot_hook.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Run the full existing suite to confirm no regression**
Run: `pytest tests/ -v`
Expected: PASS (all green — the only changed live file is `scripts/run_tournament.py`, additive)

- [ ] **Step 7: Commit**
```bash
git add tests/backtest/test_snapshot_hook.py scripts/run_tournament.py && git commit -m "feat(backtest): forward snapshot accumulator hook after tournament runs" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 33: CI smoke gate — stub system, frozen mini-corpus, regression check, workflow

Spec §10 item 2 / §8 integration testing: a no-network end-to-end run of the whole pipeline (corpus → experiment → artifacts → scores → report) on a committed 5-question corpus, gated in GitHub Actions, plus a numeric regression check of `scores.jsonl` mean `baseline_score` against a committed baseline.

**Files:**
- Create: `backtest/systems/stub.py`
- Create: `tests/backtest/test_stub_system.py`
- Create: `tests/backtest/fixtures/mini_corpus/metaculus-90001.json` (+ 90002…90005)
- Create: `backtest/corpus/fixtures.py`
- Create: `tests/backtest/test_fixture_corpus.py`
- Create: `experiments/smoke.yaml`
- Create: `tests/backtest/test_smoke_pipeline.py`
- Modify: `backtest/experiments/runner.py` (add `systems=` override to `run_experiment`)
- Modify: `scripts/backtest.py` (add `--system stub` to the `run` subcommand)
- Create: `tests/backtest/test_backtest_cli.py`
- Create: `scripts/check_regression.py`
- Create: `tests/backtest/test_check_regression.py`
- Create: `tests/backtest/fixtures/smoke_baseline.json` (generated, then committed)
- Create: `.github/workflows/backtest-smoke.yml`

- [ ] **Step 1: Write the failing test for the stub system**

Create `tests/backtest/test_stub_system.py`:

```python
"""StubSystem: deterministic, zero-network ForecastSystem for the CI smoke
gate. Never calls ctx.llm, never does research."""
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.systems.stub import StubSystem
from backtest.timetravel.asof import AsOfContext

AS_OF = datetime(2025, 2, 22, tzinfo=timezone.utc)


def _q(qtype, **kw):
    base = dict(qid="metaculus:1", platform="metaculus", native_id="1",
                qtype=qtype, title="t")
    base.update(kw)
    return BacktestQuestion(**base)


async def test_binary_returns_fixed_prob_clean_artifact():
    art = await StubSystem().forecast(_q(QType.BINARY),
                                      AsOfContext(as_of=AS_OF))
    assert art.system_name == "stub"
    assert art.qid == "metaculus:1"
    assert art.as_of == AS_OF
    assert art.forecast.binary_prob == 0.6
    assert art.contamination == "clean"
    assert art.error is None
    assert art.stages and art.stages[0].cost_usd == 0.0


async def test_mc_returns_uniform_over_options():
    q = _q(QType.MULTIPLE_CHOICE, options=("a", "b", "c", "d"))
    art = await StubSystem().forecast(q, AsOfContext(as_of=AS_OF))
    probs = art.forecast.mc_probs
    assert set(probs) == {"a", "b", "c", "d"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9


async def test_numeric_returns_valid_uniform_cdf():
    q = _q(QType.NUMERIC,
           scaling={"range_min": 0, "range_max": 100, "cdf_size": 201})
    art = await StubSystem().forecast(q, AsOfContext(as_of=AS_OF))
    cdf = art.forecast.cdf
    assert len(cdf) == 201
    assert cdf[0] == 0.0 and cdf[-1] == 1.0
    assert all(b > a for a, b in zip(cdf, cdf[1:]))      # strictly ascending
    assert all(b - a <= 0.2 for a, b in zip(cdf, cdf[1:]))  # per-bin mass cap


async def test_does_not_touch_llm_or_store():
    # ctx with NO llm and NO store: the stub must not need either.
    art = await StubSystem().forecast(_q(QType.BINARY),
                                      AsOfContext(as_of=AS_OF))
    assert art.forecast is not None


def test_fingerprint_is_deterministic_12_hex():
    fp = StubSystem().config_fingerprint()
    assert fp == StubSystem().config_fingerprint()
    assert len(fp) == 12
    int(fp, 16)   # raises if not hex
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_stub_system.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'backtest.systems.stub'`

- [ ] **Step 3: Write the stub system**

Create `backtest/systems/stub.py`:

```python
"""Deterministic no-network ForecastSystem for the CI smoke gate.

Fixed outputs: binary 0.6, MC uniform over options, numeric a uniform
201-point CDF. No research, no LLM calls (ctx.llm is never touched), no
randomness — so the smoke run's scores are bit-stable and the committed
regression baseline (tests/backtest/fixtures/smoke_baseline.json) is exact.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord)
from backtest.timetravel.asof import AsOfContext

_BINARY_P = 0.6
_CDF_SIZE = 201


class StubSystem:
    name = "stub"

    def config_fingerprint(self) -> str:
        return hashlib.sha256(b"stub-v1").hexdigest()[:12]

    async def forecast(self, q: BacktestQuestion,
                       ctx: AsOfContext) -> ForecastArtifact:
        if q.qtype == QType.BINARY:
            fc = Forecast(qtype=q.qtype, binary_prob=_BINARY_P)
        elif q.qtype == QType.MULTIPLE_CHOICE:
            n = len(q.options) or 1
            fc = Forecast(qtype=q.qtype,
                          mc_probs={o: 1.0 / n for o in q.options})
        else:  # NUMERIC / DISCRETE: uniform CDF, valid under 2026 constraints
            size = (q.scaling or {}).get("cdf_size", _CDF_SIZE)
            fc = Forecast(qtype=q.qtype,
                          cdf=tuple(i / (size - 1) for i in range(size)))
        now = datetime.now(timezone.utc).isoformat()
        return ForecastArtifact(
            qid=q.qid,
            horizon="",   # stamped by the experiment runner, which owns it
            as_of=ctx.as_of,
            system_name=self.name,
            config_fingerprint=self.config_fingerprint(),
            forecast=fc,
            stages=[StageRecord(stage="stub",
                                outputs={"note": "fixed deterministic forecast"},
                                started_at=now, ended_at=now)],
            contamination="clean",   # no model knowledge is used at all
            cutoff_mode=ctx.cutoff_mode,
        )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_stub_system.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit the stub system**
```bash
git add backtest/systems/stub.py tests/backtest/test_stub_system.py && git commit -m "feat(backtest): deterministic StubSystem for CI smoke runs" -m "test(backtest): StubSystem shape and determinism tests" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Create the five frozen mini-corpus fixture files**

Run: `mkdir -p tests/backtest/fixtures/mini_corpus`

Create `tests/backtest/fixtures/mini_corpus/metaculus-90001.json`:

```json
{
  "question": {
    "qid": "metaculus:90001",
    "platform": "metaculus",
    "native_id": "90001",
    "qtype": "binary",
    "title": "Will the Aurora index close above 8000 on 2025-02-21?",
    "description": "Synthetic fixture question for the CI smoke gate.",
    "resolution_criteria": "Resolves YES if the official closing value exceeds 8000.",
    "fine_print": "Uses the venue's official close.",
    "open_time": "2025-01-01T00:00:00+00:00",
    "scheduled_close_time": "2025-03-01T00:00:00+00:00",
    "spot_scoring_time": "2025-02-22T00:00:00+00:00",
    "actual_resolve_time": "2025-03-02T00:00:00+00:00",
    "resolution": true,
    "options": [],
    "scaling": null,
    "question_weight": 1.0,
    "provenance": {"source": "fixture", "license": "synthetic", "fetched_at": "2026-06-11T00:00:00+00:00"}
  },
  "cp_series": [
    ["2025-01-05T00:00:00+00:00", 0.55],
    ["2025-02-01T00:00:00+00:00", 0.70],
    ["2025-02-20T00:00:00+00:00", 0.82]
  ]
}
```

Create `tests/backtest/fixtures/mini_corpus/metaculus-90002.json`:

```json
{
  "question": {
    "qid": "metaculus:90002",
    "platform": "metaculus",
    "native_id": "90002",
    "qtype": "binary",
    "title": "Will the Borealis spacecraft attempt a launch before 2025-03-01?",
    "description": "Synthetic fixture question for the CI smoke gate.",
    "resolution_criteria": "Resolves YES on any launch attempt before the date.",
    "fine_print": "Static fires do not count.",
    "open_time": "2025-01-01T00:00:00+00:00",
    "scheduled_close_time": "2025-03-01T00:00:00+00:00",
    "spot_scoring_time": "2025-02-22T00:00:00+00:00",
    "actual_resolve_time": "2025-03-02T00:00:00+00:00",
    "resolution": false,
    "options": [],
    "scaling": null,
    "question_weight": 1.0,
    "provenance": {"source": "fixture", "license": "synthetic", "fetched_at": "2026-06-11T00:00:00+00:00"}
  },
  "cp_series": [
    ["2025-01-05T00:00:00+00:00", 0.40],
    ["2025-02-01T00:00:00+00:00", 0.25],
    ["2025-02-20T00:00:00+00:00", 0.12]
  ]
}
```

Create `tests/backtest/fixtures/mini_corpus/metaculus-90003.json`:

```json
{
  "question": {
    "qid": "metaculus:90003",
    "platform": "metaculus",
    "native_id": "90003",
    "qtype": "binary",
    "title": "Will the Cascadia bill pass a floor vote before 2025-03-01?",
    "description": "Synthetic fixture question for the CI smoke gate.",
    "resolution_criteria": "Resolves YES on a successful floor vote.",
    "fine_print": "Committee votes do not count.",
    "open_time": "2025-01-01T00:00:00+00:00",
    "scheduled_close_time": "2025-03-01T00:00:00+00:00",
    "spot_scoring_time": "2025-02-22T00:00:00+00:00",
    "actual_resolve_time": "2025-03-02T00:00:00+00:00",
    "resolution": true,
    "options": [],
    "scaling": null,
    "question_weight": 1.0,
    "provenance": {"source": "fixture", "license": "synthetic", "fetched_at": "2026-06-11T00:00:00+00:00"}
  },
  "cp_series": [
    ["2025-01-05T00:00:00+00:00", 0.50],
    ["2025-02-01T00:00:00+00:00", 0.60],
    ["2025-02-20T00:00:00+00:00", 0.66]
  ]
}
```

Create `tests/backtest/fixtures/mini_corpus/metaculus-90004.json`:

```json
{
  "question": {
    "qid": "metaculus:90004",
    "platform": "metaculus",
    "native_id": "90004",
    "qtype": "multiple_choice",
    "title": "How many named Delta-class storms will form before 2025-03-01?",
    "description": "Synthetic fixture question for the CI smoke gate.",
    "resolution_criteria": "Resolves to the bucket containing the official count.",
    "fine_print": "Counts at the scheduled close time.",
    "open_time": "2025-01-01T00:00:00+00:00",
    "scheduled_close_time": "2025-03-01T00:00:00+00:00",
    "spot_scoring_time": "2025-02-22T00:00:00+00:00",
    "actual_resolve_time": "2025-03-02T00:00:00+00:00",
    "resolution": "4-6",
    "options": ["0-3", "4-6", "7 or more"],
    "scaling": null,
    "question_weight": 1.0,
    "provenance": {"source": "fixture", "license": "synthetic", "fetched_at": "2026-06-11T00:00:00+00:00"}
  },
  "cp_series": [
    ["2025-01-05T00:00:00+00:00", {"0-3": 0.20, "4-6": 0.50, "7 or more": 0.30}],
    ["2025-02-20T00:00:00+00:00", {"0-3": 0.15, "4-6": 0.60, "7 or more": 0.25}]
  ]
}
```

Create `tests/backtest/fixtures/mini_corpus/metaculus-90005.json`:

```json
{
  "question": {
    "qid": "metaculus:90005",
    "platform": "metaculus",
    "native_id": "90005",
    "qtype": "numeric",
    "title": "What will the Eastlake activity index be on 2025-03-01?",
    "description": "Synthetic fixture question for the CI smoke gate.",
    "resolution_criteria": "Resolves to the official index value at close.",
    "fine_print": "Rounded to one decimal place.",
    "open_time": "2025-01-01T00:00:00+00:00",
    "scheduled_close_time": "2025-03-01T00:00:00+00:00",
    "spot_scoring_time": "2025-02-22T00:00:00+00:00",
    "actual_resolve_time": "2025-03-02T00:00:00+00:00",
    "resolution": 42.0,
    "options": [],
    "scaling": {"range_min": 0, "range_max": 100, "zero_point": null,
                "open_lower_bound": false, "open_upper_bound": false,
                "cdf_size": 201, "unit": "index points"},
    "question_weight": 1.0,
    "provenance": {"source": "fixture", "license": "synthetic", "fetched_at": "2026-06-11T00:00:00+00:00"}
  },
  "cp_series": [
    ["2025-01-15T00:00:00+00:00", 38.0],
    ["2025-02-15T00:00:00+00:00", 41.5]
  ]
}
```

(The numeric `cp_series` is a CP median, stored for completeness; current scoring does not consume it.)

- [ ] **Step 7: Write the failing test for the fixture loader**

Create `tests/backtest/test_fixture_corpus.py`:

```python
"""The committed 5-question mini corpus and its loader — the substrate of
the CI smoke gate. 3 binary, 1 MC, 1 numeric; all resolved; all with a CP
series."""
from datetime import datetime, timezone
from pathlib import Path

from backtest.core import QType
from backtest.corpus.fixtures import load_fixture_corpus
from backtest.corpus.store import CorpusStore

FIXTURES = Path(__file__).parent / "fixtures" / "mini_corpus"


def test_loads_five_questions_with_resolutions(tmp_path):
    store = CorpusStore(tmp_path / "store")
    assert load_fixture_corpus(store, FIXTURES) == 5
    qs = store.query(platforms=["metaculus"], resolved_only=True)
    assert len(qs) == 5
    by_type: dict = {}
    for q in qs:
        by_type.setdefault(q.qtype, []).append(q)
    assert len(by_type[QType.BINARY]) == 3
    assert len(by_type[QType.MULTIPLE_CHOICE]) == 1
    assert len(by_type[QType.NUMERIC]) == 1
    assert all(q.resolution is not None for q in qs)
    mc = by_type[QType.MULTIPLE_CHOICE][0]
    assert mc.resolution in mc.options
    num = by_type[QType.NUMERIC][0]
    assert num.scaling["cdf_size"] == 201


def test_horizon_window_fits_close_minus_7d(tmp_path):
    store = CorpusStore(tmp_path / "store")
    load_fixture_corpus(store, FIXTURES)
    q = store.get_question("metaculus:90001")
    assert q.open_time <= q.scheduled_close_time
    # close-7d (2025-02-22) must be inside [open, close) so the smoke
    # experiment's single horizon never skips a fixture question
    from datetime import timedelta
    assert q.scheduled_close_time - timedelta(days=7) >= q.open_time


def test_cp_series_step_lookup(tmp_path):
    store = CorpusStore(tmp_path / "store")
    load_fixture_corpus(store, FIXTURES)
    t = datetime(2025, 2, 22, tzinfo=timezone.utc)
    assert store.value_at("metaculus:90001", "cp", t) == 0.82
    early = datetime(2025, 1, 10, tzinfo=timezone.utc)
    assert store.value_at("metaculus:90001", "cp", early) == 0.55
    mc = store.value_at("metaculus:90004", "cp", t)
    assert mc == {"0-3": 0.15, "4-6": 0.60, "7 or more": 0.25}
```

- [ ] **Step 8: Run test to verify it fails**
Run: `pytest tests/backtest/test_fixture_corpus.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'backtest.corpus.fixtures'`

- [ ] **Step 9: Write the fixture loader**

Create `backtest/corpus/fixtures.py`:

```python
"""Loader for the committed mini-corpus fixtures (CI smoke gate).

Fixture format — one JSON file per question:

    {
      "question": { ...BacktestQuestion fields; datetimes as iso8601
                    strings; qtype as the QType enum value... },
      "cp_series": [[iso_timestamp, value], ...]    # optional
    }

Also runnable as a module to build a store on disk (used by the CI
workflow):  python -m backtest.corpus.fixtures <fixtures_dir> <store_root>
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore

_DT_FIELDS = ("open_time", "scheduled_close_time", "spot_scoring_time",
              "actual_resolve_time")


def _question_from_fixture(d: dict) -> BacktestQuestion:
    kw = dict(d)
    kw["qtype"] = QType(kw["qtype"])
    for f in _DT_FIELDS:
        if kw.get(f):
            kw[f] = datetime.fromisoformat(kw[f])
    kw["options"] = tuple(kw.get("options") or ())
    return BacktestQuestion(**kw)


def load_fixture_corpus(store: CorpusStore, fixtures_dir: Path) -> int:
    """Load every *.json fixture into `store`; returns the count loaded."""
    n = 0
    for path in sorted(Path(fixtures_dir).glob("*.json")):
        blob = json.loads(path.read_text())
        q = _question_from_fixture(blob["question"])
        store.upsert_question(q)
        series = blob.get("cp_series") or []
        if series:
            store.put_series(q.qid, "cp", [(ts, v) for ts, v in series])
        n += 1
    return n


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Load committed mini-corpus fixtures into a CorpusStore")
    ap.add_argument("fixtures_dir", type=Path)
    ap.add_argument("store_root", type=Path)
    args = ap.parse_args()
    count = load_fixture_corpus(CorpusStore(args.store_root), args.fixtures_dir)
    print(f"loaded {count} fixture questions into {args.store_root}")
```

- [ ] **Step 10: Run test to verify it passes**
Run: `pytest tests/backtest/test_fixture_corpus.py -v`
Expected: PASS (3 passed)

- [ ] **Step 11: Commit fixtures and loader**
```bash
git add tests/backtest/fixtures/mini_corpus backtest/corpus/fixtures.py tests/backtest/test_fixture_corpus.py && git commit -m "feat(backtest): frozen 5-question mini corpus + fixture loader for CI" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 12: Create the smoke experiment spec**

Run: `mkdir -p experiments`

Create `experiments/smoke.yaml`:

```yaml
# CI smoke gate experiment. Runs against the committed mini corpus
# (tests/backtest/fixtures/mini_corpus) with the deterministic StubSystem
# injected via `scripts/backtest.py run --system stub` — variants and
# baselines below are therefore IGNORED in CI (the --system override
# replaces the full system list). No network, no LLM spend.
name: ci-smoke
corpus:
  platforms: [metaculus]
  resolved_only: true
horizons: [close-7d]
cutoff_mode: strict
profile: smoke
variants: []
baselines: []
seed: 0
```

- [ ] **Step 13: Write the failing end-to-end pipeline test**

Create `tests/backtest/test_smoke_pipeline.py`:

```python
"""End-to-end smoke pipeline: fixture corpus -> run_experiment with the
StubSystem override -> artifacts.jsonl + scores.jsonl + manifest.json ->
build_report. Exactly what .github/workflows/backtest-smoke.yml runs in CI.
No network, no LLM."""
import json
from pathlib import Path

from backtest.corpus.fixtures import load_fixture_corpus
from backtest.corpus.store import CorpusStore
from backtest.experiments.runner import run_experiment
from backtest.experiments.spec import ExperimentSpec
from backtest.reports.report import build_report
from backtest.systems.stub import StubSystem

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "mini_corpus"
SMOKE_YAML = REPO / "experiments" / "smoke.yaml"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


async def test_smoke_run_with_stub_override_produces_run_outputs(tmp_path):
    store = CorpusStore(tmp_path / "store")
    assert load_fixture_corpus(store, FIXTURES) == 5
    spec = ExperimentSpec.from_yaml(SMOKE_YAML)
    out = tmp_path / "runs"

    await run_experiment(spec, store, out, live_llm=False, yes=True,
                         systems=[StubSystem()])

    run_dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert (rd / "manifest.json").exists()

    artifacts = _read_jsonl(rd / "artifacts.jsonl")
    assert len(artifacts) == 5                      # 5 questions x 1 horizon
    assert {a["system_name"] for a in artifacts} == {"stub"}
    assert all(a["forecast"] is not None for a in artifacts)

    scores = _read_jsonl(rd / "scores.jsonl")
    assert scores
    assert all("baseline_score" in s for s in scores)

    report_path = build_report(rd)
    assert report_path.exists()
    assert report_path.read_text().strip()
```

- [ ] **Step 14: Run test to verify it fails**
Run: `pytest tests/backtest/test_smoke_pipeline.py -v`
Expected: FAIL with `TypeError: run_experiment() got an unexpected keyword argument 'systems'`

- [ ] **Step 15: Add the `systems` override to `run_experiment`**

Modify `backtest/experiments/runner.py` (built in the experiments-layer work; this is the same shape as the parallel fix to phase-4's Task 30). Two edits:

(a) Extend the signature with one keyword-only parameter (keep every existing parameter exactly as is):

```python
async def run_experiment(spec, store, out_dir: Path, *, live_llm: bool = True,
                         yes: bool = False,
                         systems: list | None = None) -> RunResult:
```

(b) The runner's existing construction lines

```python
    systems = _build_systems(spec, store)
    run_id = _run_id(spec, systems)
```

collide with the new parameter name; replace them with (the local is renamed `systems_map`):

```python
    # CI smoke override: when `systems` is supplied it is the COMPLETE set
    # of systems to run — spec.variants and spec.baselines are ignored.
    systems_map = (_build_systems(spec, store) if systems is None
                   else {s.name: s for s in systems})
    run_id = _run_id(spec, systems_map)
```

and update the consuming loop inside the question/horizon iteration to read the renamed dict:

```python
            for name, system in systems_map.items():
```

Everything downstream — manifest, artifacts, scoring — treats injected systems identically to constructed ones (they flow through the same dict).

- [ ] **Step 16: Run test to verify it passes**
Run: `pytest tests/backtest/test_smoke_pipeline.py -v`
Expected: PASS (1 passed)
Also run: `pytest tests/backtest -v` — Expected: PASS (no regression in the earlier runner tests, which never pass `systems=`)

- [ ] **Step 17: Commit the runner override + smoke spec**
```bash
git add backtest/experiments/runner.py experiments/smoke.yaml tests/backtest/test_smoke_pipeline.py && git commit -m "feat(backtest): systems override in run_experiment + ci-smoke experiment spec" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 18: Write the failing CLI test for `--system stub`**

Create `tests/backtest/test_backtest_cli.py`:

```python
"""The `run` subcommand must expose --system stub (CI smoke gate). Help-text
only — never executes a run, never touches the network."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_run_subcommand_exposes_system_stub_flag():
    out = subprocess.run(
        [sys.executable, "scripts/backtest.py", "run", "--help"],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert "--system" in out.stdout
    assert "stub" in out.stdout
```

- [ ] **Step 19: Run the test**
Run: `pytest tests/backtest/test_backtest_cli.py -v`
Expected: PASS if the `run` subparser already carries `--system` (the flag ships with the phase-4 CLI, Task 31); FAIL with `AssertionError` only if the flag is missing — Step 20 then aligns the CLI.

- [ ] **Step 20: Confirm the run subparser exposes `--system stub` (added with the CLI) and wire StubSystem**

The `--system` flag is added with the CLI itself (phase-4's Task 31 builds the `run`/`replay` subparsers in a shared for-loop whose subparser local is `sp`, and adds this flag there). This step VERIFIES that wiring and aligns it if it drifted — do not re-add a duplicate flag. The exact code that must be present in `scripts/backtest.py`:

(a) Inside the run/replay subparser for-loop, gated to `run` only:

```python
if name == "run":
    sp.add_argument("--system", choices=["stub"], default=None,
                    help="override systems under test; 'stub' runs the "
                         "deterministic no-network StubSystem (CI smoke gate)")
```

(b) In `_cmd_run`, immediately before/around the `run_experiment(...)` call — this is where Task 33's `backtest/systems/stub.py:StubSystem` (created in Steps 1–5 above) gets wired in:

```python
systems = None
if getattr(args, "system", None) == "stub":
    from backtest.systems.stub import StubSystem
    systems = [StubSystem()]
result = asyncio.run(run_experiment(spec, store, Path(args.out),
                                    live_llm=live_llm, yes=args.yes,
                                    systems=systems))
```

- [ ] **Step 21: Run test to verify it passes**
Run: `pytest tests/backtest/test_backtest_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 22: Commit the CLI flag**
```bash
git add scripts/backtest.py tests/backtest/test_backtest_cli.py && git commit -m "feat(backtest): --system stub CLI override for no-network CI runs" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 23: Write the failing test for the regression checker**

Create `tests/backtest/test_check_regression.py`:

```python
"""scripts/check_regression.py: compare a run's mean baseline_score in
scores.jsonl against a committed reference, one-sided with tolerance
(improvements never fail)."""
import json

from scripts.check_regression import check, main, mean_baseline_score


def _scores(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _baseline(path, mean, tol):
    path.write_text(json.dumps({"mean_baseline_score": mean, "tolerance": tol}))
    return path


def test_mean_ignores_rows_without_numeric_baseline_score(tmp_path):
    p = _scores(tmp_path / "s.jsonl",
                [{"baseline_score": 10.0}, {"baseline_score": None},
                 {"other": 1}, {"baseline_score": 20},
                 {"baseline_score": True}])     # bool is NOT a score
    assert mean_baseline_score(p) == 15.0


def test_within_tolerance_passes(tmp_path):
    s = _scores(tmp_path / "s.jsonl",
                [{"baseline_score": 10.0}, {"baseline_score": 20.0}])
    b = _baseline(tmp_path / "b.json", 15.5, 1.0)
    ok, msg = check(s, b)
    assert ok
    assert "OK" in msg


def test_drop_beyond_tolerance_fails(tmp_path):
    s = _scores(tmp_path / "s.jsonl",
                [{"baseline_score": 10.0}, {"baseline_score": 20.0}])   # mean 15
    b = _baseline(tmp_path / "b.json", 30.0, 1.0)
    ok, msg = check(s, b)
    assert not ok
    assert "REGRESSION" in msg


def test_improvement_never_fails(tmp_path):
    s = _scores(tmp_path / "s.jsonl", [{"baseline_score": 50.0}])
    b = _baseline(tmp_path / "b.json", 10.0, 1.0)
    ok, _ = check(s, b)
    assert ok


def test_main_update_then_pass_then_fail(tmp_path):
    good = _scores(tmp_path / "good.jsonl",
                   [{"baseline_score": 10.0}, {"baseline_score": 20.0}])
    b = tmp_path / "baseline.json"
    rc = main(["--scores", str(good), "--baseline", str(b),
               "--update-baseline", "--tolerance", "1.0"])
    assert rc == 0
    assert json.loads(b.read_text()) == {"mean_baseline_score": 15.0,
                                         "tolerance": 1.0}
    assert main(["--scores", str(good), "--baseline", str(b)]) == 0
    bad = _scores(tmp_path / "bad.jsonl", [{"baseline_score": 1.0}])
    assert main(["--scores", str(bad), "--baseline", str(b)]) == 1
```

- [ ] **Step 24: Run test to verify it fails**
Run: `pytest tests/backtest/test_check_regression.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'scripts.check_regression'`

- [ ] **Step 25: Write the regression checker**

Create `scripts/check_regression.py`:

```python
"""CI regression gate: compare a smoke run's mean baseline_score (from a
run directory's scores.jsonl) against a committed reference value.

One-sided with tolerance: fails ONLY when the mean drops more than
`tolerance` below the reference — improvements always pass (then refresh
the reference with --update-baseline). The smoke run is deterministic
(StubSystem on the frozen mini corpus), so any drop is a real change in
scoring/runner behavior, not noise.

Usage:
  python scripts/check_regression.py --scores RUN_DIR/scores.jsonl \
      --baseline tests/backtest/fixtures/smoke_baseline.json
  python scripts/check_regression.py --scores ... --baseline ... \
      --update-baseline --tolerance 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def mean_baseline_score(scores_path: Path) -> float:
    vals = []
    for line in Path(scores_path).read_text().splitlines():
        if not line.strip():
            continue
        v = json.loads(line).get("baseline_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    if not vals:
        raise SystemExit(f"no baseline_score values found in {scores_path}")
    return sum(vals) / len(vals)


def check(scores_path: Path, baseline_path: Path) -> tuple[bool, str]:
    ref = json.loads(Path(baseline_path).read_text())
    mean = mean_baseline_score(scores_path)
    floor = ref["mean_baseline_score"] - ref["tolerance"]
    ok = mean >= floor
    msg = (f"mean baseline_score={mean:.4f} "
           f"reference={ref['mean_baseline_score']:.4f} "
           f"tolerance={ref['tolerance']} floor={floor:.4f} "
           f"-> {'OK' if ok else 'REGRESSION'}")
    return ok, msg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, type=Path,
                    help="path to a run's scores.jsonl")
    ap.add_argument("--baseline", required=True, type=Path,
                    help="committed reference JSON "
                         "{mean_baseline_score, tolerance}")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write the current mean as the new reference")
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="tolerance written on --update-baseline")
    args = ap.parse_args(argv)

    if args.update_baseline:
        mean = mean_baseline_score(args.scores)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(
            {"mean_baseline_score": mean, "tolerance": args.tolerance},
            indent=2) + "\n")
        print(f"baseline updated: mean_baseline_score={mean:.6f} "
              f"tolerance={args.tolerance}")
        return 0

    ok, msg = check(args.scores, args.baseline)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 26: Run test to verify it passes**
Run: `pytest tests/backtest/test_check_regression.py -v`
Expected: PASS (5 passed)

- [ ] **Step 27: Commit the regression checker**
```bash
git add scripts/check_regression.py tests/backtest/test_check_regression.py && git commit -m "feat(backtest): scores.jsonl regression checker for the CI gate" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 28: Write the GitHub Actions workflow**

Create `.github/workflows/backtest-smoke.yml`:

```yaml
name: backtest-smoke
on:
  pull_request: {}
  workflow_dispatch: {}

concurrency:
  group: backtest-smoke-${{ github.ref }}
  cancel-in-progress: true

jobs:
  smoke:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - name: Backtest unit tests
        run: pytest tests/backtest -q
      - name: Build fixture corpus
        run: python -m backtest.corpus.fixtures tests/backtest/fixtures/mini_corpus data/backtest/ci-smoke
      - name: Smoke experiment (stub system, no network, no spend)
        run: python scripts/backtest.py run --spec experiments/smoke.yaml --corpus data/backtest/ci-smoke --out logs/backtest/ci-smoke --system stub --yes
      - name: Build report
        run: |
          RUN_DIR=$(ls -d logs/backtest/ci-smoke/*/ | head -n 1)
          python scripts/backtest.py report "$RUN_DIR"
      - name: Regression check vs committed baseline
        # mean-with-tolerance vs a committed value, NOT a paired significance test — sufficient because the stub run is deterministic; live regression gating would need paired stats from backtest/scoring/stats.py
        run: |
          RUN_DIR=$(ls -d logs/backtest/ci-smoke/*/ | head -n 1)
          python scripts/check_regression.py --scores "${RUN_DIR%/}/scores.jsonl" --baseline tests/backtest/fixtures/smoke_baseline.json
```

- [ ] **Step 29: Reconcile the workflow's CLI invocations against the real CLI**
Run: `python scripts/backtest.py run --help && python scripts/backtest.py report --help`
Expected: exit 0 for both. Confirm the spelling of `--spec`, `--corpus`, `--out`, `--system`, `--yes`, and the `report` subcommand's positional run-dir argument as implemented in `scripts/backtest.py`. If any flag is spelled differently there, update the two workflow `run:` lines AND the local commands in the next step to match the implemented spelling (the CLI is the contract; the workflow follows it).

- [ ] **Step 30: Validate the workflow YAML parses**
Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/backtest-smoke.yml')); print('yaml ok')"`
Expected: prints `yaml ok`

- [ ] **Step 31: Generate the committed baseline by running the exact CI commands locally**
Run (from the repo root — outputs land under gitignored `data/backtest/` and `logs/backtest/`):
```bash
rm -rf data/backtest/ci-smoke logs/backtest/ci-smoke
python -m backtest.corpus.fixtures tests/backtest/fixtures/mini_corpus data/backtest/ci-smoke
python scripts/backtest.py run --spec experiments/smoke.yaml --corpus data/backtest/ci-smoke --out logs/backtest/ci-smoke --system stub --yes
RUN_DIR=$(ls -d logs/backtest/ci-smoke/*/ | head -n 1)
python scripts/backtest.py report "$RUN_DIR"
python scripts/check_regression.py --scores "${RUN_DIR%/}/scores.jsonl" --baseline tests/backtest/fixtures/smoke_baseline.json --update-baseline --tolerance 1.0
python scripts/check_regression.py --scores "${RUN_DIR%/}/scores.jsonl" --baseline tests/backtest/fixtures/smoke_baseline.json
```
Expected: every command exits 0; the final line prints `... -> OK`; `tests/backtest/fixtures/smoke_baseline.json` now exists with the deterministic mean and `"tolerance": 1.0`.

- [ ] **Step 32: Commit the workflow and baseline**
```bash
git add .github/workflows/backtest-smoke.yml tests/backtest/fixtures/smoke_baseline.json && git commit -m "feat(backtest): CI smoke gate workflow with frozen baseline regression check" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
### Task 34: Calibration flywheel — export run artifacts into `scripts/fit_calibration.py` input format

Spec §10 item 3: a backtest run yields hundreds of resolved (forecast, outcome) pairs at once; export them in the exact row shape that `scripts/fit_calibration.py`'s `join_resolutions(logs, resolutions)` consumes, so `bot/calibration.py:fit_platt` can be fitted offline on backtest output instead of waiting months for live history.

What `scripts/fit_calibration.py` actually consumes (read it — `join_resolutions`, lines 16–25): a list of dict rows where it reads `row.get("type") == "binary"`, `row.get("question_id")` (used only as a dict key into `resolutions`), and `row.get("final")` (float, may be None → skipped); plus a `resolutions` dict mapping `question_id -> 1/0`. The live rows are written by `bot/ensemble_bot.py` `_log()` (lines 265–276) with keys `question_id`, `id_of_question`, `page_url`, `type`, `final`, `members`, `models`, `config_hash`, `research_providers`, and for binary additionally `raw`, `spread`.

**Files:**
- Create: `backtest/experiments/export_calibration.py`
- Create: `tests/backtest/test_export_calibration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_export_calibration.py`:

```python
"""Calibration flywheel: a run directory's artifacts.jsonl (+ resolutions
from scores.jsonl) round-trips into scripts/fit_calibration.py's
join_resolutions exactly like live logs/forecasts.jsonl rows do."""
import json
from datetime import datetime, timezone

from backtest.core import (Forecast, ForecastArtifact, QType, StageRecord,
                           artifact_to_dict)
from backtest.experiments.export_calibration import (
    artifacts_to_calibration_rows, resolutions_from_rows,
    write_calibration_export)
from bot.logging_store import read_forecasts
from scripts.fit_calibration import join_resolutions

AS_OF = datetime(2025, 2, 22, tzinfo=timezone.utc)


def _art(qid, qtype, forecast, stages=None, error=None):
    return ForecastArtifact(
        qid=qid, horizon="close-7d", as_of=AS_OF, system_name="stub",
        config_fingerprint="abc123def456", forecast=forecast,
        stages=stages or [], error=error)


def _run_dir(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    arts = [
        _art("metaculus:90001", QType.BINARY,
             Forecast(qtype=QType.BINARY, binary_prob=0.8),
             stages=[StageRecord(stage="aggregation",
                                 outputs={"aggregate_raw": 0.74})]),
        _art("metaculus:90002", QType.BINARY,
             Forecast(qtype=QType.BINARY, binary_prob=0.3)),
        _art("metaculus:90005", QType.NUMERIC,
             Forecast(qtype=QType.NUMERIC,
                      cdf=tuple(i / 200 for i in range(201)))),
        _art("metaculus:90003", QType.BINARY, None, error="boom"),
    ]
    (rd / "artifacts.jsonl").write_text(
        "".join(json.dumps(artifact_to_dict(a)) + "\n" for a in arts))
    scores = [
        {"qid": "metaculus:90001", "resolution": True, "baseline_score": 12.0},
        {"qid": "metaculus:90002", "resolution": False, "baseline_score": 30.0},
        {"qid": "metaculus:90005", "resolution": 42.0, "baseline_score": 5.0},
    ]
    (rd / "scores.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in scores))
    return rd


def test_rows_carry_fit_calibration_fields(tmp_path):
    rows = artifacts_to_calibration_rows(_run_dir(tmp_path))
    assert len(rows) == 3                      # failed artifact skipped
    by_qid = {r["question_id"]: r for r in rows}
    r1 = by_qid["metaculus:90001"]
    assert r1["type"] == "binary"
    assert r1["final"] == 0.8
    assert r1["raw"] == 0.74                   # from the aggregation stage
    assert r1["resolution"] == 1               # bool -> 1/0 for binary
    assert r1["config_hash"] == "abc123def456"
    r2 = by_qid["metaculus:90002"]
    assert r2["resolution"] == 0
    assert r2["raw"] is None                   # no aggregation stage logged
    r5 = by_qid["metaculus:90005"]
    assert r5["type"] == "numeric"
    assert r5["final"] is None                 # join skips non-binary anyway


def test_round_trip_into_fit_calibration_join(tmp_path):
    rows = artifacts_to_calibration_rows(_run_dir(tmp_path))
    preds, outcomes = join_resolutions(rows, resolutions_from_rows(rows))
    assert preds == [0.8, 0.3]
    assert outcomes == [1, 0]


def test_written_export_is_forecasts_jsonl_compatible(tmp_path):
    rd = _run_dir(tmp_path)
    out = write_calibration_export(rd, tmp_path / "export" / "calib.jsonl")
    assert out.exists()
    rows = read_forecasts(out)            # the live log reader parses it
    preds, outcomes = join_resolutions(rows, resolutions_from_rows(rows))
    assert preds == [0.8, 0.3]
    assert outcomes == [1, 0]


def test_missing_scores_file_yields_resolution_none(tmp_path):
    rd = _run_dir(tmp_path)
    (rd / "scores.jsonl").unlink()
    rows = artifacts_to_calibration_rows(rd)
    assert len(rows) == 3
    assert all(r["resolution"] is None for r in rows)
    preds, outcomes = join_resolutions(rows, resolutions_from_rows(rows))
    assert preds == [] and outcomes == []      # join skips unresolved rows
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_export_calibration.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'backtest.experiments.export_calibration'`

- [ ] **Step 3: Write the exporter**

Create `backtest/experiments/export_calibration.py`:

```python
"""Calibration flywheel: convert a backtest run directory into rows that
scripts/fit_calibration.py consumes, so the binary Platt map can be fitted
offline on hundreds of resolved backtest forecasts instead of waiting on
live history (spec §10 item 3).

EXACT field mapping (live logs/forecasts.jsonl row, as written by
bot/ensemble_bot.py _log() -> exported row):

  question_id  <- ForecastArtifact.qid (string "metaculus:90001"; live logs
                  use the int post id — join_resolutions only ever uses the
                  value as a dict key into `resolutions`, so any consistent
                  hashable id works)
  type         <- forecast.qtype value: "binary" | "multiple_choice" |
                  "numeric" | "discrete" (join_resolutions filters on
                  type == "binary", exactly like live rows)
  final        <- forecast.binary_prob for binary; None otherwise
                  (join_resolutions skips rows with final None)
  raw          <- first StageRecord whose outputs contain "aggregate_raw"
                  (fallback key: "raw") — the pre-calibration aggregate,
                  mirroring the live binary rows' `raw` field; None when the
                  system logged no aggregation stage (e.g. baselines/stub)
  config_hash  <- ForecastArtifact.config_fingerprint
  resolution   <- joined from the run's scores.jsonl (rows keyed "qid" or
                  "question_id"); binary bools coerced to 1/0. NOT part of
                  the live row shape — additive, carried here so the join
                  needs no network. Missing scores.jsonl => None (and
                  join_resolutions then skips the row, same as an
                  unresolved live question).
  horizon, system, as_of  <- artifact metadata (additive; ignored by
                  fit_calibration, useful for slicing exports)

Offline flywheel usage (no network):

    from backtest.experiments.export_calibration import (
        artifacts_to_calibration_rows, resolutions_from_rows)
    from scripts.fit_calibration import join_resolutions
    from bot.calibration import MIN_FIT_N, fit_platt, save_platt

    rows = artifacts_to_calibration_rows(run_dir)
    preds, outcomes = join_resolutions(rows, resolutions_from_rows(rows))
    if len(preds) >= MIN_FIT_N:
        save_platt(fit_platt(preds, outcomes), "calibration/binary.json")

CAUTION: a Platt map fitted on backtest artifacts inherits the backtest's
biases (clean-stratum skew, missing Perplexity research track). Point the
adapter's variant at a separate calibration_dir when experimenting; only
promote to the live calibration/ dir deliberately.
"""
from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _resolution_index(run_dir: Path) -> dict[str, object]:
    """qid -> resolution, from the run's scores.jsonl (the runner records
    the resolution it scored against)."""
    out: dict[str, object] = {}
    for row in _read_jsonl(run_dir / "scores.jsonl"):
        qid = row.get("qid") or row.get("question_id")
        if qid is not None and "resolution" in row:
            out[str(qid)] = row["resolution"]
    return out


def _raw_from_stages(stages: list[dict]) -> object:
    for st in stages or []:
        outs = st.get("outputs") or {}
        if "aggregate_raw" in outs:
            return outs["aggregate_raw"]
        if "raw" in outs:
            return outs["raw"]
    return None


def artifacts_to_calibration_rows(run_dir) -> list[dict]:
    """All artifacts with a final forecast, as fit_calibration-shaped rows."""
    run_dir = Path(run_dir)
    res_idx = _resolution_index(run_dir)
    rows: list[dict] = []
    for d in _read_jsonl(run_dir / "artifacts.jsonl"):
        fc = d.get("forecast")
        if not fc:
            continue   # skipped/failed question: nothing to calibrate on
        qtype = fc.get("qtype")
        final = fc.get("binary_prob") if qtype == "binary" else None
        res = res_idx.get(str(d["qid"]))
        if qtype == "binary" and isinstance(res, bool):
            res = int(res)
        rows.append({
            "question_id": d["qid"],
            "type": qtype,
            "final": final,
            "raw": _raw_from_stages(d.get("stages")),
            "config_hash": d.get("config_fingerprint"),
            "resolution": res,
            "horizon": d.get("horizon"),
            "system": d.get("system_name"),
            "as_of": d.get("as_of"),
        })
    return rows


def resolutions_from_rows(rows: list[dict]) -> dict:
    """The `resolutions` dict join_resolutions expects, from exported rows."""
    return {r["question_id"]: r["resolution"] for r in rows
            if r.get("type") == "binary" and r.get("resolution") is not None}


def write_calibration_export(run_dir, out_path) -> Path:
    """Write rows as a forecasts-jsonl-compatible file (one JSON object per
    line — readable by bot.logging_store.read_forecasts)."""
    rows = artifacts_to_calibration_rows(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Export a backtest run's artifacts as "
                    "fit_calibration-compatible JSONL")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_path", type=Path)
    args = ap.parse_args()
    p = write_calibration_export(args.run_dir, args.out_path)
    n = len(artifacts_to_calibration_rows(args.run_dir))
    print(f"wrote {n} rows to {p}")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_export_calibration.py -v`
Expected: PASS (4 passed)
Also run: `pytest tests/test_fit_calibration.py -v` — Expected: PASS (the live join helper is untouched)

- [ ] **Step 5: Commit**
```bash
git add backtest/experiments/export_calibration.py tests/backtest/test_export_calibration.py && git commit -m "feat(backtest): calibration flywheel export from run artifacts to fit_calibration rows" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
### Task 35: Documentation — `docs/backtesting/{README,COMPATIBILITY,DATA}.md`

Spec §9: the compatibility contract future systems build against, plus the operator's README and the data-layer reference. Doc-writing task (no TDD), but each file gets a verification step. The drafted content below is complete; the verification steps reconcile the few code-derived details (CLI flag spellings, store DDL) against the implemented modules before committing — the code is authoritative, the docs must match it.

**Files:**
- Create: `docs/backtesting/README.md`
- Create: `docs/backtesting/COMPATIBILITY.md`
- Create: `docs/backtesting/DATA.md`

- [ ] **Step 1: Create the directory**
Run: `mkdir -p docs/backtesting`

- [ ] **Step 2: Write `docs/backtesting/README.md`**

````markdown
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

The harness reuses the live bot's `.env` keys (`OPENROUTER_API_KEY`,
`ASKNEWS_*`, `EXA_API_KEY`). Optional backtest-only providers:
`SERPER_API_KEY` (pragmatic mode), `NYT_API_KEY`, `GUARDIAN_API_KEY`
(first-party archives). See `.env.template`.

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
python scripts/backtest.py build-corpus --source halawi --dataset-dir data/backtest/datasets/halawi
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
seed: 0
```

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
````

- [ ] **Step 3: Verify README commands against the implemented CLI**
Run: `python scripts/backtest.py --help && python scripts/backtest.py build-corpus --help && python scripts/backtest.py run --help`
Expected: exit 0. Reconcile every flag spelling used in README code blocks (`--source`, `--tournaments`, `--dataset-dir`, `--spec`, `--corpus`, `--out`, `--system`, subcommand names, `compare`'s two positional variant names) with the actual help output; edit the README blocks to match the implemented CLI where they differ.

- [ ] **Step 4: Write `docs/backtesting/COMPATIBILITY.md`**

````markdown
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
| `BotConfig` fields (incl. `as_of: datetime \| None`) | `bot/config.py:33-52` | variant config patches; as-of threading |
| `config_hash(cfg)` — pops `as_of` before hashing | `bot/config.py:55-57` | live hash invariance with `as_of=None` |
| `CalibratedEnsembleBot._invoke_model(member_and_prompt)` | `bot/ensemble_bot.py:83-89` | wrapped by `CachedLLM` |
| `_days_remaining(close_iso, now=None)` | `bot/ensemble_bot.py:56-61` | as-of date math |
| `today=` prompt injections (3 sites) | `bot/ensemble_bot.py:128,147,165` | prompt masking with T |
| `run_binary/mc/numeric_ensemble(...)` result dataclasses | `bot/ensemble_bot.py` | adapter converts to `Forecast` |
| `run_research(question, cfg)` + `globals()` provider dispatch | `bot/research.py:89-95` | historical provider installation |
| `_prompt_args(question, cfg)` | `bot/research.py:34-40` | as-of search prompts |
| `percentiles_to_cdf(...)`, `CDF_SIZE = 201` | `bot/distributions.py` | numeric forecast conversion |
| `append_forecast(record, path)` / `read_forecasts(path)` | `bot/logging_store.py:12,20` | snapshot hook; calibration flywheel |
| `calibration_dir`, `log_path` config fields | `bot/config.py:47-48` | as-of-T calibration snapshots; log reads |
| `max_questions_per_run` cost-brake pattern | `bot/ensemble_bot.py:294-303` | precedent the runner's cost brake extends |

## 5. Scoring interface

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

## 6. Cost accounting

Every `StageRecord` reports `tokens_in`, `tokens_out`, `cost_usd`,
`started_at`, `ended_at`. Systems that route all calls through `ctx.llm`
get these filled from the cache layer; systems with non-LLM paid calls
(e.g. metered search APIs) must fill them in their own stage records.
Reports aggregate cost/latency per stage; Pareto (score-vs-cost) analyses
assume these fields are honest.
````

- [ ] **Step 5: Verify the seam inventory against the implemented code**
Run: `grep -n "as_of" bot/config.py | head -5 && grep -n "_invoke_model" bot/ensemble_bot.py | head -3 && grep -n "_research\b" bot/research.py | head -5`
Expected: each grep returns hits. Update any `file:line` reference in the seam table that drifted from the post-Phase-3 reality (the threading work moved lines; the table must point at the real locations).

- [ ] **Step 6: Write `docs/backtesting/DATA.md`**

````markdown
# Retrocast Data Layer

Everything under `data/backtest/` is a **permanent local archive** —
gitignored, never auto-deleted. Run outputs land under the experiment
`--out` directory (CI uses `logs/backtest/`, also gitignored).

## Layout

```
data/backtest/
├── corpus.db        # SQLite metadata (schema below)
├── blobs/           # bulky series + raw snapshots, referenced by relative path
│   ├── <safe-qid>__<kind>.json          # one JSON file per (qid, kind) series;
│   │                                    #   kind = "cp" | "market_price";
│   │                                    #   <safe-qid> = qid with chars outside
│   │                                    #   [alnum -_.] replaced by "_"
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
  `blobs/<safe-qid>__<kind>.json` — the full, sorted point list.
  `value_at(qid, kind, T)` is a step-function lookup: last point with
  `ts <= T`, else `None`.

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

## Provenance and licensing per source

Every stored row carries `provenance` (source, fetch_time, license note).
Obligations that survive into derived work:

| Source | Access | License / terms | Provenance notes recorded |
|---|---|---|---|
| Metaculus API | bot token; tier-aware | Metaculus ToU — AI-eval use legitimized via the Data Needs Form (api-requests@metaculus.com) | tier at fetch; withheld fields; own-forecast questions fully visible |
| Halawi HF datasets (`YuehHanChen/forecasting[_raw]`) | HF download | dataset card license as published | dataset text stored **verbatim** (may drift from current Metaculus copies); import validation results |
| ForecastBench (`forecastingresearch/forecastbench-datasets`) | git clone | **CC BY-SA 4.0** — attribution required; *published* derivatives of the data must carry the same license | freeze values; cross-platform ids |
| Polymarket (Gamma + CLOB) | public API, no auth | public API terms; prices-history is empty below ~12h fidelity for resolved markets | fidelity used; fetch window |
| Kalshi | public REST | public API; ~3-month rolling live window, then `/historical/*` | settled timestamp; candlestick interval |
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
````

- [ ] **Step 7: Verify the documented DDL against the implemented store**
Run: `python -c "import sqlite3, tempfile, pathlib; from backtest.corpus.store import CorpusStore; d = tempfile.mkdtemp(); CorpusStore(pathlib.Path(d)); con = sqlite3.connect(pathlib.Path(d) / 'corpus.db'); [print(r[0]) for r in con.execute(\"select sql from sqlite_master where type='table'\")]"`
Expected: prints the real CREATE TABLE statements. Replace the DDL block in `docs/backtesting/DATA.md` with the printed statements (reformatted, comments kept) wherever table or column names differ — the store is authoritative.

- [ ] **Step 8: Verify all three docs exist and are non-trivial**
Run: `wc -l docs/backtesting/README.md docs/backtesting/COMPATIBILITY.md docs/backtesting/DATA.md && grep -c '^#' docs/backtesting/*.md`
Expected: every file present, each well over 50 lines, each with multiple headings.

- [ ] **Step 9: Commit**
```bash
git add docs/backtesting/README.md docs/backtesting/COMPATIBILITY.md docs/backtesting/DATA.md && git commit -m "feat(backtest): backtesting docs — README, COMPATIBILITY contract, DATA reference" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
### Task 36: Housekeeping — `.gitignore`, `.env.template`, packaging, full-suite green

No TDD cycle (config files), but every change is verified by a command, and the task ends with the entire suite green.

**Files:**
- Modify: `.gitignore` (currently 7 lines)
- Modify: `.env.template` (currently 26 lines)
- Modify: `pyproject.toml` (currently 21 lines)

- [ ] **Step 1: Update `.gitignore`**

Replace `.gitignore` in full (adds the two backtest data roots; `logs/` already covers `logs/backtest/`, but the explicit entry is requested and self-documenting):

```
.venv/
__pycache__/
*.pyc
.env
logs/
logs/backtest/
.pytest_cache/
*.egg-info/
data/backtest/
```

- [ ] **Step 2: Verify ignore rules**
Run: `git check-ignore -v data/backtest/corpus.db logs/backtest/ci-smoke/x.jsonl && git status --short | grep -c "data/backtest" || true`
Expected: `git check-ignore -v` prints a matching rule for BOTH paths; the grep prints `0` (nothing under `data/backtest/` shows up as untracked).

- [ ] **Step 3: Update `.env.template`**

Replace `.env.template` in full. Note the file's binding comment style (its own header says so, and a recent fix removed inline-comment traps): comments live on their OWN line; values come directly after `=` with nothing else on the line. `EXA_API_KEY` already exists in the live-research section and is NOT duplicated — the backtest section documents its reuse:

```
# NOTE: put values directly after the '=' with nothing else on the line —
# this dotenv version does NOT strip inline '# comments' from values.

# --- Required to submit forecasts ---
# Get it: metaculus.com -> Settings -> My Forecasting Bots -> Create a Bot
# (must be a BOT account token, not your personal account)
METACULUS_TOKEN=

# --- Required for the LLM ensemble (one key covers all model families) ---
# Get it: openrouter.ai -> Keys; or Metaculus's free proxy key
# (request via the FutureEval resources page)
OPENROUTER_API_KEY=

# --- Research (free tournament credits; bot degrades gracefully without) ---
# Either OAuth (CLIENT_ID + SECRET) or API_KEY works; OAuth wins if both set.
ASKNEWS_CLIENT_ID=
ASKNEWS_SECRET=
ASKNEWS_API_KEY=

# --- Optional extra research providers (Phase 2+) ---
# EXA_API_KEY also powers the backtest's strict-mode historical research
# (endCrawlDate=T) — one key serves both live and backtest.
PERPLEXITY_API_KEY=
EXA_API_KEY=

# --- Backtesting (Retrocast) historical research — all optional ---
# The harness degrades gracefully without any of these.
# Pragmatic-mode dated search; per-doc leakage-screened; backtests only.
# Get it: serper.dev -> Dashboard -> API Key
SERPER_API_KEY=
# First-party news archives: leak-proof date spines, narrow coverage.
# Reserved for post-MVP archive providers (nyt_archive, guardian) — nothing
# in the MVP harness consumes these two keys yet.
# Get it: developer.nytimes.com -> Get Started (Archive API)
NYT_API_KEY=
# Get it: open-platform.theguardian.com/access -> register developer key
GUARDIAN_API_KEY=

# --- Safety gate: bot only submits when this is exactly "true" ---
SUBMIT_PREDICTION=false
```

- [ ] **Step 4: Verify the template parses cleanly with no inline-comment traps**
Run: `python -c "
from dotenv import dotenv_values
vals = dotenv_values('.env.template')
assert set(['SERPER_API_KEY','NYT_API_KEY','GUARDIAN_API_KEY','EXA_API_KEY']) <= set(vals), sorted(vals)
assert all((v or '') == '' or '#' not in v for v in vals.values()), vals
print('env template ok:', len(vals), 'keys')
"`
Expected: prints `env template ok: 11 keys` (no value contains a stray `#`).

- [ ] **Step 5: Update `pyproject.toml`**

Replace `pyproject.toml` in full: adds `pyyaml` (used by `backtest/experiments/spec.py:from_yaml` since the experiments layer landed — make the implicit dependency official) and packages the `backtest` package + subpackages so `pip install -e .` in CI exposes them (the old explicit `packages = ["bot"]` would silently omit the sidecar):

```toml
[project]
name = "metaculus-ensemble-bot"
version = "0.1.0"
description = "Calibrated ensemble scaffold bot for Metaculus FutureEval tournaments"
requires-python = ">=3.11"
dependencies = [
    "forecasting-tools>=0.2.40",
    "numpy>=1.26",
    "scipy>=1.11",
    "python-dotenv>=1.0",
    "httpx>=0.27",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[tool.setuptools.packages.find]
include = ["bot*", "backtest*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

(If an earlier phase already added other third-party imports under `backtest/` — check with `grep -rhE "^(import|from) " backtest/ | grep -vE "backtest|bot\.|^from __future__" | sort -u` — add any non-stdlib, non-listed package to `dependencies` here as well.)

- [ ] **Step 6: Reinstall and sanity-check the packaging**
Run: `pip install -e ".[dev]" && python -c "import bot, backtest, backtest.corpus.store, backtest.experiments.runner, yaml; print('packaging ok')"`
Expected: install succeeds; prints `packaging ok`.

- [ ] **Step 7: Run the ENTIRE test suite**
Run: `pytest tests/ -v`
Expected: PASS — all green, zero failures, zero errors (live bot tests, scoring goldens, corpus, timetravel, systems, experiments, and all Phase-5 tests). This is the final gate for the whole plan.

- [ ] **Step 8: Final verification checklist**

- [ ] Entire suite green (Step 7) — zero failures, zero errors.
- [ ] Replace placeholder dates in `backtest/corpus/model_releases.py` `RELEASE_DATES` with verified model release dates BEFORE the first stratified report (they are intentionally conservative fail-safes).

- [ ] **Step 9: Commit**
```bash
git add .gitignore .env.template pyproject.toml && git commit -m "feat(backtest): housekeeping — ignore backtest data dirs, backtest env keys, package sidecar + pyyaml" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
