# Calibrated Ensemble Scaffold Bot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reliable Metaculus tournament bot implementing System 1 from `FORECASTING-BOT-RESEARCH.md`: multi-source research → structured-scratchpad forecasts from a cross-model ensemble → robust aggregation → fitted calibration + capping → validated submission, with a logging/eval loop for self-improvement.

**Architecture:** Subclass `ForecastBot` from the `forecasting-tools` library (keeps Metaculus API handling, orchestration, retries — the "1 bug per season costs $500" layer) but own every differentiating layer ourselves: prompts, ensemble fan-out, aggregation (trimmed geometric-mean-of-odds), calibration (Platt fitted on our own resolved history, hard caps), and numeric CDF construction (PCHIP, fat tails). Everything tunable lives in one config object whose hash is logged with every forecast, so config A/B comparison and calibration fitting are possible later (self-improvement hooks).

**Tech Stack:** Python 3.12, `forecasting-tools` (Metaculus API + orchestration), `numpy`/`scipy` (aggregation, Platt, PCHIP), `pytest`, GitHub Actions (20-min cron). LLMs via OpenRouter (one key, all model families). Research via AskNews (free tournament credits) with search-LLM fallback.

**Design rules (from research, enforced throughout):**
1. Never miss a question for an avoidable reason (reliability > cleverness; −150 pts for 10 missed questions).
2. Never submit an unvalidated artifact (CDF standardizer + parsers with retry; one units bug = −80 pts).
3. Don't hardcode bias corrections without data — ship the *mechanism* (Platt) at identity, plus the fitting script. Hard caps [2%, 98%] always apply (strongest within-winner correlate, r=+0.48).
4. `SUBMIT_PREDICTION` defaults to **false**. Promotion path: unit tests → live-LLM smoke → dry-run → bot-testing-area → MiniBench → Summer 2026 tournament (the stopgap gates below).

**Stopgap gates:**
- **Gate A** (after Task 8): `pytest` fully green, no API keys needed.
- **Gate B** (Task 9): one cheap live call per ensemble model — verifies keys + model slugs (`scripts/check_models.py`).
- **Gate C** (Task 13): full pipeline dry-run against the bot-testing-area tournament; human reviews the reasoning + numbers before anything is submitted.
- **Gate D**: submit to bot-testing-area, then MiniBench for ≥1 two-week cycle.
- **Gate E**: enable the Summer 2026 seasonal tournament.

**Runtime stopgaps (built into code):** research provider fallback chain (AskNews → search-LLM → empty-research flag, never crash); parse → one reprompt retry → drop that ensemble member; minimum 3 surviving members per question else skip + loud log (skipping scores 0; a garbage forecast scores negative); CDF chain (PCHIP → forecasting-tools linear fallback → skip); per-question exception isolation (framework-provided); `max_questions_per_run` cost brake.

**File structure:**
```
forecasting/
├── pyproject.toml
├── .env.template            # every key documented; .env gitignored
├── .gitignore
├── bot/
│   ├── __init__.py
│   ├── config.py            # BotConfig dataclass, ensemble roster, config_hash()
│   ├── aggregation.py       # pools: trimmed GMO, GMO, median, mean; MC; CDF; extremize (off)
│   ├── calibration.py       # cap, Platt apply/fit/load/save
│   ├── timeutils.py         # days_until, Laplace, constant-hazard burn-down
│   ├── distributions.py     # percentiles → PCHIP → standardized 201-pt CDF
│   ├── prompts.py           # research + binary/MC/numeric scratchpad prompts
│   ├── parsing.py           # robust extraction of probability/options/percentiles
│   ├── research.py          # two-track research orchestrator with fallback chain
│   ├── ensemble_bot.py      # CalibratedEnsembleBot(ForecastBot)
│   └── logging_store.py     # JSONL forecast log incl. config hash + members
├── scripts/
│   ├── check_models.py      # Gate B: live smoke per model slug
│   ├── run_testing_area.py  # Gate C/D
│   ├── run_tournament.py    # Gate E (seasonal + MiniBench)
│   ├── fit_calibration.py   # self-improvement: fit Platt from resolved logs
│   └── benchmark.py         # config A/B vs community prediction
├── tests/                   # one test file per bot module
├── calibration/             # fitted params land here (gitignored? no — commit them)
├── logs/                    # forecasts.jsonl (gitignored)
└── .github/workflows/run_bot.yaml
```

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.template`, `bot/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: git init and base files**

```bash
cd /Users/aaronbao/Developer/sideprojects/forecasting
git init -b main
```

`pyproject.toml`:
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
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[tool.setuptools]
packages = ["bot"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
logs/
.pytest_cache/
*.egg-info/
```

`.env.template`:
```bash
# --- Required to submit forecasts ---
METACULUS_TOKEN=            # metaculus.com -> Settings -> My Forecasting Bots -> Create a Bot

# --- Required for the LLM ensemble (one key covers all model families) ---
OPENROUTER_API_KEY=         # openrouter.ai; or Metaculus's free proxy key (request via FutureEval resources page)

# --- Research (free tournament credits; bot degrades gracefully without them) ---
ASKNEWS_CLIENT_ID=
ASKNEWS_SECRET=

# --- Optional extra research providers (Phase 2+) ---
PERPLEXITY_API_KEY=
EXA_API_KEY=

# --- Safety gate: bot only submits when this is exactly "true" ---
SUBMIT_PREDICTION=false
```

`bot/__init__.py` and `tests/__init__.py`: empty files.

- [ ] **Step 2: create venv and install**

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```
Expected: installs cleanly (forecasting-tools pulls litellm, pydantic, etc.).

- [ ] **Step 3: smoke-run pytest**

Run: `.venv/bin/pytest -q`
Expected: `no tests ran` (exit 5) — fine.

- [ ] **Step 4: commit**

```bash
git add -A && git commit -m "chore: project scaffold (pyproject, env template, gitignore)"
```

---

### Task 2: Aggregation library

**Files:**
- Create: `bot/aggregation.py`
- Test: `tests/test_aggregation.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aggregation.py
import numpy as np
import pytest

from bot.aggregation import (
    aggregate_binary, aggregate_cdfs, aggregate_multiple_choice,
    extremize, geometric_mean_of_odds, median_pool, trimmed_gmo,
)


def test_gmo_known_value():
    # odds 1/9 and 2/3 -> mean log-odds -1.3013 -> p ~= 0.2139
    assert geometric_mean_of_odds([0.1, 0.4]) == pytest.approx(0.2139, abs=1e-3)


def test_gmo_single_forecast_is_identity():
    assert geometric_mean_of_odds([0.37]) == pytest.approx(0.37, abs=1e-4)


def test_trimmed_gmo_drops_extremes():
    probs = [0.05, 0.3, 0.35, 0.4, 0.95]
    assert trimmed_gmo(probs) == pytest.approx(
        geometric_mean_of_odds([0.3, 0.35, 0.4]))


def test_trimmed_gmo_small_n_falls_back_to_plain_gmo():
    probs = [0.2, 0.4, 0.6]
    assert trimmed_gmo(probs) == pytest.approx(geometric_mean_of_odds(probs))


def test_median_pool():
    assert median_pool([0.1, 0.5, 0.9]) == 0.5


def test_extremize_identity_at_a1():
    assert extremize(0.7, 1.0) == pytest.approx(0.7)


def test_extremize_pushes_away_from_half():
    assert extremize(0.7, 2.5) > 0.7
    assert extremize(0.3, 2.5) < 0.3


def test_aggregate_binary_rejects_unknown_method_and_empty():
    with pytest.raises(ValueError):
        aggregate_binary([0.5], method="nope")
    with pytest.raises(ValueError):
        aggregate_binary([], method="trimmed_gmo")


def test_aggregate_mc_floors_and_renormalizes():
    runs = [{"A": 0.998, "B": 0.001, "C": 0.001},
            {"A": 0.998, "B": 0.001, "C": 0.001}]
    agg = aggregate_multiple_choice(runs, floor=0.01)
    assert agg["B"] >= 0.009          # floored before renormalization
    assert sum(agg.values()) == pytest.approx(1.0)


def test_aggregate_cdfs_pointwise_median_stays_monotone():
    c1 = np.linspace(0, 1, 201)
    c2 = np.linspace(0, 1, 201) ** 2
    out = aggregate_cdfs([c1, c2])
    assert np.all(np.diff(out) >= 0)
    assert out[0] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_aggregation.py -q`
Expected: FAIL — `ModuleNotFoundError: bot.aggregation`

- [ ] **Step 3: Implement**

```python
# bot/aggregation.py
"""Probability pooling for ensemble forecasts.

Default: trimmed geometric mean of odds (Samotsvety's method) — drop the most
extreme forecast on each side, then average in log-odds space. GMO beat
mean/median on ~850 resolved Metaculus questions (Sevilla 2021); trimming
fixes diversity-induced underconfidence (Jose et al. 2014).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def to_log_odds(probs) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def from_log_odds(lo) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(lo, dtype=float)))


def geometric_mean_of_odds(probs) -> float:
    return float(from_log_odds(np.mean(to_log_odds(probs))))


def median_pool(probs) -> float:
    return float(np.median(np.asarray(probs, dtype=float)))


def mean_pool(probs) -> float:
    return float(np.mean(np.asarray(probs, dtype=float)))


def trimmed_gmo(probs, n_trim: int = 1) -> float:
    """Drop n_trim lowest and highest forecasts, then GMO; plain GMO if too few."""
    p = np.sort(np.asarray(probs, dtype=float))
    if len(p) > 2 * n_trim + 1:
        p = p[n_trim:-n_trim]
    return geometric_mean_of_odds(p)


def extremize(p: float, a: float) -> float:
    """Karmarkar transform t(p) = p^a / (p^a + (1-p)^a). a=1 is identity.

    OFF by default: fitted extremization overfits in-sample and hurt on recent
    Metaculus data. Available for held-out experiments only.
    """
    p = float(np.clip(p, EPS, 1 - EPS))
    num = p ** a
    return float(num / (num + (1 - p) ** a))


def aggregate_binary(probs, method: str = "trimmed_gmo") -> float:
    methods = {
        "trimmed_gmo": trimmed_gmo,
        "gmo": geometric_mean_of_odds,
        "median": median_pool,
        "mean": mean_pool,
    }
    if method not in methods:
        raise ValueError(f"unknown aggregation method: {method}")
    probs = list(probs)
    if not probs:
        raise ValueError("no forecasts to aggregate")
    return methods[method](probs)


def aggregate_multiple_choice(option_prob_dicts, floor: float = 0.01) -> dict:
    """Per-option arithmetic mean, then floor every option and renormalize.

    'Good forecasters leave some moderate probability on most options.'
    """
    if not option_prob_dicts:
        raise ValueError("no forecasts to aggregate")
    options = list(option_prob_dicts[0].keys())
    means = {o: float(np.mean([d[o] for d in option_prob_dicts])) for o in options}
    floored = {o: max(v, floor) for o, v in means.items()}
    total = sum(floored.values())
    return {o: v / total for o, v in floored.items()}


def aggregate_cdfs(cdfs) -> np.ndarray:
    """Pointwise median across member CDFs (preserves monotonicity and the
    per-step bounds each member already satisfies)."""
    arr = np.stack([np.asarray(c, dtype=float) for c in cdfs])
    return np.median(arr, axis=0)
```

- [ ] **Step 4: Run tests** — `.venv/bin/pytest tests/test_aggregation.py -q` → all PASS.

- [ ] **Step 5: Commit** — `git add bot/aggregation.py tests/test_aggregation.py && git commit -m "feat: ensemble aggregation library (trimmed GMO default)"`

---

### Task 3: Calibration layer

**Files:**
- Create: `bot/calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibration.py
import numpy as np
import pytest

from bot.calibration import (
    MIN_FIT_N, PlattParams, apply_platt, calibrate_binary, cap,
    fit_platt, load_platt, save_platt,
)


def test_cap_clamps_both_ends():
    assert cap(0.001) == 0.02
    assert cap(0.999) == 0.98
    assert cap(0.5) == 0.5


def test_platt_identity_is_noop():
    p = PlattParams()
    assert apply_platt(0.37, p) == pytest.approx(0.37, abs=1e-6)


def test_platt_negative_intercept_lowers_probability():
    p = PlattParams(a=1.0, b=-0.539)  # the '30% -> 20%' yes-bias map
    assert apply_platt(0.30, p) == pytest.approx(0.20, abs=0.01)


def test_fit_platt_recovers_known_bias():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, size=4000)
    logit = np.log(true_p / (1 - true_p))
    reported = 1 / (1 + np.exp(-(logit + 0.6)))    # bot systematically too-yes
    outcomes = rng.binomial(1, true_p)
    fitted = fit_platt(reported, outcomes)
    assert fitted.a == pytest.approx(1.0, abs=0.15)
    assert fitted.b == pytest.approx(-0.6, abs=0.15)  # map undoes the bias


def test_fit_platt_refuses_small_samples():
    with pytest.raises(ValueError):
        fit_platt([0.5] * (MIN_FIT_N - 1), [1] * (MIN_FIT_N - 1))


def test_calibrate_binary_identity_params_just_caps():
    assert calibrate_binary(0.995, None) == 0.98
    assert calibrate_binary(0.5, PlattParams()) == 0.5


def test_save_and_load_roundtrip(tmp_path):
    p = PlattParams(a=1.2, b=-0.3, n_fit=150)
    path = tmp_path / "binary.json"
    save_platt(p, path)
    loaded = load_platt(path)
    assert loaded == p


def test_load_missing_file_returns_identity(tmp_path):
    assert load_platt(tmp_path / "nope.json").is_identity()
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# bot/calibration.py
"""Post-aggregation calibration: fitted Platt map, then hard caps.

Don't hardcode bias corrections without data: the Platt layer
sigmoid(a*logit(p) + b) subsumes yes-bias correction (intercept b) and
hedging/overconfidence correction (slope a). It ships at identity and is
fitted by scripts/fit_calibration.py once >= MIN_FIT_N of OUR OWN forecasts
have resolved (+0.016 Brier on Metaculus bot data). Hard caps are tail-risk
insurance under log scoring and ALWAYS apply last (winner correlate r=+0.48).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

MIN_FIT_N = 100


@dataclass
class PlattParams:
    a: float = 1.0
    b: float = 0.0
    n_fit: int = 0

    def is_identity(self) -> bool:
        return self.a == 1.0 and self.b == 0.0


def cap(p: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return float(min(max(p, lo), hi))


def apply_platt(p: float, params: PlattParams) -> float:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    z = params.a * (np.log(p) - np.log1p(-p)) + params.b
    return float(1.0 / (1.0 + np.exp(-z)))


def fit_platt(predictions, outcomes) -> PlattParams:
    """Minimize log loss of sigmoid(a*logit(p)+b) against resolved outcomes."""
    preds = np.clip(np.asarray(predictions, dtype=float), 1e-4, 1 - 1e-4)
    y = np.asarray(outcomes, dtype=float)
    if len(preds) < MIN_FIT_N:
        raise ValueError(
            f"need >= {MIN_FIT_N} resolved forecasts to fit, got {len(preds)}")
    lo = np.log(preds) - np.log1p(-preds)

    def nll(params):
        z = params[0] * lo + params[1]
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return PlattParams(a=float(res.x[0]), b=float(res.x[1]), n_fit=len(preds))


def calibrate_binary(p: float, params: PlattParams | None,
                     lo: float = 0.02, hi: float = 0.98) -> float:
    if params is not None and not params.is_identity():
        p = apply_platt(p, params)
    return cap(p, lo, hi)


def load_platt(path) -> PlattParams:
    """Fitted params if present; identity otherwise (stopgap default)."""
    path = Path(path)
    if not path.exists():
        return PlattParams()
    d = json.loads(path.read_text())
    return PlattParams(a=d["a"], b=d["b"], n_fit=d.get("n_fit", 0))


def save_platt(params: PlattParams, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"a": params.a, "b": params.b, "n_fit": params.n_fit}))
```

- [ ] **Step 4: Run tests** → PASS. (The fit test is statistical; tolerance ±0.15 with n=4000 and a fixed seed is stable.)

- [ ] **Step 5: Commit** — `git commit -m "feat: calibration layer (Platt fit/apply, hard caps)"`

---

### Task 4: Time/hazard utilities

**Files:**
- Create: `bot/timeutils.py`
- Test: `tests/test_timeutils.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_timeutils.py
from datetime import datetime, timezone

import pytest

from bot.timeutils import days_until, laplace_rate, remaining_window_probability


def test_days_until():
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    close = datetime(2026, 6, 11, tzinfo=timezone.utc)
    assert days_until(close, now) == pytest.approx(10.0)


def test_laplace_rule_of_succession():
    assert laplace_rate(0, 0) == pytest.approx(0.5)
    assert laplace_rate(1, 1) == pytest.approx(2 / 3)
    assert laplace_rate(0, 10) == pytest.approx(1 / 12)
    with pytest.raises(ValueError):
        laplace_rate(5, 3)


def test_constant_hazard_burndown():
    # P=0.5 over 100 days; 50 days left -> 1 - 0.5**0.5 ~= 0.2929
    assert remaining_window_probability(0.5, 100, 50) == pytest.approx(0.2929, abs=1e-3)
    assert remaining_window_probability(0.5, 100, 100) == pytest.approx(0.5)
    assert remaining_window_probability(0.5, 100, 0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        remaining_window_probability(0.5, 0, 0)
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

```python
# bot/timeutils.py
"""Time/hazard math for deadline ('will X happen by DATE') questions."""
from __future__ import annotations

from datetime import datetime, timezone


def days_until(close_time: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    return (close_time - now).total_seconds() / 86400.0


def laplace_rate(k: int, n: int) -> float:
    """Rule of succession: P(event in next period) = (k+1)/(n+2)."""
    if k < 0 or n < 0 or k > n:
        raise ValueError("need 0 <= k <= n")
    return (k + 1) / (n + 2)


def remaining_window_probability(p_full_window: float, total_days: float,
                                 remaining_days: float) -> float:
    """Constant-hazard burn-down. Solves (1-r)^T = 1-P for the per-day hazard,
    then returns P over the remaining window: 1-(1-P)^(t/T). Absent news, a
    deadline question's probability must decay — sitting still implicitly
    raises the hazard estimate daily."""
    if total_days <= 0:
        raise ValueError("total_days must be positive")
    p = min(max(p_full_window, 0.0), 1.0 - 1e-9)
    t = min(max(remaining_days, 0.0), total_days)
    return 1.0 - (1.0 - p) ** (t / total_days)
```

- [ ] **Step 4: Run tests** → PASS.  - [ ] **Step 5: Commit** — `git commit -m "feat: hazard/Laplace time utilities"`

---

### Task 5: Numeric distribution machinery

**Files:**
- Create: `bot/distributions.py`
- Test: `tests/test_distributions.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_distributions.py
import numpy as np
import pytest

from bot.distributions import (
    CDF_SIZE, MAX_STEP, MIN_STEP, percentiles_to_cdf, repair_percentiles,
    standardize_cdf, validate_cdf,
)

DECLARED = {2.5: 10, 10: 20, 25: 30, 50: 40, 75: 50, 90: 60, 97.5: 70}


def test_happy_path_closed_bounds():
    cdf = percentiles_to_cdf(DECLARED, 0, 100, open_lower=False, open_upper=False)
    assert len(cdf) == CDF_SIZE
    assert cdf[0] == pytest.approx(0.0)
    assert cdf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cdf) >= MIN_STEP - 1e-12)
    # median knot: value 40 sits at grid index 80 of linspace(0,100,201)
    assert cdf[80] == pytest.approx(0.5, abs=0.02)


def test_open_bounds_keep_mass_outside():
    cdf = percentiles_to_cdf(DECLARED, 0, 100, open_lower=True, open_upper=True)
    assert cdf[0] >= 0.001 - 1e-9
    assert cdf[-1] <= 0.999 + 1e-9


def test_non_monotone_input_is_repaired_not_fatal():
    bad = {10: 30, 50: 20, 90: 60}   # 50th percentile below 10th
    cdf = percentiles_to_cdf(bad, 0, 100, open_lower=False, open_upper=False)
    validate_cdf(cdf, open_lower=False, open_upper=False)  # should not raise


def test_spike_distribution_respects_max_step():
    spike = {10: 49.9, 50: 50.0, 90: 50.1}
    cdf = percentiles_to_cdf(spike, 0, 100, open_lower=False, open_upper=False)
    assert np.diff(cdf).max() <= MAX_STEP + 1e-9


def test_repair_percentiles_sorts_and_strictly_increases():
    out = repair_percentiles({50: 20, 10: 20, 90: 19}, 0, 100)
    vals = list(out.values())
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_validate_rejects_garbage():
    with pytest.raises(ValueError):
        validate_cdf(np.zeros(CDF_SIZE), open_lower=False, open_upper=False)
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

```python
# bot/distributions.py
"""LLM percentiles -> validated, PCHIP-smoothed, Metaculus-compliant CDF.

Bots' worst deficit vs Pros is numeric questions (-23.2 head-to-head); much of
it is plumbing. This module: (1) repairs non-monotone elicited percentiles
(~30% of naive runs fail platform validation), (2) interpolates with PCHIP
(shape-preserving; the official template's linear interpolation wastes
information), (3) enforces the Metaculus API constraint set: 201 points,
monotone, step in [5e-05, 0.2], closed bounds pinned to 0/1, open bounds
keep >= 0.001 mass outside.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

CDF_SIZE = 201
MIN_STEP = 5.0e-05
MAX_STEP = 0.2
MIN_OUTSIDE = 0.001


def repair_percentiles(declared: dict, range_min: float, range_max: float) -> dict:
    """Sort by percentile; clamp values inside the range (with margin); force
    strictly increasing values by minimal jitter. Raises if unrepairable."""
    if not declared:
        raise ValueError("no percentiles declared")
    span = range_max - range_min
    if span <= 0:
        raise ValueError("range_max must exceed range_min")
    eps = max(span * 1e-6, 1e-12)
    lo, hi = range_min + eps, range_max - eps
    items = sorted(declared.items())
    values = [min(max(float(v), lo), hi) for _, v in items]
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] + eps
    values[-1] = min(values[-1], hi)
    for i in range(len(values) - 2, -1, -1):
        if values[i] >= values[i + 1]:
            values[i] = values[i + 1] - eps
    if values[0] < lo - 1e-9:
        raise ValueError("percentiles cannot fit inside question range")
    return {pct: v for (pct, _), v in zip(items, values)}


def percentiles_to_cdf(declared: dict, range_min: float, range_max: float,
                       open_lower: bool, open_upper: bool) -> np.ndarray:
    d = repair_percentiles(declared, range_min, range_max)
    first = MIN_OUTSIDE if open_lower else 0.0
    last = (1.0 - MIN_OUTSIDE) if open_upper else 1.0
    xs = [range_min] + list(d.values()) + [range_max]
    ys = [first] + [min(max(p / 100.0, first + 1e-4), last - 1e-4)
                    for p in d.keys()] + [last]
    for i in range(1, len(ys)):                 # strictness after clamping
        if ys[i] <= ys[i - 1]:
            ys[i] = ys[i - 1] + 1e-6
    interp = PchipInterpolator(xs, ys)
    grid = np.linspace(range_min, range_max, CDF_SIZE)
    return standardize_cdf(interp(grid), open_lower, open_upper)


def standardize_cdf(cdf, open_lower: bool, open_upper: bool) -> np.ndarray:
    c = np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0)
    c = np.maximum.accumulate(c)
    first = MIN_OUTSIDE if open_lower else 0.0
    last = (1.0 - MIN_OUTSIDE) if open_upper else 1.0
    interior = (c - c[0]) / max(c[-1] - c[0], 1e-12)   # rescale between pins
    c = first + interior * (last - first)
    d = np.diff(c)
    total = last - first
    for _ in range(10):                                # step bounds + mass conservation
        d = np.clip(d, MIN_STEP, MAX_STEP)
        excess = d.sum() - total
        if abs(excess) < 1e-12:
            break
        if excess > 0:
            room = d - MIN_STEP
            d = d - room * min(excess / max(room.sum(), 1e-12), 1.0)
        else:
            room = MAX_STEP - d
            d = d + room * min(-excess / max(room.sum(), 1e-12), 1.0)
    c = np.concatenate([[first], first + np.cumsum(d)])
    c[-1] = last
    validate_cdf(c, open_lower, open_upper)
    return c


def validate_cdf(c, open_lower: bool, open_upper: bool) -> None:
    c = np.asarray(c, dtype=float)
    if len(c) != CDF_SIZE:
        raise ValueError(f"cdf must have {CDF_SIZE} points, got {len(c)}")
    d = np.diff(c)
    if d.min() < MIN_STEP - 1e-9:
        raise ValueError(f"cdf step below minimum: {d.min()}")
    if d.max() > MAX_STEP + 1e-9:
        raise ValueError(f"cdf step above maximum: {d.max()}")
    lo_ok = c[0] >= MIN_OUTSIDE - 1e-9 if open_lower else abs(c[0]) < 1e-9
    hi_ok = c[-1] <= 1 - MIN_OUTSIDE + 1e-9 if open_upper else abs(c[-1] - 1) < 1e-9
    if not (lo_ok and hi_ok):
        raise ValueError("cdf endpoint violates bound rules")
```

- [ ] **Step 4: Run tests** → PASS.  - [ ] **Step 5: Commit** — `git commit -m "feat: PCHIP numeric CDF construction + Metaculus constraint standardizer"`

---

### Task 6: Prompts

**Files:**
- Create: `bot/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
from bot.prompts import (
    BINARY_PROMPT, MC_PROMPT, NUMERIC_PROMPT,
    RESEARCH_PROMPT_CURRENT, RESEARCH_PROMPT_HISTORICAL,
)

COMMON = dict(
    question_text="Will X happen?", background_info="Some background.",
    resolution_criteria="Resolves YES if X.", fine_print="None.",
    research="Report.", today="2026-06-11", close_date="2026-07-01",
    days_remaining=20.0,
)


def test_binary_prompt_renders_and_has_contract_line():
    text = BINARY_PROMPT.format(**COMMON)
    assert "Probability: ZZ%" in text
    assert "reference class" in text.lower()
    assert "premortem" in text.lower()
    assert "status quo" in text.lower()


def test_mc_prompt_renders():
    text = MC_PROMPT.format(**COMMON, options="A, B, C")
    assert "moderate probability on most options" in text


def test_numeric_prompt_renders_with_all_percentiles():
    text = NUMERIC_PROMPT.format(
        **COMMON, units="widgets", lower_bound_message="cannot be below 0",
        upper_bound_message="cannot be above 100")
    for p in ("2.5", "10", "25", "50", "75", "90", "97.5"):
        assert f"Percentile {p}:" in text


def test_research_prompts_render():
    args = dict(question_text="Q", resolution_criteria="R", fine_print="F",
                today="2026-06-11")
    assert "historical" in RESEARCH_PROMPT_HISTORICAL.format(**args).lower()
    assert "news" in RESEARCH_PROMPT_CURRENT.format(**args).lower()
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement** — full prompt text (the core IP; synthesized from the template's measured elements + Panshul42's checklist + GJP science; see research doc §7.2):

```python
# bot/prompts.py
"""Forecasting prompts. Every element here has measured support — see
FORECASTING-BOT-RESEARCH.md section 7.2. Deliberately excluded: 'think
Bayesianly' instructions (measured harm, arXiv 2506.01578)."""

_HEADER = """You are a professional forecaster with a strong track record, known both for
calibration and for being decisively right when the evidence warrants it.

QUESTION
{question_text}

BACKGROUND
{background_info}

RESOLUTION CRITERIA (these decide the outcome — forecast these criteria
literally, not the question's title or vibe)
{resolution_criteria}

FINE PRINT
{fine_print}

RESEARCH REPORT (from your assistant; weigh source quality yourself)
{research}

Today's date: {today}
Question closes: {close_date} ({days_remaining:.0f} days remaining)
"""

_CORE_STEPS = """Work through these steps, writing each out briefly:

1. RESTATE: the resolution criteria in your own words — exact thresholds,
   dates, and any fine-print trap that could cause a surprising resolution.
2. STATUS CHECK: based only on the research report, could this already be
   effectively decided? Do not assume an event happened without evidence.
3. OUTSIDE VIEW FIRST: name 2-3 reference classes with rough base rates
   (counts where possible: "happened X of the last N times"). State the base
   rate you anchor on. Outside view first, usually.
4. STATUS QUO: what happens if nothing changes before resolution? Weight it —
   the world changes more slowly than headlines suggest.
5. INSIDE VIEW: strongest case for each outcome. Grade each piece of evidence
   STRONG (multiple independent reliable sources / direct causal mechanism /
   structural factors), MODERATE, or WEAK (anecdote, speculation, single
   unnamed source). Weigh accordingly.
6. TIME & SCOPE: how much of the plausible window remains; does remaining
   time still allow the event? Check: would your answer change appropriately
   if the deadline were doubled or halved? It should.
7. PREMORTEM: it is resolution day and you were badly wrong — what is the
   single most likely reason? Adjust if that scenario deserves more weight.
"""

BINARY_PROMPT = (_HEADER + _CORE_STEPS + """
8. FINAL CALIBRATION: state an initial probability, then check it against the
   evidence grades and the base rate. Sanity line: "out of 100 similar
   situations, this happens about X times." Use 1% granularity — do not round
   to the nearest 5 or 10.

The very last line of your answer must be exactly:
Probability: ZZ%
""")

MC_PROMPT = (_HEADER + _CORE_STEPS + """
8. FINAL CALIBRATION: distribute probability across ALL options. Good
   forecasters leave moderate probability on most options — unexpected
   outcomes happen. Options sum to 100%. Use 1% granularity.

The options are: {options}

The last lines of your answer must be exactly one line per option, in the
order given above:
OptionName: XX%
""")

NUMERIC_PROMPT = (_HEADER + """Units for your answer: {units}
{lower_bound_message}
{upper_bound_message}

Work through these steps, writing each out briefly:

1. RESTATE: what exactly is measured, in what units, by when, per the
   resolution criteria. Never use scientific notation.
2. STATUS QUO: the value if nothing changes.
3. TREND: the value if the current trend continues to resolution.
4. EXPERTS & MARKETS: what relevant expert estimates or market prices imply.
5. LOW SCENARIO: a plausible unexpected path to a LOW value.
6. HIGH SCENARIO: a plausible unexpected path to a HIGH value.
7. TAIL CHECK: good forecasters set WIDE intervals — the single most common
   error on numeric questions is tails that are too thin. Your 2.5th and
   97.5th percentiles should feel uncomfortably far apart.

The last lines of your answer must be exactly (values strictly increasing,
plain numbers, no commas or scientific notation):
Percentile 2.5: XX
Percentile 10: XX
Percentile 25: XX
Percentile 50: XX
Percentile 75: XX
Percentile 90: XX
Percentile 97.5: XX
""")

RESEARCH_PROMPT_HISTORICAL = """You are a research assistant to a forecaster. For the question below, produce
a concise OUTSIDE-VIEW report: relevant historical precedents and how often
similar events happened (rough base rates with counts), structural factors,
and how comparable situations resolved. Do not forecast.

Question: {question_text}
Resolution criteria: {resolution_criteria}
{fine_print}
Today: {today}"""

RESEARCH_PROMPT_CURRENT = """You are a research assistant to a forecaster. For the question below, produce
a concise INSIDE-VIEW report of the latest relevant news and current state:
most recent developments (with dates), key actors and their stated
intentions, scheduled events before the close date, and whether the
resolution criteria would be met if things stand as they are today. Flag
opinion vs fact; prefer identifiable expert sources. Do not forecast.

Question: {question_text}
Resolution criteria: {resolution_criteria}
{fine_print}
Today: {today}"""
```

- [ ] **Step 4: Run tests** → PASS.  - [ ] **Step 5: Commit** — `git commit -m "feat: scratchpad prompts (binary/MC/numeric) and two-track research prompts"`

---

### Task 7: Parsing

**Files:**
- Create: `bot/parsing.py`
- Test: `tests/test_parsing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parsing.py
import pytest

from bot.parsing import ParseError, parse_binary, parse_multiple_choice, parse_percentiles


def test_parse_binary_takes_last_match_and_scales():
    text = "Initial estimate... Probability: 60%\nAfter premortem...\nProbability: 57%"
    assert parse_binary(text) == pytest.approx(0.57)


def test_parse_binary_handles_decimals_and_clamps():
    assert parse_binary("Probability: 99.95%") == pytest.approx(0.999)
    assert parse_binary("Probability: 0%") == pytest.approx(0.001)


def test_parse_binary_raises_when_absent():
    with pytest.raises(ParseError):
        parse_binary("I cannot answer this.")


def test_parse_mc_extracts_all_options_case_insensitive():
    text = "reasoning...\noption a: 20%\nOption B: 30%\nOPTION C: 50%"
    out = parse_multiple_choice(text, ["Option A", "Option B", "Option C"])
    assert out == pytest.approx({"Option A": 0.2, "Option B": 0.3, "Option C": 0.5})


def test_parse_mc_renormalizes_sloppy_sums():
    text = "Option A: 50%\nOption B: 60%"
    out = parse_multiple_choice(text, ["Option A", "Option B"])
    assert sum(out.values()) == pytest.approx(1.0)


def test_parse_mc_raises_on_missing_option():
    with pytest.raises(ParseError):
        parse_multiple_choice("Option A: 100%", ["Option A", "Option B"])


def test_parse_percentiles_handles_commas_and_negatives():
    text = """Percentile 2.5: -1,200\nPercentile 10: -100\nPercentile 25: 0
Percentile 50: 1,250.5\nPercentile 75: 2000\nPercentile 90: 3,000
Percentile 97.5: 10,000"""
    out = parse_percentiles(text)
    assert out[2.5] == -1200
    assert out[50] == 1250.5
    assert len(out) == 7


def test_parse_percentiles_raises_when_too_few():
    with pytest.raises(ParseError):
        parse_percentiles("Percentile 50: 10")
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

```python
# bot/parsing.py
"""Robust extraction of forecasts from LLM text. One units/parse bug cost a
Fall-2025 bot ~80 points; parsers raise ParseError so callers can reprompt
once, then drop the ensemble member (never submit garbage)."""
from __future__ import annotations

import re


class ParseError(ValueError):
    pass


_NUM = r"(-?[\d,]+(?:\.\d+)?)"


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def parse_binary(text: str) -> float:
    """Last 'Probability: ZZ%' in the text, as a probability in [0.001, 0.999]."""
    matches = re.findall(r"Probability:\s*" + _NUM + r"\s*%", text, re.IGNORECASE)
    if not matches:
        raise ParseError("no 'Probability: ZZ%' line found")
    p = _to_float(matches[-1]) / 100.0
    return min(max(p, 0.001), 0.999)


def parse_multiple_choice(text: str, options: list[str]) -> dict:
    """Last 'OptionName: XX%' per option (case-insensitive); renormalized."""
    out = {}
    for opt in options:
        pattern = re.escape(opt) + r"\s*:\s*" + _NUM + r"\s*%"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            raise ParseError(f"option not found in output: {opt!r}")
        out[opt] = max(_to_float(matches[-1]) / 100.0, 0.0)
    total = sum(out.values())
    if total <= 0:
        raise ParseError("all options parsed as zero")
    return {k: v / total for k, v in out.items()}


def parse_percentiles(text: str) -> dict:
    """All 'Percentile P: value' lines; requires >= 5 distinct percentiles."""
    matches = re.findall(
        r"Percentile\s+(\d+(?:\.\d+)?)\s*:\s*" + _NUM, text, re.IGNORECASE)
    out = {}
    for pct, val in matches:
        out[float(pct)] = _to_float(val)   # later occurrences overwrite earlier
    if len(out) < 5:
        raise ParseError(f"only {len(out)} percentiles found, need >= 5")
    return out
```

- [ ] **Step 4: Run tests** → PASS.  - [ ] **Step 5: Commit** — `git commit -m "feat: forecast output parsers with ParseError contract"`

---

### Task 8: Config

**Files:**
- Create: `bot/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
from bot.config import DEFAULT_CONFIG, BotConfig, EnsembleMember, config_hash


def test_default_config_has_diverse_ensemble():
    families = {m.model.split("/")[0] for m in DEFAULT_CONFIG.ensemble}
    assert len(families) >= 2          # decorrelation requires >= 2 model families
    assert sum(m.runs for m in DEFAULT_CONFIG.ensemble) >= 5


def test_config_hash_is_stable_and_sensitive():
    h1 = config_hash(DEFAULT_CONFIG)
    h2 = config_hash(DEFAULT_CONFIG)
    assert h1 == h2 and len(h1) == 12
    other = BotConfig(ensemble=[EnsembleMember("openai/gpt-5.4", 1, 0.5)])
    assert config_hash(other) != h1


def test_submit_defaults_false(monkeypatch):
    monkeypatch.delenv("SUBMIT_PREDICTION", raising=False)
    assert BotConfig().submit_predictions is False
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

```python
# bot/config.py
"""Single source of truth for every tunable. The config hash is logged with
each forecast so calibration fitting and A/B comparison can condition on the
exact configuration that produced a forecast (self-improvement hook)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EnsembleMember:
    model: str            # OpenRouter slug, e.g. "anthropic/claude-opus-4-6"
    runs: int = 2
    temperature: float = 0.5


def _default_ensemble() -> list[EnsembleMember]:
    # 3 families x 2 runs = 6 members. Verify slugs with scripts/check_models.py
    # (Gate B) before first live run; update freely — everything downstream is
    # config-driven.
    return [
        EnsembleMember("openai/gpt-5.4", runs=2, temperature=0.5),
        EnsembleMember("anthropic/claude-opus-4-6", runs=2, temperature=0.5),
        EnsembleMember("google/gemini-3.1-pro", runs=2, temperature=0.5),
    ]


@dataclass
class BotConfig:
    ensemble: list[EnsembleMember] = field(default_factory=_default_ensemble)
    aggregation_method: str = "trimmed_gmo"
    min_members: int = 3              # skip question if fewer members survive
    cap_lo: float = 0.02
    cap_hi: float = 0.98
    mc_floor: float = 0.01
    research_providers: tuple = ("asknews", "search_llm")
    search_model: str = "perplexity/sonar-pro"     # via OpenRouter
    parser_model: str = "openai/gpt-5.4-mini"      # cheap reprompt/repair calls
    max_questions_per_run: int = 50                # cost brake
    calibration_dir: str = "calibration"
    log_path: str = "logs/forecasts.jsonl"

    @property
    def submit_predictions(self) -> bool:
        return os.getenv("SUBMIT_PREDICTION", "false").strip().lower() == "true"


def config_hash(cfg: BotConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


DEFAULT_CONFIG = BotConfig()
```

- [ ] **Step 4: Run tests** → PASS.  - [ ] **Step 5: Commit** — `git commit -m "feat: central bot config with hash for forecast provenance"`

**=== GATE A: `.venv/bin/pytest -q` fully green; zero API keys required so far. ===**

---

### Task 9: Model preflight script (Gate B)

**Files:**
- Create: `scripts/check_models.py`

- [ ] **Step 1: Implement** (no unit test — this IS a live test; costs ~$0.02 total)

```python
# scripts/check_models.py
"""Gate B: verify every configured model slug answers a trivial prompt.
Run after setting OPENROUTER_API_KEY. Catches wrong slugs / dead models /
quota issues BEFORE a tournament question does."""
import asyncio

from dotenv import load_dotenv

from bot.config import DEFAULT_CONFIG

load_dotenv()


async def main() -> None:
    from forecasting_tools import GeneralLlm
    models = [m.model for m in DEFAULT_CONFIG.ensemble]
    models += [DEFAULT_CONFIG.search_model, DEFAULT_CONFIG.parser_model]
    ok = True
    for slug in models:
        try:
            llm = GeneralLlm(model=f"openrouter/{slug}", temperature=0)
            out = await llm.invoke("Reply with the single word: ready")
            print(f"  OK   {slug}: {out.strip()[:40]!r}")
        except Exception as e:
            ok = False
            print(f"  FAIL {slug}: {type(e).__name__}: {e}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run** (requires `OPENROUTER_API_KEY` in `.env`): `.venv/bin/python scripts/check_models.py`
Expected: `OK` for all five slugs; fix slugs in `bot/config.py` until green.

- [ ] **Step 3: Commit** — `git commit -m "feat: Gate B model preflight script"`

---

### Task 10: Research orchestrator

**Files:**
- Create: `bot/research.py`
- Test: `tests/test_research.py`

- [ ] **Step 1: Write the failing tests** (providers mocked; no keys needed)

```python
# tests/test_research.py
import pytest

import bot.research as research_mod
from bot.config import BotConfig
from bot.research import ResearchResult, run_research

QUESTION = dict(question_text="Q?", resolution_criteria="R", fine_print="F")


async def test_uses_first_working_provider(monkeypatch):
    async def fake_asknews(q, cfg):
        return "asknews report"
    monkeypatch.setattr(research_mod, "_asknews_research", fake_asknews)
    out = await run_research(QUESTION, BotConfig())
    assert isinstance(out, ResearchResult)
    assert out.provider == "asknews"
    assert "asknews report" in out.text


async def test_falls_back_when_provider_raises(monkeypatch):
    async def boom(q, cfg):
        raise RuntimeError("asknews down")
    async def fake_search(q, cfg):
        return "search-llm report"
    monkeypatch.setattr(research_mod, "_asknews_research", boom)
    monkeypatch.setattr(research_mod, "_search_llm_research", fake_search)
    out = await run_research(QUESTION, BotConfig())
    assert out.provider == "search_llm"


async def test_returns_empty_flagged_result_when_all_fail(monkeypatch):
    async def boom(q, cfg):
        raise RuntimeError("down")
    monkeypatch.setattr(research_mod, "_asknews_research", boom)
    monkeypatch.setattr(research_mod, "_search_llm_research", boom)
    out = await run_research(QUESTION, BotConfig())
    assert out.provider == "none"
    assert out.text == ""
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

```python
# bot/research.py
"""Two-track research (outside-view/historical + inside-view/current) with a
provider fallback chain. A research failure must NEVER cost us the question:
worst case we forecast on model knowledge with an empty-research flag.
Distinct-source count was the #1 Fall-2025 winner predictor."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date

from bot.config import BotConfig
from bot.prompts import RESEARCH_PROMPT_CURRENT, RESEARCH_PROMPT_HISTORICAL

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    text: str
    provider: str          # "asknews" | "search_llm" | "none"


def _prompt_args(question: dict) -> dict:
    return dict(
        question_text=question["question_text"],
        resolution_criteria=question.get("resolution_criteria", ""),
        fine_print=question.get("fine_print", ""),
        today=date.today().isoformat(),
    )


async def _asknews_research(question: dict, cfg: BotConfig) -> str:
    if not (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")):
        raise RuntimeError("AskNews credentials not configured")
    from forecasting_tools import AskNewsSearcher
    searcher = AskNewsSearcher()
    return await searcher.get_formatted_news_async(question["question_text"])


async def _search_llm_research(question: dict, cfg: BotConfig) -> str:
    from forecasting_tools import GeneralLlm
    llm = GeneralLlm(model=f"openrouter/{cfg.search_model}", temperature=0.1)
    args = _prompt_args(question)
    current = await llm.invoke(RESEARCH_PROMPT_CURRENT.format(**args))
    historical = await llm.invoke(RESEARCH_PROMPT_HISTORICAL.format(**args))
    return (f"## Outside view (historical)\n{historical}\n\n"
            f"## Inside view (current)\n{current}")


_PROVIDERS = {
    "asknews": _asknews_research,
    "search_llm": _search_llm_research,
}


async def run_research(question: dict, cfg: BotConfig) -> ResearchResult:
    for name in cfg.research_providers:
        fn = {"asknews": _asknews_research,
              "search_llm": _search_llm_research}[name]
        # late-bind through module attributes so tests can monkeypatch:
        fn = globals()[f"_{name}_research"]
        try:
            text = await fn(question, cfg)
            if text and text.strip():
                return ResearchResult(text=text, provider=name)
        except Exception as e:
            logger.warning("research provider %s failed: %s", name, e)
    logger.error("ALL research providers failed for question: %s",
                 question.get("question_text", "")[:80])
    return ResearchResult(text="", provider="none")
```

- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Verify the AskNews call name against the installed library** (it changes between versions):
Run: `.venv/bin/python -c "from forecasting_tools import AskNewsSearcher; print([m for m in dir(AskNewsSearcher) if 'news' in m.lower() or 'call' in m.lower()])"`
Adjust `_asknews_research` to the available method (e.g., `call_preconfigured_version("asknews/news-summaries", ...)`) if needed; re-run tests.
- [ ] **Step 6: Commit** — `git commit -m "feat: two-track research orchestrator with fallback chain"`

---

### Task 11: Forecast logging store

**Files:**
- Create: `bot/logging_store.py`
- Test: `tests/test_logging_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_logging_store.py
import json

from bot.logging_store import append_forecast, read_forecasts


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "f.jsonl"
    rec = {"question_id": 1, "type": "binary", "final": 0.42,
           "members": [0.4, 0.45], "config_hash": "abc123def456"}
    append_forecast(rec, path)
    append_forecast({**rec, "question_id": 2}, path)
    rows = read_forecasts(path)
    assert len(rows) == 2
    assert rows[0]["question_id"] == 1
    assert "logged_at" in rows[0]


def test_read_missing_file_returns_empty(tmp_path):
    assert read_forecasts(tmp_path / "nope.jsonl") == []
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

```python
# bot/logging_store.py
"""Append-only JSONL forecast log. This is the substrate for ALL
self-improvement: calibration fitting, config A/B comparison, and error
post-mortems condition on what was forecast, by which members, under which
config hash."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def append_forecast(record: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "logged_at": datetime.now(timezone.utc).isoformat()}
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_forecasts(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
```

- [ ] **Step 4: Run tests** → PASS.  - [ ] **Step 5: Commit** — `git commit -m "feat: JSONL forecast provenance log"`

---

### Task 12: The ensemble bot

**Files:**
- Create: `bot/ensemble_bot.py`
- Test: `tests/test_ensemble_bot.py`

- [ ] **Step 1: Read the installed framework's exact signatures** (they drift between versions; lock to what's installed):

```bash
.venv/bin/python - <<'EOF'
import inspect
from forecasting_tools import ForecastBot, BinaryQuestion, NumericDistribution, ReasonedPrediction
print(inspect.signature(ForecastBot.__init__))
print([m for m in dir(ForecastBot) if m.startswith("_run_forecast")])
print(inspect.signature(NumericDistribution.__init__) if hasattr(NumericDistribution, "__init__") else "pydantic")
print([f for f in getattr(NumericDistribution, "model_fields", {})])
EOF
```
Record the output; adapt field names in Step 3 accordingly.

- [ ] **Step 2: Write the failing test** (LLM fully mocked)

```python
# tests/test_ensemble_bot.py
import pytest

from bot.config import BotConfig, EnsembleMember
from bot.ensemble_bot import CalibratedEnsembleBot

CANNED = "reasoning...\nstatus quo...\npremortem...\nProbability: 64%"


@pytest.fixture
def bot(monkeypatch):
    cfg = BotConfig(ensemble=[EnsembleMember("fake/model-a", runs=3),
                              EnsembleMember("fake/model-b", runs=3)])
    b = CalibratedEnsembleBot(bot_config=cfg, publish_reports_to_metaculus=False)

    async def fake_invoke(self, prompt):
        return CANNED
    # patch the LLM call at the bot's seam, not inside litellm:
    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", fake_invoke,
                        raising=True)
    return b


async def test_binary_pipeline_aggregates_and_caps(bot):
    result = await bot.run_binary_ensemble(
        question_text="Q?", background_info="", resolution_criteria="R",
        fine_print="", research="news", close_iso="2026-07-01T00:00:00Z")
    # six identical members at 0.64 -> trimmed GMO = 0.64, inside caps
    assert result.final == pytest.approx(0.64, abs=1e-6)
    assert len(result.member_probs) == 6


async def test_skips_when_too_few_members_survive(bot, monkeypatch):
    async def garbage(self, prompt):
        return "I refuse."
    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", garbage)
    with pytest.raises(RuntimeError, match="only 0 ensemble members"):
        await bot.run_binary_ensemble(
            question_text="Q?", background_info="", resolution_criteria="R",
            fine_print="", research="", close_iso="2026-07-01T00:00:00Z")
```

- [ ] **Step 3: Implement** — the bot separates a pure, testable ensemble core (`run_binary_ensemble`, etc.) from thin framework adapters (`_run_forecast_on_binary` and friends):

```python
# bot/ensemble_bot.py
"""CalibratedEnsembleBot: research -> K runs across model families ->
trimmed-GMO aggregation -> Platt + caps -> validated submission.

Framework integration: subclasses forecasting_tools.ForecastBot with
research_reports_per_question=1 and predictions_per_research_report=1; the
ensemble fan-out is OURS (inside run_*_ensemble), not the framework's, so
aggregation and calibration stay under our control and unit tests.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from bot.aggregation import (aggregate_binary, aggregate_cdfs,
                             aggregate_multiple_choice)
from bot.calibration import calibrate_binary, load_platt
from bot.config import BotConfig, DEFAULT_CONFIG, config_hash
from bot.distributions import percentiles_to_cdf
from bot.logging_store import append_forecast
from bot.parsing import (ParseError, parse_binary, parse_multiple_choice,
                         parse_percentiles)
from bot.prompts import BINARY_PROMPT, MC_PROMPT, NUMERIC_PROMPT
from bot.research import run_research

logger = logging.getLogger(__name__)


@dataclass
class BinaryEnsembleResult:
    final: float
    aggregate_raw: float
    member_probs: list = field(default_factory=list)
    member_models: list = field(default_factory=list)


def _days_remaining(close_iso: str) -> float:
    try:
        close = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
        from bot.timeutils import days_until
        return max(days_until(close), 0.0)
    except Exception:
        return 0.0


class CalibratedEnsembleBot:
    """Core engine. (Framework adapter subclass added in Step 5 below keeps
    this class import-light and unit-testable without forecasting-tools.)"""

    def __init__(self, bot_config: BotConfig | None = None, **kwargs):
        self.cfg = bot_config or DEFAULT_CONFIG
        self.cfg_hash = config_hash(self.cfg)
        self._platt_binary = load_platt(
            Path(self.cfg.calibration_dir) / "binary.json")

    async def _invoke_model(self, member_slug_prompt) -> str:
        """Single seam for all LLM calls (monkeypatch target in tests)."""
        member, prompt = member_slug_prompt
        from forecasting_tools import GeneralLlm
        llm = GeneralLlm(model=f"openrouter/{member.model}",
                         temperature=member.temperature, timeout=600)
        return await llm.invoke(prompt)

    async def _fan_out(self, prompt: str):
        """Run every ensemble member; return (texts, models) of survivors."""
        jobs, models = [], []
        for member in self.cfg.ensemble:
            for _ in range(member.runs):
                jobs.append(self._invoke_model((member, prompt)))
                models.append(member.model)
        results = await asyncio.gather(*jobs, return_exceptions=True)
        texts, kept_models = [], []
        for r, m in zip(results, models):
            if isinstance(r, Exception):
                logger.warning("member %s failed: %s", m, r)
            else:
                texts.append(r)
                kept_models.append(m)
        return texts, kept_models

    async def run_binary_ensemble(self, *, question_text, background_info,
                                  resolution_criteria, fine_print, research,
                                  close_iso) -> BinaryEnsembleResult:
        prompt = BINARY_PROMPT.format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research, today=date.today().isoformat(),
            close_date=close_iso[:10], days_remaining=_days_remaining(close_iso))
        texts, models = await self._fan_out(prompt)
        probs, kept = [], []
        for t, m in zip(texts, models):
            try:
                probs.append(parse_binary(t))
                kept.append(m)
            except ParseError as e:
                logger.warning("parse failure from %s: %s", m, e)
        if len(probs) < self.cfg.min_members:
            raise RuntimeError(
                f"only {len(probs)} ensemble members survived "
                f"(min {self.cfg.min_members}); skipping question")
        raw = aggregate_binary(probs, self.cfg.aggregation_method)
        final = calibrate_binary(raw, self._platt_binary,
                                 self.cfg.cap_lo, self.cfg.cap_hi)
        return BinaryEnsembleResult(final=final, aggregate_raw=raw,
                                    member_probs=probs, member_models=kept)
```

(MC and numeric ensemble methods follow the identical fan-out → parse → aggregate pattern: `parse_multiple_choice` → `aggregate_multiple_choice` → floor+renormalize; `parse_percentiles` → per-percentile median across members → `percentiles_to_cdf` with the question's bounds. Include them in the same file with the same structure as `run_binary_ensemble` — copy the body, swap the parse/aggregate calls. Numeric fallback chain: if `percentiles_to_cdf` raises, retry with framework's `NumericDistribution` linear construction from the median percentiles; if that fails, raise RuntimeError to skip.)

- [ ] **Step 4: Run tests** → PASS (`pytest tests/test_ensemble_bot.py -q`).

- [ ] **Step 5: Add the framework adapter subclass** in the same file (exact field names confirmed in Step 1):

```python
def build_framework_bot(cfg: BotConfig | None = None):
    """Factory returning a forecasting_tools.ForecastBot subclass instance
    wrapping CalibratedEnsembleBot. Defined inside a factory so importing
    bot.ensemble_bot never requires forecasting-tools at test time."""
    from forecasting_tools import (BinaryQuestion, ForecastBot,
                                   MultipleChoiceQuestion, NumericQuestion,
                                   ReasonedPrediction)

    engine = CalibratedEnsembleBot(bot_config=cfg)

    class FrameworkBot(ForecastBot):
        async def run_research(self, question) -> str:
            res = await run_research(
                dict(question_text=question.question_text,
                     resolution_criteria=question.resolution_criteria or "",
                     fine_print=question.fine_print or ""), engine.cfg)
            return res.text

        async def _run_forecast_on_binary(self, question: BinaryQuestion,
                                          research: str) -> ReasonedPrediction:
            r = await engine.run_binary_ensemble(
                question_text=question.question_text,
                background_info=question.background_info or "",
                resolution_criteria=question.resolution_criteria or "",
                fine_print=question.fine_print or "",
                research=research,
                close_iso=str(question.scheduled_resolution_time or ""))
            append_forecast({"question_id": question.id_of_post,
                             "type": "binary", "final": r.final,
                             "raw": r.aggregate_raw,
                             "members": r.member_probs,
                             "models": r.member_models,
                             "config_hash": engine.cfg_hash},
                            engine.cfg.log_path)
            return ReasonedPrediction(
                prediction_value=r.final,
                reasoning=f"ensemble={r.member_probs} -> raw={r.aggregate_raw:.3f} "
                          f"-> calibrated+capped={r.final:.3f}")

        # _run_forecast_on_multiple_choice / _run_forecast_on_numeric: same
        # adapter pattern over the MC/numeric ensemble methods, constructing
        # PredictedOptionList / NumericDistribution per the field names
        # recorded in Step 1.

    return FrameworkBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=(cfg or DEFAULT_CONFIG).submit_predictions,
        skip_previously_forecasted_questions=True,
    )
```

- [ ] **Step 6: Run full test suite** → PASS.  - [ ] **Step 7: Commit** — `git commit -m "feat: calibrated ensemble bot core + framework adapter"`

---

### Task 13: Run scripts (Gates C–E)

**Files:**
- Create: `scripts/run_testing_area.py`, `scripts/run_tournament.py`

- [ ] **Step 1: Implement testing-area runner**

```python
# scripts/run_testing_area.py
"""Gate C/D: run the full pipeline against the Metaculus bot-testing-area
tournament. Dry-run by default; prints every would-be submission for human
review. Set SUBMIT_PREDICTION=true in .env to actually publish there."""
import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from bot.config import DEFAULT_CONFIG
from bot.ensemble_bot import build_framework_bot

TESTING_AREA = "bot-testing-area"      # slug; ID 32977


async def main():
    bot = build_framework_bot(DEFAULT_CONFIG)
    reports = await bot.forecast_on_tournament(
        TESTING_AREA, return_exceptions=True)
    for r in reports:
        if isinstance(r, Exception):
            print(f"FAILED: {r}")
        else:
            print(getattr(r, "summary", None) or r)
    submitted = DEFAULT_CONFIG.submit_predictions
    print(f"\nsubmit_predictions={submitted} "
          f"({'published' if submitted else 'DRY RUN — nothing submitted'})")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Implement tournament runner** — identical structure over the live tournaments:

```python
# scripts/run_tournament.py
"""Gate E: forecast the live seasonal FutureEval tournament + MiniBench.
Refuses to publish unless SUBMIT_PREDICTION=true (defense in depth: the
flag also gates publish_reports_to_metaculus inside build_framework_bot)."""
import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from bot.config import DEFAULT_CONFIG
from bot.ensemble_bot import build_framework_bot


async def main():
    from forecasting_tools import MetaculusApi
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


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Gate C — dry run** (needs METACULUS_TOKEN + OPENROUTER_API_KEY):
`.venv/bin/python scripts/run_testing_area.py`
Expected: one report per practice question type; human reviews reasoning, member spread, final numbers, CDF shape. Fix anything alarming before proceeding.
- [ ] **Step 4: Gate D — set `SUBMIT_PREDICTION=true`, rerun, verify forecasts appear on the bot-testing-area question pages.** Then revert to `false`.
- [ ] **Step 5: Commit** — `git commit -m "feat: testing-area and tournament runners with submit gates"`

---

### Task 14: GitHub Actions automation

**Files:**
- Create: `.github/workflows/run_bot.yaml`

- [ ] **Step 1: Implement**

```yaml
name: run-bot
on:
  schedule:
    - cron: "7,27,47 * * * *"     # every 20 min, offset from the hour
  workflow_dispatch: {}

concurrency:
  group: run-bot
  cancel-in-progress: false

jobs:
  forecast:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e .
      - run: python scripts/run_tournament.py
        env:
          METACULUS_TOKEN: ${{ secrets.METACULUS_TOKEN }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          ASKNEWS_CLIENT_ID: ${{ secrets.ASKNEWS_CLIENT_ID }}
          ASKNEWS_SECRET: ${{ secrets.ASKNEWS_SECRET }}
          SUBMIT_PREDICTION: ${{ vars.SUBMIT_PREDICTION }}
```

- [ ] **Step 2: Commit** — `git commit -m "feat: 20-minute tournament cron workflow"`
(Enabling requires a GitHub repo + secrets — see "what I need from you." A local `launchd`/cron alternative works identically: run `scripts/run_tournament.py` every 20 min.)

---

### Task 15: Self-improvement loop scripts

**Files:**
- Create: `scripts/fit_calibration.py`, `scripts/benchmark.py`
- Test: `tests/test_fit_calibration.py` (pure-function part)

- [ ] **Step 1: Write the failing test for the resolution-joining logic**

```python
# tests/test_fit_calibration.py
from scripts.fit_calibration import join_resolutions


def test_join_resolutions_matches_on_question_id():
    logs = [{"question_id": 1, "type": "binary", "final": 0.8},
            {"question_id": 2, "type": "binary", "final": 0.3},
            {"question_id": 3, "type": "numeric", "final": None}]
    resolutions = {1: 1, 2: 0}            # question_id -> outcome (1=yes)
    preds, outcomes = join_resolutions(logs, resolutions)
    assert preds == [0.8, 0.3]
    assert outcomes == [1, 0]
```

- [ ] **Step 2: Verify failure** (needs `scripts/__init__.py` — create it empty).

- [ ] **Step 3: Implement**

```python
# scripts/fit_calibration.py
"""Self-improvement loop: fit the binary Platt map from this bot's own
resolved forecasts. Refuses below MIN_FIT_N (identity stays in effect —
that's the stopgap, not an error). Rerun monthly / after each MiniBench."""
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from bot.calibration import MIN_FIT_N, fit_platt, save_platt
from bot.config import DEFAULT_CONFIG
from bot.logging_store import read_forecasts


def join_resolutions(logs: list, resolutions: dict):
    preds, outcomes = [], []
    for row in logs:
        if row.get("type") != "binary":
            continue
        res = resolutions.get(row.get("question_id"))
        if res is not None and row.get("final") is not None:
            preds.append(float(row["final"]))
            outcomes.append(int(res))
    return preds, outcomes


async def fetch_resolutions(question_ids: list) -> dict:
    """question_id -> 1/0 for resolved binary questions, via Metaculus API."""
    from forecasting_tools import MetaculusApi
    out = {}
    for qid in question_ids:
        try:
            q = MetaculusApi.get_question_by_post_id(qid)
            res = getattr(q, "resolution_string", None) or getattr(q, "resolution", None)
            if res is not None and str(res).lower() in ("yes", "no"):
                out[qid] = 1 if str(res).lower() == "yes" else 0
        except Exception as e:
            print(f"  skip {qid}: {e}")
    return out


async def main():
    logs = read_forecasts(DEFAULT_CONFIG.log_path)
    ids = sorted({r["question_id"] for r in logs if r.get("type") == "binary"})
    print(f"{len(logs)} logged forecasts, {len(ids)} unique binary questions")
    resolutions = await fetch_resolutions(ids)
    preds, outcomes = join_resolutions(logs, resolutions)
    print(f"{len(preds)} resolved binary forecasts available")
    if len(preds) < MIN_FIT_N:
        print(f"insufficient data (< {MIN_FIT_N}); identity calibration "
              f"remains in effect — this is the designed stopgap")
        sys.exit(0)
    params = fit_platt(preds, outcomes)
    save_platt(params, f"{DEFAULT_CONFIG.calibration_dir}/binary.json")
    print(f"fitted: a={params.a:.3f} b={params.b:.3f} n={params.n_fit} -> saved")


if __name__ == "__main__":
    asyncio.run(main())
```

```python
# scripts/benchmark.py
"""Config A/B vs the community prediction on open main-site questions, using
forecasting-tools' Benchmarker. Regression test, not the objective — top
bots win by disagreeing with the CP *well*. Needs ~300+ questions to separate
similar configs (with 100 there's ~30% chance the worse bot wins)."""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from bot.config import DEFAULT_CONFIG
from bot.ensemble_bot import build_framework_bot


async def main():
    from forecasting_tools import Benchmarker
    bot = build_framework_bot(DEFAULT_CONFIG)
    benchmarker = Benchmarker(
        forecast_bots=[bot],
        number_of_questions_to_use=100,    # raise to 300+ for real decisions
        file_path_to_save_reports="logs/benchmarks/",
    )
    results = await benchmarker.run_benchmark()
    for b in results:
        print(b.name, "avg expected baseline score:",
              b.average_expected_baseline_score)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run the unit test** → PASS. Live scripts run at Gate D+.
- [ ] **Step 5: Commit** — `git commit -m "feat: calibration fitting + CP benchmark self-improvement loop"`

---

## Post-MVP roadmap (already-designed extension points, not in this plan's tasks)

1. **Supervisor agent** (System 2 element): trigger when `max(member_probs) − min(member_probs) > 0.35`; one frontier-model call that reads all member reasonings + runs follow-up research. Seam: between `_fan_out` and `aggregate_binary`.
2. **Market-odds enrichment**: add a third research provider injecting Polymarket/Manifold/Metaculus-CP odds into the research report (adjacentwire placed #11 on this change alone).
3. **Per-question-type Platt maps + MC/numeric calibration** once log volume allows.
4. **Resolution-source verifier**: fetch the URL named in resolution criteria; ask "how would this resolve today?"; inject into research.
5. **Triage tiers**: cheap classifier call routing near-certain questions to a 2-member fast path.
6. **Config A/B at scale** via `benchmark.py` with 300+ questions per arm.

## Self-review (done at planning time)

- Spec coverage: research pipeline ✔ (Task 10), ensemble ✔ (12), aggregation ✔ (2), calibration + caps ✔ (3), numeric CDF ✔ (5), prompts ✔ (6), parsing ✔ (7), config/provenance ✔ (8, 11), submission + gates ✔ (13), automation ✔ (14), self-improvement ✔ (15), stopgaps ✔ (gates + runtime chains).
- Known intentional deferrals: MC/numeric ensemble methods are specified by pattern in Task 12 (same structure as binary, different parse/aggregate calls — all of which are fully implemented and tested in Tasks 2, 5, 7); framework field names are pinned by the live-inspection steps (12.1, 10.5) because forecasting-tools' API drifts between releases. These are verification steps, not placeholders.
- Type consistency: `BotConfig`/`EnsembleMember` fields match across Tasks 8–15; parser names match between Tasks 7 and 12; `PlattParams` round-trips between Tasks 3 and 15.
