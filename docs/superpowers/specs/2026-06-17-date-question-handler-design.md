# Date-Question Handler — Design (2026-06-17)

Adds a full-ensemble `_run_forecast_on_date` to the live bot so date questions
no longer crash.

## Problem

The Gate-C dry run (2026-06-11, `logs/gate_c_dryrun.log:445`) crashed on the
date practice question (qid 43324) with:

```
NotImplementedError: Subclass must implement this method
```

`build_framework_bot` (`bot/ensemble_bot.py`) implements
`_run_forecast_on_binary`, `_run_forecast_on_multiple_choice`, and
`_run_forecast_on_numeric`, but not `_run_forecast_on_date`. The base
`forecasting_tools.ForecastBot._run_forecast_on_date`
(`forecast_bots/forecast_bot.py:531`) raises `NotImplementedError`, so any date
question kills that forecast. Date questions are "supported but not launched" in
the Summer-2026 season (`research-notes/05-metaculus-mechanics.md`), so this is
unlikely to fire in the live seasonal tournament — but it WILL fail any date
practice question and is an unguarded landmine if one ever launches.

## Goal / non-goals

**Goal:** date questions are forecast through the same ensemble → median →
PCHIP-distribution pipeline as numeric questions, returning a valid
`NumericDistribution` over unix timestamps.

**Non-goals:** conditional questions (`_run_forecast_on_conditional`; separate,
also unimplemented, but did NOT crash Gate C because the base class provides a
default). The OpenRouter weekly-credit cap (account action, not code).

## Contract

From `forecast_bots/forecast_bot.py:531` and the official Summer-2026 template
(`forecast_bots/official_bots/template_bot_2026_summer.py:416`):

```python
async def _run_forecast_on_date(
    self, question: DateQuestion, research: str
) -> ReasonedPrediction[NumericDistribution]:
    # Return a numeric distribution of TIMESTAMPS
```

`DateQuestion` (`data_models/questions.py:473`) fields used: `lower_bound:
datetime`, `upper_bound: datetime`, `open_lower_bound: bool`,
`open_upper_bound: bool`, `zero_point: float | None` (None for dates),
`cdf_size: int = 201`.

The canonical template approach: elicit dates from the LLM, parse to
`datetime`, convert each to `.timestamp()`, build a numeric distribution over
those timestamps. We mirror this but route through the repo's ensemble +
`_median_per_percentile` + `PchipNumericDistribution` instead of a single
model + `NumericDistribution.from_question`.

## Constraints (repo doctrine)

- Core engine (`CalibratedEnsembleBot`) stays framework-free and
  offline-unit-testable; `_invoke_model` is the monkeypatch seam.
- Parsers raise `ParseError` so a bad member is dropped, never submitted.
- `bot/` live path for existing question types stays bit-identical (the new
  prompt is a new constant; existing templates and rendered prompts are
  untouched, so the `llm_cache` keys for binary/MC/numeric do not move).

## Design

Four additions, each parallel to the numeric path.

### 1. `DATE_PROMPT` — `bot/prompts.py`

Clone of `NUMERIC_PROMPT` (`bot/prompts.py:72`) minus the `{units}` line, with
date-framed steps and the wide-tail reminder retained. Final block elicits the
repo's standard 7 percentiles (2.5/10/25/50/75/90/97.5) as dates, in
chronological order:

```
The last lines of your answer must be exactly (dates strictly increasing,
format YYYY-MM-DD, append THH:MM:SSZ in UTC only if the hour matters):
Percentile 2.5: YYYY-MM-DD
Percentile 10: YYYY-MM-DD
Percentile 25: YYYY-MM-DD
Percentile 50: YYYY-MM-DD
Percentile 75: YYYY-MM-DD
Percentile 90: YYYY-MM-DD
Percentile 97.5: YYYY-MM-DD
```

Placeholders: the shared `_HEADER` set plus `{lower_bound_message}` and
`{upper_bound_message}` — same as numeric minus `{units}`. We keep 2.5→97.5
(not the template's 10→90) for consistency with the numeric path and the repo's
fat-tail emphasis; `_median_per_percentile` and `percentiles_to_cdf` need ≥5
shared percentiles, which 7 satisfies.

### 2. `parse_date_percentiles(text) -> dict[float, float]` — `bot/parsing.py`

Same `Percentile P:` line structure as `parse_percentiles`, but the value is a
date string, returned as a **unix timestamp (float seconds)**:

```python
_DATE = r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?Z?)?)"

def parse_date_percentiles(text: str) -> dict:
    matches = re.findall(r"Percentile\s+(\d+(?:\.\d+)?)\s*:\s*" + _DATE,
                         text, re.IGNORECASE)
    out = {}
    for pct, datestr in matches:
        out[float(pct)] = _date_to_timestamp(datestr)  # later overwrite earlier
    if len(out) < 5:
        raise ParseError(f"only {len(out)} date percentiles found, need >= 5")
    return out
```

`_date_to_timestamp` parses with `datetime.fromisoformat` (normalizing a
trailing `Z` → `+00:00`) and, **if the result is naive, forces
`tzinfo=timezone.utc` before `.timestamp()`**. This is the one real correctness
trap: a naive `.timestamp()` uses the machine's local TZ, which is
non-deterministic across machines and would silently corrupt both forecasts and
`llm_cache` reproducibility. Unparseable dates raise `ParseError` (member
dropped).

The naive→UTC normalization is factored into a shared helper
`to_utc_timestamp(dt: datetime) -> float` (also in `bot/parsing.py`), so the
**question bounds and the elicited percentile dates use the identical TZ rule**
— see §4. `_date_to_timestamp` is `to_utc_timestamp(datetime.fromisoformat(...))`.

### 3. `run_date_ensemble(...)` — `CalibratedEnsembleBot`, `bot/ensemble_bot.py`

Clone of `run_numeric_ensemble` (`bot/ensemble_bot.py:192`):
- add `"date": DATE_PROMPT` to the `_template()` map (`bot/ensemble_bot.py:92`)
  and import `DATE_PROMPT`;
- render `DATE_PROMPT` with the standard header fields + bound messages (no
  `units`);
- fan out via `_fan_out`, parse with `parse_date_percentiles`;
- `median_pcts = _median_per_percentile(dicts)` (timestamps are floats — works
  unchanged);
- return the existing `NumericEnsembleResult` (percentile→value; values are
  timestamps). No new dataclass.

### 4. `_run_forecast_on_date` handler — `build_framework_bot`

Clone of `_run_forecast_on_numeric` (`bot/ensemble_bot.py:371`):
- import `DateQuestion` in the factory's `forecasting_tools` import;
- build bound messages from the datetimes (closed vs open):
  "The outcome can not be earlier/later than {lower/upper_bound}." vs
  "The value is likely not earlier/later than …";
- call `engine.run_date_ensemble(research=…, lower_bound_message=…,
  upper_bound_message=…, **_common_fields(question))`;
- `_log(question, "date", r.median_percentiles, r.member_percentile_dicts,
  r.member_models)`;
- construct `PchipNumericDistribution` with **timestamp bounds normalized
  through the same UTC rule as the percentiles**:
  `lower_bound=to_utc_timestamp(question.lower_bound)`,
  `upper_bound=to_utc_timestamp(question.upper_bound)`,
  `declared_percentiles=[Percentile(pct/100, value=ts) …]`,
  `zero_point=question.zero_point`, `cdf_size=question.cdf_size`;
- return `ReasonedPrediction(prediction_value=dist, reasoning=…)`.

`PchipNumericDistribution.get_cdf()` (`bot/ensemble_bot.py:241`) takes the PCHIP
path because `zero_point is None` and `cdf_size == 201`, building a
uniform-in-time linear grid between the two bound timestamps — correct for
dates.

## Data flow

```
DateQuestion
  → DATE_PROMPT (per member) → _fan_out → N texts
  → parse_date_percentiles (dates → UTC timestamps) per survivor
  → _median_per_percentile → {pct: median_timestamp}
  → PchipNumericDistribution(declared=timestamps, bounds=question bounds as ts)
  → get_cdf() PCHIP over [lower_ts, upper_ts] → 201-pt CDF
  → ReasonedPrediction[NumericDistribution]
```

## Error handling

- Per-member parse failure → `ParseError` → member dropped (`_parse_survivors`).
- Fewer than `min_members` survivors → `RuntimeError` skip (same fail-safe as
  every other type; surfaces as an error artifact, never a garbage forecast).
- Naive datetime (parsed percentile OR question bound) → forced UTC via
  `to_utc_timestamp` (determinism + bound/percentile TZ consistency).
- Log-scaled date question (`zero_point` set, unusual) → `PchipNumericDistribution`
  falls back to framework CDF construction — safe by existing design.

## Testing (TDD; tests written first)

New tests in `tests/test_parsing.py` and `tests/test_ensemble_bot.py`
(+ a handler test where forecasting_tools is available):

1. **`parse_date_percentiles`**
   - 7 `YYYY-MM-DD` lines → correct UTC timestamps.
   - `THH:MM:SSZ` form parsed correctly.
   - later duplicate percentile overwrites earlier.
   - `<5` percentiles → `ParseError`; unparseable date → `ParseError`.
   - **TZ-independence:** monkeypatch `os.environ["TZ"]` / assert the timestamp
     equals the UTC-computed value regardless of local zone (the determinism
     trap).
2. **`run_date_ensemble`** — monkeypatch `_invoke_model` to return canned date
   percentiles from N members; assert medianed timestamps and `member_models`
   kept; assert `<min_members` raises.
3. **Framework handler** — build a `DateQuestion` (bounds e.g. 2026-01-01 →
   2027-01-01), stub the engine, assert the returned `NumericDistribution`'s
   `get_cdf()` passes `validate_cdf` and is monotone with endpoints honoring
   open/closed bounds.

Full suite (`pytest`, ~495 tests) must stay green; the binary/MC/numeric
prompt-byte-identity golden tests must still pass (new constant, no edits to
existing templates).

## Files touched

- `bot/prompts.py` — add `DATE_PROMPT`.
- `bot/parsing.py` — add `parse_date_percentiles`, `_date_to_timestamp`,
  `to_utc_timestamp`.
- `bot/ensemble_bot.py` — add `run_date_ensemble`, `"date"` template entry,
  `_run_forecast_on_date` handler, `DateQuestion` import.
- `tests/test_parsing.py`, `tests/test_ensemble_bot.py` — new tests.
