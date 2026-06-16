# Date-Question Handler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full-ensemble `_run_forecast_on_date` so date questions forecast through the same pipeline as numeric questions instead of crashing with `NotImplementedError`.

**Architecture:** Elicit date percentiles from the ensemble → parse each to a UTC unix timestamp → per-percentile median across members → build the existing `PchipNumericDistribution` over timestamps (bounds also converted to timestamps via one shared UTC rule) → return `ReasonedPrediction[NumericDistribution]`. Mirrors the numeric path exactly.

**Tech Stack:** Python 3.12, `forecasting_tools` (in `.venv`), `pytest` (`asyncio_mode = "auto"`), `numpy`/`scipy` (existing distribution machinery).

**Spec:** `docs/superpowers/specs/2026-06-17-date-question-handler-design.md`

**Run tests with:** `.venv/bin/python -m pytest`

---

## File Structure

- `bot/parsing.py` — add `to_utc_timestamp`, `_date_to_timestamp`, `parse_date_percentiles`. (Date → timestamp lives next to the other parsers.)
- `bot/prompts.py` — add `DATE_PROMPT` constant.
- `bot/ensemble_bot.py` — add `run_date_ensemble` engine method, `"date"` entry in `_template`, the `_run_forecast_on_date` framework handler, and imports.
- `tests/test_parsing.py` — date-parser tests.
- `tests/test_ensemble_bot.py` — `run_date_ensemble` engine test.
- `tests/test_framework_adapter.py` — `_run_forecast_on_date` handler test.

---

## Task 1: Date parsing (`bot/parsing.py`)

**Files:**
- Modify: `bot/parsing.py` (append after `parse_percentiles`, currently ending at line 53)
- Test: `tests/test_parsing.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parsing.py`. Update the import on line 3 to include the new function:

```python
# line 3 becomes:
from bot.parsing import (ParseError, parse_binary, parse_date_percentiles,
                         parse_multiple_choice, parse_percentiles)
```

Add at the end of the file:

```python
from datetime import datetime, timezone


def test_parse_date_percentiles_returns_utc_timestamps():
    text = """reasoning...
Percentile 2.5: 2026-02-01
Percentile 10: 2026-03-01
Percentile 25: 2026-04-01
Percentile 50: 2026-06-01
Percentile 75: 2026-08-01
Percentile 90: 2026-10-01
Percentile 97.5: 2026-12-01"""
    out = parse_date_percentiles(text)
    assert len(out) == 7
    assert out[50.0] == datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()
    assert out[2.5] < out[50.0] < out[97.5]


def test_parse_date_percentiles_handles_iso_hour_with_z():
    text = ("Percentile 10: 2026-01-01T06:00:00Z\n"
            "Percentile 25: 2026-01-02\nPercentile 50: 2026-01-03\n"
            "Percentile 75: 2026-01-04\nPercentile 90: 2026-01-05")
    out = parse_date_percentiles(text)
    assert out[10.0] == datetime(2026, 1, 1, 6, 0, 0,
                                 tzinfo=timezone.utc).timestamp()


def test_parse_date_percentiles_later_line_overwrites():
    text = ("Percentile 50: 2026-01-01\nPercentile 10: 2026-01-02\n"
            "Percentile 25: 2026-01-03\nPercentile 75: 2026-01-04\n"
            "Percentile 90: 2026-01-05\nPercentile 50: 2026-07-07")
    out = parse_date_percentiles(text)
    assert out[50.0] == datetime(2026, 7, 7, tzinfo=timezone.utc).timestamp()


def test_parse_date_percentiles_raises_when_too_few():
    with pytest.raises(ParseError):
        parse_date_percentiles("Percentile 50: 2026-01-01")


def test_parse_date_percentiles_raises_on_unparseable_date():
    text = ("Percentile 2.5: 2026-13-99\nPercentile 10: 2026-03-01\n"
            "Percentile 25: 2026-04-01\nPercentile 50: 2026-06-01\n"
            "Percentile 75: 2026-08-01\nPercentile 90: 2026-10-01")
    with pytest.raises(ParseError):
        parse_date_percentiles(text)


def test_parse_date_percentiles_is_timezone_independent(monkeypatch):
    import time
    text = "\n".join(f"Percentile {p}: 2026-01-01"
                     for p in (2.5, 10, 25, 50, 75, 90, 97.5))
    expected = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        out = parse_date_percentiles(text)
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()
    assert out[50.0] == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_parsing.py -k date -v`
Expected: FAIL — `ImportError: cannot import name 'parse_date_percentiles'`.

- [ ] **Step 3: Implement the parsers**

Add `from datetime import datetime, timezone` to the imports at the top of `bot/parsing.py` (it currently imports only `re`), then append after `parse_percentiles` (line 53):

```python
def to_utc_timestamp(dt: datetime) -> float:
    """Unix seconds. A naive datetime is assumed UTC (NOT machine-local):
    naive datetime.timestamp() uses the local TZ, which is non-deterministic
    across machines and would corrupt llm_cache replay and the alignment
    between question bounds and elicited percentiles."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_DATE = r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?Z?)?)"


def _date_to_timestamp(s: str) -> float:
    s = s.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise ParseError(f"unparseable date: {s!r}") from e
    return to_utc_timestamp(dt)


def parse_date_percentiles(text: str) -> dict:
    """All 'Percentile P: <date>' lines as {percentile: unix_timestamp}.
    Dates may be YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS[Z]; bare dates default to
    midnight UTC. Requires >= 5 distinct percentiles (else ParseError, so the
    member is dropped rather than submitting garbage)."""
    matches = re.findall(
        r"Percentile\s+(\d+(?:\.\d+)?)\s*:\s*" + _DATE, text, re.IGNORECASE)
    out = {}
    for pct, datestr in matches:
        out[float(pct)] = _date_to_timestamp(datestr)   # later overwrite earlier
    if len(out) < 5:
        raise ParseError(f"only {len(out)} date percentiles found, need >= 5")
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_parsing.py -v`
Expected: PASS (all date tests + the pre-existing parsing tests).

- [ ] **Step 5: Commit**

```bash
git add bot/parsing.py tests/test_parsing.py
git commit -m "feat(parsing): date-percentile parser -> UTC unix timestamps

parse_date_percentiles + to_utc_timestamp; naive datetimes forced to UTC so
.timestamp() is machine-TZ-independent (determinism + llm_cache replay).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `DATE_PROMPT` + `run_date_ensemble` (`bot/prompts.py`, `bot/ensemble_bot.py`)

**Files:**
- Modify: `bot/prompts.py` (append `DATE_PROMPT` after `NUMERIC_PROMPT`, line 98)
- Modify: `bot/ensemble_bot.py` — imports (lines 23-25), `_template` map (line 92-93), add `run_date_ensemble` after `run_numeric_ensemble` (line 212)
- Test: `tests/test_ensemble_bot.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ensemble_bot.py`:

```python
CANNED_DATE = """reasoning...
Percentile 2.5: 2026-02-01
Percentile 10: 2026-03-01
Percentile 25: 2026-04-01
Percentile 50: 2026-06-01
Percentile 75: 2026-08-01
Percentile 90: 2026-10-01
Percentile 97.5: 2026-12-01"""


async def test_date_pipeline_medians_timestamps(monkeypatch):
    from datetime import datetime, timezone
    bot = make_bot(monkeypatch, CANNED_DATE)
    result = await bot.run_date_ensemble(
        **COMMON, lower_bound_message="not before 2026-01-01",
        upper_bound_message="not after 2027-01-01")
    assert result.median_percentiles[50.0] == \
        datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()
    assert 2.5 in result.median_percentiles
    assert 97.5 in result.median_percentiles
    assert len(result.member_models) == 6
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ensemble_bot.py::test_date_pipeline_medians_timestamps -v`
Expected: FAIL — `AttributeError: 'CalibratedEnsembleBot' object has no attribute 'run_date_ensemble'`.

- [ ] **Step 3a: Add `DATE_PROMPT` to `bot/prompts.py`**

Append after `NUMERIC_PROMPT` (after line 98):

```python
DATE_PROMPT = (_HEADER + """{lower_bound_message}
{upper_bound_message}

Work through these steps, writing each out briefly:

1. RESTATE: what exact event the resolution criteria date, and by when it
   would be known. Express everything as calendar dates.
2. TIME TO KNOW: when will the outcome actually be settled?
3. STATUS QUO / TREND: the resolution date if nothing changes, and if the
   current trend continues.
4. EXPERTS & MARKETS: what relevant expert estimates, schedules, or markets
   imply about timing.
5. EARLY SCENARIO: a plausible unexpected path to a MUCH EARLIER date.
6. LATE SCENARIO: a plausible unexpected path to a MUCH LATER date.
7. TAIL CHECK: good forecasters set WIDE intervals — the single most common
   error is tails that are too narrow. Your 2.5th and 97.5th percentile dates
   should feel uncomfortably far apart.

The last lines of your answer must be exactly (dates in strictly increasing
chronological order, format YYYY-MM-DD; append THH:MM:SSZ in UTC only if the
hour matters; nothing else on these lines):
Percentile 2.5: YYYY-MM-DD
Percentile 10: YYYY-MM-DD
Percentile 25: YYYY-MM-DD
Percentile 50: YYYY-MM-DD
Percentile 75: YYYY-MM-DD
Percentile 90: YYYY-MM-DD
Percentile 97.5: YYYY-MM-DD
""")
```

- [ ] **Step 3b: Wire it into the engine in `bot/ensemble_bot.py`**

Update the prompts import (line 25):

```python
from bot.prompts import BINARY_PROMPT, DATE_PROMPT, MC_PROMPT, NUMERIC_PROMPT
```

Update the parsing import (lines 23-24) to add the date parser and the helper:

```python
from bot.parsing import (ParseError, parse_binary, parse_date_percentiles,
                         parse_multiple_choice, parse_percentiles,
                         to_utc_timestamp)
```

Update the `_template` map (lines 92-93) to add the `"date"` entry:

```python
        return {"binary": BINARY_PROMPT, "mc": MC_PROMPT,
                "numeric": NUMERIC_PROMPT, "date": DATE_PROMPT}[qtype]
```

Add the engine method immediately after `run_numeric_ensemble` (after line 212), inside the `CalibratedEnsembleBot` class:

```python
    async def run_date_ensemble(self, *, question_text, background_info,
                                resolution_criteria, fine_print, research,
                                close_iso, lower_bound_message,
                                upper_bound_message) -> NumericEnsembleResult:
        prompt = self._template("date").format(
            question_text=question_text, background_info=background_info,
            resolution_criteria=resolution_criteria, fine_print=fine_print,
            research=research, playbook_section=self._playbook_section(),
            today=(self.cfg.as_of.date() if self.cfg.as_of
                   else date.today()).isoformat(),
            close_date=str(close_iso)[:10],
            days_remaining=_days_remaining(close_iso, now=self.cfg.as_of),
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message)
        texts, models = await self._fan_out(prompt)
        dicts, kept = self._parse_survivors(texts, models, parse_date_percentiles)
        median_pcts = _median_per_percentile(dicts)
        return NumericEnsembleResult(median_percentiles=median_pcts,
                                     member_percentile_dicts=dicts,
                                     member_models=kept)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ensemble_bot.py -v`
Expected: PASS (the new date test + all existing engine tests).

- [ ] **Step 5: Commit**

```bash
git add bot/prompts.py bot/ensemble_bot.py tests/test_ensemble_bot.py
git commit -m "feat(ensemble): DATE_PROMPT + run_date_ensemble (timestamp medians)

Date-native elicitation through the existing ensemble; parses to UTC
timestamps and medians per percentile, reusing NumericEnsembleResult.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `_run_forecast_on_date` framework handler (`bot/ensemble_bot.py`)

**Files:**
- Modify: `bot/ensemble_bot.py` — `build_framework_bot` forecasting_tools import (lines 274-276), add handler after `_run_forecast_on_numeric` (after line 400)
- Test: `tests/test_framework_adapter.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_framework_adapter.py`:

```python
async def test_date_handler_returns_valid_timestamp_distribution(
        monkeypatch, tmp_path):
    from datetime import datetime, timezone

    import numpy as np
    from forecasting_tools import DateQuestion

    from bot.distributions import validate_cdf

    cfg = BotConfig(ensemble=[EnsembleMember("fake/model-a", runs=3),
                              EnsembleMember("fake/model-b", runs=3)],
                    log_path=str(tmp_path / "forecasts.jsonl"))
    canned = ("reasoning...\n"
              "Percentile 2.5: 2026-02-01\nPercentile 10: 2026-03-01\n"
              "Percentile 25: 2026-04-01\nPercentile 50: 2026-06-01\n"
              "Percentile 75: 2026-08-01\nPercentile 90: 2026-10-01\n"
              "Percentile 97.5: 2026-12-01")

    async def fake_invoke(self, member_and_prompt):
        return canned

    async def fake_run_research(question, cfg_):
        return ResearchResult(text="stub", provider="asknews",
                              providers=("asknews",))

    monkeypatch.setattr(CalibratedEnsembleBot, "_invoke_model", fake_invoke,
                        raising=True)
    import bot.ensemble_bot as eb
    monkeypatch.setattr(eb, "run_research", fake_run_research)

    bot = build_framework_bot(cfg)
    q = DateQuestion(
        question_text="When?", id_of_post=7, background_info="",
        resolution_criteria="", fine_print="",
        lower_bound=datetime(2026, 1, 1, tzinfo=timezone.utc),
        upper_bound=datetime(2027, 1, 1, tzinfo=timezone.utc),
        open_lower_bound=False, open_upper_bound=False)
    research = await bot.run_research(q)
    pred = await bot._run_forecast_on_date(q, research)

    cdf = pred.prediction_value.get_cdf()
    assert len(cdf) == 201
    # closed bounds: probabilities pinned 0..1, monotone, valid steps
    validate_cdf(np.array([p.percentile for p in cdf]),
                 open_lower=False, open_upper=False)
    # the distribution lives in timestamp space, within the bound timestamps
    lo = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    hi = datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp()
    assert cdf[0].value == pytest.approx(lo)
    assert cdf[-1].value == pytest.approx(hi)
    # logged as a date forecast
    rows = read_forecasts(cfg.log_path)
    assert rows[-1]["type"] == "date"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest "tests/test_framework_adapter.py::test_date_handler_returns_valid_timestamp_distribution" -v`
Expected: FAIL — `AttributeError: 'FrameworkBot' object has no attribute '_run_forecast_on_date'`.

- [ ] **Step 3: Implement the handler**

Add `DateQuestion` to the forecasting_tools import inside `build_framework_bot` (lines 274-276):

```python
    from forecasting_tools import (BinaryQuestion, DateQuestion, ForecastBot,
                                   MultipleChoiceQuestion, NumericQuestion,
                                   PredictedOptionList, ReasonedPrediction)
```

Add the handler inside `class FrameworkBot`, immediately after `_run_forecast_on_numeric` (after line 400):

```python
        async def _run_forecast_on_date(
                self, question: DateQuestion, research: str
        ) -> ReasonedPrediction:
            lo = to_utc_timestamp(question.lower_bound)
            hi = to_utc_timestamp(question.upper_bound)
            lo_msg = (f"The outcome can not be earlier than {question.lower_bound}."
                      if not question.open_lower_bound else
                      f"The outcome is likely not earlier than {question.lower_bound}.")
            hi_msg = (f"The outcome can not be later than {question.upper_bound}."
                      if not question.open_upper_bound else
                      f"The outcome is likely not later than {question.upper_bound}.")
            r = await engine.run_date_ensemble(
                research=research, lower_bound_message=lo_msg,
                upper_bound_message=hi_msg, **_common_fields(question))
            _log(question, "date", r.median_percentiles,
                 r.member_percentile_dicts, r.member_models)
            dist = PchipNumericDistribution(
                declared_percentiles=[
                    Percentile(percentile=pct / 100.0, value=val)
                    for pct, val in sorted(r.median_percentiles.items())],
                open_lower_bound=question.open_lower_bound,
                open_upper_bound=question.open_upper_bound,
                lower_bound=lo, upper_bound=hi,
                zero_point=question.zero_point,
                cdf_size=question.cdf_size)
            return ReasonedPrediction(
                prediction_value=dist,
                reasoning=f"median date percentiles (unix ts) across "
                          f"{len(r.member_percentile_dicts)} members: "
                          f"{r.median_percentiles}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_framework_adapter.py -v`
Expected: PASS (new date handler test + all existing adapter tests).

- [ ] **Step 5: Commit**

```bash
git add bot/ensemble_bot.py tests/test_framework_adapter.py
git commit -m "feat(bot): _run_forecast_on_date — fixes Gate-C date-question crash

Mirrors the numeric handler; bounds + percentiles share one UTC timestamp
rule. Closes NotImplementedError on date questions (gate_c_dryrun.log:445).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Full-suite regression gate

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all pre-existing tests (~495) plus the new date tests. In
particular the prompt byte-identity / config-hash golden tests must still pass:
`DATE_PROMPT` is a NEW constant and no existing template was edited, so binary/
MC/numeric rendered prompts and their `llm_cache` keys are unchanged.

- [ ] **Step 2: If anything fails, fix and re-run**

Triage: a failure in `test_parsing.py` / `test_ensemble_bot.py` /
`test_framework_adapter.py` is in-scope — fix it. A failure elsewhere means an
unintended change to a shared symbol (most likely the parsing or prompts
imports) — re-check Task 2 Step 3b. Do not edit unrelated tests to make them
pass.

- [ ] **Step 3: No commit needed** (Task 4 changes nothing if green).

---

## Post-plan (manual, out of band)

Re-running Gate C to confirm the date question (qid 43324) no longer crashes
requires a live OpenRouter key with sufficient weekly credit — currently
exhausted (HTTP 402). That is an account action for the owner, tracked
separately from this code change. Once credit is restored:
`.venv/bin/python scripts/run_testing_area.py` (with the env loaded) should pull
the practice questions and forecast the date question without
`NotImplementedError`.
