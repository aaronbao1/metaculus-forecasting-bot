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


# ---------------------------------------------------------------------------
# Naive-datetime safety: backtest as_of coercion
# ---------------------------------------------------------------------------

def test_days_remaining_naive_as_of_with_aware_close():
    """Naive as_of (e.g. from a backtest fixture) must not silently return 0.0.
    The fix coerces naive inputs to UTC so the arithmetic succeeds."""
    naive_now = datetime(2024, 6, 15, 12, 0)           # no tzinfo
    aware_close = datetime(2024, 6, 25, 12, 0, tzinfo=timezone.utc)
    result = _days_remaining(aware_close.isoformat(), now=naive_now)
    assert result == pytest.approx(10.0, abs=1e-6)


def test_days_remaining_naive_close_iso_with_aware_now():
    """A close_iso string that round-trips through fromisoformat without a tz
    component (rare but possible) is also coerced to UTC correctly."""
    aware_now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    # fromisoformat of a plain ISO string produces a naive datetime
    naive_close_iso = "2024-06-25T12:00:00"             # no Z / +00:00
    result = _days_remaining(naive_close_iso, now=aware_now)
    assert result == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# B1 — horizon_to_asof with naive open + aware close must not raise TypeError
# ---------------------------------------------------------------------------

def test_horizon_to_asof_naive_open_aware_close_no_crash():
    """62 corpus questions have naive open_time + aware scheduled_close_time.
    horizon_to_asof must normalise naive datetimes to UTC before arithmetic
    instead of crashing with TypeError."""
    from backtest.core import BacktestQuestion, QType
    from backtest.timetravel.asof import horizon_to_asof

    q = BacktestQuestion(
        qid="test:1", platform="test", native_id="1",
        qtype=QType.BINARY, title="T?",
        open_time=datetime(2024, 1, 1, 0, 0),                 # naive
        scheduled_close_time=datetime(2024, 7, 1, tzinfo=timezone.utc),  # aware
    )

    # All geo-k and close-Nd horizons must not raise
    for h in ["geo-1", "geo-2", "geo-3", "geo-4", "geo-5",
              "close-7d", "close-30d", "close-365d"]:
        result = horizon_to_asof(q, h)
        # When a value is returned (not None) it must be tz-aware
        if result is not None:
            assert result.tzinfo is not None, (
                f"horizon_to_asof({h!r}) returned naive datetime: {result!r}")


def test_horizon_to_asof_geo_naive_open_returns_aware_datetime():
    """geo-1 with naive open + aware close returns a tz-aware datetime (not None,
    not naive) — specifically it is the expected T_1 value."""
    from backtest.core import BacktestQuestion, QType
    from backtest.timetravel.asof import horizon_to_asof

    # span = 181 days (Jan 1 → Jul 1, 2024, leap year)
    open_dt = datetime(2024, 1, 1, 0, 0)          # naive
    close_dt = datetime(2024, 7, 1, tzinfo=timezone.utc)  # aware
    q = BacktestQuestion(
        qid="test:2", platform="test", native_id="2",
        qtype=QType.BINARY, title="T?",
        open_time=open_dt, scheduled_close_time=close_dt,
    )

    result = horizon_to_asof(q, "geo-1")
    assert result is not None
    assert result.tzinfo is not None, "returned datetime must be tz-aware"
