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
