# tests/test_ensemble_bot_coverage.py
import pytest
from bot.config import BotConfig
from bot.ensemble_bot import build_framework_bot


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # build_framework_bot constructs the framework's Metaculus client, which reads
    # these at construction time; set dummies so the bot builds offline.
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


class FakeQ:
    def __init__(self, qid, already=False):
        self.id_of_post = qid
        self.already_forecasted = already


def _bot(cfg):
    return build_framework_bot(cfg)


async def test_paginates_and_caps(monkeypatch):
    cfg = BotConfig(max_questions_to_fetch=500, max_questions_per_run=3,
                    forecast_retry_attempts=0)
    bot = _bot(cfg)
    captured = {}

    async def fake_get(api_filter, num_questions=None, randomly_sample=False,
                       error_if_question_target_missed=True):
        captured.update(num=num_questions, err=error_if_question_target_missed,
                        tourneys=api_filter.allowed_tournaments,
                        statuses=api_filter.allowed_statuses)
        return [FakeQ(i) for i in range(5)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)

    async def fake_fq(questions, return_exceptions=False):
        return [f"ok-{q.id_of_post}" for q in questions]
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)

    reports = await bot.forecast_on_tournament(33022, return_exceptions=True)
    assert len(reports) == 3                      # capped
    assert captured["num"] == 500                 # explicit fetch target -> pagination
    assert captured["err"] is False               # must not raise on shortfall
    assert captured["tourneys"] == [33022]
    assert captured["statuses"] == ["open"]
    s = bot._last_run_stats
    assert s["questions_fetched"] == 5
    assert s["truncated_count"] == 2
    assert s["forecast_ok"] == 3
    assert s["tournament"] == "33022"


async def test_skips_already_forecasted_before_forecasting(monkeypatch):
    cfg = BotConfig(max_questions_per_run=50, forecast_retry_attempts=0)
    bot = _bot(cfg)

    async def fake_get(api_filter, **kw):
        return [FakeQ(1, already=True), FakeQ(2), FakeQ(3, already=True), FakeQ(4)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)

    seen = {}
    async def fake_fq(questions, return_exceptions=False):
        seen["ids"] = [q.id_of_post for q in questions]
        return [f"ok-{q.id_of_post}" for q in questions]
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)

    await bot.forecast_on_tournament(33022, return_exceptions=True)
    assert seen["ids"] == [2, 4]                   # skip-filtered, aligns with reports
    assert bot._last_run_stats["questions_after_skip"] == 2


async def test_raises_when_not_return_exceptions(monkeypatch):
    cfg = BotConfig(forecast_retry_attempts=0)
    bot = _bot(cfg)
    async def fake_get(api_filter, **kw):
        return [FakeQ(1)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)
    async def fake_fq(questions, return_exceptions=False):
        return [RuntimeError("boom")]
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)
    with pytest.raises(RuntimeError, match="boom"):
        await bot.forecast_on_tournament(33022, return_exceptions=False)


async def test_retries_transient_failures(monkeypatch):
    cfg = BotConfig(max_questions_per_run=50, forecast_retry_attempts=2)
    bot = _bot(cfg)
    async def fake_get(api_filter, **kw):
        return [FakeQ(1), FakeQ(2), FakeQ(3)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)

    calls = {"n": 0}
    async def fake_fq(questions, return_exceptions=False):
        calls["n"] += 1
        out = []
        for q in questions:
            if q.id_of_post == 2 and calls["n"] == 1:   # q2 fails only first time
                out.append(RuntimeError("transient"))
            else:
                out.append(f"ok-{q.id_of_post}")
        return out
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)

    reports = await bot.forecast_on_tournament(33022, return_exceptions=True)
    assert reports == ["ok-1", "ok-2", "ok-3"]      # q2 recovered, spliced in order
    s = bot._last_run_stats
    assert s["retried"] == 1 and s["retry_recovered"] == 1
    assert s["forecast_failed"] == 0


async def test_unrecovered_failure_stays_exception(monkeypatch):
    cfg = BotConfig(forecast_retry_attempts=1)
    bot = _bot(cfg)
    async def fake_get(api_filter, **kw):
        return [FakeQ(1), FakeQ(2)]
    monkeypatch.setattr(bot.metaculus_client, "get_questions_matching_filter", fake_get)
    async def fake_fq(questions, return_exceptions=False):
        return [RuntimeError("always") if q.id_of_post == 2 else f"ok-{q.id_of_post}"
                for q in questions]
    monkeypatch.setattr(bot, "forecast_questions", fake_fq)
    reports = await bot.forecast_on_tournament(33022, return_exceptions=True)
    assert reports[0] == "ok-1"
    assert isinstance(reports[1], BaseException)
    assert bot._last_run_stats["forecast_failed"] == 1
    assert bot._last_run_stats["retried"] == 1 and bot._last_run_stats["retry_recovered"] == 0
