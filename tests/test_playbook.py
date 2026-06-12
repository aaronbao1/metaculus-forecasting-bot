"""Task 2: playbook loader + prompt injection + per-qtype prompt overrides.
The load-bearing property: with playbook_path=None and no overrides, rendered
prompts are BYTE-IDENTICAL to the pre-playbook era (sha256 goldens captured
2026-06-11 before the change) — protects llm_cache hits and live-path parity."""
import asyncio
import hashlib
import textwrap
from datetime import datetime, timezone

import pytest

from bot.config import BotConfig, EnsembleMember
from bot.ensemble_bot import CalibratedEnsembleBot
from bot.playbook import load_playbook

GOLDEN = {
    "binary": "affa913d7a7135e7aa80c7705a3cb18b3806ecb8379713edd10bf3660751ba7b",
    "mc": "f0dd618f64360582bcae60283f28921829d3469aa52a7fc66caa630056272a53",
    "numeric": "84cb5426218e2512f4b9bbdd9ae83a73b18f9a676db69306f7be62739a7b1a09",
}

PLAYBOOK = textwrap.dedent("""\
    # Forecasting Playbook

    ## Promoted

    - For sanctions questions, check the OFAC press page first. <!-- id:pb-1 source:run-1 date:2026-01-01 -->
    - Decay deadline-bound probabilities absent news. <!-- id:pb-2 source:run-1 date:2026-01-01 -->

    ## Candidates

    - Never inject me. <!-- id:pb-3 source:run-1 date:2026-01-01 -->
    """)


def test_load_playbook_missing_path_is_empty(tmp_path):
    assert load_playbook(tmp_path / "nope.md") == ""


def test_load_playbook_promoted_only_comments_stripped(tmp_path):
    p = tmp_path / "pb.md"
    p.write_text(PLAYBOOK)
    text = load_playbook(p)
    assert "OFAC" in text and "Decay deadline-bound" in text
    assert "Never inject me" not in text
    assert "<!--" not in text


def test_load_playbook_truncates_at_bullet_boundary(tmp_path):
    p = tmp_path / "pb.md"
    p.write_text(PLAYBOOK)
    text = load_playbook(p, max_chars=60)
    assert "OFAC" in text
    assert "Decay" not in text          # second bullet would exceed the cap


def _capture_prompt(cfg, qtype):
    bot = CalibratedEnsembleBot(cfg)
    captured = {}

    async def fake(member_and_prompt):
        captured["prompt"] = member_and_prompt[1]
        return {"binary": "Probability: 50%",
                "mc": "A: 50%\nB: 50%",
                "numeric": ("Percentile 2.5: 1\nPercentile 10: 2\n"
                            "Percentile 25: 3\nPercentile 50: 4\n"
                            "Percentile 75: 5\nPercentile 90: 6\n"
                            "Percentile 97.5: 7")}[qtype]

    bot._invoke_model = fake
    common = dict(question_text="Will X happen?", background_info="BG",
                  resolution_criteria="RC", fine_print="FP", research="RES",
                  close_iso="2026-02-01T00:00:00+00:00")
    if qtype == "binary":
        asyncio.run(bot.run_binary_ensemble(**common))
    elif qtype == "mc":
        asyncio.run(bot.run_mc_ensemble(options=["A", "B"], **common))
    else:
        asyncio.run(bot.run_numeric_ensemble(
            units="u", lower_bound_message="LB", upper_bound_message="UB",
            **common))
    return captured["prompt"]


@pytest.mark.parametrize("qtype", ["binary", "mc", "numeric"])
def test_prompts_byte_identical_when_off(qtype):
    cfg = BotConfig(ensemble=[EnsembleMember("m", runs=1)], min_members=1,
                    as_of=datetime(2026, 1, 2, tzinfo=timezone.utc))
    prompt = _capture_prompt(cfg, qtype)
    assert hashlib.sha256(prompt.encode()).hexdigest() == GOLDEN[qtype]


def test_playbook_injected_between_research_and_date(tmp_path):
    p = tmp_path / "pb.md"
    p.write_text(PLAYBOOK)
    cfg = BotConfig(ensemble=[EnsembleMember("m", runs=1)], min_members=1,
                    playbook_path=str(p),
                    as_of=datetime(2026, 1, 2, tzinfo=timezone.utc))
    prompt = _capture_prompt(cfg, "binary")
    assert "FORECASTING PLAYBOOK" in prompt
    assert "OFAC" in prompt and "Never inject me" not in prompt
    assert (prompt.index("RES") < prompt.index("FORECASTING PLAYBOOK")
            < prompt.index("Today's date:"))


def test_empty_promoted_section_keeps_prompt_byte_identical(tmp_path):
    p = tmp_path / "pb.md"
    p.write_text("# Forecasting Playbook\n\n## Promoted\n\n## Candidates\n")
    cfg = BotConfig(ensemble=[EnsembleMember("m", runs=1)], min_members=1,
                    playbook_path=str(p),
                    as_of=datetime(2026, 1, 2, tzinfo=timezone.utc))
    prompt = _capture_prompt(cfg, "binary")
    assert hashlib.sha256(prompt.encode()).hexdigest() == GOLDEN["binary"]


def test_prompt_overrides_replace_only_named_qtype():
    tmpl = "CUSTOM {question_text} {research}\nProbability line please."
    cfg = BotConfig(ensemble=[EnsembleMember("m", runs=1)], min_members=1,
                    prompt_overrides={"binary": tmpl},
                    as_of=datetime(2026, 1, 2, tzinfo=timezone.utc))
    prompt = _capture_prompt(cfg, "binary")
    assert prompt == "CUSTOM Will X happen? RES\nProbability line please."
    mc_prompt = _capture_prompt(cfg, "mc")
    assert hashlib.sha256(mc_prompt.encode()).hexdigest() == GOLDEN["mc"]
