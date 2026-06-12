from bot.prompts import (
    BINARY_PROMPT, MC_PROMPT, NUMERIC_PROMPT,
    RESEARCH_PROMPT_CURRENT, RESEARCH_PROMPT_HISTORICAL,
)

COMMON = dict(
    question_text="Will X happen?", background_info="Some background.",
    resolution_criteria="Resolves YES if X.", fine_print="None.",
    research="Report.", today="2026-06-11", close_date="2026-07-01",
    days_remaining=20.0, playbook_section="",
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
