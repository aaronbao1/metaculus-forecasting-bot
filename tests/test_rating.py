"""Task 10: GABRIEL-style evidence rater — LLM-seam injected, never raises
per item, consensus across n_samples."""
import asyncio

from bot.rating import filter_dossier, rate_items, split_dossier_items

DOSSIER = (
    "Historical news snapshots (AskNews archive):\n"
    "- [2026-01-01] OPEC meets — production cut agreed (http://a)\n"
    "- [2026-01-02] Celebrity gossip — unrelated fluff (http://b)\n"
    "- [2026-01-03] Oil prices spike on cut news (http://c)\n")

QCTX = {"question_text": "Will WTI exceed $100?",
        "resolution_criteria": "Resolves YES if WTI settles above $100."}


def _scoring_invoke(scores_by_needle, jitter_by_sample=0):
    async def invoke(prompt, sample_idx):
        for needle, score in scores_by_needle.items():
            if needle in prompt:
                s = score + sample_idx * jitter_by_sample
                return f'{{"scores": {{"relevance": {s}}}}}'
        return '{"scores": {"relevance": 50}}'
    return invoke


def test_split_and_rejoin_lossless():
    header, items = split_dossier_items(DOSSIER)
    assert len(items) == 3
    assert header.startswith("Historical news")
    assert (header + "\n".join(items) + "\n") == DOSSIER


def test_rate_items_scores_and_consensus():
    invoke = _scoring_invoke({"OPEC": 90, "gossip": 10}, jitter_by_sample=4)
    ratings = asyncio.run(rate_items(
        ["OPEC article", "gossip article"], ("relevance",), QCTX, invoke,
        n_samples=2))
    assert ratings[0].scores["relevance"] == 92.0          # mean(90, 94)
    assert ratings[0].std["relevance"] == 2.0
    assert ratings[1].scores["relevance"] == 12.0


def test_rate_items_malformed_json_is_none_not_raise():
    async def invoke(prompt, sample_idx):
        if "bad" in prompt:
            return "not json at all"
        return '{"scores": {"relevance": 80}}'
    ratings = asyncio.run(rate_items(["good", "bad"], ("relevance",), QCTX,
                                     invoke))
    assert ratings[0] is not None and ratings[1] is None


def test_filter_dossier_drops_low_relevance():
    invoke = _scoring_invoke({"OPEC": 90, "gossip": 10, "spike": 75})
    filtered, ratings = asyncio.run(filter_dossier(
        DOSSIER, QCTX, invoke, threshold=40))
    assert "OPEC" in filtered and "spike" in filtered
    assert "gossip" not in filtered
    assert "relevance-filtered" in filtered
    assert len(ratings) == 3


def test_filter_dossier_unparsed_items_kept():
    async def invoke(prompt, sample_idx):
        return "garbage"
    filtered, _ = asyncio.run(filter_dossier(DOSSIER, QCTX, invoke))
    for needle in ("OPEC", "gossip", "spike"):     # fail-open: keep all
        assert needle in filtered


def test_filter_dossier_everything_dropped_is_empty():
    invoke = _scoring_invoke({"http": 5})          # matches all items
    filtered, _ = asyncio.run(filter_dossier(DOSSIER, QCTX, invoke))
    assert filtered == ""


def test_filter_dossier_no_bullets_passthrough():
    text = "Crosswalk-matched market priced P(Yes)=40%."
    filtered, ratings = asyncio.run(filter_dossier(
        text, QCTX, _scoring_invoke({})))
    assert filtered == text and ratings == []
