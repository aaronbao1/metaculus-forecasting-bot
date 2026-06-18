from scripts.dashboard import summarize_forecasts


def test_summarize_forecasts_counts_types_providers_and_timeline():
    rows = [
        {"type": "binary", "final": 0.2, "config_hash": "aaa",
         "research_providers": "asknews,search_llm,markets",
         "logged_at": "2026-06-11T16:32:02.7+00:00"},
        {"type": "binary", "final": 0.9, "config_hash": "aaa",
         "research_providers": "search_llm,markets",
         "logged_at": "2026-06-11T17:00:00+00:00"},
        {"type": "numeric", "final": {"50": 3.0}, "config_hash": "bbb",
         "research_providers": "", "logged_at": "2026-06-12T09:00:00+00:00"},
    ]
    s = summarize_forecasts(rows)
    assert s["total"] == 3
    assert s["by_type"] == {"binary": 2, "numeric": 1}
    assert s["provider_coverage"]["markets"] == 2
    assert s["provider_coverage"]["asknews"] == 1
    assert s["provider_coverage"]["none"] == 1
    assert s["by_config"] == {"aaa": 2, "bbb": 1}
    assert s["binary_finals"] == [0.2, 0.9]
    assert s["timeline"] == [("2026-06-11", 2), ("2026-06-12", 1)]
    assert s["last_forecast"] == "2026-06-12T09:00:00+00:00"


def test_summarize_forecasts_empty():
    s = summarize_forecasts([])
    assert s["total"] == 0
    assert s["binary_finals"] == []
    assert s["last_forecast"] is None
