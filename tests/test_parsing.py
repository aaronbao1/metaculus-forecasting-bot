from datetime import datetime, timezone

import pytest

from bot.parsing import (ParseError, parse_binary, parse_date_percentiles,
                         parse_multiple_choice, parse_percentiles)


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

