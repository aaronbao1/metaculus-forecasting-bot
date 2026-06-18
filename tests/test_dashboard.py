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


# Task 2: Backtest-run aggregator
import json
from scripts.dashboard import summarize_backtest_scores, load_backtest_runs


def test_summarize_backtest_scores_means_and_counts():
    scores = [
        {"brier": 0.10, "log_score": -0.3, "baseline_score": 12.0,
         "h2h_market": 5.0, "h2h_cp": None},
        {"brier": 0.20, "log_score": -0.5, "baseline_score": 8.0,
         "h2h_market": 3.0, "h2h_cp": 2.0},
        {"brier": None, "log_score": None, "baseline_score": None,
         "h2h_market": None, "h2h_cp": None},  # unscored row ignored
    ]
    manifest = {"run_id": "current-bot-clean-0-91c43068",
                "spec_name": "current-bot-clean", "actual_cost_usd": 1.23}
    s = summarize_backtest_scores(scores, manifest)
    assert s["run_id"] == "current-bot-clean-0-91c43068"
    assert s["n_scored"] == 2
    assert s["mean_brier"] == 0.15
    assert s["mean_log_score"] == -0.4
    assert s["mean_h2h_market"] == 4.0
    assert s["mean_h2h_cp"] == 2.0          # only the one non-null
    assert s["actual_cost_usd"] == 1.23


def test_load_backtest_runs_discovers_and_sorts(tmp_path):
    root = tmp_path / "backtest"
    run = root / "spec" / "spec-0-abc"
    run.mkdir(parents=True)
    (run / "scores.jsonl").write_text(
        json.dumps({"brier": 0.1, "log_score": -0.2, "baseline_score": 1.0,
                    "h2h_market": None, "h2h_cp": None}) + "\n")
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "spec-0-abc", "spec_name": "spec",
                    "actual_cost_usd": 0.5}))
    runs = load_backtest_runs(root)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "spec-0-abc"
    assert runs[0]["n_scored"] == 1


def test_load_backtest_runs_missing_root_returns_empty(tmp_path):
    assert load_backtest_runs(tmp_path / "nope") == []


# Task 3: Roadmap parser
from scripts.dashboard import parse_roadmap


def test_parse_roadmap_extracts_rank_title_and_tag():
    md = (
        "# heading noise\n"
        "### 1. Ship a calibration layer now: Platt vs extremization "
        "— **IMMEDIATE, $0 replay**\n"
        "some body text\n"
        "### 2. Swap regressed ensemble members — newest != best\n"
        "### 10. Research composition: restore outside-view track\n"
    )
    items = parse_roadmap(md)
    assert [i["rank"] for i in items] == [1, 2, 10]
    assert items[0]["title"].startswith("Ship a calibration layer now")
    assert items[0]["tag"] == "IMMEDIATE, $0 replay"
    assert items[1]["tag"] == ""          # no bold tag
    assert "Swap regressed ensemble members" in items[1]["title"]


def test_parse_roadmap_empty_text():
    assert parse_roadmap("") == []


# Task 4: OpenRouter usage
from scripts.dashboard import parse_openrouter_key, fetch_openrouter_usage


def test_parse_openrouter_key_maps_fields():
    data = {"data": {"usage": 120.37, "limit": 100, "limit_remaining": 29.29,
                     "usage_daily": 0.5, "usage_weekly": 70.7,
                     "usage_monthly": 120.37, "limit_reset": "weekly"}}
    u = parse_openrouter_key(data)
    assert u["usage"] == 120.37
    assert u["limit"] == 100
    assert u["limit_remaining"] == 29.29
    assert u["usage_weekly"] == 70.7
    assert u["limit_reset"] == "weekly"


def test_parse_openrouter_key_tolerates_missing_fields():
    u = parse_openrouter_key({"data": {"usage": 5.0}})
    assert u["usage"] == 5.0
    assert u["limit"] is None
    assert u["usage_weekly"] is None


def test_fetch_openrouter_usage_no_key_returns_none():
    assert fetch_openrouter_usage(None) is None
    assert fetch_openrouter_usage("") is None


# Task 5: Inline-SVG chart + numeric helpers
from scripts.dashboard import svg_bars, svg_gauge, histogram, esc


def test_histogram_buckets_into_ten_bins():
    h = histogram([0.05, 0.15, 0.15, 0.99])
    assert len(h) == 10
    assert h[0] == ("0.0–0.1", 1)
    assert h[1] == ("0.1–0.2", 2)
    assert h[9] == ("0.9–1.0", 1)


def test_svg_bars_renders_each_pair():
    out = svg_bars([("binary", 2), ("numeric", 1)])
    assert out.startswith("<svg")
    assert "binary" in out and "numeric" in out
    assert out.count("<rect") == 2


def test_svg_bars_empty_is_safe():
    assert svg_bars([]).startswith("<svg")


def test_svg_gauge_clamps_and_colors():
    assert svg_gauge(50, 100).startswith("<svg")
    assert svg_gauge(200, 100).startswith("<svg")   # over-limit clamps, no crash
    assert svg_gauge(1, 0).startswith("<svg")        # zero limit, no div-by-zero


def test_esc_escapes_html():
    assert esc("<a>&\"") == "&lt;a&gt;&amp;&quot;"


# Task 6: HTML page renderer
from scripts.dashboard import render_page


def _ctx():
    return {
        "generated_at": "2026-06-18T00:00:00+00:00",
        "status": {"workflow_state": "disabled", "submit_prediction": "false",
                   "origin": "in sync", "last_forecast": "2026-06-12T09:00:00+00:00",
                   "total": 58},
        "costs": {"usage": 120.37, "limit": 100, "limit_remaining": 29.29,
                  "usage_weekly": 70.7, "usage_daily": 0.5, "usage_monthly": 120.37,
                  "limit_reset": "weekly"},
        "forecasts": {"total": 58, "by_type": {"binary": 20, "numeric": 24},
                      "by_config": {"aaa": 58},
                      "provider_coverage": {"asknews": 30, "markets": 58, "none": 2},
                      "binary_finals": [0.2, 0.9], "timeline": [("2026-06-11", 58)],
                      "last_forecast": "2026-06-12T09:00:00+00:00"},
        "backtests": [{"run_id": "current-bot-clean-0-91c43068",
                       "spec_name": "current-bot-clean", "n_scored": 121,
                       "n_total": 485, "mean_brier": 0.094, "mean_log_score": -0.3,
                       "mean_baseline": 12.0, "mean_h2h_market": 5.31,
                       "mean_h2h_cp": None, "actual_cost_usd": 1.23,
                       "report_path": "logs/backtest/x/report.md"}],
        "roadmap": [{"rank": 1, "title": "Ship calibration layer", "tag": "IMMEDIATE, $0"}],
    }


def test_render_page_is_self_contained_html_with_all_panels():
    html = render_page(_ctx())
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower()        # no JS
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")  # no external links
    for marker in ("API costs", "Questions", "Backtest", "Future directions",
                   "current-bot-clean-0-91c43068", "Ship calibration layer", "29.29"):
        assert marker in html


def test_render_page_handles_missing_costs():
    ctx = _ctx()
    ctx["costs"] = None
    html = render_page(ctx)
    assert "unavailable" in html.lower()


# Task 7: main() wiring
from scripts.dashboard import build_context


def test_build_context_assembles_all_sections(tmp_path, monkeypatch):
    # forecasts log
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "forecasts.jsonl").write_text(
        json.dumps({"type": "binary", "final": 0.3, "config_hash": "aaa",
                    "research_providers": "asknews,markets",
                    "logged_at": "2026-06-11T16:32:02+00:00"}) + "\n")
    # backtest run
    run = tmp_path / "logs" / "backtest" / "spec" / "spec-0-abc"
    run.mkdir(parents=True)
    (run / "scores.jsonl").write_text(
        json.dumps({"brier": 0.1, "log_score": -0.2, "baseline_score": 1.0,
                    "h2h_market": 2.0, "h2h_cp": None}) + "\n")
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "spec-0-abc", "spec_name": "spec",
                    "actual_cost_usd": 0.4}))
    # roadmap
    notes = tmp_path / "research-notes"
    notes.mkdir()
    (notes / "08-improvement-roadmap.md").write_text("### 1. Do the thing — **$0**\n")

    # no network: force costs None
    import scripts.dashboard as _dash
    monkeypatch.setattr(_dash, "fetch_openrouter_usage", lambda *a, **k: None)
    ctx = build_context(root=tmp_path, openrouter_key=None, generated_at="T")
    assert ctx["forecasts"]["total"] == 1
    assert ctx["backtests"][0]["run_id"] == "spec-0-abc"
    assert ctx["roadmap"][0]["title"] == "Do the thing"
    assert ctx["costs"] is None
    assert ctx["generated_at"] == "T"


def test_build_context_missing_files_is_safe(tmp_path, monkeypatch):
    import scripts.dashboard as _dash
    monkeypatch.setattr(_dash, "fetch_openrouter_usage", lambda *a, **k: None)
    ctx = build_context(root=tmp_path, openrouter_key=None, generated_at="T")
    assert ctx["forecasts"]["total"] == 0
    assert ctx["backtests"] == []
    assert ctx["roadmap"] == []
