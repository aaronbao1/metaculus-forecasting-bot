from bot.logging_store import append_forecast, read_forecasts


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "f.jsonl"
    rec = {"question_id": 1, "type": "binary", "final": 0.42,
           "members": [0.4, 0.45], "config_hash": "abc123def456"}
    append_forecast(rec, path)
    append_forecast({**rec, "question_id": 2}, path)
    rows = read_forecasts(path)
    assert len(rows) == 2
    assert rows[0]["question_id"] == 1
    assert "logged_at" in rows[0]


def test_read_missing_file_returns_empty(tmp_path):
    assert read_forecasts(tmp_path / "nope.jsonl") == []


from bot.logging_store import append_run, read_runs


def test_append_and_read_runs_roundtrip(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_run({"tournament": "33022", "truncated_count": 0}, p)
    append_run({"tournament": "minibench", "truncated_count": 3}, p)
    rows = read_runs(p)
    assert [r["tournament"] for r in rows] == ["33022", "minibench"]
    assert rows[1]["truncated_count"] == 3
    assert "logged_at" in rows[0]          # timestamp injected like append_forecast


def test_read_runs_missing_file(tmp_path):
    assert read_runs(tmp_path / "nope.jsonl") == []
