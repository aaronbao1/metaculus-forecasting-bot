"""scripts/check_regression.py: compare a run's mean baseline_score in
scores.jsonl against a committed reference, one-sided with tolerance
(improvements never fail)."""
import json

from scripts.check_regression import check, main, mean_baseline_score


def _scores(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _baseline(path, mean, tol):
    path.write_text(json.dumps({"mean_baseline_score": mean, "tolerance": tol}))
    return path


def test_mean_ignores_rows_without_numeric_baseline_score(tmp_path):
    p = _scores(tmp_path / "s.jsonl",
                [{"baseline_score": 10.0}, {"baseline_score": None},
                 {"other": 1}, {"baseline_score": 20},
                 {"baseline_score": True}])     # bool is NOT a score
    assert mean_baseline_score(p) == 15.0


def test_within_tolerance_passes(tmp_path):
    s = _scores(tmp_path / "s.jsonl",
                [{"baseline_score": 10.0}, {"baseline_score": 20.0}])
    b = _baseline(tmp_path / "b.json", 15.5, 1.0)
    ok, msg = check(s, b)
    assert ok
    assert "OK" in msg


def test_drop_beyond_tolerance_fails(tmp_path):
    s = _scores(tmp_path / "s.jsonl",
                [{"baseline_score": 10.0}, {"baseline_score": 20.0}])   # mean 15
    b = _baseline(tmp_path / "b.json", 30.0, 1.0)
    ok, msg = check(s, b)
    assert not ok
    assert "REGRESSION" in msg


def test_improvement_never_fails(tmp_path):
    s = _scores(tmp_path / "s.jsonl", [{"baseline_score": 50.0}])
    b = _baseline(tmp_path / "b.json", 10.0, 1.0)
    ok, _ = check(s, b)
    assert ok


def test_main_update_then_pass_then_fail(tmp_path):
    good = _scores(tmp_path / "good.jsonl",
                   [{"baseline_score": 10.0}, {"baseline_score": 20.0}])
    b = tmp_path / "baseline.json"
    rc = main(["--scores", str(good), "--baseline", str(b),
               "--update-baseline", "--tolerance", "1.0"])
    assert rc == 0
    assert json.loads(b.read_text()) == {"mean_baseline_score": 15.0,
                                         "tolerance": 1.0}
    assert main(["--scores", str(good), "--baseline", str(b)]) == 0
    bad = _scores(tmp_path / "bad.jsonl", [{"baseline_score": 1.0}])
    assert main(["--scores", str(bad), "--baseline", str(b)]) == 1
