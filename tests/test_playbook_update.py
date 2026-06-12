"""Task 17: playbook reflector — failure facets to candidate deltas."""
import asyncio
import json

from scripts.playbook_update import update_playbook

PLAYBOOK = """# Forecasting Playbook

## Promoted

- Always check whether the event already happened. <!-- id:pb-old source:x date:2026-06-01 -->

## Candidates
"""

FAILURES = {
    "run_dir": "logs/backtest/run-1",
    "clusters": [{"label": "research misses",
                  "description": "single-query dossiers miss imminent news",
                  "members": [0]}],
    "facets": [{"summary": "missed an imminent product launch",
                "primary_cause": "research-miss", "baseline_score": -80.0}],
}


def _invoke_returning(bullets):
    async def invoke(prompt, sample_idx):
        return json.dumps({"bullets": bullets})
    return invoke


def test_appends_deduped_candidates_with_provenance(tmp_path):
    pb = tmp_path / "pb.md"
    pb.write_text(PLAYBOOK)
    invoke = _invoke_returning([
        {"lesson": "Generate multiple search queries per question.",
         "applies_to": "research-miss"},
        {"lesson": "Always check whether the event already happened.",
         "applies_to": "status"},                       # dup of promoted
        {"lesson": "Cross-check market price before finalizing.",
         "applies_to": "calibration"},
    ])
    added = asyncio.run(update_playbook(FAILURES, pb, invoke,
                                        today="2026-06-11"))
    assert added == 2
    text = pb.read_text()
    cand_section = text.split("## Candidates")[1]
    assert "Generate multiple search queries" in cand_section
    assert "Cross-check market price" in cand_section
    assert "already happened" not in cand_section       # deduped
    assert "source:logs/backtest/run-1" in cand_section
    assert "date:2026-06-11" in cand_section
    # Promoted untouched
    promoted = text.split("## Promoted")[1].split("## Candidates")[0]
    assert sum(1 for ln in promoted.splitlines()
               if ln.startswith("- ")) == 1


def test_idempotent_rerun_adds_nothing(tmp_path):
    pb = tmp_path / "pb.md"
    pb.write_text(PLAYBOOK)
    invoke = _invoke_returning([{"lesson": "Use base rates.",
                                 "applies_to": "all"}])
    assert asyncio.run(update_playbook(FAILURES, pb, invoke)) == 1
    assert asyncio.run(update_playbook(FAILURES, pb, invoke)) == 0


def test_max_new_honored(tmp_path):
    pb = tmp_path / "pb.md"
    pb.write_text(PLAYBOOK)
    invoke = _invoke_returning([{"lesson": f"Lesson number {i}.",
                                 "applies_to": "x"} for i in range(10)])
    assert asyncio.run(update_playbook(FAILURES, pb, invoke,
                                       max_new=3)) == 3


def test_cli_replay_mode_cache_miss_is_friendly(tmp_path, capsys):
    import json as _json

    from scripts.playbook_update import main
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    failures = run_dir / "failures.json"
    failures.write_text(_json.dumps(FAILURES))
    pb = tmp_path / "pb.md"
    pb.write_text(PLAYBOOK)
    rc = main([str(failures), "--playbook", str(pb)])   # replay-only default
    assert rc == 1
    out = capsys.readouterr().out
    assert "--live" in out                              # actionable hint
    assert pb.read_text() == PLAYBOOK                   # untouched


def test_garbage_reflection_adds_nothing(tmp_path):
    pb = tmp_path / "pb.md"
    pb.write_text(PLAYBOOK)

    async def invoke(prompt, sample_idx):
        return "not json"
    assert asyncio.run(update_playbook(FAILURES, pb, invoke)) == 0
    assert pb.read_text() == PLAYBOOK
