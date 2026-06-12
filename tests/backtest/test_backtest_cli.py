"""The `run` subcommand must expose --system stub (CI smoke gate). Help-text
only — never executes a run, never touches the network."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_run_subcommand_exposes_system_stub_flag():
    out = subprocess.run(
        [sys.executable, "scripts/backtest.py", "run", "--help"],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert "--system" in out.stdout
    assert "stub" in out.stdout
