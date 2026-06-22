"""Local auto-refreshing dashboard for the forecasting bot.

Run:  .venv/bin/python scripts/dashboard.py [--port 8787] [--refresh-interval 60]
Then open http://127.0.0.1:8787  (localhost only; no secrets are served).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=False)

from bot.dashboard import collectors  # noqa: E402
from bot.dashboard.state import DashboardState  # noqa: E402
from bot.dashboard.server import serve  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

TOURNAMENTS = ["33022", "minibench"]


def build_collectors_map() -> dict:
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    mc_token = os.getenv("METACULUS_TOKEN", "")
    return {
        "cost": lambda: collectors.fetch_openrouter_budget(or_key),
        "coverage": lambda: collectors.fetch_metaculus_coverage(mc_token, TOURNAMENTS),
        "resolved": lambda: collectors.fetch_recent_resolved(mc_token, TOURNAMENTS),
        "runs": lambda: collectors.fetch_github_runs(),
        "armed": lambda: collectors.fetch_armed_state(),
        "run_summary": lambda: collectors.fetch_run_summary(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--refresh-interval", type=float, default=60.0)
    args = ap.parse_args(argv)

    state = DashboardState(build_collectors_map(), interval=args.refresh_interval)
    state.refresh_once()
    state.start_background()
    logger.info("dashboard at http://127.0.0.1:%d  (refresh %ss)",
                args.port, args.refresh_interval)
    try:
        serve(state, port=args.port)
    except KeyboardInterrupt:
        state.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
