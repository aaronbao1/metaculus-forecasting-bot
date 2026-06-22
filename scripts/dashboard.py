"""Local auto-refreshing dashboard for the forecasting bot.

Run:  .venv/bin/python scripts/dashboard.py [--port 8787] [--tick 15]
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
    # (collector, refresh interval seconds). Metaculus polls least often.
    return {
        "cost": (lambda: collectors.fetch_openrouter_budget(or_key), 60),
        "runs": (lambda: collectors.fetch_github_runs(), 60),
        "run_summary": (lambda: collectors.fetch_run_summary(), 60),
        "armed": (lambda: collectors.fetch_armed_state(), 120),
        "coverage": (lambda: collectors.fetch_metaculus_coverage(mc_token, TOURNAMENTS), 300),
        "resolved": (lambda: collectors.fetch_recent_resolved(mc_token, TOURNAMENTS), 1800),
        "forecasts": (lambda: collectors.fetch_forecast_ledger(mc_token, TOURNAMENTS), 120),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--tick", type=float, default=15.0,
                    help="how often the loop checks for due collectors")
    args = ap.parse_args(argv)

    state = DashboardState(build_collectors_map(), tick=args.tick)
    state.refresh_once()
    state.start_background()
    logger.info("dashboard at http://127.0.0.1:%d  (tick %ss)", args.port, args.tick)
    try:
        serve(state, port=args.port)
    except KeyboardInterrupt:
        state.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
