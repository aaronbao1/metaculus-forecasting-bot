"""Gate E: forecast the live seasonal FutureEval tournament + MiniBench.
Refuses to publish unless SUBMIT_PREDICTION=true (defense in depth: the
flag also gates publish_reports_to_metaculus inside build_framework_bot).

Post-run, the forward snapshot accumulator (backtest spec §10 item 1)
archives every question this run touched into the backtest corpus at
data/backtest/ — the Metaculus token retains full visibility into
own-forecast questions forever, so every live run grows the corpus.
Guarded: a snapshot failure can NEVER fail the tournament run.
Disable with SNAPSHOT_AFTER_RUN=false (default: true). Note: on ephemeral
CI runners the archive is discarded with the container — the accumulator
pays off on machines with a persistent data/backtest/."""
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from bot.config import DEFAULT_CONFIG
from bot.ensemble_bot import build_framework_bot
from bot.logging_store import append_run, read_forecasts

logger = logging.getLogger(__name__)


def post_run_snapshot(records: list[dict],
                      store_root: str = "data/backtest") -> None:
    """Archive every question id appearing in `records` into the backtest
    corpus. NEVER raises — log-and-continue only (the tournament result is
    already in; the snapshot is a bonus, not a dependency)."""
    if os.getenv("SNAPSHOT_AFTER_RUN", "true").strip().lower() != "true":
        logger.info("post-run snapshot disabled (SNAPSHOT_AFTER_RUN != true)")
        return
    try:
        # Imported at call time: the live bot must keep working even if the
        # backtest sidecar is broken or absent.
        from backtest.corpus.snapshot import snapshot_questions
        from backtest.corpus.store import CorpusStore

        ids = sorted({int(r["question_id"]) for r in records
                      if r.get("question_id") is not None})
        if not ids:
            logger.info("post-run snapshot: no question ids in this run")
            return
        store = CorpusStore(Path(store_root))
        report = snapshot_questions(store, ids)
        logger.info("post-run snapshot: %d archived, %d errors",
                    report.fetched, len(report.errors))
    except Exception:
        logger.exception("post-run snapshot failed (tournament run unaffected)")


async def _run_one_tournament(bot, tournament) -> list:
    """Forecast one tournament, print the summary, and persist the coverage
    run-summary. An append failure never fails the run (it already succeeded)."""
    reports = await bot.forecast_on_tournament(tournament, return_exceptions=True)
    errors = [r for r in reports if isinstance(r, BaseException)]
    print(f"{tournament}: {len(reports) - len(errors)} ok, {len(errors)} failed")
    for e in errors:
        print(f"  ERROR: {e}")
    stats = getattr(bot, "_last_run_stats", None)
    if stats:
        try:
            append_run(stats)
        except Exception:
            logger.exception("run-summary append failed (run unaffected)")
    return reports


async def main() -> list[dict]:
    """Run the tournament and return only the records appended by this run."""
    from forecasting_tools import MetaculusApi
    n_before = len(read_forecasts(DEFAULT_CONFIG.log_path))
    bot = build_framework_bot(DEFAULT_CONFIG)
    for tournament in (MetaculusApi.CURRENT_AI_COMPETITION_ID,
                       MetaculusApi.CURRENT_MINIBENCH_ID):
        await _run_one_tournament(bot, tournament)
    # Only the records appended by THIS run (the log is append-only).
    return read_forecasts(DEFAULT_CONFIG.log_path)[n_before:]


if __name__ == "__main__":
    new_records = asyncio.run(main())
    post_run_snapshot(new_records)
