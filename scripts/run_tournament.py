"""Gate E: forecast the live seasonal FutureEval tournament + MiniBench.
Refuses to publish unless SUBMIT_PREDICTION=true (defense in depth: the
flag also gates publish_reports_to_metaculus inside build_framework_bot)."""
import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from bot.config import DEFAULT_CONFIG
from bot.ensemble_bot import build_framework_bot


async def main():
    from forecasting_tools import MetaculusApi
    bot = build_framework_bot(DEFAULT_CONFIG)
    for tournament in (MetaculusApi.CURRENT_AI_COMPETITION_ID,
                       MetaculusApi.CURRENT_MINIBENCH_ID):
        reports = await bot.forecast_on_tournament(
            tournament, return_exceptions=True)
        errors = [r for r in reports if isinstance(r, Exception)]
        print(f"{tournament}: {len(reports) - len(errors)} ok, "
              f"{len(errors)} failed")
        for e in errors:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())
