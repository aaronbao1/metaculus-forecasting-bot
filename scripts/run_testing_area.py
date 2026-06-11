"""Gate C/D: run the full pipeline against the Metaculus bot-testing-area
tournament. Dry-run by default; prints every would-be submission for human
review. Set SUBMIT_PREDICTION=true in .env to actually publish there."""
import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from bot.config import DEFAULT_CONFIG
from bot.ensemble_bot import build_framework_bot

TESTING_AREA = "bot-testing-area"      # slug; ID 32977


async def main():
    bot = build_framework_bot(DEFAULT_CONFIG)
    reports = await bot.forecast_on_tournament(
        TESTING_AREA, return_exceptions=True)
    for r in reports:
        if isinstance(r, Exception):
            print(f"FAILED: {r}")
        else:
            print(getattr(r, "summary", None) or r)
    submitted = DEFAULT_CONFIG.submit_predictions
    print(f"\nsubmit_predictions={submitted} "
          f"({'published' if submitted else 'DRY RUN — nothing submitted'})")


if __name__ == "__main__":
    asyncio.run(main())
