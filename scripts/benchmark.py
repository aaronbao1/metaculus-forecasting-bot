"""Config A/B vs the community prediction on open main-site questions, using
forecasting-tools' Benchmarker. Regression test, not the objective — top
bots win by disagreeing with the CP *well*. Needs ~300+ questions to separate
similar configs (with 100 there's ~30% chance the worse bot wins)."""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from bot.config import DEFAULT_CONFIG
from bot.ensemble_bot import build_framework_bot


async def main():
    from forecasting_tools import Benchmarker
    bot = build_framework_bot(DEFAULT_CONFIG)
    benchmarker = Benchmarker(
        forecast_bots=[bot],
        number_of_questions_to_use=100,    # raise to 300+ for real decisions
        file_path_to_save_reports="logs/benchmarks/",
    )
    results = await benchmarker.run_benchmark()
    for b in results:
        print(b.name, "avg expected baseline score:",
              b.average_expected_baseline_score)


if __name__ == "__main__":
    asyncio.run(main())
