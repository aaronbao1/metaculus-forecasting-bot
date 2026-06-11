"""Self-improvement loop: fit the binary Platt map from this bot's own
resolved forecasts. Refuses below MIN_FIT_N (identity stays in effect —
that's the stopgap, not an error). Rerun monthly / after each MiniBench."""
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from bot.calibration import MIN_FIT_N, fit_platt, save_platt
from bot.config import DEFAULT_CONFIG
from bot.logging_store import read_forecasts


def join_resolutions(logs: list, resolutions: dict):
    preds, outcomes = [], []
    for row in logs:
        if row.get("type") != "binary":
            continue
        res = resolutions.get(row.get("question_id"))
        if res is not None and row.get("final") is not None:
            preds.append(float(row["final"]))
            outcomes.append(int(res))
    return preds, outcomes


async def fetch_resolutions(question_ids: list) -> dict:
    """question_id -> 1/0 for resolved binary questions, via Metaculus API."""
    from forecasting_tools import MetaculusApi
    out = {}
    for qid in question_ids:
        try:
            q = MetaculusApi.get_question_by_post_id(qid)
            res = getattr(q, "resolution_string", None)
            if res is not None and str(res).lower() in ("yes", "no"):
                out[qid] = 1 if str(res).lower() == "yes" else 0
        except Exception as e:
            print(f"  skip {qid}: {e}")
    return out


async def main():
    logs = read_forecasts(DEFAULT_CONFIG.log_path)
    ids = sorted({r["question_id"] for r in logs if r.get("type") == "binary"})
    print(f"{len(logs)} logged forecasts, {len(ids)} unique binary questions")
    resolutions = await fetch_resolutions(ids)
    preds, outcomes = join_resolutions(logs, resolutions)
    print(f"{len(preds)} resolved binary forecasts available")
    if len(preds) < MIN_FIT_N:
        print(f"insufficient data (< {MIN_FIT_N}); identity calibration "
              f"remains in effect — this is the designed stopgap")
        sys.exit(0)
    params = fit_platt(preds, outcomes)
    save_platt(params, f"{DEFAULT_CONFIG.calibration_dir}/binary.json")
    print(f"fitted: a={params.a:.3f} b={params.b:.3f} n={params.n_fit} -> saved")


if __name__ == "__main__":
    asyncio.run(main())
