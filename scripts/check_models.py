"""Gate B: verify every configured model slug answers a trivial prompt.
Run after setting OPENROUTER_API_KEY. Catches wrong slugs / dead models /
quota issues BEFORE a tournament question does."""
import asyncio

from dotenv import load_dotenv

from bot.config import DEFAULT_CONFIG

load_dotenv()


async def main() -> None:
    from forecasting_tools import GeneralLlm
    models = [m.model for m in DEFAULT_CONFIG.ensemble]
    models += [DEFAULT_CONFIG.search_model, DEFAULT_CONFIG.parser_model,
               DEFAULT_CONFIG.supervisor_model]
    ok = True
    for slug in dict.fromkeys(models):     # dedupe, keep order
        try:
            llm = GeneralLlm(model=f"openrouter/{slug}", temperature=0)
            out = await llm.invoke("Reply with the single word: ready")
            print(f"  OK   {slug}: {out.strip()[:40]!r}")
        except Exception as e:
            ok = False
            print(f"  FAIL {slug}: {type(e).__name__}: {e}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
