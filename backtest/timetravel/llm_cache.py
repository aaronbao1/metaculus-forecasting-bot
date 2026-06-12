"""Content-addressed completion cache around the ensemble's single LLM seam.
Key = sha256(f"{model}|{temperature}|{run_index}|{sha256(prompt)}"), one JSON
file per completion. live=False is replay-only (CacheMiss on any uncached
call) — that is what makes post-LLM reruns (aggregation, caps, calibration)
cost $0, and lets CI replay a frozen corpus with no keys at all."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CacheMiss(Exception):
    """Raised in live=False (replay) mode when a completion is not cached."""


def cache_key(model: str, prompt: str, temperature: float,
              run_index: int = 0) -> str:
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    return hashlib.sha256(
        f"{model}|{temperature}|{run_index}|{prompt_sha}".encode()).hexdigest()


class CachedLLM:

    def __init__(self, cache_dir: Path, live: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.live = live

    async def invoke(self, model: str, prompt: str, temperature: float,
                     run_index: int = 0) -> str:
        key = cache_key(model, prompt, temperature, run_index)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["completion"]
        if not self.live:
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            raise CacheMiss(f"model={model} temperature={temperature} "
                            f"run_index={run_index} "
                            f"prompt_sha={prompt_sha[:12]}")
        completion, tokens_in, tokens_out, cost_usd = await self._call_live(
            model, prompt, temperature)
        record = {"model": model, "temperature": temperature,
                  "run_index": run_index,
                  "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(),
                  "completion": completion, "tokens_in": tokens_in,
                  "tokens_out": tokens_out, "cost_usd": cost_usd}
        tmp = path.with_suffix(".json.tmp")     # atomic: never a torn cache file
        tmp.write_text(json.dumps(record))
        tmp.replace(path)
        return completion

    async def _call_live(self, model: str, prompt: str, temperature: float):
        """Single live seam (monkeypatch target in tests).

        Cost is captured via MonetaryCostManager (litellm callback) rather
        than by inspecting the response object.  GeneralLlm.invoke() returns a
        plain str, so getattr(resp, 'cost', ...) always returns 0; the
        MonetaryCostManager context records what litellm's internal callback
        actually tracked.  Cost is a lower bound — litellm may not price all
        model slugs (unknown slugs produce cost_usd=0 with a warning in logs).
        Token counts remain 0 for the same reason; they are not used by the
        spend-brake and are included for forward-compatibility only.
        """
        from forecasting_tools import GeneralLlm, MonetaryCostManager
        llm = GeneralLlm(model=f"openrouter/{model}", temperature=temperature,
                         timeout=900)
        with MonetaryCostManager() as cost_mgr:
            resp = await llm.invoke(prompt)
        completion = resp if isinstance(resp, str) else str(resp)
        cost_usd = float(cost_mgr.current_usage or 0.0)
        # Tokens are not reliably extractable from a plain-str response;
        # record 0 and let the cost figure carry the spend signal.
        return completion, 0, 0, cost_usd
