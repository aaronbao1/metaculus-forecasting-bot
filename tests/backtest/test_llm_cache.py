"""CachedLLM: content-addressed on (model, temperature, run_index,
prompt sha). Live calls go through forecasting_tools.GeneralLlm with the
openrouter/ prefix (monkeypatched here — never the network); live=False is
replay-only and raises CacheMiss on any uncached call."""
import hashlib
import json

import pytest

from backtest.timetravel.llm_cache import CachedLLM, CacheMiss, cache_key


class FakeGeneralLlm:
    constructed = []

    def __init__(self, model, temperature, timeout=None):
        FakeGeneralLlm.constructed.append(
            dict(model=model, temperature=temperature))

    async def invoke(self, prompt):
        return f"completion for: {prompt}"


@pytest.fixture
def llm(tmp_path, monkeypatch):
    FakeGeneralLlm.constructed = []
    monkeypatch.setattr("forecasting_tools.GeneralLlm", FakeGeneralLlm)
    return CachedLLM(tmp_path / "cache")


def test_key_derivation_is_content_addressed():
    prompt_sha = hashlib.sha256(b"P").hexdigest()
    expected = hashlib.sha256(f"m|0.5|2|{prompt_sha}".encode()).hexdigest()
    assert cache_key("m", "P", 0.5, 2) == expected
    assert cache_key("m", "P", 0.5, 3) != expected      # run_index in key
    assert cache_key("m", "P", 0.7, 2) != expected      # temperature in key
    assert cache_key("m2", "P", 0.5, 2) != expected     # model in key
    assert cache_key("m", "Q", 0.5, 2) != expected      # prompt in key


async def test_live_call_writes_cache_file_with_full_record(llm, tmp_path):
    out = await llm.invoke("openai/gpt-5.4-mini", "P", 0.0, run_index=1)
    assert out == "completion for: P"
    assert FakeGeneralLlm.constructed == [
        dict(model="openrouter/openai/gpt-5.4-mini", temperature=0.0)]
    key = cache_key("openai/gpt-5.4-mini", "P", 0.0, 1)
    rec = json.loads((tmp_path / "cache" / f"{key}.json").read_text())
    assert rec == {"model": "openai/gpt-5.4-mini", "temperature": 0.0,
                   "run_index": 1,
                   "prompt_sha": hashlib.sha256(b"P").hexdigest(),
                   "completion": "completion for: P",
                   "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}


async def test_second_call_hits_cache_not_the_llm(llm):
    await llm.invoke("m", "P", 0.0)
    n = len(FakeGeneralLlm.constructed)
    assert await llm.invoke("m", "P", 0.0) == "completion for: P"
    assert len(FakeGeneralLlm.constructed) == n         # no new live call


async def test_distinct_run_index_is_a_distinct_completion(llm):
    await llm.invoke("m", "P", 0.0, run_index=0)
    await llm.invoke("m", "P", 0.0, run_index=1)
    assert len(FakeGeneralLlm.constructed) == 2


async def test_replay_mode_raises_cache_miss(tmp_path):
    replay = CachedLLM(tmp_path / "cache", live=False)
    with pytest.raises(CacheMiss):
        await replay.invoke("m", "P", 0.0)


async def test_replay_mode_serves_cached_completions(llm, tmp_path):
    await llm.invoke("m", "P", 0.0)
    replay = CachedLLM(tmp_path / "cache", live=False)
    assert await replay.invoke("m", "P", 0.0) == "completion for: P"


async def test_tokens_and_cost_recorded_when_response_carries_them(
        tmp_path, monkeypatch):
    class Usage:
        prompt_tokens = 11
        completion_tokens = 7

    class Resp(str):
        usage = Usage()
        cost = 0.0042

    class RichLlm:
        def __init__(self, model, temperature, timeout=None):
            pass

        async def invoke(self, prompt):
            return Resp("rich completion")

    monkeypatch.setattr("forecasting_tools.GeneralLlm", RichLlm)
    llm = CachedLLM(tmp_path / "cache")
    assert await llm.invoke("m", "P", 0.0) == "rich completion"
    key = cache_key("m", "P", 0.0, 0)
    rec = json.loads((tmp_path / "cache" / f"{key}.json").read_text())
    assert rec["tokens_in"] == 11 and rec["tokens_out"] == 7
    assert rec["cost_usd"] == pytest.approx(0.0042)
