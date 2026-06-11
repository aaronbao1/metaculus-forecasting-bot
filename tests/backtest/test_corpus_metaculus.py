from datetime import datetime, timezone

import pytest

import backtest.corpus.metaculus as metaculus_mod
from backtest.corpus.metaculus import fetch_metaculus
from backtest.corpus.store import CorpusStore

LIST_PAGE = {
    "next": None,
    "results": [{"id": 31678, "title": "Will X happen before 2027?"},
                {"id": 31999, "title": "Locked-down question?"}],
}

# Full-visibility detail (own-forecast question: token sees everything)
DETAIL_FULL = {
    "id": 31678,
    "title": "Will X happen before 2027?",
    "status": "resolved",
    "question": {
        "type": "binary",
        "title": "Will X happen before 2027?",
        "description": "Background paragraph.",
        "resolution_criteria": "Resolves YES if X occurs before 2027-01-01.",
        "fine_print": "Standard fine print.",
        "open_time": "2026-04-01T00:00:00Z",
        "scheduled_close_time": "2026-06-01T00:00:00Z",
        "spot_scoring_time": "2026-05-15T00:00:00Z",
        "actual_resolve_time": "2026-06-02T12:00:00Z",
        "resolution": "yes",
        "question_weight": 0.75,
        "aggregations": {"recency_weighted": {"history": [
            {"start_time": 1775000000.0, "end_time": 1775500000.0,
             "centers": [0.40], "means": [0.41]},
            {"start_time": 1775500000.0, "end_time": None,
             "centers": [0.55]},
        ]}},
    },
}

# Degraded detail (post-lockdown plain-token tier: nulls are WITHHELD)
DETAIL_DEGRADED = {
    "id": 31999,
    "title": "Locked-down question?",
    "status": "resolved",
    "question": {
        "type": "binary",
        "description": None,
        "resolution_criteria": "",
        "open_time": "2026-01-01T00:00:00Z",
        "scheduled_close_time": "2026-03-01T00:00:00Z",
        "actual_resolve_time": "2026-03-02T00:00:00Z",
        "resolution": None,
        "aggregations": {"recency_weighted": {"history": None}},
    },
}

# Multiple-choice detail: per-option CP values keyed by option name
DETAIL_MC = {
    "id": 32500,
    "title": "Who wins the 2026 race?",
    "status": "resolved",
    "question": {
        "type": "multiple_choice",
        "title": "Who wins the 2026 race?",
        "description": "Background.",
        "resolution_criteria": "Resolves to the winner.",
        "open_time": "2026-01-01T00:00:00Z",
        "scheduled_close_time": "2026-06-01T00:00:00Z",
        "actual_resolve_time": "2026-06-02T00:00:00Z",
        "resolution": "Alice",
        "options": ["Alice", "Bob"],
        "aggregations": {"recency_weighted": {"history": [
            {"start_time": 1775000000.0, "end_time": None,
             "centers": [0.3, 0.7]},
        ]}},
    },
}

# Resolved BEFORE open -> quarantined (resolution-date validation, spec §3.4)
DETAIL_BAD_DATES = {
    "id": 31700,
    "title": "Resolved before it opened?",
    "status": "resolved",
    "question": {
        "type": "binary",
        "description": "Background.",
        "open_time": "2026-04-01T00:00:00Z",
        "scheduled_close_time": "2026-06-01T00:00:00Z",
        "actual_resolve_time": "2026-03-01T00:00:00Z",   # before open
        "resolution": "yes",
    },
}


@pytest.fixture
def no_throttle(monkeypatch):
    calls = []

    async def fake_throttle():
        calls.append(1)

    monkeypatch.setattr(metaculus_mod, "_throttle", fake_throttle)
    return calls


@pytest.fixture
def fake_api(monkeypatch):
    seen = {"headers": [], "params": []}

    async def fake_get(url, params=None, headers=None, client=None):
        seen["headers"].append(headers)
        seen["params"].append(params)
        if url.endswith("/posts/"):
            return LIST_PAGE
        if url.endswith("/posts/31678/"):
            return DETAIL_FULL
        if url.endswith("/posts/31999/"):
            return DETAIL_DEGRADED
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    return seen


async def test_fetch_ingests_full_and_degraded(tmp_path, monkeypatch,
                                               fake_api, no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store, tournaments=["minibench"])
    assert report.fetched == 2
    assert report.withheld == {"description": 1, "resolution": 1,
                               "aggregations": 1}
    assert fake_api["headers"][0] == {"Authorization": "Token test-token"}
    assert fake_api["params"][0]["tournaments"] == "minibench"
    assert fake_api["params"][0]["statuses"] == "resolved"

    full = store.get_question("metaculus:31678")
    assert full.resolution is True            # "yes" -> bool
    assert full.question_weight == 0.75
    assert full.provenance["withheld"] == []
    assert full.open_time == datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert full.spot_scoring_time == datetime(2026, 5, 15,
                                              tzinfo=timezone.utc)

    degraded = store.get_question("metaculus:31999")
    assert degraded.resolution is None        # withheld, NOT resolved-no
    assert degraded.description == ""
    assert set(degraded.provenance["withheld"]) == \
        {"description", "resolution", "aggregations"}
    assert no_throttle, "self-throttle must run between detail GETs"


async def test_cp_history_flattened_to_step_points(tmp_path, monkeypatch,
                                                   fake_api, no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    store = CorpusStore(tmp_path)
    await fetch_metaculus(store)
    # interval entry start_time <= t < end_time, flattened to
    # (start_time, value) step points; "centers" preferred over "means"
    t_mid = datetime.fromtimestamp(1775200000.0, tz=timezone.utc)
    assert store.value_at("metaculus:31678", "cp", t_mid) == 0.40
    t_late = datetime.fromtimestamp(1775600000.0, tz=timezone.utc)
    assert store.value_at("metaculus:31678", "cp", t_late) == 0.55
    t_early = datetime.fromtimestamp(1774000000.0, tz=timezone.utc)
    assert store.value_at("metaculus:31678", "cp", t_early) is None


async def test_mc_cp_values_stored_as_option_dict(tmp_path, monkeypatch,
                                                  no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/"):
            return {"next": None,
                    "results": [{"id": 32500, "title": "Who wins?"}]}
        if url.endswith("/posts/32500/"):
            return DETAIL_MC
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store)
    assert report.fetched == 1
    # MC CP entries are stored as option -> probability dicts
    t = datetime.fromtimestamp(1775500000.0, tz=timezone.utc)
    assert store.value_at("metaculus:32500", "cp", t) == \
        {"Alice": 0.3, "Bob": 0.7}


async def test_resolve_before_open_is_quarantined(tmp_path, monkeypatch,
                                                  no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/"):
            return {"next": None,
                    "results": [{"id": 31700, "title": "Bad dates?"}]}
        if url.endswith("/posts/31700/"):
            return DETAIL_BAD_DATES
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store)
    assert report.fetched == 0
    assert any("quarantined metaculus:31700" in e for e in report.errors)
    assert store.get_question("metaculus:31700") is None


async def test_missing_token_raises(tmp_path, monkeypatch, fake_api):
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        await fetch_metaculus(CorpusStore(tmp_path))


async def test_detail_failure_is_isolated(tmp_path, monkeypatch,
                                          no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    async def flaky_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/"):
            return LIST_PAGE
        if url.endswith("/posts/31678/"):
            return DETAIL_FULL
        raise RuntimeError("503 from Metaculus")

    monkeypatch.setattr(metaculus_mod, "_get_json", flaky_get)
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store)
    assert report.fetched == 1
    assert any("31999" in e for e in report.errors)
    assert store.get_question("metaculus:31678") is not None


async def test_limit_caps_detail_fetches(tmp_path, monkeypatch, fake_api,
                                         no_throttle):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    report = await fetch_metaculus(CorpusStore(tmp_path), limit=1)
    assert report.fetched == 1


# ---------------------------------------------------------------------------
# Fix 1: CP gap handling — sentinel (end_time, None) so value_at returns
# None inside a gap between intervals and after a terminated history.
# History: interval A 100→150, gap 150–300, interval B 300→null.
# ---------------------------------------------------------------------------

DETAIL_GAPPED_CP = {
    "id": 40000,
    "title": "Gapped CP question",
    "status": "resolved",
    "question": {
        "type": "binary",
        "description": "Background.",
        "resolution_criteria": "Resolves YES if ...",
        "open_time": "2020-01-01T00:00:00Z",
        "scheduled_close_time": "2030-01-01T00:00:00Z",
        "actual_resolve_time": "2029-01-01T00:00:00Z",
        "resolution": "yes",
        "aggregations": {"recency_weighted": {"history": [
            # interval A: [100, 150)
            {"start_time": 100.0, "end_time": 150.0, "centers": [0.30]},
            # gap: 150–300 (no interval covers this range)
            # interval B: [300, ∞)
            {"start_time": 300.0, "end_time": None, "centers": [0.70]},
        ]}},
    },
}


async def test_cp_gap_returns_none_inside_gap(tmp_path, monkeypatch,
                                              no_throttle):
    """value_at in a gap between terminated and next interval returns None."""
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/"):
            return {"next": None,
                    "results": [{"id": 40000, "title": "Gapped CP question"}]}
        if url.endswith("/posts/40000/"):
            return DETAIL_GAPPED_CP
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store)
    assert report.fetched == 1

    from datetime import datetime, timezone

    # Inside interval A → value present
    t_in_a = datetime.fromtimestamp(120.0, tz=timezone.utc)
    assert store.value_at("metaculus:40000", "cp", t_in_a) == 0.30

    # Inside the gap (after A ended, before B started) → None
    t_gap = datetime.fromtimestamp(200.0, tz=timezone.utc)
    assert store.value_at("metaculus:40000", "cp", t_gap) is None

    # Inside interval B → value present
    t_in_b = datetime.fromtimestamp(350.0, tz=timezone.utc)
    assert store.value_at("metaculus:40000", "cp", t_in_b) == 0.70


# ---------------------------------------------------------------------------
# Fix 2: scaling normalization — _ingest_detail must normalise raw API fields
# to the BacktestQuestion.scaling contract keys.
# ---------------------------------------------------------------------------

DETAIL_NUMERIC = {
    "id": 50000,
    "title": "Numeric question",
    "status": "resolved",
    "question": {
        "type": "numeric",
        "description": "Background.",
        "resolution_criteria": "Resolves to the measured value.",
        "open_time": "2026-01-01T00:00:00Z",
        "scheduled_close_time": "2026-12-01T00:00:00Z",
        "actual_resolve_time": "2026-12-02T00:00:00Z",
        "resolution": "42.5",
        "scaling": {
            "range_min": 0.0,
            "range_max": 100.0,
            "zero_point": None,
        },
        "open_lower_bound": False,
        "open_upper_bound": True,
        "inbound_outcome_count": 101,
        "unit": "percent",
        "aggregations": {"recency_weighted": {"history": []}},
    },
}


async def test_scaling_normalized_to_contract_keys(tmp_path, monkeypatch,
                                                   no_throttle):
    """Numeric question scaling stored under the BacktestQuestion contract keys."""
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/"):
            return {"next": None,
                    "results": [{"id": 50000, "title": "Numeric question"}]}
        if url.endswith("/posts/50000/"):
            return DETAIL_NUMERIC
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = await fetch_metaculus(store)
    assert report.fetched == 1

    q = store.get_question("metaculus:50000")
    assert q is not None
    sc = q.scaling
    assert sc is not None

    # Exactly the contract keys — no extras, none missing
    expected_keys = {
        "range_min", "range_max", "zero_point",
        "open_lower_bound", "open_upper_bound",
        "cdf_size", "unit",
    }
    assert set(sc.keys()) == expected_keys

    # Values mapped correctly from the fixture
    assert sc["range_min"] == 0.0
    assert sc["range_max"] == 100.0
    assert sc["zero_point"] is None
    assert sc["open_lower_bound"] is False
    assert sc["open_upper_bound"] is True
    assert sc["cdf_size"] == 101
    assert sc["unit"] == "percent"


async def test_scaling_none_for_binary_question(tmp_path, monkeypatch,
                                                fake_api, no_throttle):
    """Binary questions must store scaling=None (not the raw dict)."""
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    store = CorpusStore(tmp_path)
    await fetch_metaculus(store, limit=1)
    q = store.get_question("metaculus:31678")
    assert q is not None
    assert q.scaling is None


async def test_scaling_cdf_size_fallback(tmp_path, monkeypatch, no_throttle):
    """When inbound_outcome_count is absent, cdf_size defaults to 201."""
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    import copy
    detail = copy.deepcopy(DETAIL_NUMERIC)
    del detail["question"]["inbound_outcome_count"]

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/"):
            return {"next": None,
                    "results": [{"id": 50000, "title": "Numeric question"}]}
        if url.endswith("/posts/50000/"):
            return detail
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    await fetch_metaculus(store)
    q = store.get_question("metaculus:50000")
    assert q.scaling["cdf_size"] == 201
