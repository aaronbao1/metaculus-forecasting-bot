"""Task 9: FRED fetcher — mocked HTTP seam, no network."""
import asyncio

from backtest.corpus import fred as fred_mod
from backtest.corpus.store import CorpusStore


def _payloads(sid):
    return {
        fred_mod.FRED_SERIES_URL: {
            "seriess": [{"id": sid, "title": "Consumer Price Index",
                         "units": "Index 1982-1984=100"}]},
        fred_mod.FRED_OBS_URL: {
            "observations": [
                {"date": "2026-01-01", "value": "310.5"},
                {"date": "2026-02-01", "value": "."},      # missing marker
                {"date": "2026-03-01", "value": "312.1"},
            ]},
    }


def test_fetch_fred_stores_question_and_series(tmp_path, monkeypatch):
    store = CorpusStore(tmp_path / "corpus")
    payloads = _payloads("CPIAUCSL")

    async def fake_get(url, params):
        assert params["api_key"] == "k"
        return payloads[url]

    monkeypatch.setattr(fred_mod, "_http_get_json", fake_get)
    report = asyncio.run(fred_mod.fetch_fred(store, ["CPIAUCSL"],
                                             api_key="k"))
    assert report.fetched == 1 and not report.errors
    q = store.get_question("fred:CPIAUCSL")
    assert q.title == "Consumer Price Index"
    pts = store.series_points("fred:CPIAUCSL", "target")
    assert [v for _, v in pts] == [310.5, 312.1]


def test_fetch_fred_without_key_reports_error(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    store = CorpusStore(tmp_path / "corpus")
    report = asyncio.run(fred_mod.fetch_fred(store, ["X"]))
    assert report.fetched == 0
    assert "FRED_API_KEY" in report.errors[0]
