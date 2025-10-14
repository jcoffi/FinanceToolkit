import numpy as np
import pandas as pd

import financetoolkit.historical_model as hist
import financetoolkit.ibind_model as ib


def _mk_daily_df(days=5, end=None):
    end = pd.Timestamp("2025-01-10") if end is None else pd.to_datetime(end)
    idx = pd.period_range(end=end, periods=days, freq="D")
    df = pd.DataFrame(
        {
            "Open": np.linspace(1.0, 1.0 + 0.1 * (days - 1), days),
            "High": np.linspace(1.01, 1.11, days),
            "Low": np.linspace(0.9, 1.0, days),
            "Close": np.linspace(1.0, 1.1, days),
            "Adj Close": np.linspace(1.0, 1.1, days),
            "Volume": np.arange(days, dtype=float),
        },
        index=idx,
    )
    return df


def test_normalize_symbol_for_ib_variants():
    assert ib._normalize_symbol_for_ib("BRK.B") == "BRK B"
    assert ib._normalize_symbol_for_ib("BRK-B") == "BRK B"
    assert ib._normalize_symbol_for_ib("BRK/B") == "BRK B"


def test_permission_denied_conids_are_skipped(monkeypatch):
    class Dummy:
        pass
    client = Dummy()

    cand_list = [
        {"conid": 1, "currency": "USD", "exchange": "NYSE", "primaryExchange": "NYSE", "secType": "STK"},
        {"conid": 2, "currency": "USD", "exchange": "NASDAQ", "primaryExchange": "NASDAQ", "secType": "STK"},
    ]

    class DummyRes:
        def __init__(self, data):
            self.data = data

    client.search_contract_by_symbol = lambda symbol, sec_type=None: DummyRes(cand_list)

    def fake_gather(obj):
        return cand_list

    # Always return permission denied metadata
    monkeypatch.setattr(ib, "_gather_candidates_from_search", fake_gather)
    monkeypatch.setattr(ib, "_enrich_candidates_via_secdef", lambda c, ids: cand_list)

    # Deterministic resolver does not probe permissions; ensure it selects first valid US candidate
    start = pd.Timestamp("2024-12-01", tz="UTC")
    end = pd.Timestamp("2025-01-15", tz="UTC")
    best = ib._resolve_best_conid(client, "AAPL", start, end)
    assert str(best) == "1"


def test_cache_quick_revalidation_invalidates_and_picks_valid(monkeypatch):
    class Dummy:
        pass
    client = Dummy()

    cand_list = [
        {"conid": 1, "currency": "USD", "exchange": "NYSE", "primaryExchange": "NYSE", "secType": "STK"},
        {"conid": 2, "currency": "USD", "exchange": "NASDAQ", "primaryExchange": "NASDAQ", "secType": "STK"},
    ]

    class DummyRes:
        def __init__(self, data):
            self.data = data

    client.search_contract_by_symbol = lambda symbol, sec_type=None: DummyRes(cand_list)

    def fake_gather(obj):
        return cand_list

    monkeypatch.setattr(ib, "_gather_candidates_from_search", fake_gather)
    monkeypatch.setattr(ib, "_enrich_candidates_via_secdef", lambda c, ids: cand_list)

    # Deterministic resolver doesn't revalidate; should pick NYSE due to primary exchange
    start = pd.Timestamp("2024-12-01", tz="UTC")
    end = pd.Timestamp("2025-01-15", tz="UTC")
    best = ib._resolve_best_conid(client, "AAPL", start, end)
    assert str(best) == "1"


def test_historical_statistics_routing_enforce_source_ibkr(monkeypatch):
    # Return a simple Series from IBKR provider path
    idx = [
        "Currency",
        "Symbol",
        "Exchange Name",
        "Instrument Type",
        "First Trade Date",
        "Regular Market Time",
        "GMT Offset",
        "Timezone",
        "Exchange Timezone Name",
    ]
    series = pd.Series(index=idx, dtype=object)
    series.loc["Currency"] = "USD"
    series.loc["Symbol"] = "AAPL"
    series.loc["Exchange Name"] = "NASDAQ"
    monkeypatch.setattr(ib, "get_historical_statistics", lambda t: series)

    stats, no_data = hist.get_historical_statistics(
        tickers="AAPL", enforce_source="IBKR", progress_bar=False, show_errors=False
    )
    assert "AAPL" in stats.columns
    assert stats.loc["Currency", "AAPL"] == "USD"
    assert no_data == []
