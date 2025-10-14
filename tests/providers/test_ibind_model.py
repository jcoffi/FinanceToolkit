import pandas as pd
import numpy as np
import types

import financetoolkit.ibind_model as ib


def _mk_daily_df(days=5, start=None):
    end = pd.Timestamp("2025-01-10") if start is None else start + pd.Timedelta(days=days)
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


def test_compute_coverage_periodindex():
    df = _mk_daily_df(20)
    start = df.index.to_timestamp().min()
    end = df.index.to_timestamp().max()
    cov = ib._compute_coverage(df, start, end)
    assert 0.9 <= cov <= 1.0


def test_history_payload_to_df_variants():
    # dict with data list
    payload = {"data": [
        {"t": "2025-01-05", "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.05, "v": 10},
        {"t": "2025-01-06", "o": 1.06, "h": 1.12, "l": 0.95, "c": 1.08, "v": 12},
    ]}
    df = ib._history_payload_to_df(payload)
    assert not df.empty
    assert set(["Open","High","Low","Close","Adj Close","Volume"]).issubset(df.columns)
    assert isinstance(df.index, pd.PeriodIndex) and df.index.freqstr == 'D'

    # list directly
    payload2 = [
        {"t": "2025-01-05", "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.05, "v": 10},
        {"t": "2025-01-06", "o": 1.06, "h": 1.12, "l": 0.95, "c": 1.08, "v": 12},
    ]
    df2 = ib._history_payload_to_df(payload2)
    assert not df2.empty
    assert isinstance(df2.index, pd.PeriodIndex)


def test_resolve_best_conid_scoring_prefers_coverage(monkeypatch):
    # Simulate candidates with different coverage
    class Dummy:
        pass
    client = Dummy()

    # Monkeypatch candidate gathering to fixed set
    cand_list = [
        {"conid": 1, "currency": "USD", "exchange": "NASDAQ", "listingExchange": "NASDAQ", "primaryExchange": "NASDAQ", "secType": "STK"},
        {"conid": 2, "currency": "USD", "exchange": "NYSE", "listingExchange": "NYSE", "primaryExchange": "NYSE", "secType": "STK"},
    ]

    def fake_gather(obj):
        return cand_list

    class DummyRes:
        def __init__(self, data):
            self.data = data
    # Provide search_contract_by_symbol so fallback path yields our candidates
    client.search_contract_by_symbol = lambda symbol, sec_type=None: DummyRes(cand_list)


    # conid 1 has fewer bars, conid 2 has more bars
    def fake_probe(client, conid, period):
        if str(conid) == '1':
            return _mk_daily_df(5), {"mktDataDelay": 0}
        return _mk_daily_df(20), {"mktDataDelay": 0}

    monkeypatch.setattr(ib, "_gather_candidates_from_search", fake_gather)
    monkeypatch.setattr(ib, "_enrich_candidates_via_secdef", lambda c, ids: cand_list)
    monkeypatch.setattr(ib, "_probe_candidate_history", fake_probe)

    start = pd.Timestamp("2024-12-01").tz_localize("UTC")
    end = pd.Timestamp("2025-01-15").tz_localize("UTC")
    best = ib._resolve_best_conid(client, "AAPL", start, end)
    assert str(best) == "2"
