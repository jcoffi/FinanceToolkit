import numpy as np
import pandas as pd

import financetoolkit.ibind_model as ib


def _mk_daily_df(days=10, end=None):
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


def test_probe_backoff_skips_problematic_conid(monkeypatch):
    class Dummy:
        pass
    client = Dummy()

    # Two candidates; conid 1 is backoffed, should skip to 2
    cand_list = [
        {"conid": 1, "currency": "USD", "exchange": "NYSE", "primaryExchange": "NYSE", "secType": "STK"},
        {"conid": 2, "currency": "USD", "exchange": "NASDAQ", "primaryExchange": "NASDAQ", "secType": "STK"},
    ]

    class DummyRes:
        def __init__(self, data):
            self.data = data

    client.search_contract_by_symbol = lambda symbol, sec_type=None: DummyRes(cand_list)

    monkeypatch.setattr(ib, "_gather_candidates_from_search", lambda obj: cand_list)
    monkeypatch.setattr(ib, "_enrich_candidates_via_secdef", lambda c, ids: cand_list)

    # Backoff removed in deterministic resolver; ensure deterministic pick honors US primaries
    start = pd.Timestamp("2024-12-01", tz="UTC")
    end = pd.Timestamp("2025-01-15", tz="UTC")
    best = ib._resolve_best_conid(client, "AAPL", start, end)
    assert str(best) == "1"  # NYSE preferred when both are valid

