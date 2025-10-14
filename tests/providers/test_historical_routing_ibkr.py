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


def test_enforce_source_ibkr_calls_provider(monkeypatch):
    # Monkeypatch provider to return a valid dataset regardless of ibind availability
    df = _mk_daily_df(10)
    monkeypatch.setattr(ib, "get_historical_data", lambda **kwargs: df)

    data, no_data = hist.get_historical_data(
        tickers=["AAPL"],
        enforce_source="IBKR",
        start="2024-01-01",
        end="2024-01-31",
        interval="1d",
        progress_bar=False,
        show_errors=False,
    )
    assert "AAPL" in data.columns.get_level_values(1)
    # Ensure the required columns exist for the ticker
    # Columns are (Field, Ticker) in the historical_model combined frame
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        assert (col, "AAPL") in data.columns
    assert no_data == []
