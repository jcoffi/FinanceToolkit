import pandas as pd
import numpy as np
import financetoolkit.historical_model as hist
import financetoolkit.ibind_model as ib


def _mk_daily_df(days=5):
    idx = pd.period_range(end=pd.Timestamp("2024-02-01"), periods=days, freq="D")
    return pd.DataFrame(
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


def test_default_uses_ibkr_when_configured(monkeypatch):
    # Simulate configured OAuth via env and ibind available
    monkeypatch.setenv("IBIND_USE_OAUTH", "1")
    monkeypatch.setenv("IBIND_OAUTH1A_CONSUMER_KEY", "x")
    monkeypatch.setenv("IBIND_OAUTH1A_ACCESS_TOKEN", "y")
    monkeypatch.setenv("IBIND_OAUTH1A_ACCESS_TOKEN_SECRET", "z")

    # Make ibind_model return non-empty to be chosen first
    df = _mk_daily_df(10)
    monkeypatch.setattr(ib, "get_historical_data", lambda **kwargs: df)

    # Make FMP return empty to ensure not accidentally used
    import financetoolkit.fmp_model as fmp
    monkeypatch.setattr(fmp, "get_historical_data", lambda **kwargs: pd.DataFrame())

    data, no_data = hist.get_historical_data(
        tickers=["AAPL"],
        api_key="DUMMY",
        enforce_source=None,
        start="2024-01-01",
        end="2024-01-31",
        interval="1d",
        progress_bar=False,
        show_errors=False,
    )

    assert no_data == []
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        assert (col, "AAPL") in data.columns
