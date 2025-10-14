import pandas as pd
import numpy as np
import types
import financetoolkit.historical_model as hist


def _mk_intraday_df(points=5, freq="min"):
    idx = pd.period_range(end=pd.Timestamp("2025-01-10 16:00"), periods=points, freq=freq)
    return pd.DataFrame(
        {
            "Open": np.linspace(100.0, 100.0 + 0.1 * (points - 1), points),
            "High": np.linspace(100.1, 100.2, points),
            "Low": np.linspace(99.9, 100.0, points),
            "Close": np.linspace(100.0, 100.1, points),
            "Adj Close": np.linspace(100.0, 100.1, points),
            "Volume": np.arange(points, dtype=float),
        },
        index=idx,
    )


def test_intraday_ibkr_first(monkeypatch):
    # IBKR returns data; ensure FMP isn't used
    ib_df = _mk_intraday_df(points=8, freq="min")

    # Patch ibind intraday to return data
    import financetoolkit.ibind_model as ib
    monkeypatch.setattr(ib, "get_intraday_data", lambda **kwargs: ib_df)

    # Patch FMP intraday to raise if called
    import financetoolkit.fmp_model as fmp
    def _no_call(**kwargs):
        raise AssertionError("FMP intraday should not be called when IBKR returns data")
    monkeypatch.setattr(fmp, "get_intraday_data", _no_call)

    data, no_data = hist.get_historical_data(
        tickers=["AAPL"],
        api_key="DUMMY",
        enforce_source=None,
        start="2024-01-01",
        end="2024-01-10",
        interval="1min",
        progress_bar=False,
        show_errors=False,
    )

    assert no_data == []
    # Ensure ticker present and required OHLCV columns exist
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        assert (col, "AAPL") in data.columns


def test_intraday_fallback_to_fmp_on_error(monkeypatch):
    # IBKR raises; FMP returns data
    import financetoolkit.ibind_model as ib
    monkeypatch.setattr(ib, "get_intraday_data", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("perm denied")))

    fmp_df = _mk_intraday_df(points=6, freq="min")
    import financetoolkit.fmp_model as fmp
    monkeypatch.setattr(fmp, "get_intraday_data", lambda **kwargs: fmp_df)

    data, no_data = hist.get_historical_data(
        tickers=["AAPL"],
        api_key="DUMMY",
        enforce_source=None,
        start="2024-01-01",
        end="2024-01-10",
        interval="1min",
        progress_bar=False,
        show_errors=False,
    )

    assert no_data == []
    # Ensure it used FMP path: columns present for ticker
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        assert (col, "AAPL") in data.columns
