import types
import pandas as pd


def test_ibkr_options_puts(monkeypatch):
    # Configure environment as if OAuth is set
    monkeypatch.setenv("IBIND_USE_OAUTH", "1")
    monkeypatch.setenv("IBIND_OAUTH1A_ACCESS_TOKEN", "x")
    monkeypatch.setenv("IBIND_OAUTH1A_ACCESS_TOKEN_SECRET", "y")
    monkeypatch.setenv("IBIND_OAUTH1A_CONSUMER_KEY", "z")

    import financetoolkit.options.options_model as om

    fake_ib = types.SimpleNamespace()

    def fake_expiries(ticker: str):
        return ["2025-12-19"]

    # Return a puts chain when put_option=True is requested upstream
    def fake_chain(tickers, expiration_date, put_option=False):
        assert put_option is True  # ensure puts branch tested
        idx = pd.MultiIndex.from_product([[tickers[0]], [105.0]], names=["Ticker", "Strike Price"])
        df = pd.DataFrame(
            {
                "Contract Symbol": ["AAPL251219P00105000"],
                "Currency": ["USD"],
                "Last Price": [2.34],
                "Change": [0.02],
                "Percent Change": [0.9],
                "Volume": [15],
                "Open Interest": [120],
                "Bid": [2.30],
                "Ask": [2.40],
                "Expiration": ["2025-12-19"],
                "Last Trade Date": ["2025-10-11"],
                "Implied Volatility": [0.27],
                "In The Money": [False],
            },
            index=idx,
        )
        return df

    fake_ib.get_option_expiry_dates = fake_expiries
    fake_ib.get_option_chains = fake_chain

    monkeypatch.setattr(om, "ibind_model", fake_ib, raising=False)
    monkeypatch.setattr(om, "_ENABLE_IBKR", True, raising=False)

    expiries = om.get_option_expiry_dates("AAPL", enforce_source=None)
    assert expiries == ["2025-12-19"]

    df = om.get_option_chains(["AAPL"], expiration_date="2025-12-19", put_option=True, enforce_source=None)
    assert not df.empty
    assert df.index.names == ["Ticker", "Strike Price"]
    assert (df["Contract Symbol"].iloc[0]).endswith("P00105000")


def test_ibkr_options_multi_tickers(monkeypatch):
    # Configure environment as if OAuth is set
    monkeypatch.setenv("IBIND_USE_OAUTH", "1")
    monkeypatch.setenv("IBIND_OAUTH1A_ACCESS_TOKEN", "x")
    monkeypatch.setenv("IBIND_OAUTH1A_ACCESS_TOKEN_SECRET", "y")
    monkeypatch.setenv("IBIND_OAUTH1A_CONSUMER_KEY", "z")

    import financetoolkit.options.options_model as om

    fake_ib = types.SimpleNamespace()

    def fake_expiries(ticker: str):
        return ["2025-12-19"]

    def fake_chain(tickers, expiration_date, put_option=False):
        # Produce rows for multiple tickers
        frames = []
        for t in tickers:
            idx = pd.MultiIndex.from_product([[t], [100.0]], names=["Ticker", "Strike Price"])
            df = pd.DataFrame(
                {
                    "Contract Symbol": [f"{t}251219C00100000"],
                    "Currency": ["USD"],
                    "Last Price": [1.23],
                    "Change": [0.01],
                    "Percent Change": [0.8],
                    "Volume": [10],
                    "Open Interest": [100],
                    "Bid": [1.20],
                    "Ask": [1.25],
                    "Expiration": ["2025-12-19"],
                    "Last Trade Date": ["2025-10-10"],
                    "Implied Volatility": [0.25],
                    "In The Money": [False],
                },
                index=idx,
            )
            frames.append(df)
        return pd.concat(frames)

    fake_ib.get_option_expiry_dates = fake_expiries
    fake_ib.get_option_chains = fake_chain

    monkeypatch.setattr(om, "ibind_model", fake_ib, raising=False)
    monkeypatch.setattr(om, "_ENABLE_IBKR", True, raising=False)

    expiries = om.get_option_expiry_dates("AAPL", enforce_source=None)
    assert expiries == ["2025-12-19"]

    df = om.get_option_chains(["AAPL", "MSFT"], expiration_date="2025-12-19", put_option=False, enforce_source=None)
    assert not df.empty
    assert set(df.index.get_level_values(0)) == {"AAPL", "MSFT"}
    assert df.index.names == ["Ticker", "Strike Price"]
