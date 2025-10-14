import types
import pandas as pd


def test_ibkr_permission_denied_fallback(monkeypatch):
    # Simulate OAuth configured and ibind_model present
    monkeypatch.setenv("IBIND_USE_OAUTH", "1")
    monkeypatch.setenv("IBIND_OAUTH1A_ACCESS_TOKEN", "x")
    monkeypatch.setenv("IBIND_OAUTH1A_ACCESS_TOKEN_SECRET", "y")
    monkeypatch.setenv("IBIND_OAUTH1A_CONSUMER_KEY", "z")

    import financetoolkit.options.options_model as om

    # Fake ibind_model that returns empty to simulate permission denied/no data
    fake_ib = types.SimpleNamespace()

    def fake_expiries(ticker: str):
        return []  # empty => forces fallback

    def fake_chain(tickers, expiration_date, put_option=False):
        return pd.DataFrame()  # empty => forces fallback

    fake_ib.get_option_expiry_dates = fake_expiries
    fake_ib.get_option_chains = fake_chain

    # Monkeypatch options_model to route to our fake
    monkeypatch.setattr(om, "ibind_model", fake_ib, raising=False)
    monkeypatch.setattr(om, "_ENABLE_IBKR", True, raising=False)

    # Monkeypatch yfinance Ticker to return deterministic options
    class DummyChain:
        def __init__(self):
            import pandas as pd
            self.calls = pd.DataFrame(
                {
                    "contractSymbol": ["AAPL251219C00100000"],
                    "strike": [100.0],
                    "currency": ["USD"],
                    "lastPrice": [1.23],
                    "change": [0.01],
                    "percentChange": [0.8],
                    "volume": [10],
                    "openInterest": [100],
                    "bid": [1.20],
                    "ask": [1.25],
                    "contractSize": ["REG"],
                    "expiration": ["2025-12-19"],
                    "lastTradeDate": [1697040000],
                    "impliedVolatility": [0.25],
                    "inTheMoney": [False],
                }
            )
            self.puts = self.calls.copy()

    class DummyTicker:
        def __init__(self, ticker):
            self._ticker = ticker
            self.options = ["2025-12-19"]

        def option_chain(self, expiration_date):
            return DummyChain()

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", DummyTicker)

    # With enforce_source=None, IBKR path is tried first, but since it returns empty, code falls back to Yahoo
    expiries = om.get_option_expiry_dates("AAPL", enforce_source=None)
    assert expiries == ["2025-12-19"]

    df = om.get_option_chains(["AAPL"], expiration_date="2025-12-19", put_option=False, enforce_source=None)
    assert not df.empty
    assert df.index.names == ["Ticker", "Strike Price"]


def test_ibkr_schema_columns_types(monkeypatch):
    import financetoolkit.options.options_model as om

    # Provide a fake ibind_model that returns a full DataFrame with correct schema
    fake_ib = types.SimpleNamespace()

    def fake_expiries(ticker: str):
        return ["2025-12-19"]

    def fake_chain(tickers, expiration_date, put_option=False):
        idx = pd.MultiIndex.from_product([["AAPL"], [100.0]], names=["Ticker", "Strike Price"])
        df = pd.DataFrame(
            {
                "Contract Symbol": ["AAPL251219C00100000"],
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
        return df

    fake_ib.get_option_expiry_dates = fake_expiries
    fake_ib.get_option_chains = fake_chain

    monkeypatch.setattr(om, "ibind_model", fake_ib, raising=False)
    monkeypatch.setattr(om, "_ENABLE_IBKR", True, raising=False)

    expiries = om.get_option_expiry_dates("AAPL", enforce_source=None)
    assert expiries == ["2025-12-19"]

    df = om.get_option_chains(["AAPL"], expiration_date="2025-12-19", put_option=False, enforce_source=None)

    expected_cols = {
        "Contract Symbol",
        "Currency",
        "Last Price",
        "Change",
        "Percent Change",
        "Volume",
        "Open Interest",
        "Bid",
        "Ask",
        "Expiration",
        "Last Trade Date",
        "Implied Volatility",
        "In The Money",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df.index.names == ["Ticker", "Strike Price"]
