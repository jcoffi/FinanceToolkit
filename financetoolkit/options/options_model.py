"""Options Model"""

import pandas as pd
import yfinance as yf

# Optional IBKR/iBind provider for options
try:  # pragma: no cover - import guard
    from financetoolkit import ibind_model
    _ENABLE_IBKR = True
except Exception:  # noqa: BLE001
    ibind_model = None  # type: ignore
    _ENABLE_IBKR = False


def get_option_expiry_dates(ticker: str, enforce_source: str | None = None) -> list[str]:
    """
    Retrieve available option expiry dates for a given ticker symbol.

    Routing rules:
    - enforce_source == "IBKR": try IBKR/iBind only
    - enforce_source == "YahooFinance": use yfinance only
    - enforce_source is None: IBKR first (if available/configured), then fallback to yfinance

    Returns a list of 'YYYY-MM-DD' strings. Empty list when unavailable.
    """
    # IBKR first when requested or by default
    if (enforce_source in (None, "IBKR")) and _ENABLE_IBKR:
        try:
            if hasattr(ibind_model, "get_option_expiry_dates"):
                expiries = ibind_model.get_option_expiry_dates(ticker)
                if expiries:
                    return expiries
        except Exception:  # noqa: S110
            ...  # fall through to Yahoo

    # Yahoo fallback or enforced
    if enforce_source in (None, "YahooFinance"):
        try:
            return yf.Ticker(ticker).options
        except Exception:
            return []

    return []


def _normalize_yf_options_df(options_df: pd.DataFrame) -> pd.DataFrame:
    options_df = options_df.rename(
        columns={
            "contractSymbol": "Contract Symbol",
            "strike": "Strike",
            "currency": "Currency",
            "lastPrice": "Last Price",
            "change": "Change",
            "percentChange": "Percent Change",
            "volume": "Volume",
            "openInterest": "Open Interest",
            "bid": "Bid",
            "ask": "Ask",
            "contractSize": "Contract Size",
            "expiration": "Expiration",
            "lastTradeDate": "Last Trade Date",
            "impliedVolatility": "Implied Volatility",
            "inTheMoney": "In The Money",
        }
    )
    if "Contract Size" in options_df.columns:
        options_df = options_df.drop(columns="Contract Size")
    options_df = options_df.set_index("Strike")
    return options_df


def get_option_chains(
    tickers: list[str],
    expiration_date: str,
    put_option: bool = False,
    enforce_source: str | None = None,
) -> pd.DataFrame:
    """
    Retrieve option chains (calls or puts) for a list of tickers and a specific expiration date.

    Routing rules:
    - enforce_source == "IBKR": try IBKR/iBind only
    - enforce_source == "YahooFinance": use yfinance only
    - enforce_source is None: IBKR first (if available/configured), then fallback to yfinance

    Returns a concatenated DataFrame indexed by (Ticker, Strike Price).
    """
    result_dict: dict[str, pd.DataFrame] = {}

    # Attempt IBKR/iBind provider
    if (enforce_source in (None, "IBKR")) and _ENABLE_IBKR:
        try:
            if hasattr(ibind_model, "get_option_chains"):
                ib_df = ibind_model.get_option_chains(
                    tickers=tickers,
                    expiration_date=expiration_date,
                    put_option=put_option,
                )
                if isinstance(ib_df, pd.DataFrame) and not ib_df.empty:
                    return ib_df
        except Exception:  # noqa: S110
            ...  # Fall through to Yahoo

    # Fallback to Yahoo Finance
    if enforce_source in (None, "YahooFinance"):
        for ticker in tickers:
            chain = yf.Ticker(ticker).option_chain(expiration_date)
            options_df = chain.puts if put_option else chain.calls
            options_df = _normalize_yf_options_df(options_df)
            result_dict[ticker] = options_df

        result_final = pd.concat(result_dict)
        if "Last Trade Date" in result_final.columns:
            result_final["Last Trade Date"] = pd.to_datetime(
                result_final["Last Trade Date"], unit="s", errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        result_final.index.names = ["Ticker", "Strike Price"]
        return result_final

    # No data
    return pd.DataFrame()
