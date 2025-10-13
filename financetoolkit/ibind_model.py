"""IBKR (iBind) Provider Module

Optional provider that uses Voyz/ibind to access IBKR Web API via OAuth 1.0a.
This module is imported by FinanceToolkit only when enforce_source == "IBKR".
It has no hard dependency on ibind; if ibind or OAuth config is missing, it
returns empty DataFrames so the caller can handle fallbacks.
"""

__docformat__ = "google"

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from financetoolkit import helpers
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# pylint: disable=too-many-arguments


def _ibind_available() -> bool:
    try:
        import ibind  # noqa: F401
    except Exception:  # pragma: no cover - optional dep
        return False
    return True


def _oauth_configured() -> bool:
    # Minimal set for OAuth 1.0a in iBind; library may accept more variants.
    required_env = [
        "IBIND_USE_OAUTH",
        "IBIND_OAUTH1A_ACCESS_TOKEN",
        "IBIND_OAUTH1A_ACCESS_TOKEN_SECRET",
        "IBIND_OAUTH1A_CONSUMER_KEY",
    ]
    return all(os.environ.get(k) for k in required_env)


def _mk_period_index(idx: pd.Index, freq: str = "D") -> pd.PeriodIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx)
        except Exception:  # pragma: no cover
            idx = pd.to_datetime([])
    with pd.option_context("mode.chained_assignment", None):
        return idx.to_period(freq=freq)


def get_historical_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    risk_free_rate: pd.DataFrame = pd.DataFrame(),
    include_dividends: bool = True,
    divide_ohlc_by: int | float | None = None,
    sleep_timer: bool = True,  # kept for signature parity
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """Fetch historical OHLCV via IBKR using iBind if available.

    Returns a DataFrame with PeriodIndex(freq='D') and columns:
    [Open, High, Low, Close, Adj Close, Volume, (optional) Dividends],
    enriched with helpers.enrich_historical_data.
    """
    if interval in ["yearly", "quarterly"]:
        interval = "1d"

    if not _ibind_available() or not _oauth_configured():
        # Silent no-op to allow other sources/fallbacks to work
        return pd.DataFrame()

    # Compute extended window like other providers to stabilize returns
    if end is not None:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    else:
        end_dt = datetime.today()
        end = end_dt.strftime("%Y-%m-%d")

    if start is not None:
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
        except ValueError:
            return pd.DataFrame()
    else:
        # default lookback 10y to align with toolkit behavior
        start_dt = end_dt.replace(year=end_dt.year - 10)
        start = start_dt.strftime("%Y-%m-%d")

    bar = interval if interval in {"1min", "5min", "15min", "30min", "1h", "4h", "1d"} else "1d"
    # Map daily/weekly/monthly/yearly to acceptable iBind duration strings
    # We keep it simple and request full window with daily bars.
    period = "1000d"

    try:
        from ibind import IbkrClient  # type: ignore

        client = IbkrClient(use_oauth=True)
        # Ensure brokerage session is available for market data
        # If iBind is configured to init brokerage during oauth_init, this is a no-op
        if hasattr(client, "initialize_brokerage_session"):
            try:
                client.initialize_brokerage_session()
            except Exception:  # pragma: no cover
                pass

        # iBind convenience: directly by symbol
        # Some environments require conid lookup; iBind abstracts this.
        result = client.marketdata_history_by_symbol(
            ticker, period=period, bar=bar
        )
        data: Any = getattr(result, "data", None)
        if not data:
            return pd.DataFrame()

        # Expected shape from iBind: list of dicts with fields like t, o, h, l, c, v
        df = pd.DataFrame(data)
        # Common field names per ibind examples
        rename_map = {
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
            "adj": "Adj Close",
            "t": "Date",
        }
        df = df.rename(columns=rename_map)

        # If Adj Close not provided, mirror Close
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"].to_numpy()

        # Build index
        idx_col = "Date" if "Date" in df.columns else None
        if idx_col is None:
            # Try 'time' or 'ts'
            for cand in ("time", "ts"):
                if cand in df.columns:
                    idx_col = cand
                    break
        if idx_col is None:
            return pd.DataFrame()

        df[idx_col] = pd.to_datetime(df[idx_col], utc=False, errors="coerce")
        df = df.set_index(idx_col).sort_index()
        df.index = _mk_period_index(df.index, freq="D")

        # Keep required columns
        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        present = [c for c in cols if c in df.columns]
        if len(present) < 5:  # require at least OHLCV; Adj Close may be synthesized
            return pd.DataFrame()
        df = df[[c for c in cols if c in df.columns]]

        if divide_ohlc_by:
            np.seterr(divide="ignore", invalid="ignore")
            df = df.div(divide_ohlc_by)

        # Include dividends column if available from API; else set to 0 if requested
        if include_dividends and "Dividends" not in df.columns:
            df["Dividends"] = 0.0

        df = df.loc[~df.index.duplicated(keep="first")]

        df = helpers.enrich_historical_data(
            historical_data=df,
            start=start,
            end=end,
            return_column=return_column,
            risk_free_rate=risk_free_rate,
        )

        return df

    except Exception as e:  # pragma: no cover - robust optional integration
        logger.warning("IBKR/iBind historical fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()


def get_historical_statistics(ticker: str) -> pd.Series:
    """Return basic instrument statistics schema as Series.

    IBKR endpoints do not expose a 1:1 mapping for all fields used in Toolkit.
    We attempt to fill what is commonly available and return NaN for others to
    keep schema compatibility.
    """
    index = [
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
    stats = pd.Series([np.nan] * len(index), index=index, dtype=object)

    if not _ibind_available() or not _oauth_configured():
        return stats

    try:
        from ibind import IbkrClient  # type: ignore

        client = IbkrClient(use_oauth=True)
        # Lookup contract by symbol to obtain currency/exchange
        con = client.stock_conid_by_symbol(ticker)
        data = getattr(con, "data", None)
        if isinstance(data, dict):
            # Common keys in IBKR responses
            stats.loc["Symbol"] = data.get("symbol") or ticker
            stats.loc["Currency"] = data.get("currency")
            stats.loc["Exchange Name"] = data.get("exchange") or data.get("listingExchange")
            stats.loc["Instrument Type"] = data.get("secType") or "STK"
            # Dates: leave as NaN if not readily available
            # Timezone information typically not exposed directly
        return stats
    except Exception as e:  # pragma: no cover
        logger.warning("IBKR/iBind stats fetch failed for %s: %s", ticker, e)
        return stats
