"""IBKR (iBind) Provider Module

Optional provider that uses Voyz/ibind to access IBKR Web API via OAuth 1.0a.
This module is imported by FinanceToolkit only when enforce_source == "IBKR".
It has no hard dependency on ibind; if ibind or OAuth config is missing, it
returns empty DataFrames so the caller can handle fallbacks.
"""

from __future__ import annotations

__docformat__ = "google"

import importlib
import os
from contextlib import suppress
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from financetoolkit import helpers
from financetoolkit.utilities import logger_model

try:
    import pandas_market_calendars as pmc  # type: ignore
except Exception:  # pragma: no cover
    pmc = None

try:
    from ibind.client import ibkr_utils as _ibkr_utils  # type: ignore
except Exception:  # pragma: no cover
    _ibkr_utils = None

logger = logger_model.get_logger()

# pylint: disable=too-many-arguments
def check_dependencies() -> tuple[object | None, object | None]:
    """Return optional dependencies as detected at import time: (pmc, _ibkr_utils)."""
    return pmc, _ibkr_utils


_EPOCH_MS_THRESHOLD = 1_000_000_000_000  # ms median threshold to detect epoch timestamps vs seconds
_CLASS_SUFFIX_MIN = 1  # min length of class suffix when normalizing tickers (e.g., BRK B)
_CLASS_SUFFIX_MAX = 3  # max length of class suffix
_CLASS_SUFFIX_MED = 2  # typical split length used to detect class suffix pattern
_MIN_REQUIRED_COLUMNS = 5  # minimum expected OHLCV-like columns in returned payload
_DATE8_LEN = 8  # length of date strings formatted as YYYYMMDD



def _ibind_available() -> bool:
    try:
        return importlib.util.find_spec("ibind") is not None
    except Exception:
        return False

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

def _gather_candidates_from_search(obj: Any) -> list[dict]:
    out: list[dict] = []
    def rec(x: Any):
        if isinstance(x, dict):
            # normalize dict with conid
            if x.get("conid"):
                out.append(x)
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)
    rec(obj)
    # deduplicate by conid preserving order
    seen = set()
    uniq: list[dict] = []
    for d in out:
        c = str(d.get("conid"))
        if c and c not in seen:
            seen.add(c)
            uniq.append(d)
    return uniq

_US_PRIMARY_EXCHANGES = ["NYSE", "NASDAQ", "ARCA", "BATS", "CBOE", "AMEX"]

def _exchange_rank(candidate: dict) -> tuple[int, int]:
    # primaryExchange first, then exchange. SMART is a router, lowest rank.
    prim = (candidate.get("primaryExchange") or candidate.get("primary_exchange") or "")
    exch = (candidate.get("exchange") or candidate.get("listingExchange") or "")
    def rank(name: str) -> int:
        n = (name or "").upper()
        if n in _US_PRIMARY_EXCHANGES:
            return 3
        if n == "SMART":
            return 1
        return 2 if n else 0
    return (rank(str(prim)), rank(str(exch)))

def _expected_trading_days(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> int:
    # Try exchange calendar for XNYS; fallback to pandas bdate_range
    try:
        if pmc is None:
            raise RuntimeError("pmc not available")
        cal = pmc.get_calendar("XNYS")
        sched = cal.schedule(start_date=start_dt.date(), end_date=end_dt.date())
        return int(len(sched))
    except Exception:
        return int(len(pd.bdate_range(start_dt, end_dt)))

# Scoring constants for exchange rank comparisons
_RANK_PRIMARY = 3
_RANK_EXCH_HIGH = 3
_RANK_EXCH_MED = 2

def _compute_coverage(bars: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> float:
    if bars.empty:
        return 0.0
    # bars indexed by PeriodIndex('D') or datetime; count unique trading days in window
    idx = bars.index
    dt_index = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else pd.DatetimeIndex(idx)
    # Normalize all to UTC naive for safe comparison
    if getattr(dt_index, "tz", None) is not None:
        dt_index = dt_index.tz_convert("UTC").tz_localize(None)
    sd = start_dt.tz_convert("UTC").tz_localize(None) if getattr(start_dt, "tz", None) is not None else start_dt
    ed = end_dt.tz_convert("UTC").tz_localize(None) if getattr(end_dt, "tz", None) is not None else end_dt
    mask = (dt_index >= sd) & (dt_index <= ed)
    dt_index = dt_index[mask]
    covered = dt_index.normalize().unique().size
    expected = max(_expected_trading_days(sd, ed), 1)
    return min(1.0, covered / expected)

def _history_payload_to_df(payload: Any, freq: str = "D") -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    if isinstance(payload, dict):
        data = payload.get("data") or payload.get("points") or payload.get("bars")
    elif isinstance(payload, list):
        data = payload
    else:
        data = None
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    rename_map = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "adj": "Adj Close", "t": "Date"}
    df = df.rename(columns=rename_map)
    # Timestamp handling
    idx_col = "Date" if "Date" in df.columns else (
        "time" if "time" in df.columns else ("ts" if "ts" in df.columns else None)
    )
    if not idx_col:
        return pd.DataFrame()
    ser = df[idx_col]
    if np.issubdtype(ser.dtype, np.number):
        unit = "ms"
        with suppress(Exception):
            unit = "ms" if float(pd.Series(ser).median()) > _EPOCH_MS_THRESHOLD else "s"
        df[idx_col] = pd.to_datetime(ser, unit=unit, utc=False, errors="coerce")
    else:
        df[idx_col] = pd.to_datetime(ser, utc=False, errors="coerce")
    df = df.set_index(idx_col).sort_index()
    df.index = _mk_period_index(df.index, freq=freq)
    # Ensure Adj Close present; default to Close when absent
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df

def _fetch_history(client, conid: str, period: str = "1y") -> dict | None:
    """Fetch raw history payload from IBKR API."""
    res = client.marketdata_history_by_conid(conid=str(conid), bar="1d", period=period)
    return getattr(res, "data", None)


def _extract_metadata(payload: dict | None) -> dict:
    """Extract metadata from history payload."""
    meta = {}
    if isinstance(payload, dict):
        # capture some metadata if present
        keys = (
            "mktDataDelay",
            "primaryExchange",
            "exchange",
            "listingExchange",
            "symbol",
            "text",
            "error",
            "message",
        )
        for k in keys:
            if k in payload:
                meta[k] = payload[k]
        # detect permission errors in known fields
        msg = str(payload.get("text") or payload.get("error") or payload.get("message") or "").lower()
        if "permission" in msg:
            meta["no_permission"] = True
    return meta


def _normalize_bars(payload: dict | None) -> pd.DataFrame:
    """Normalize payload to OHLCV DataFrame."""
    return _history_payload_to_df(payload)


def _probe_candidate_history(client, conid: str, period: str = "1y") -> tuple[pd.DataFrame, dict]:
    try:
        payload = _fetch_history(client, conid, period)
        meta = _extract_metadata(payload)
        df = _normalize_bars(payload)
        return df, meta
    except Exception as e:
        logger.debug("_probe_candidate_history failed for conid %s: %s", conid, e)
        return pd.DataFrame(), {}

def _enrich_candidates_via_secdef(client, conids: list[str]) -> list[dict]:
    if not conids:
        return []
    try:
        res = client.security_definition_by_conid(conids)
        data = getattr(res, "data", None)
        out: list[dict] = []
        if isinstance(data, dict) and isinstance(data.get("secdef"), list):
            for item in data["secdef"]:
                if isinstance(item, dict) and item.get("conid"):
                    d = {
                        "conid": item.get("conid"),
                        "currency": item.get("currency"),
                        "primaryExchange": item.get("primaryExchange") or item.get("primary_exchange"),
                        "exchange": item.get("exchange") or item.get("listingExchange"),
                        "listingExchange": item.get("listingExchange"),
                        "symbol": item.get("symbol") or item.get("localSymbol"),
                        "localSymbol": item.get("localSymbol"),
                        "tradingClass": item.get("tradingClass"),
                        "secType": item.get("secType")
                    }
                    out.append(d)
        return out
    except Exception:
        return []

def _normalize_symbol_for_ib(sym: str) -> str:
    s = sym.strip()
    # Replace common class separators with space (e.g., BRK.B -> BRK B, BRK-B -> BRK B, BRK/B -> BRK B)
    for ch in (".", "-", "/", ":"):
        parts = s.split(ch)
        if len(parts) == _CLASS_SUFFIX_MED and _CLASS_SUFFIX_MIN <= len(parts[1]) <= _CLASS_SUFFIX_MAX:
            s = " ".join(parts)
            break
    return s

def _is_adr(cand: dict) -> bool:
    s = (cand.get("localSymbol") or cand.get("tradingClass") or cand.get("symbol") or "")
    return "ADR" in str(s).upper()

def resolve_conid_simple(client, ticker: str) -> list[dict]:
    """Return a small list of candidate dicts (enriched when possible) for a ticker.
    - StockQuery path first when ibkr_utils is available
    - Fallback to search_contract_by_symbol with sec_type in [None, 'STK', 'IND'] once
    No probing, no internal caching/backoff.
    """
    sym = ticker.strip()
    if sym.startswith("^"):
        sym = sym[1:]
    sym = _normalize_symbol_for_ib(sym)

    conids: list[str] = []
    try:
        if _ibkr_utils is not None:
            q = _ibkr_utils.StockQuery(symbol=sym)
            res = client.stock_conid_by_symbol([q], default_filtering=True)
            data = getattr(res, "data", None)
            if isinstance(data, dict):
                for _, v in data.items():
                    if isinstance(v, (int, str)):
                        conids.append(str(v))
            elif isinstance(data, list):
                conids = [str(x) for x in data]
    except Exception as e:
        logger.debug("resolve_conid_simple search fallback failed for %s: %s", sym, e)
        ...

    if not conids:
        # attempt generic search first, then typed fallbacks
        try:
            res = client.search_contract_by_symbol(symbol=sym)
            cand = _gather_candidates_from_search(getattr(res, "data", None))
            conids = [str(d.get("conid")) for d in cand if d.get("conid")]
        except Exception as e:
            logger.debug("resolve_conid_simple generic search failed for %s: %s", sym, e)
            conids = []
        for sec_type in ("STK", "IND"):
            if conids:
                break
            try:
                res2 = client.search_contract_by_symbol(symbol=sym, sec_type=sec_type)
                cand2 = _gather_candidates_from_search(getattr(res2, "data", None))
                conids = [str(d.get("conid")) for d in cand2 if d.get("conid")]
            except Exception as e:
                logger.debug("resolve_conid_simple typed search failed for %s sec_type=%s: %s", sym, sec_type, e)
                continue

    # Dedup small set
    conids = [c for i, c in enumerate(conids) if c and c not in conids[:i]]
    if not conids:
        return []

    # Enrich minimal fields needed for scoring
    cands = _enrich_candidates_via_secdef(client, conids[:10]) or [{"conid": c} for c in conids[:5]]
    return cands


def score_candidates(candidates: list[dict]) -> list[tuple[float, dict]]:
    """Pure scoring: US-first, USD-first, primaryExchange over SMART, small ADR bonus."""
    def exch_name(d: dict) -> str:
        return str(
            d.get("primaryExchange")
            or d.get("primary_exchange")
            or d.get("exchange")
            or d.get("listingExchange")
            or ""
        )

    def currency(d: dict) -> str:
        return str(d.get("currency") or "").upper()

    out: list[tuple[float, dict]] = []
    for d in candidates:
        prim, exch = _exchange_rank(d)
        score = 0.0
        # Prefer primary US exchanges
        score += 1.0 if prim == _RANK_PRIMARY else 0.0
        # Prefer known US exchanges and penalize SMART router as sole exchange
        score += 0.3 if exch == _RANK_EXCH_HIGH else (0.1 if exch == _RANK_EXCH_MED else 0.0)
        # Prefer USD
        score += 0.3 if currency(d) == "USD" else 0.0
        # ADR slight bonus as final tie-breaker when already US/USD
        is_adr = _is_adr(d)
        score += 0.05 if is_adr and prim == _RANK_PRIMARY and currency(d) == "USD" else 0.0
        out.append((score, d))
    # Highest first
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def pick_best_conid(candidates: list[dict]) -> str | None:
    ranked = score_candidates(candidates)
    if not ranked:
        return None
    return str(ranked[0][1].get("conid")) if ranked[0][1].get("conid") else None


def _resolve_best_conid(client, ticker: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> str | None:
    """Compatibility wrapper: use simplified resolution without probing/backoff."""
    cands = resolve_conid_simple(client, ticker)
    return pick_best_conid(cands)

def _resolve_conid_for_symbol(client, ticker: str) -> str | None:
    """Resolve a ticker to best IBKR conid using probing and US defaults.

    - No hard filtering on secType; we rely on coverage/exchange/currency scoring.
    - Caret prefix is stripped only for the query symbol but indices are not forced.
    """
    # Use a broad resolver that probes chartability and ranks candidates.
    end_dt = pd.Timestamp.utcnow().normalize()
    start_dt = end_dt - pd.Timedelta(days=365)
    try:
        return _resolve_best_conid(client, ticker, start_dt, end_dt)
    except Exception:
        return None

def get_intraday_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1min",
    return_column: str = "Close",
    sleep_timer: bool = True,
) -> pd.DataFrame:
    """Intraday OHLCV via IBKR/iBind if available.

    Returns PeriodIndex with freq 'min' or 'h' depending on interval, columns:
    [Open, High, Low, Close, Adj Close, Volume]. Empty DataFrame when unavailable.
    """
    if not _ibind_available() or not _oauth_configured():
        return pd.DataFrame()

    # Map Toolkit intervals to IBKR bars
    interval_map = {
        "1min": ("1min", "min"),
        "5min": ("5min", "min"),
        "15min": ("15min", "min"),
        "30min": ("30min", "min"),
        "1hour": ("1h", "h"),
        "4hour": ("4h", "h"),
    }
    if interval not in interval_map:
        return pd.DataFrame()

    bar, freq = interval_map[interval]

    # Default window per provider constraints
    try:
        end_dt = pd.to_datetime(end) if end is not None else pd.Timestamp.utcnow()
        if start is not None:
            start_dt = pd.to_datetime(start)
            if start_dt > end_dt:
                return pd.DataFrame()
        else:
            # Short default window for intraday
            start_dt = end_dt - pd.Timedelta(days=5)
        period = "5d"
    except Exception as e:
        logger.debug("get_intraday_data date parsing failed for %s: %s", ticker, e)
        return pd.DataFrame()

    try:
        from ibind import IbkrClient  # type: ignore  # noqa: PLC0415
        client = IbkrClient(use_oauth=True)
        conid = _resolve_conid_for_symbol(client, ticker)
        if not conid:
            return pd.DataFrame()
        res = client.marketdata_history_by_conid(conid=str(conid), bar=bar, period=period)
        payload = getattr(res, "data", None)
        df = _history_payload_to_df(payload, freq=freq)
        if df.empty:
            return df
        # Ensure columns present
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        df = df[cols]
        # Trim to requested window if provided
        if start is not None or end is not None:
            ts = df.index.to_timestamp()
            mask = (ts >= start_dt) & (ts <= end_dt)
            df = df[mask]
        return df
    except Exception as e:
        logger.debug("get_intraday_data failed for %s: %s", ticker, e)
        return pd.DataFrame()

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
    use_cached_data: bool | str = False,
    cached_data_location: str = "cached",
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
    # Prefer supported WebAPI periods (e.g., 1w, 1m, 3m, 6m, 1y, 5y, 10y, ytd)
    period = "1y"

    try:
        from ibind import IbkrClient  # type: ignore  # noqa: PLC0415

        from financetoolkit.utilities import cache_model  # noqa: E402, PLC0415

        client = IbkrClient(use_oauth=True)
        # Resolve conid (supports indices like TNX/VIX and equities)
        conid = _resolve_conid_for_symbol(client, ticker)
        if not conid:
            return pd.DataFrame()

        # Caching: load if enabled
        cache_key = f"ibkr_daily_{ticker.upper()}_{period}_{bar}.pickle"
        if use_cached_data:
            cached = cache_model.load_cached_data(
                cached_data_location=cached_data_location if isinstance(use_cached_data, str) else "cached",
                file_name=cache_key,
                method="pandas",
                return_empty_type=pd.DataFrame(),
            )
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                df = cached
                # Enrich and return
                df = helpers.enrich_historical_data(
                    historical_data=df,
                    start=start,
                    end=end,
                    return_column=return_column,
                    risk_free_rate=risk_free_rate,
                )
                return df

        # Fetch history by conid
        result = client.marketdata_history_by_conid(
            conid=str(conid), period=period, bar=bar
        )
        payload: Any = getattr(result, "data", None)
        if not payload:
            return pd.DataFrame()

        # iBind/IBKR returns a dict with a nested 'data' list for history
        if isinstance(payload, dict):
            data = payload.get("data") or payload.get("points") or payload.get("bars")
        elif isinstance(payload, list):
            data = payload
        else:
            data = None
        if not isinstance(data, list) or not data:
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

        # Build index: attempt epoch ms/s handling if numeric
        idx_col = "Date" if "Date" in df.columns else None
        if idx_col is None:
            for cand in ("time", "ts"):
                if cand in df.columns:
                    idx_col = cand
                    break
        if idx_col is None:
            return pd.DataFrame()

        ser = df[idx_col]
        if np.issubdtype(ser.dtype, np.number):
            try:
                unit = "ms" if float(pd.Series(ser).median()) > _EPOCH_MS_THRESHOLD else "s"
            except Exception:
                unit = "ms"
            df[idx_col] = pd.to_datetime(ser, unit=unit, utc=False, errors="coerce")
        else:
            df[idx_col] = pd.to_datetime(ser, utc=False, errors="coerce")
        df = df.set_index(idx_col).sort_index()
        df.index = _mk_period_index(df.index, freq="D")

        # Keep required columns
        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        present = [c for c in cols if c in df.columns]
        if len(present) < _MIN_REQUIRED_COLUMNS:  # require at least OHLCV; Adj Close may be synthesized
            return pd.DataFrame()
        df = df[[c for c in cols if c in df.columns]]

        if divide_ohlc_by:
            np.seterr(divide="ignore", invalid="ignore")
            df = df.div(divide_ohlc_by)

        # Include dividends column if available from API; else set to 0 if requested
        if include_dividends and "Dividends" not in df.columns:
            df["Dividends"] = 0.0

        df = df.loc[~df.index.duplicated(keep="first")]

        # Save raw before enrichment if caching is enabled
        try:
            if use_cached_data:
                from financetoolkit.utilities import cache_model  # noqa: E402, PLC0415
                cache_model.save_cached_data(
                    cached_data=df,
                    cached_data_location=cached_data_location if isinstance(use_cached_data, str) else "cached",
                    file_name=cache_key,
                    method="pandas",
                    include_message=False,
                )
        except Exception as e:  # noqa: S110
            logger.debug('ibind: suppressed non-fatal exception: %s', e)

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
        from ibind import IbkrClient  # type: ignore  # noqa: PLC0415

        client = IbkrClient(use_oauth=True)
        # Lookup contract by symbol to obtain currency/exchange
        # Normalize caret-prefix if present
        sym = ticker[1:] if ticker.startswith("^") else ticker
        # Try stock first, then index
        try:
            res = client.stock_conid_by_symbol(sym)
            data = getattr(res, "data", None)
            if isinstance(data, dict):
                stats.loc["Symbol"] = data.get("symbol") or sym
                stats.loc["Currency"] = data.get("currency")
                stats.loc["Exchange Name"] = data.get("exchange") or data.get("listingExchange")
                stats.loc["Instrument Type"] = data.get("secType") or "STK"
                return stats
        except Exception as e:  # noqa: S110
            logger.debug('ibind: suppressed non-fatal exception: %s', e)
        try:
            res = client.search_contract_by_symbol(symbol=sym, sec_type="IND")
            data = getattr(res, "data", None)
            # Best-effort extraction of basic fields if present
            if isinstance(data, dict):
                stats.loc["Symbol"] = data.get("symbol") or sym
            return stats
        except Exception:
            return stats
    except Exception as e:  # pragma: no cover
        logger.warning("IBKR/iBind stats fetch failed for %s: %s", ticker, e)
        return stats

# -------------------------
# Options (Expiries, Chains)
# -------------------------

def _get_ibkr_client():
    if not _ibind_available() or not _oauth_configured():
        return None
    try:  # pragma: no cover - runtime dependency
        from ibind import IbkrClient  # type: ignore  # noqa: PLC0415
        client = IbkrClient()
        # Best-effort: ensure brokerage session
        with suppress(Exception):
            client.initialize_brokerage_session()
        return client
    except Exception:
        return None

def _yyyy_mm_dd(date_str: str) -> str:
    try:
        return pd.to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception:
        return ""

def _yyyymmdd(date_str: str) -> str:
    try:
        return pd.to_datetime(date_str).strftime("%Y%m%d")
    except Exception:
        return ""

def _mon_yy(date_str: str) -> str:
    try:
        ts = pd.to_datetime(date_str)
        return ts.strftime("%b%y").upper()
    except Exception:
        return ""

def get_option_expiry_dates(ticker: str) -> list[str]:
    """Return available option expiration dates for a ticker via IBKR/iBind.

    Returns list of YYYY-MM-DD. Empty list when unavailable or not configured.
    """
    client = _get_ibkr_client()
    if client is None:
        return []

    # Resolve underlying conid using recent window
    end_dt = pd.Timestamp.utcnow().normalize()
    start_dt = end_dt - pd.Timedelta(days=30)
    try:
        conid = _resolve_best_conid(client, ticker, start_dt, end_dt)
    except Exception:
        conid = None
    if not conid:
        return []

    # Prefer strikes endpoint that often returns expirations list
    try:
        res = client.search_strikes_by_conid(conid=str(conid), sec_type="OPT", exchange="SMART")
        data = getattr(res, "data", None)
        expirations = []
        if isinstance(data, dict):
            # Common CPAPI shape: { "strikes": [...], "expirations": ["YYYYMMDD", ...] }
            raw = data.get("expirations") or data.get("expirationDates")
            if isinstance(raw, list):
                expirations = [_yyyy_mm_dd(str(x)) if len(str(x)) == _DATE8_LEN else _yyyy_mm_dd(x) for x in raw]
        # Deduplicate and sort ascending
        expirations = sorted({d for d in expirations if d})
        if expirations:
            return expirations
    except Exception as e:  # noqa: S110
            logger.debug('ibind: suppressed non-fatal exception: %s', e)

    # Fallback: try secdef info for a few nearby months and collect maturityDate
    outs: set[str] = set()
    try:
        base = pd.Timestamp.utcnow().normalize().to_period("M").start_time
        for add in range(0, 8):  # next ~8 months
            m = (base + pd.DateOffset(months=add)).strftime("%b%y").upper()
            try:
                res2 = client.search_secdef_info_by_conid(
                    conid=str(conid), sec_type="OPT", month=m, exchange="SMART"
                )
                data2 = getattr(res2, "data", None)
                if isinstance(data2, list):
                    for item in data2:
                        md = item.get("maturityDate") or item.get("lastTradingDay")
                        if md:
                            # md may be YYYYMMDD
                            ds = _yyyy_mm_dd(str(md)) if len(str(md)) == _DATE8_LEN else _yyyy_mm_dd(str(md))
                            if ds:
                                outs.add(ds)
            except Exception:  # noqa: S112
                continue
    except Exception as e:  # noqa: S110
            logger.debug('ibind: suppressed non-fatal exception: %s', e)

    return sorted(outs)

def get_option_chains(
    tickers: list[str],
    expiration_date: str,
    put_option: bool = False,
) -> pd.DataFrame:
    """Fetch option chains for tickers at a given expiration via IBKR/iBind.

    Returns a DataFrame indexed by (Ticker, Strike Price) with columns matching
    Toolkit expectations. Empty on error or when not configured.
    """
    client = _get_ibkr_client()
    if client is None:
        return pd.DataFrame()

    mon_code = _mon_yy(expiration_date)  # e.g., AUG25
    expiry_yyyymmdd = _yyyymmdd(expiration_date)  # e.g., 20250816
    right = "P" if put_option else "C"

    fields = [
        "31",   # last_price
        "84",   # bid_price
        "86",   # ask_price
        "87",   # volume
        "7638", # option_open_interest
        "82",   # change
        "83",   # change_percent
        "7714", # last_trading_date
        "7633", # implied_vol_percent
        "55",   # symbol
    ]

    result_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        # Resolve underlying conid
        end_dt = pd.Timestamp.utcnow().normalize()
        start_dt = end_dt - pd.Timedelta(days=30)
        try:
            under_conid = _resolve_best_conid(client, ticker, start_dt, end_dt)
        except Exception:
            under_conid = None
        if not under_conid:
            continue

        # Obtain strikes set for expiration
        strikes: list[float] = []
        try:
            res = client.search_strikes_by_conid(
                conid=str(under_conid), sec_type="OPT", month=mon_code or None, exchange="SMART"
            )
            data = getattr(res, "data", None)
            if isinstance(data, dict):
                strikes_raw = data.get("strikes") or []
                if isinstance(strikes_raw, list):
                    strikes = [float(s) for s in strikes_raw if s is not None]
        except Exception:
            # Try without month hint
            try:
                res = client.search_strikes_by_conid(conid=str(under_conid), sec_type="OPT", exchange="SMART")
                data = getattr(res, "data", None)
                if isinstance(data, dict):
                    strikes_raw = data.get("strikes") or []
                    if isinstance(strikes_raw, list):
                        strikes = [float(s) for s in strikes_raw if s is not None]
            except Exception:
                strikes = []

        if not strikes:
            continue

        # For each strike, resolve option conid for the specified right and month
        contracts: list[dict] = []
        for strike in strikes:
            try:
                res2 = client.search_secdef_info_by_conid(
                    conid=str(under_conid),
                    sec_type="OPT",
                    month=mon_code or None,
                    exchange="SMART",
                    strike=str(strike),
                    right=right,
                )
                data2 = getattr(res2, "data", None)
                if isinstance(data2, list) and data2:
                    # Take the first match per strike/right
                    item = data2[0]
                    oc = {
                        "option_conid": str(item.get("conid")),
                        "strike": float(strike),
                        "currency": item.get("currency"),
                        "expiration": expiry_yyyymmdd,
                    }
                    contracts.append(oc)
            except Exception:  # noqa: S112
                continue

        # Snapshot market data for all option conids in batches
        rows: list[dict] = []
        for i in range(0, len(contracts), 50):
            batch = contracts[i : i + 50]
            conids = [c["option_conid"] for c in batch if c.get("option_conid")]
            if not conids:
                continue
            try:
                md = client.live_marketdata_snapshot(conids=conids, fields=fields)
                payload = getattr(md, "data", None)
                if not isinstance(payload, list):
                    continue
                # Map each returned conid to its contract
                by_conid = {c["option_conid"]: c for c in batch}
                for entry in payload:
                    # entry fields typically include 'conid'
                    econid = str(entry.get("conid")) if isinstance(entry, dict) else None
                    meta = by_conid.get(econid, {})
                    if not econid:
                        continue
                    row = {
                        "Contract Symbol": str(entry.get("symbol") or econid),
                        "Strike": meta.get("strike"),
                        "Currency": meta.get("currency"),
                        "Last Price": entry.get("last_price"),
                        "Change": entry.get("change"),
                        "Percent Change": entry.get("change_percent"),
                        "Volume": entry.get("volume"),
                        "Open Interest": entry.get("option_open_interest"),
                        "Bid": entry.get("bid_price"),
                        "Ask": entry.get("ask_price"),
                        "Expiration": _yyyy_mm_dd(meta.get("expiration", "")),
                        "Last Trade Date": _yyyy_mm_dd(str(entry.get("last_trading_date", ""))),
                        "Implied Volatility": entry.get("implied_vol_percent"),
                        # In The Money computed later, requires underlying
                    }
                    row["_econid"] = econid
                    rows.append(row)
            except Exception:  # noqa: S112
                continue

        if not rows:
            continue

        # Compute In The Money using underlying last price
        try:
            und_md = client.live_marketdata_snapshot(conids=[str(under_conid)], fields=["31"])  # last_price
            und_price = None
            pdata = getattr(und_md, "data", None)
            if isinstance(pdata, list) and pdata:
                und_price = pdata[0].get("last_price")
        except Exception:
            und_price = None

        for row in rows:
            strike_val = row.get("Strike")
            if und_price is not None and strike_val is not None:
                row["In The Money"] = bool((und_price > strike_val) if right == "C" else (und_price < strike_val))
            else:
                row["In The Money"] = None

        df = pd.DataFrame(rows)
        if df.empty:
            continue
        # Set index to Strike and multiindex with ticker
        df = df.drop(columns=[c for c in ["_econid"] if c in df.columns])
        df = df.set_index("Strike", drop=True)
        df.index.name = "Strike Price"
        df.insert(0, "Ticker", ticker)
        df = df.set_index("Ticker", append=True)
        df = df.reorder_levels(["Ticker", "Strike Price"]).sort_index()
        result_frames.append(df)

    if not result_frames:
        return pd.DataFrame()

    final = pd.concat(result_frames, axis=0)
    # Ensure columns ordering (match yfinance normalization as much as possible)
    cols = [
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
    ]
    # Insert Strike via index; already in index name
    # Reindex to include all columns (missing will be added as NaN)
    final = final.reindex(columns=cols)
    return final

