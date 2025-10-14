from __future__ import annotations
"""IBKR (iBind) Provider Module

Optional provider that uses Voyz/ibind to access IBKR Web API via OAuth 1.0a.
This module is imported by FinanceToolkit only when enforce_source == "IBKR".
It has no hard dependency on ibind; if ibind or OAuth config is missing, it
returns empty DataFrames so the caller can handle fallbacks.
"""

__docformat__ = "google"

import os
import time
import random
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


# Simple TTL cache for conid resolution to reduce probing
_CONID_CACHE: dict[str, tuple[str, float]] = {}
_CONID_TTL_SECONDS = 7 * 24 * 3600


def _now_ts() -> float:
    try:
        return pd.Timestamp.utcnow().timestamp()
    except Exception:
        import time as _t
        return _t.time()


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
        import pandas_market_calendars as pmc  # type: ignore
        cal = pmc.get_calendar("XNYS")
        sched = cal.schedule(start_date=start_dt.date(), end_date=end_dt.date())
        return int(len(sched))
    except Exception:
        return int(len(pd.bdate_range(start_dt, end_dt)))


def _compute_coverage(bars: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> float:
    if bars.empty:
        return 0.0
    # bars indexed by PeriodIndex('D') or datetime; count unique trading days in window
    idx = bars.index
    if isinstance(idx, pd.PeriodIndex):
        dt_index = idx.to_timestamp()
    else:
        dt_index = pd.DatetimeIndex(idx)
    # Normalize all to UTC naive for safe comparison
    if getattr(dt_index, 'tz', None) is not None:
        dt_index = dt_index.tz_convert('UTC').tz_localize(None)
    sd = start_dt.tz_convert('UTC').tz_localize(None) if getattr(start_dt, 'tz', None) is not None else start_dt
    ed = end_dt.tz_convert('UTC').tz_localize(None) if getattr(end_dt, 'tz', None) is not None else end_dt
    mask = (dt_index >= sd) & (dt_index <= ed)
    dt_index = dt_index[mask]
    covered = dt_index.normalize().unique().size
    expected = max(_expected_trading_days(sd, ed), 1)
    return min(1.0, covered / expected)


def _history_payload_to_df(payload: Any) -> pd.DataFrame:
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
    idx_col = "Date" if "Date" in df.columns else ("time" if "time" in df.columns else ("ts" if "ts" in df.columns else None))
    if not idx_col:
        return pd.DataFrame()
    ser = df[idx_col]
    if np.issubdtype(ser.dtype, np.number):
        unit = "ms"
        try:
            unit = "ms" if float(pd.Series(ser).median()) > 1e12 else "s"
        except Exception:
            pass
        df[idx_col] = pd.to_datetime(ser, unit=unit, utc=False, errors="coerce")
    else:
        df[idx_col] = pd.to_datetime(ser, utc=False, errors="coerce")
    df = df.set_index(idx_col).sort_index()
    df.index = _mk_period_index(df.index, freq="D")
    # Ensure Adj Close present; default to Close when absent
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df


def _probe_candidate_history(client, conid: str, period: str = "1y") -> tuple[pd.DataFrame, dict]:
    try:
        res = client.marketdata_history_by_conid(conid=str(conid), bar="1d", period=period)
        payload = getattr(res, "data", None)
        meta = {}
        if isinstance(payload, dict):
            # capture some metadata if present
            for k in ("mktDataDelay", "primaryExchange", "exchange", "listingExchange", "symbol", "text", "error", "message"):
                if k in payload:
                    meta[k] = payload[k]
            # detect permission errors in known fields
            msg = str(payload.get("text") or payload.get("error") or payload.get("message") or "").lower()
            if "permission" in msg:
                meta["no_permission"] = True
        df = _history_payload_to_df(payload)
        return df, meta
    except Exception:
        return pd.DataFrame(), {}


def _enrich_candidates_via_secdef(client, conids: list[str]) -> list[dict]:
    if not conids:
        return []
    try:
        res = client.security_definition_by_conid(conids)
        data = getattr(res, 'data', None)
        out: list[dict] = []
        if isinstance(data, dict) and isinstance(data.get('secdef'), list):
            for item in data['secdef']:
                if isinstance(item, dict) and item.get('conid'):
                    d = {
                        'conid': item.get('conid'),
                        'currency': item.get('currency'),
                        'primaryExchange': item.get('primaryExchange') or item.get('primary_exchange'),
                        'exchange': item.get('exchange') or item.get('listingExchange'),
                        'listingExchange': item.get('listingExchange'),
                        'symbol': item.get('symbol') or item.get('localSymbol'),
                        'localSymbol': item.get('localSymbol'),
                        'tradingClass': item.get('tradingClass'),
                        'secType': item.get('secType')
                    }
                    out.append(d)
        return out
    except Exception:
        return []


def _resolve_best_conid(client, ticker: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> str | None:
    # Cache check (US scope cache key)
    key = f"US::{ticker.upper()}"
    ts_now = _now_ts()
    cached = _CONID_CACHE.get(key)
    if cached and (ts_now - cached[1]) < _CONID_TTL_SECONDS:
        return cached[0]

    sym = ticker.strip()
    if sym.startswith("^"):
        sym = sym[1:]

    # Primary path: get concrete conids via stock_conid_by_symbol
    conids: list[str] = []
    try:
        from ibind.client import ibkr_utils as _utils
        q = _utils.StockQuery(symbol=sym)
        res = client.stock_conid_by_symbol([q], default_filtering=True)
        data = getattr(res, 'data', None)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, str)):
                    conids.append(str(v))
        elif isinstance(data, list):
            conids = [str(x) for x in data]
    except Exception:
        pass

    # Fall back to generic contract search to scrape any conids we can find
    if not conids:
        try:
            res = client.search_contract_by_symbol(symbol=sym)
            cand = _gather_candidates_from_search(getattr(res, 'data', None))
            conids = [str(d.get('conid')) for d in cand if d.get('conid')]
        except Exception:
            pass
        if not conids:
            for sec_type in ('STK','IND'):
                try:
                    res = client.search_contract_by_symbol(symbol=sym, sec_type=sec_type)
                    cand = _gather_candidates_from_search(getattr(res, 'data', None))
                    conids += [str(d.get('conid')) for d in cand if d.get('conid')]
                except Exception:
                    pass
    # Dedup
    conids = [c for i, c in enumerate(conids) if c and c not in conids[:i]]
    if not conids:
        return None

    # Enrich with secdef to get primaryExchange/currency/tradingClass
    candidates = _enrich_candidates_via_secdef(client, conids[:10])
    if not candidates:
        # As an escape hatch, create minimal candidate dicts from conids
        candidates = [{'conid': c} for c in conids[:5]]

    # Stage-1 filters: prefer US primaryExchange and USD
    def exch_name(d: dict) -> str:
        return str(d.get("primaryExchange") or d.get("primary_exchange") or d.get("exchange") or d.get("listingExchange") or "")

    def currency(d: dict) -> str:
        return str(d.get("currency") or "").upper()

    def apply_filters(cands: list[dict], require_us_exch: bool, require_usd: bool) -> list[dict]:
        out: list[dict] = []
        for d in cands:
            ex = exch_name(d).upper()
            cur = currency(d)
            if require_us_exch and ex not in _US_PRIMARY_EXCHANGES:
                continue
            if require_usd and cur != "USD":
                continue
            out.append(d)
        return out

    for require_us_exch, require_usd in [(True, True), (False, True), (True, False), (False, False)]:
        filtered = apply_filters(candidates, require_us_exch, require_usd)
        if filtered:
            candidates = filtered
            break

    # Probe and score
    probe_period_long = "1y"
    probe_period_short = "1m"

    def _sectype_weight_for(tkr: str, cand: dict) -> float:
        st = (cand.get('secType') or '').upper()
        if not st:
            # Heuristic: caret implies index intent; else equity default
            if tkr.startswith('^'):
                st = 'IND'
            else:
                st = 'STK'
        if st == 'STK':
            return 1.0
        if st == 'ETF':
            return 0.95
        if st == 'IND':
            return 0.9
        if st == 'CFD':
            return 0.6
        if st in {'WAR', 'WARRANT', 'OPT', 'OPTION', 'FUT', 'FUTURE'}:
            return 0.4
        return 0.8

    scored: list[tuple[float, dict, pd.DataFrame, dict]] = []
    for d in candidates[:5]:
        conid = str(d.get('conid'))
        df_long, meta = _probe_candidate_history(client, conid, period=probe_period_long)
        # skip if no data or if permissions are missing for this conid
        if df_long.empty or (meta.get("no_permission") is True):
            # basic pacing/backoff to avoid hammering endpoints on repeated failures
            time.sleep(0.05 + random.random() * 0.1)
            continue
        df_short, _ = _probe_candidate_history(client, conid, period=probe_period_short)
        time.sleep(0.02 + random.random() * 0.05)
        last_dt = pd.Timestamp.utcnow().tz_convert('UTC').normalize()
        try:
            last_idx = df_long.index.to_timestamp()
            if getattr(last_idx, 'tz', None) is None:
                last_idx = last_idx.tz_localize('UTC')
            last_dt = last_idx.max().normalize()
        except Exception:
            pass
        # normalize start_dt to UTC naive for comparison
        sd = start_dt.tz_localize('UTC') if getattr(start_dt, 'tz', None) is None else start_dt.tz_convert('UTC')
        start_long = max(sd, last_dt - pd.Timedelta(days=365))
        start_short = max(sd, last_dt - pd.Timedelta(days=45))
        cov_long = _compute_coverage(df_long, start_long, last_dt)
        cov_short = _compute_coverage(df_short if not df_short.empty else df_long, start_short, last_dt)
        coverage_norm = 0.7 * cov_long + 0.3 * cov_short
        prim_rank, exch_rank = _exchange_rank(d)
        exch_rank_norm = 1.0 if prim_rank == 3 else (0.7 if exch_rank == 3 else (0.5 if exch_rank == 2 else (0.3 if exch_rank == 1 else 0.0)))
        currency_match = 1.0 if currency(d) == "USD" else 0.0
        try:
            age_days = max(0, (pd.Timestamp.now().normalize() - last_dt).days)
            recency_norm = max(0.0, min(1.0, 1.0 - age_days / 30.0))
        except Exception:
            recency_norm = 0.0
        delay = 0
        try:
            delay = int(meta.get("mktDataDelay", 0))
        except Exception:
            delay = 0
        delay_norm = min(1.0, max(0.0, delay / 15.0))
        stw = _sectype_weight_for(ticker if isinstance(ticker, str) else str(ticker), d)
        score = 1.00 * coverage_norm + 0.10 * (1.0 if prim_rank == 3 else 0.0) + 0.05 * exch_rank_norm + 0.05 * currency_match + 0.05 * recency_norm + 0.05 * stw - 0.05 * delay_norm
        scored.append((score, d, df_long, meta))

    if not scored:
        return None

    scored.sort(key=lambda x: (x[0], len(x[2])), reverse=True)
    best_conid = str(scored[0][1].get("conid"))
    _CONID_CACHE[key] = (best_conid, ts_now)
    return best_conid

    # Stage-1 filters: try US primary exchange and USD currency
    def exch_name(d: dict) -> str:
        return str(d.get("primaryExchange") or d.get("primary_exchange") or d.get("exchange") or d.get("listingExchange") or "")

    def currency(d: dict) -> str:
        return str(d.get("currency") or "").upper()

    def apply_filters(candidates: list[dict], require_us_exch: bool, require_usd: bool) -> list[dict]:
        out: list[dict] = []
        for d in candidates:
            ex = exch_name(d).upper()
            cur = currency(d)
            if require_us_exch and ex not in _US_PRIMARY_EXCHANGES:
                continue
            if require_usd and cur != "USD":
                continue
            out.append(d)
        return out

    candidates = cand
    for require_us_exch, require_usd in [(True, True), (False, True), (True, False), (False, False)]:
        filtered = apply_filters(candidates, require_us_exch, require_usd)
        if filtered:
            candidates = filtered
            break

    # Probe up to 5 candidates; compute coverage scores over 1y and ~30 trading days
    probe_period_long = "1y"
    probe_period_short = "1m"

    scored: list[tuple[float, dict, pd.DataFrame, dict]] = []
    count = 0
    for d in candidates:
        if count >= 5:
            break
        conid = str(d.get("conid"))
        df_long, meta = _probe_candidate_history(client, conid, period=probe_period_long)
        if df_long.empty:
            # skip non-chartable quickly
            count += 1
            continue
        # recent window coverage
        df_short, _ = _probe_candidate_history(client, conid, period=probe_period_short)
        # Compute expected bars using calendar between inferred window
        # Derive approximate end date as last index date
        last_dt = pd.Timestamp.now().normalize()
        try:
            if not df_long.empty:
                last_dt = df_long.index.to_timestamp().max().normalize()  # type: ignore
        except Exception:
            pass
        start_long = max(start_dt, last_dt - pd.Timedelta(days=365))
        start_short = max(start_dt, last_dt - pd.Timedelta(days=45))
        cov_long = _compute_coverage(df_long, start_long, last_dt)
        cov_short = _compute_coverage(df_short if not df_short.empty else df_long, start_short, last_dt)
        coverage_norm = 0.7 * cov_long + 0.3 * cov_short
        # Exchange ranking
        prim_rank, exch_rank = _exchange_rank(d)
        exch_rank_norm = 1.0 if prim_rank == 3 else (0.7 if exch_rank == 3 else (0.5 if exch_rank == 2 else (0.3 if exch_rank == 1 else 0.0)))
        # Currency match
        currency_match = 1.0 if currency(d) == "USD" else 0.0
        # Recency
        try:
            age_days = max(0, (pd.Timestamp.now().normalize() - last_dt).days)
            recency_norm = max(0.0, min(1.0, 1.0 - age_days / 30.0))
        except Exception:
            recency_norm = 0.0
        # Delay
        delay = 0
        try:
            delay = int(meta.get("mktDataDelay", 0))
        except Exception:
            delay = 0
        delay_norm = min(1.0, max(0.0, delay / 15.0))  # rough scale

        score = 1.00 * coverage_norm + 0.10 * (1.0 if prim_rank == 3 else 0.0) + 0.05 * exch_rank_norm + 0.05 * currency_match + 0.05 * recency_norm - 0.05 * delay_norm
        scored.append((score, d, df_long, meta))
        count += 1

    if not scored:
        return None

    scored.sort(key=lambda x: (x[0], len(x[2])), reverse=True)
    best_conid = str(scored[0][1].get("conid"))

    _CONID_CACHE[key] = (best_conid, ts_now)
    return best_conid


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
        from ibind import IbkrClient  # type: ignore
        from financetoolkit.utilities import cache_model

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
                unit = "ms" if float(pd.Series(ser).median()) > 1e12 else "s"
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

        # Save raw before enrichment if caching is enabled
        try:
            if use_cached_data:
                from financetoolkit.utilities import cache_model
                cache_model.save_cached_data(
                    cached_data=df,
                    cached_data_location=cached_data_location if isinstance(use_cached_data, str) else "cached",
                    file_name=cache_key,
                    method="pandas",
                    include_message=False,
                )
        except Exception:
            pass

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
        # Normalize caret-prefix if present
        sym = ticker[1:] if ticker.startswith('^') else ticker
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
        except Exception:
            pass
        try:
            res = client.search_contract_by_symbol(symbol=sym, sec_type='IND')
            data = getattr(res, 'data', None)
            # Best-effort extraction of basic fields if present
            if isinstance(data, dict):
                stats.loc["Symbol"] = data.get("symbol") or sym
            return stats
        except Exception:
            return stats
    except Exception as e:  # pragma: no cover
        logger.warning("IBKR/iBind stats fetch failed for %s: %s", ticker, e)
        return stats
