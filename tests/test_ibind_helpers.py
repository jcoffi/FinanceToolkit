import pandas as pd
import numpy as np
import pytest

from financetoolkit.financetoolkit import ibind_model as ib


def test_parse_dates_defaults_and_explicit():
    # explicit
    s, e, sd, ed = ib._parse_dates("2020-01-01", "2020-12-31")
    assert s == "2020-01-01"
    assert e == "2020-12-31"
    assert sd.year == 2020 and ed.year == 2020

    # defaults (end None -> today, start None -> 10y back)
    s2, e2, sd2, ed2 = ib._parse_dates(None, None)
    assert isinstance(s2, str) and isinstance(e2, str)
    assert (ed2 - sd2).days >= 365 * 9


def test_determine_bar_and_period():
    b, p = ib._determine_bar_and_period("1d")
    assert b == "1d"
    assert p == "1y"
    b2, p2 = ib._determine_bar_and_period("unknown")
    assert b2 == "1d"


def test_history_payload_to_df_variants():
    # numeric epoch seconds
    payload = [{"t": 1609459200, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]
    df = ib._history_payload_to_df(payload, freq="D")
    assert isinstance(df, pd.DataFrame)
    assert "Open" in df.columns and "Adj Close" in df.columns

    # dict with 'data' key
    payload2 = {"data": payload}
    df2 = ib._history_payload_to_df(payload2)
    assert not df2.empty


def test_load_and_save_cache_cycle(tmp_path, monkeypatch):
    # We'll simulate cache_model with a minimal wrapper in the utilities module
    calls = {}

    class DummyCache:
        @staticmethod
        def load_cached_data(cached_data_location, file_name, method, return_empty_type):
            calls['loaded'] = (cached_data_location, file_name)
            return pd.DataFrame()

        @staticmethod
        def save_cached_data(cached_data, cached_data_location, file_name, method, include_message):
            calls['saved'] = (cached_data_location, file_name)

    monkeypatch.setitem(__import__('sys').modules, 'financetoolkit.utilities.cache_model', DummyCache)

    key = "test_cache_key"
    df = pd.DataFrame({"A": [1, 2, 3]})
    # load should return None (Dummy returns empty DataFrame)
    loaded = ib._load_cached_if_enabled(True, str(tmp_path), key)
    assert loaded is None
    # save should not raise
    ib._save_cached_if_enabled(True, str(tmp_path), key, df)
    assert 'saved' in calls
