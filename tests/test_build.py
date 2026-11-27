"""Unit tests for src.build and the src.panel helpers it relies on (Sections 2-3), synthetic tables only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.build import (ea_window_mask, earnings_dates, security_table, shares_panel, sic_to_ff49,
                       unadjust_dividends)
from src.panel import compound_monthly, daily_returns, last_day_flag, month_ends, month_id, to_wide


# ------------------------------------------------------------------ panel helpers
def test_to_wide_pivots_and_reindexes() -> None:
    bars = pd.DataFrame({"date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-04"]),
                         "security": ["A", "A", "B"], "close": [1.0, 2.0, 3.0], "volume": [10, 20, 30]})
    dates = pd.DatetimeIndex(["2021-01-04", "2021-01-05", "2021-01-06"])
    w = to_wide(bars, "close", dates=dates, securities=pd.Index(["A", "B", "C"]))
    assert w.shape == (3, 3) and w.dtypes.eq(np.float32).all()
    assert w.at[dates[1], "A"] == 2.0 and np.isnan(w.at[dates[1], "B"]) and w["C"].isna().all()
    assert to_wide(bars, "volume").dtypes.eq(np.float64).all()
    with pytest.raises(ValueError):
        to_wide(bars.iloc[:0], "close")


def test_daily_returns_require_previous_bar() -> None:
    close = pd.DataFrame({"a": [10.0, 11.0, np.nan, 12.1, 13.31]})
    ret = daily_returns(close)
    assert np.isnan(ret["a"].iloc[[0, 2, 3]]).all()
    np.testing.assert_allclose(ret["a"].iloc[[1, 4]], [0.1, 0.1], rtol=1e-6)


def test_month_ends_last_day_flag_and_month_id(bdays: pd.DatetimeIndex) -> None:
    me = month_ends(bdays)
    assert list(me) == list(pd.DatetimeIndex(["2021-01-29", "2021-02-26", "2021-03-31"]))
    flag = last_day_flag(bdays)
    assert list(bdays[flag]) == list(me)
    mid = month_id(bdays)
    assert mid[0] == 2021 * 12 and mid[-1] == 2021 * 12 + 2


def test_compound_monthly_min_days(bdays: pd.DatetimeIndex) -> None:
    ret = pd.DataFrame({"a": 0.01, "b": np.nan}, index=bdays)
    ret.loc[bdays[:2], "b"] = 0.05
    out = compound_monthly(ret, min_days=2)
    jan_n = (bdays.month == 1).sum()
    assert out["a"].iloc[0] == pytest.approx(1.01 ** jan_n - 1)
    assert out["b"].iloc[0] == pytest.approx(1.05 ** 2 - 1)
    assert out["b"].iloc[1:].isna().all()
    assert list(out.index) == list(month_ends(bdays))


# ------------------------------------------------------------------ security table and adjustments
def test_security_table_latest_ticker_first() -> None:
    u = pd.DataFrame({"security": ["1", "1", "1", "2"], "ticker": ["OLD", "NEW", "NEW", "X"],
                      "cik": ["c1", "c1", "c1", "c2"], "primary_exchange": ["XNAS", "XNYS", "XNYS", "XASE"],
                      "last_seen": pd.to_datetime(["2015-12-31", "2018-12-31", "2017-12-31", "2020-12-31"]),
                      "first_seen": pd.to_datetime(["2010-12-31", "2016-12-31", "2016-12-31", "2019-12-31"])})
    t = security_table(u)
    assert t.at["1", "tickers"] == ["NEW", "OLD"]
    assert t.at["1", "primary_exchange"] == "XNYS"
    assert t.at["1", "last_seen"] == pd.Timestamp("2018-12-31")
    assert t.at["1", "first_seen"] == pd.Timestamp("2010-12-31")


def test_unadjust_dividends_recovers_unadjusted_close() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    unadj = pd.DataFrame({"s": np.linspace(10, 12, len(dates))}, index=dates)
    f_jan, f_feb = 0.99, 0.98
    F = np.where(dates < "2020-01-15", f_jan * f_feb, np.where(dates < "2020-02-03", f_feb, 1.0))
    adj = unadj.mul(F, axis=0)
    divs = pd.DataFrame({"ticker": ["T", "T", "T"], "ex_dividend_date": ["2020-01-15", "2020-02-03", "2020-02-10"],
                         "historical_adjustment_factor": [f_jan * f_feb, f_feb, np.nan],
                         "distribution_type": ["recurring", None, "recurring"]})
    secs = pd.DataFrame({"tickers": [["T"]]}, index=["s"])
    out = unadjust_dividends(adj, secs, divs, window_end="2020-03-31")
    np.testing.assert_allclose(out["s"].to_numpy(), unadj["s"].to_numpy())


def test_shares_panel_latest_quarter_split_adjusted_and_stale() -> None:
    dates = pd.bdate_range("2020-01-01", "2021-06-30")
    secs = pd.DataFrame({"cik": ["c1"], "tickers": [["T"]], "first_seen": [pd.Timestamp("2019-12-31")]}, index=["s"])
    income = pd.DataFrame({"cik": ["c1", "c1"], "period_end": ["2020-03-31", "2020-06-30"],
                           "basic_shares_outstanding": [100.0, 110.0]})
    splits = pd.DataFrame({"ticker": ["T"], "execution_date": ["2020-05-15"], "split_to": [2.0], "split_from": [1.0]})
    out = shares_panel(dates, secs, income, splits, window_end="2021-06-30")["s"]
    assert out[dates < "2020-03-31"].isna().all()
    assert (out[(dates >= "2020-03-31") & (dates < "2020-06-30")] == 200.0).all()  # 100 x 2-for-1 split later
    after = (dates >= "2020-06-30") & (dates <= pd.Timestamp("2020-06-30") + pd.Timedelta(days=12 * 31))
    assert (out[after] == 110.0).all()
    assert out[dates > pd.Timestamp("2020-06-30") + pd.Timedelta(days=12 * 31)].isna().all()


def test_sic_to_ff49_first_match_other_and_missing() -> None:
    ranges = [(1, 100, 199), (2, 150, 299), (3, 1000, 1000)]
    out = sic_to_ff49(pd.Series(["120", "250", "1000", "5000", None, "x"]), ranges)
    np.testing.assert_array_equal(out.iloc[:4].to_numpy(), [1, 2, 3, 49])
    assert out.iloc[4:].isna().all()


# ------------------------------------------------------------------ earnings announcements
def test_earnings_dates_prefers_early_8k_else_periodic_filing() -> None:
    income = pd.DataFrame({"cik": ["a", "b", "c"], "period_end": ["2020-03-31"] * 3})
    filings = pd.DataFrame({
        "cik": ["a", "a", "a", "b", "c", "c"],
        "form_type": ["8-K", "8-K", "10-Q", "10-Q", "10-Q", "8-K"],
        "filing_date": ["2020-04-05", "2020-04-25", "2020-05-08", "2020-05-01", "2020-05-01", "2020-05-20"],
    })
    ea = earnings_dates(income, filings).set_index("cik")["ea_date"]
    assert ea["a"] == pd.Timestamp("2020-04-25")  # the 8-K before period end + 10 days is ignored
    assert ea["b"] == pd.Timestamp("2020-05-01")  # no 8-K: 10-Q filing date
    assert ea["c"] == pd.Timestamp("2020-05-01")  # 8-K after the 10-Q: 10-Q filing date


def test_ea_window_mask_three_trading_days_and_weekend_shift() -> None:
    dates = pd.bdate_range("2020-04-01", "2020-04-30")
    secs = pd.DataFrame({"cik": ["a", "b", "z"]}, index=["s1", "s2", "s3"])
    ea = pd.DataFrame({"cik": ["a", "b"], "ea_date": pd.to_datetime(["2020-04-15", "2020-04-18"])})  # 18th = Sat
    mask = ea_window_mask(dates, secs, ea)
    assert list(dates[mask[:, 0]]) == list(pd.to_datetime(["2020-04-14", "2020-04-15", "2020-04-16"]))
    assert list(dates[mask[:, 1]]) == list(pd.to_datetime(["2020-04-17", "2020-04-20", "2020-04-21"]))
    assert not mask[:, 2].any()
