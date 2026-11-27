"""Unit tests for src.fundamentals: statement merge, TTM sums, SUE and point-in-time alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.fundamentals import add_ttm_and_growth, merge_statements, sue, to_monthly


def _quarters(n: int, start: str = "2018-03-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="QE")


def test_merge_statements_one_row_per_cik_quarter() -> None:
    income = pd.DataFrame({"cik": ["a", "a", "a"], "period_end": ["2020-03-31", "2020-03-31", "2020-06-30"],
                           "revenue": [1.0, 2.0, 3.0], "tickers": [["A"]] * 3, "filing_date": ["x"] * 3})
    balance = pd.DataFrame({"cik": ["a", "b"], "period_end": ["2020-03-31", "2020-03-31"],
                            "total_assets": [10.0, 20.0], "fiscal_year": [2020, 2020]})
    cash = pd.DataFrame({"cik": ["a"], "period_end": ["2020-06-30"], "dividends": [-1.0]})
    out = merge_statements(income, balance, cash)
    assert len(out) == 3
    assert not {"tickers", "filing_date", "fiscal_year"} & set(out.columns)
    row = out[(out["cik"] == "a") & (out["period_end"] == "2020-03-31")].iloc[0]
    assert row["revenue"] == 2.0 and row["total_assets"] == 10.0  # duplicate quarter: last record
    assert np.isnan(out[out["cik"] == "b"]["revenue"]).all()


def test_add_ttm_and_growth_sums_lags_and_gaps() -> None:
    pe = _quarters(9)
    q = pd.DataFrame({"cik": "a", "period_end": pe, "revenue": np.arange(1.0, 10.0),
                      "total_assets": np.arange(100.0, 109.0), "basic_shares_outstanding": 10.0,
                      "net_income_loss_attributable_common_shareholders": [np.nan] + [2.0] * 8,
                      "consolidated_net_income_loss": 5.0})
    out = add_ttm_and_growth(q).reset_index(drop=True)
    assert out["revenue_ttm"].iloc[:3].isna().all()
    np.testing.assert_allclose(out["revenue_ttm"].iloc[3:], [10, 14, 18, 22, 26, 30])
    np.testing.assert_allclose(out["revenue_ttm_l4"].iloc[7:], [10, 14])
    np.testing.assert_allclose(out["total_assets_l4"].iloc[4:], np.arange(100.0, 105.0))
    np.testing.assert_allclose(out["eps_q"], [0.5] + [0.2] * 8)  # falls back to consolidated net income


def test_add_ttm_large_gap_is_not_a_ttm() -> None:
    pe = pd.DatetimeIndex(["2018-03-31", "2018-06-30", "2018-09-30", "2019-12-31"])
    q = pd.DataFrame({"cik": "a", "period_end": pe, "revenue": 1.0, "basic_shares_outstanding": 1.0,
                      "net_income_loss_attributable_common_shareholders": 1.0, "consolidated_net_income_loss": 1.0})
    assert add_ttm_and_growth(q)["revenue_ttm"].isna().all()


def test_add_ttm_one_missing_quarter_breaks_four_consecutive_quarters() -> None:
    pe = _quarters(9)
    q = pd.DataFrame({"cik": "a", "period_end": pe, "revenue": np.arange(1.0, 10.0), "basic_shares_outstanding": 1.0,
                      "net_income_loss_attributable_common_shareholders": 1.0, "consolidated_net_income_loss": 1.0})
    gap = add_ttm_and_growth(q.drop(index=5)).reset_index(drop=True)  # quarter 2019-06-30 missing
    np.testing.assert_allclose(gap["revenue_ttm"].iloc[3:5], [10, 14])
    # rows 5..7 (2019-09-30 .. 2020-03-31) have no four consecutive quarters ending there
    assert gap["revenue_ttm"].iloc[5:].isna().all()


def test_add_ttm_is_per_cik() -> None:
    pe = _quarters(4)
    q = pd.DataFrame({"cik": ["a"] * 4 + ["b"] * 4, "period_end": list(pe) * 2, "revenue": [1.0] * 4 + [2.0] * 4,
                      "basic_shares_outstanding": 1.0, "net_income_loss_attributable_common_shareholders": 1.0,
                      "consolidated_net_income_loss": 1.0})
    out = add_ttm_and_growth(q)
    assert out.groupby("cik")["revenue_ttm"].last().to_dict() == {"a": 4.0, "b": 8.0}
    assert out.groupby("cik")["revenue_ttm"].apply(lambda s: s.isna().sum()).to_dict() == {"a": 3, "b": 3}


def test_sue_standardizes_by_past_eight_changes(rng: np.random.Generator) -> None:
    n = 16
    eps = np.cumsum(rng.standard_normal(n))
    q = pd.DataFrame({"cik": "a", "period_end": _quarters(n), "eps_q": eps})
    out = sue(q).to_numpy()
    chg = eps[4:] - eps[:-4]  # change of quarter i (i >= 4) is chg[i - 4]
    for i in range(n):
        past = chg[max(0, i - 4 - 8):max(0, i - 4)]
        if i < 4 or len(past) < 6:
            assert np.isnan(out[i])
        else:
            assert out[i] == pytest.approx(chg[i - 4] / np.std(past, ddof=1))


def test_to_monthly_point_in_time_lag_and_staleness() -> None:
    q = pd.DataFrame({"cik": ["a", "a"], "period_end": pd.to_datetime(["2020-03-31", "2020-06-30"]),
                      "x": [1.0, 2.0]})
    months = pd.date_range("2020-06-30", "2022-01-31", freq="ME")
    cik_of = pd.Series({"s1": "a", "s2": "b"})
    out = to_monthly(q, ["x"], months, cik_of)["x"]
    assert config.FUND_LAG_MONTHS == 4
    assert out.loc[:"2020-06-30", "s1"].isna().all()
    assert out.at[pd.Timestamp("2020-07-31"), "s1"] == 1.0  # March quarter public 4 months later
    assert out.at[pd.Timestamp("2020-09-30"), "s1"] == 1.0
    assert out.at[pd.Timestamp("2020-10-31"), "s1"] == 2.0
    assert out["s2"].isna().all()
    last_ok = pd.Timestamp("2020-06-30") + pd.Timedelta(days=(12 + 4) * 31)
    assert (out.loc[out.index <= last_ok, "s1"].dropna() > 0).all()
    assert out.loc[out.index > last_ok, "s1"].isna().all()  # stale quarter discarded
