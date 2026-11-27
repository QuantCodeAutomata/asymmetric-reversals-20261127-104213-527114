"""Unit tests for src.universe: investable universe hysteresis and winsorized weights (Sections 2-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.universe import dividend_factor, investable_mask, nyse_percentile, split_factor_after, winsorized_weights


def _two_stock_me(shares: list[float]) -> pd.DataFrame:
    """Target stock with the given share of aggregate ME, plus one filler stock holding the rest."""
    s = np.array(shares)
    months = pd.date_range("2020-01-31", periods=len(s), freq="ME")
    return pd.DataFrame({"target": s * 1e9, "filler": (1 - np.nan_to_num(s)) * 1e9}, index=months)


def test_investable_mask_entry_exit_hysteresis() -> None:
    assert config.UNIV_ENTRY_SHARE == pytest.approx(3e-5) and config.UNIV_EXIT_SHARE == pytest.approx(2e-5)
    shares = [2.5e-5, 3.5e-5, 2.5e-5, 2.05e-5, 1.5e-5, 2.5e-5, 2.9e-5, 3.1e-5]
    mask = investable_mask(_two_stock_me(shares))
    # below entry -> out; above entry -> in; between exit and entry -> stays in; below exit -> out;
    # between thresholds again -> stays out until it exceeds the entry share
    assert mask["target"].tolist() == [False, True, True, True, False, False, False, True]
    assert mask["filler"].all()


def test_investable_mask_missing_me_resets_membership() -> None:
    mask = investable_mask(_two_stock_me([3.5e-5, np.nan, 2.5e-5]))
    assert mask["target"].tolist() == [True, False, False]


def test_investable_mask_custom_thresholds() -> None:
    me = pd.DataFrame({"a": [10.0, 4.0, 2.0], "b": [90.0, 96.0, 98.0]})
    mask = investable_mask(me, entry=0.05, exit_=0.03)
    assert mask["a"].tolist() == [True, True, False]


def test_nyse_percentile_and_winsorized_weights_cap_at_80th_nyse() -> None:
    cols = [f"s{i}" for i in range(12)]
    me = pd.DataFrame([np.r_[np.arange(1.0, 11.0), 100.0, 200.0]], columns=cols,
                      index=pd.DatetimeIndex(["2020-01-31"]))
    is_nyse = pd.Series([True] * 10 + [False] * 2, index=cols)
    mask = pd.DataFrame(True, index=me.index, columns=cols)
    cap = np.quantile(np.arange(1.0, 11.0), 0.8)
    assert nyse_percentile(me, mask, is_nyse, 0.8).iloc[0] == pytest.approx(cap)
    w = winsorized_weights(me, mask, is_nyse)
    np.testing.assert_allclose(w.iloc[0].to_numpy(), np.minimum(me.iloc[0].to_numpy(), cap))
    assert config.WEIGHT_WINSOR_PCT == 0.80
    # stocks outside the mask get NaN weights and do not enter the NYSE percentile
    mask2 = mask.copy()
    mask2.loc[:, ["s9", "s8"]] = False
    w2 = winsorized_weights(me, mask2, is_nyse)
    cap2 = np.quantile(np.arange(1.0, 9.0), 0.8)
    assert w2[["s8", "s9"]].isna().all(axis=None)
    assert w2["s11"].iloc[0] == pytest.approx(cap2)
    assert w2["s0"].iloc[0] == 1.0


@pytest.mark.filterwarnings("ignore:Downcasting object dtype:FutureWarning")
def test_winsorized_weights_missing_nyse_flag_is_not_nyse() -> None:
    me = pd.DataFrame([[1.0, 2.0, 3.0, 50.0]], columns=list("abcd"))
    is_nyse = pd.Series([True, True, True], index=list("abc"))  # "d" unknown -> treated as non-NYSE
    w = winsorized_weights(me, pd.DataFrame(True, index=me.index, columns=me.columns), is_nyse)
    assert w["d"].iloc[0] == pytest.approx(np.quantile([1.0, 2.0, 3.0], 0.8))


# ------------------------------------------------------------------ price adjustment helpers
def test_dividend_factor_products_of_events_after_date() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    ex = np.array(["2020-02-03", "2020-01-15"], dtype="datetime64[ns]")  # unsorted on purpose
    f_jan, f_feb = 0.99, 0.98
    hist = np.array([f_feb, f_jan * f_feb])  # cumulative factors (this and later events)
    F = dividend_factor(dates, ex, hist, pd.Timestamp("2020-03-31"))
    expected = np.where(dates < "2020-01-15", f_jan * f_feb, np.where(dates < "2020-02-03", f_feb, 1.0))
    np.testing.assert_allclose(F, expected)
    # relative to a window end between the two ex-dates, the later event is not applied
    end = pd.Timestamp("2020-01-31")
    F_mid = dividend_factor(dates, ex, hist, end)
    inside = dates <= end
    np.testing.assert_allclose(F_mid[inside], np.where(dates[inside] < "2020-01-15", f_jan, 1.0))


def test_split_factor_after_window() -> None:
    splits = np.array(["2020-02-15", "2020-08-01"], dtype="datetime64[ns]")
    assert split_factor_after(splits, np.array([2.0, 3.0]), pd.Timestamp("2020-12-31")) == 1.0
    assert split_factor_after(splits, np.array([2.0, 3.0]), pd.Timestamp("2020-06-30")) == 3.0
    assert split_factor_after(splits, np.array([2.0, 3.0]), pd.Timestamp("2019-06-30")) == 6.0
