"""Properties of the shared pipeline on a reduced market-data slice (a few large stocks, 2021-2022).

The slice is loaded once per session through data/data_loader.py and cached in pytest's cache
directory; the tests are skipped when DATALOADER_API_KEY is not set.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

from data import data_loader as dl
from src.decomposition import wls_cross_section
from src.panel import daily_returns, month_ends
from src.pipeline import prev_month_index
from src.signals import month_blocks, res_sys

load_dotenv()
pytestmark = [
    pytest.mark.market_data,
    pytest.mark.skipif(not os.environ.get("DATALOADER_API_KEY"), reason="DATALOADER_API_KEY is not set"),
]

SLICE_TICKERS = ["AAPL", "MSFT", "IBM", "JPM", "XOM", "KO", "PFE", "WMT"]
SLICE_START, SLICE_END = "2021-01-01", "2022-12-30"


@pytest.fixture(scope="session")
def close_slice(request: pytest.FixtureRequest) -> pd.DataFrame:
    """Dividend-adjusted closes (dates x tickers) of the slice, cached between sessions."""
    path = request.config.cache.mkdir("asymrev_slice") / "close.parquet"
    if path.exists():
        return pd.read_parquet(path)
    bars, meta = dl.load_ohlcv(SLICE_TICKERS, SLICE_START, SLICE_END, identity_date=SLICE_END)
    assert not meta["failed"], meta["failed"]
    close = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    close.to_parquet(path)
    return close


def test_slice_bars_are_complete_and_positive(close_slice: pd.DataFrame) -> None:
    assert set(close_slice.columns) == set(SLICE_TICKERS)
    assert close_slice.index.min() >= pd.Timestamp(SLICE_START)
    assert close_slice.index.max() <= pd.Timestamp(SLICE_END)
    assert 480 <= len(close_slice) <= 505  # about 252 trading days per year
    assert (close_slice > 0).all().all()


def test_daily_returns_plausible(close_slice: pd.DataFrame) -> None:
    ret = daily_returns(close_slice)
    assert ret.iloc[0].isna().all()
    vals = ret.iloc[1:].to_numpy()
    assert np.isfinite(vals).all()
    assert np.abs(vals).max() < 0.5
    assert 0.005 < np.nanstd(vals) < 0.05


def test_weights_and_characteristics_use_previous_month_end(close_slice: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex(close_slice.index)
    mends = month_ends(dates)
    pm = prev_month_index(dates, mends)
    valid = pm >= 0
    # the information month-end strictly precedes every day it is applied to (no look-ahead)
    assert (mends[pm[valid]] < dates[valid]).all()
    assert (dates[~valid] <= mends[0]).all()


def test_wls_residuals_orthogonal_on_real_returns(close_slice: pd.DataFrame) -> None:
    ret = daily_returns(close_slice).iloc[1:].to_numpy()
    n = ret.shape[1]
    rng = np.random.default_rng(0)
    Z = np.column_stack([np.ones(n), rng.standard_normal(n)])
    w = np.full(n, 1.0 / n)
    f, fitted, rank = wls_cross_section(ret[10], Z, w)
    eps = ret[10] - fitted
    assert rank == Z.shape[1]
    np.testing.assert_allclose(Z.T @ (w * eps), 0.0, atol=1e-8)


def test_monthly_res_sys_skip_last_day(close_slice: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex(close_slice.index)
    ret = daily_returns(close_slice).fillna(0.0).to_numpy(dtype="float64")
    eps = 0.5 * ret
    blocks = month_blocks(dates)
    out = res_sys(eps, ret - eps, blocks)
    s, e = blocks[3]
    # Equations 10-11: the last trading day of the month is skipped
    full = month_blocks(dates, skip_last=False)[3]
    assert full == (s, e + 1)
    np.testing.assert_allclose(out["RES"][3], eps[s:e].sum(axis=0), atol=1e-8)
    np.testing.assert_allclose(out["RES"][3] + out["SYS"][3], ret[s:e].sum(axis=0), atol=1e-8)
