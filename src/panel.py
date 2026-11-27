"""Wide daily and monthly panels (dates x securities) built from long bar tables.

All panels share one date index (trading days) and one column index
(security keys), so every later step works with aligned numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_wide(bars: pd.DataFrame, field: str, dates: pd.DatetimeIndex | None = None,
            securities: pd.Index | None = None) -> pd.DataFrame:
    """Pivot a long table with columns ``date``, ``security`` and ``field`` to dates x securities.

    Args:
        bars: long table (one row per security-date).
        field: value column to pivot.
        dates, securities: optional target axes (missing cells become NaN).
    """
    if bars.empty:
        raise ValueError("empty bar table")
    wide = bars.pivot(index="date", columns="security", values=field)
    wide.columns = wide.columns.astype(str)
    if dates is not None or securities is not None:
        wide = wide.reindex(index=dates if dates is not None else wide.index,
                            columns=securities if securities is not None else wide.columns)
    return wide.astype("float32" if field != "volume" else "float64")


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from (dividend- and split-adjusted) closes.

    A return is defined only when the previous trading day also has a bar, so a
    gap in trading never produces a multi-day return attributed to one day.
    """
    ret = close / close.shift(1) - 1.0
    return ret.astype("float32")


def month_ends(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in ``dates``."""
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(dates.to_period("M")).max().values)


def month_id(dates: pd.DatetimeIndex) -> np.ndarray:
    """Integer month index (year*12 + month - 1) of each date."""
    return (dates.year * 12 + dates.month - 1).to_numpy()


def last_day_flag(dates: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array: True on the last trading day of each month."""
    mid = month_id(dates)
    flag = np.ones(len(dates), dtype=bool)
    flag[:-1] = mid[:-1] != mid[1:]
    return flag


def compound_monthly(ret: pd.DataFrame, min_days: int = 1) -> pd.DataFrame:
    """Monthly compounded returns from daily returns (month-end trading-day index).

    A month is NaN for a security with fewer than ``min_days`` valid daily returns.
    """
    per = ret.index.to_period("M")
    logr = np.log1p(ret.astype("float64"))
    out = np.expm1(logr.groupby(per).sum(min_count=min_days))
    out.index = month_ends(ret.index)
    return out
