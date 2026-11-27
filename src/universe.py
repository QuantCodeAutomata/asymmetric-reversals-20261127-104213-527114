"""Market equity, investable universe and regression/portfolio weights (Section 2-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def dividend_factor(dates: pd.DatetimeIndex, ex_dates: np.ndarray, hist_factor: np.ndarray,
                    window_end: pd.Timestamp) -> np.ndarray:
    """Cumulative dividend back-adjustment factor F_t relative to ``window_end``.

    Dividend-adjusted closes equal unadjusted closes times F_t, where F_t is the
    product of the per-event factors of all ex-dates in (t, window_end].
    ``hist_factor`` is the provider's cumulative factor of an ex-date (product of
    the event factors of that and all later ex-dates).
    """
    order = np.argsort(ex_dates)
    ex, hf = ex_dates[order], hist_factor[order]
    # cumulative factor of the first ex-date strictly after a date; 1 if none
    after_end = np.searchsorted(ex, np.datetime64(window_end), side="right")
    h_end = hf[after_end] if after_end < len(ex) else 1.0
    idx = np.searchsorted(ex, dates.values, side="right")
    h_t = np.where(idx < len(ex), hf[np.minimum(idx, len(ex) - 1)], 1.0)
    return (h_t / h_end).astype("float64")


def split_factor_after(period_end: np.ndarray, split_dates: np.ndarray, ratios: np.ndarray,
                       window_end: pd.Timestamp) -> np.ndarray:
    """Product of split ratios (split_to/split_from) executed in (period_end, window_end]."""
    out = np.ones(len(period_end))
    for d, r in zip(split_dates, ratios):
        if d <= np.datetime64(window_end):
            out[period_end < d] *= r
    return out


def investable_mask(me: pd.DataFrame, entry: float = config.UNIV_ENTRY_SHARE,
                    exit_: float = config.UNIV_EXIT_SHARE) -> pd.DataFrame:
    """Hysteresis universe filter (Section 2).

    A stock enters when its market equity exceeds ``entry`` of aggregate market
    capitalization and leaves when it falls below ``exit_``.
    """
    share = me.div(me.sum(axis=1), axis=0).to_numpy()
    inside = np.zeros(share.shape, dtype=bool)
    prev = np.zeros(share.shape[1], dtype=bool)
    for m in range(share.shape[0]):
        s = share[m]
        valid = np.isfinite(s)
        cur = valid & ((s > entry) | (prev & (s >= exit_)))
        inside[m] = cur
        prev = cur
    return pd.DataFrame(inside, index=me.index, columns=me.columns)


def nyse_percentile(values: pd.DataFrame, mask: pd.DataFrame, is_nyse: pd.Series, q: float) -> pd.Series:
    """Cross-sectional ``q`` percentile of ``values`` among NYSE stocks in ``mask``, per row."""
    nyse = mask & is_nyse.reindex(values.columns).fillna(False).to_numpy()[None, :]
    v = values.where(nyse)
    return v.quantile(q, axis=1)


def winsorized_weights(me: pd.DataFrame, mask: pd.DataFrame, is_nyse: pd.Series,
                       q: float = config.WEIGHT_WINSOR_PCT) -> pd.DataFrame:
    """Market equity winsorized at the ``q`` NYSE percentile (Section 3); NaN outside ``mask``."""
    cap = nyse_percentile(me, mask, is_nyse, q)
    w = me.clip(upper=cap, axis=0).where(mask)
    return w
