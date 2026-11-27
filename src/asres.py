"""Out-of-sample asymmetric residual return forecast ASRES (Section 7, Equations 25-28)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.irf import daily_dl_regression, forward_sum, lag_window, spline_basis


def asres_basis() -> tuple[np.ndarray, np.ndarray]:
    """Degree-1 spline basis over lags 2..252 (Equation 25).

    The paper does not state how the lag-1 knot of Equation (22) is treated
    when the forecast starts at lag 2; the knot at lag 1 is dropped, leaving
    26 tent functions with the first one peaking at lag 2.
    Returns (lags, basis) with basis of shape (251, 26).
    """
    knots = tuple(k for k in config.SPLINE_KNOTS if k >= config.ASRES_FIRST_LAG)
    if knots[0] != config.ASRES_FIRST_LAG:
        knots = (config.ASRES_FIRST_LAG,) + knots
    lags = np.arange(config.ASRES_FIRST_LAG, config.N_LAGS + 1)
    return lags, spline_basis(knots, lags)


def forecast_coefficients(exret: np.ndarray, w_day: np.ndarray, eps_t: np.ndarray, r_t: np.ndarray,
                          days: np.ndarray, horizon: int = config.FORECAST_HORIZON) -> np.ndarray:
    """Daily spline Fama-MacBeth coefficients of R^21_{i,t} on lagged signed shocks (Equation 25).

    Regressors: eps~+ and eps~- (lags 2..252) and pooled r~ (lags 2..252),
    projected on the spline basis; intercept included. Returns (len(days), 3, 251)
    implied lag coefficients [eps+, eps-, r~].
    """
    _, B = asres_basis()
    R21 = forward_sum(exret, horizon)
    e = np.nan_to_num(eps_t.astype("float32"))
    r = np.nan_to_num(r_t.astype("float32"))
    shocks = [np.maximum(e, 0), np.minimum(e, 0), r]
    return daily_dl_regression(R21, w_day, shocks, days, config.ASRES_FIRST_LAG, config.N_LAGS, basis=B)


def expanding_lambda(lam: np.ndarray, reg_days: np.ndarray, dates: pd.DatetimeIndex, mends: pd.DatetimeIndex,
                     horizon: int = config.FORECAST_HORIZON) -> np.ndarray:
    """Expanding-window average of the residual lag coefficients available at each formation month (Equation 26).

    For formation month m only regressions whose 21-day holding period ends no
    later than the end of month m-1 enter the average (Section 7), so no
    dependent variable realized after formation is used.

    Returns (M, 2, L) array (NaN when no regression is available).
    """
    end_idx = reg_days + horizon - 1  # last day of each regression's holding period
    end_dates = dates[np.minimum(end_idx, len(dates) - 1)]
    good = np.isfinite(lam[:, :2]).all(axis=(1, 2)) & (end_idx < len(dates))
    M, L = len(mends), lam.shape[2]
    out = np.full((M, 2, L), np.nan)
    csum = np.zeros((2, L))
    cnt = 0
    order = np.argsort(end_dates.values)
    j = 0
    for m in range(1, M):
        cutoff = mends[m - 1]
        while j < len(order) and end_dates[order[j]] <= cutoff:
            if good[order[j]]:
                csum += lam[order[j], :2]
                cnt += 1
            j += 1
        if cnt:
            out[m] = csum / cnt
    return out


def asres_signal(lam_bar: np.ndarray, eps_t: np.ndarray, t_m: np.ndarray, short_lag: int = config.ASRES_SHORT_LAG,
                 long_weight: float = config.ASRES_LONG_WEIGHT) -> dict[str, np.ndarray]:
    """mu^{eps,full}, mu^{eps,21} and ASRES for every formation month (Equations 27-28).

    Args:
        lam_bar: (M, 2, L) expanding-window coefficients on eps~+ and eps~- for lags 2..252.
        eps_t: (T, N) earnings-adjusted residuals.
        t_m: (M,) index of the first trading day of holding month m+1 (-1 if none).
    Returns:
        dict of (M, N) arrays ``mu_full``, ``mu_21``, ``ASRES``.
    """
    M = lam_bar.shape[0]
    N = eps_t.shape[1]
    first = config.ASRES_FIRST_LAG
    L = lam_bar.shape[2]
    n_short = short_lag - first + 1
    e = np.nan_to_num(eps_t.astype("float64"))
    out = {k: np.full((M, N), np.nan) for k in ("mu_full", "mu_21", "ASRES")}
    for m in range(M):
        t = t_m[m]
        if t < 0 or not np.isfinite(lam_bar[m]).all():
            continue
        win = lag_window(e, t, first, first + L - 1)  # (L, N), row j = lag first + j
        pos, neg = np.maximum(win, 0), np.minimum(win, 0)
        contrib = lam_bar[m, 0][:, None] * pos + lam_bar[m, 1][:, None] * neg
        full = contrib.sum(axis=0)
        short = contrib[:n_short].sum(axis=0)
        out["mu_full"][m], out["mu_21"][m] = full, short
        out["ASRES"][m] = short + long_weight * (full - short)
    return out


def first_day_next_month(dates: pd.DatetimeIndex, mends: pd.DatetimeIndex) -> np.ndarray:
    """Index of the first trading day after each month end (-1 for the last month)."""
    idx = np.searchsorted(dates.values, mends.values, side="right")
    return np.where(idx < len(dates), idx, -1)


def size_groups(me: np.ndarray, univ: np.ndarray, is_nyse: np.ndarray,
                groups: dict[str, tuple[float, float]] = config.SIZE_GROUPS) -> dict[str, np.ndarray]:
    """NYSE market-equity percentile groups (Table 9): label masks (M, N) per group, (lo, hi] percentiles."""
    M, N = me.shape
    out = {g: np.zeros((M, N), dtype=bool) for g in groups}
    for m in range(M):
        v = univ[m] & is_nyse & np.isfinite(me[m])
        if v.sum() < 20:
            continue
        for g, (lo, hi) in groups.items():
            qlo = np.quantile(me[m, v], lo)
            sel = univ[m] & np.isfinite(me[m]) & (me[m] > qlo)
            if hi < 1.0:
                sel &= me[m] <= np.quantile(me[m, v], hi)
            out[g][m] = sel
    return out
