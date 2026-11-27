"""Daily cross-sectional return decomposition (Section 3, Equations 3-9).

r_{t+1} = Z_t f_{t+1} + eps_{t+1},  Z_t = [I_t, X_t],
f_{t+1} = (Z_t' Omega_t Z_t)^{-1} Z_t' Omega_t r_{t+1}      (Equation 3, WLS)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def wls_cross_section(r: np.ndarray, Z: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """One WLS cross-sectional regression (custom implementation of Equation 3).

    Solved by least squares on sqrt(w)-scaled rows, which handles rank deficiency
    (e.g. empty industries) with the minimum-norm solution.

    Args:
        r: (N,) returns; Z: (N, K) regressors; w: (N,) non-negative weights.
    Returns:
        f: (K,) coefficients, fitted values Z f, and the numerical rank of sqrt(w) Z.
    """
    if r.ndim != 1 or Z.shape[0] != r.shape[0] or w.shape != r.shape:
        raise ValueError("shape mismatch in wls_cross_section")
    sw = np.sqrt(w)
    f, _, rank, _ = np.linalg.lstsq(Z * sw[:, None], r * sw, rcond=None)
    return f, Z @ f, int(rank)


def industry_dummies(ind: np.ndarray, n_ind: int = config.N_INDUSTRIES) -> np.ndarray:
    """(N, n_ind) 0/1 matrix from industry numbers 1..n_ind."""
    out = np.zeros((len(ind), n_ind))
    out[np.arange(len(ind)), ind.astype(int) - 1] = 1.0
    return out


def daily_decomposition(exret: np.ndarray, month_of_day: np.ndarray, weights_m: np.ndarray,
                        chars_m: np.ndarray, incl_m: np.ndarray, ind_m: np.ndarray | None,
                        day_range: range) -> dict[str, np.ndarray]:
    """Daily WLS decomposition of excess returns into fitted and residual parts.

    For each day t+1 of month m the regressors are the month m-1 standardized
    characteristics (and FF49 industry dummies) and the weights the month m-1
    winsorized market equity, normalized to sum to one (Section 3). Without
    industries an intercept is included.

    Args:
        exret: (T, N) daily excess returns.
        month_of_day: (T,) index into the monthly arrays of the *previous* month end
            (the information set used on that day); -1 when unavailable.
        weights_m: (M, N) winsorized ME at month ends (NaN outside the universe).
        chars_m: (M, N, K) standardized characteristics.
        incl_m: (M, K) characteristic inclusion flags.
        ind_m: (M, N) FF49 industry numbers or None for the characteristics-only model.
        day_range: days to process.
    Returns:
        dict with ``eps`` and ``fitted`` (T, N) float32 (NaN outside regressions),
        ``rss``, ``tss``, ``nobs``, ``npar`` per day (weighted with the normalized
        weights) and ``rss_eq``, ``tss_eq`` for the equal-weighted (OLS) fit.
    """
    T, N = exret.shape
    eps = np.full((T, N), np.nan, dtype=np.float32)
    fit = np.full((T, N), np.nan, dtype=np.float32)
    stats = {k: np.full(T, np.nan) for k in ("rss", "tss", "nobs", "npar", "rss_eq", "tss_eq")}
    for t in day_range:
        m = month_of_day[t]
        if m < 0:
            continue
        w = weights_m[m]
        r = exret[t]
        cols = np.flatnonzero(incl_m[m])
        X = chars_m[m][:, cols]
        ok = np.isfinite(w) & (w > 0) & np.isfinite(r) & np.isfinite(X).all(axis=1)
        if ind_m is not None:
            ok &= np.isfinite(ind_m[m])
        idx = np.flatnonzero(ok)
        if len(idx) < 3 * (len(cols) + (config.N_INDUSTRIES if ind_m is not None else 1)) // 2:
            continue
        if ind_m is not None:
            D = industry_dummies(ind_m[m][idx])
            D = D[:, D.sum(axis=0) > 0]
            Z = np.hstack([D, X[idx]])
        else:
            Z = np.hstack([np.ones((len(idx), 1)), X[idx]])
        wn = w[idx] / w[idx].sum()
        rr = r[idx].astype("float64")
        f, rhat, rank = wls_cross_section(rr, Z, wn)
        e = rr - rhat
        eps[t, idx], fit[t, idx] = e, rhat
        rbar = (wn * rr).sum()
        stats["rss"][t] = (wn * e * e).sum()
        stats["tss"][t] = (wn * (rr - rbar) ** 2).sum()
        stats["nobs"][t], stats["npar"][t] = len(idx), rank
        # equal-weighted OLS fit of the same model (Figure 1 robustness)
        f0, rhat0, _ = wls_cross_section(rr, Z, np.full(len(idx), 1.0 / len(idx)))
        stats["rss_eq"][t] = ((rr - rhat0) ** 2).mean()
        stats["tss_eq"][t] = ((rr - rr.mean()) ** 2).mean()
    return {"eps": eps, "fitted": fit, **stats}


def zero_ea_residuals(exret: np.ndarray, eps: np.ndarray, ea_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Earnings-announcement adjustment (Equations 5-6).

    eps_tilde = 0 inside the [-1, +1] announcement window, else eps;
    r_tilde = r - eps_tilde (systematic shock).
    """
    eps_t = np.where(ea_mask & np.isfinite(eps), 0.0, eps).astype(np.float32)
    r_t = (exret - eps_t).astype(np.float32)
    r_t[~np.isfinite(eps_t)] = np.nan
    return eps_t, r_t


def annual_adj_r2(dates: pd.DatetimeIndex, rss: np.ndarray, tss: np.ndarray, nobs: np.ndarray,
                  npar: np.ndarray) -> pd.Series:
    """Annual adjusted R^2 (Equation 7): 1 - (RSS_y/(N_y-K_y)) / (TSS_y/(N_y-D_y)).

    RSS and TSS are daily sums of normalized-weight squared deviations, summed
    within the year; N_y stock-days, K_y estimated parameters, D_y trading days.
    """
    df = pd.DataFrame({"rss": rss, "tss": tss, "n": nobs, "k": npar}, index=dates).dropna()
    g = df.groupby(df.index.year)
    out = 1 - (g["rss"].sum() / (g["n"].sum() - g["k"].sum())) / (g["tss"].sum() / (g["n"].sum() - g.size()))
    return out.rename("adj_r2")


def ff3_timeseries_fit(exret: np.ndarray, ff: np.ndarray, month_key: np.ndarray, w_day: np.ndarray,
                       min_obs: int = config.MIN_DAILY_OBS) -> dict[str, np.ndarray]:
    """Stock-by-stock FF3 time-series benchmark (Section 3.1, Figure 1 right panel).

    Within each calendar month every stock's daily excess returns are regressed
    on an intercept and the daily MKT, SMB, HML factors (4 parameters per
    stock-month; at least ``min_obs`` days). Daily weighted RSS/TSS use the same
    normalized weights ``w_day`` (T, N) as the cross-sectional model.
    """
    T, N = exret.shape
    out = {k: np.full(T, np.nan) for k in ("rss", "tss", "nobs", "npar", "rss_eq", "tss_eq")}
    resid = np.full((T, N), np.nan)
    bounds = np.flatnonzero(np.r_[True, month_key[1:] != month_key[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        R = exret[a:b].astype("float64")
        F = np.hstack([np.ones((b - a, 1)), ff[a:b]])
        valid = np.isfinite(R) & np.isfinite(w_day[a:b])
        cnt = valid.sum(axis=0)
        full = np.flatnonzero(cnt == b - a)
        if full.size and b - a >= min_obs:
            beta, *_ = np.linalg.lstsq(F, R[:, full], rcond=None)
            resid[a:b, full] = R[:, full] - F @ beta
        for j in np.flatnonzero((cnt >= min_obs) & (cnt < b - a)):
            v = valid[:, j]
            beta, *_ = np.linalg.lstsq(F[v], R[v, j], rcond=None)
            resid[a:b, j][v] = R[v, j] - F[v] @ beta
        npar_month = 4 * ((cnt >= min_obs).sum())
        for t in range(a, b):
            ok = np.isfinite(resid[t]) & np.isfinite(w_day[t])
            if ok.sum() < 3:
                continue
            wn = w_day[t, ok] / w_day[t, ok].sum()
            r = exret[t, ok].astype("float64")
            e = resid[t, ok]
            out["rss"][t] = (wn * e * e).sum()
            out["tss"][t] = (wn * (r - (wn * r).sum()) ** 2).sum()
            out["rss_eq"][t] = (e * e).mean()
            out["tss_eq"][t] = ((r - r.mean()) ** 2).mean()
            out["nobs"][t] = ok.sum()
            out["npar"][t] = npar_month / (b - a)
    return out
