"""Daily distributed-lag Fama-MacBeth regressions, spline basis and impulse responses (Sections 6-7).

Shocks of days before a stock's first observation (or outside the decomposition
sample) enter the lag windows as zero; the paper does not specify how missing
days inside the 252-day lookback are treated.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import norm

from src import config
from src.stats import nw_mean


def spline_basis(knots: tuple[int, ...] = config.SPLINE_KNOTS, lags: np.ndarray | None = None) -> np.ndarray:
    """Degree-1 B-spline (tent) basis B(tau) (Equation 22, Figure A.5).

    Uses scipy.interpolate.BSpline.design_matrix with degree 1 and the knot
    vector padded once at both ends, so the K = len(knots) tent functions peak
    at the knots and form a partition of unity over [knots[0], knots[-1]].

    Returns:
        (len(lags), K) matrix; row j is B(lags[j])'.
    """
    kn = np.asarray(knots, dtype=float)
    lags = np.arange(int(kn[0]), int(kn[-1]) + 1) if lags is None else np.asarray(lags)
    t = np.r_[kn[0], kn, kn[-1]]
    B = BSpline.design_matrix(lags.astype(float), t, 1).toarray()
    assert B.shape == (len(lags), len(kn))
    assert np.allclose(B.sum(axis=1), 1.0)
    return B


def signed_shocks(eps_t: np.ndarray, r_t: np.ndarray) -> dict[str, np.ndarray]:
    """eps~+, eps~-, r~+, r~- and pooled r~ (Section 6.1); missing values set to 0, float32 (T, N)."""
    e = np.nan_to_num(eps_t.astype("float32"))
    r = np.nan_to_num(r_t.astype("float32"))
    return {"eps_pos": np.maximum(e, 0), "eps_neg": np.minimum(e, 0),
            "r_pos": np.maximum(r, 0), "r_neg": np.minimum(r, 0), "r": r}


def lag_window(x: np.ndarray, t: int, first_lag: int, last_lag: int) -> np.ndarray:
    """(last_lag - first_lag + 1, N) array whose row j is x[t - (first_lag + j)] (zeros before the sample)."""
    L = last_lag - first_lag + 1
    out = np.zeros((L, x.shape[1]), dtype=x.dtype)
    hi = t - first_lag  # index of lag first_lag
    lo = t - last_lag
    a = max(lo, 0)
    if hi >= a:
        blk = x[a:hi + 1][::-1]  # row 0 = lag first_lag
        out[: len(blk)] = blk
    return out


def daily_dl_regression(y: np.ndarray, w_day: np.ndarray, shocks: list[np.ndarray], days: np.ndarray,
                        first_lag: int, last_lag: int, basis: np.ndarray | None = None) -> np.ndarray:
    """Daily cross-sectional WLS of y_t on an intercept and lagged shocks (Equations 19, 23, 25).

    Args:
        y: (T, N) dependent variable for day t (daily excess return r_{i,t} or R^21_{i,t}).
        w_day: (T, N) regression weights on day t (NaN/0 = not in the cross-section).
        shocks: list of S (T, N) shock panels (zero where missing).
        days: trading-day indices t to estimate.
        first_lag, last_lag: lag range tau.
        basis: optional (L, K) spline basis; when given the regressors are the
            projections g = sum_tau B(tau) x_{t-tau} (Equation 23) and the
            returned coefficients are the implied lag coefficients
            lambda_tau = B(tau)' gamma.

    Returns:
        (len(days), S, L) lag coefficients lambda_{tau,t} (NaN on infeasible days).
    """
    L = last_lag - first_lag + 1
    S = len(shocks)
    out = np.full((len(days), S, L), np.nan)
    for j, t in enumerate(days):
        w = w_day[t]
        ok = np.isfinite(y[t]) & np.isfinite(w) & (w > 0)
        n = int(ok.sum())
        regs = []
        for x in shocks:
            win = lag_window(x[:, ok], t, first_lag, last_lag).astype("float64")  # (L, n)
            regs.append((basis.T @ win).T if basis is not None else win.T)
        X = np.column_stack([np.ones(n)] + regs)
        if n < X.shape[1] + 10:
            continue
        ww = w[ok] / w[ok].sum()
        sw = np.sqrt(ww)
        Xw = X * sw[:, None]
        yw = y[t, ok].astype("float64") * sw
        # WLS normal equations (Equation 19/23): (X'WX) b = X'Wy
        XtX = Xw.T @ Xw
        coef = np.linalg.lstsq(XtX, Xw.T @ yw, rcond=None)[0][1:]
        if basis is not None:
            K = basis.shape[1]
            out[j] = np.stack([basis @ coef[s * K:(s + 1) * K] for s in range(S)])
        else:
            out[j] = coef.reshape(S, L)
    return out


def impulse_response(lam: np.ndarray, sign: float, shock: float = config.SHOCK_SIZE, conf: float = 0.95
                     ) -> dict[str, np.ndarray]:
    """Average impulse response of a +/-1% shock (Equations 20-21).

    Args:
        lam: (T, H) daily lag coefficients lambda_{h,t} estimated on day t, h = 1..H
            (row t holds the regression of day t).
        sign: +1 for a positive shock, -1 for a negative shock.

    For every start date t, IR_t(tau) = (1 + sign*shock) prod_{h=1}^{tau}
    (1 + sign*shock*lambda_{h,t+h}) - 1; IR(tau) is the average over t, with
    Newey-West 95% bands from the time series of IR_t(tau).

    Returns:
        dict with ``ir`` (H+1,) including tau = 0, ``lo`` and ``hi`` bands, ``n`` starts used.
    """
    T, H = lam.shape
    starts = T - H
    if starts <= 2:
        raise ValueError("too few days for the impulse response")
    paths = np.full((starts, H + 1), np.nan)
    for t in range(starts):
        steps = 1 + sign * shock * lam[t + 1 + np.arange(H), np.arange(H)]
        paths[t] = (1 + sign * shock) * np.r_[1.0, np.cumprod(steps)] - 1
    good = np.isfinite(paths).all(axis=1)
    paths = paths[good]
    z = float(norm.ppf(0.5 + conf / 2))
    ir, lo, hi = np.empty(H + 1), np.empty(H + 1), np.empty(H + 1)
    for h in range(H + 1):
        m, t_stat = nw_mean(paths[:, h])
        se = abs(m / t_stat) if np.isfinite(t_stat) and t_stat != 0 else 0.0
        ir[h], lo[h], hi[h] = m, m - z * se, m + z * se
    return {"ir": ir, "lo": lo, "hi": hi, "n": int(good.sum()), "paths_mean": paths.mean(axis=0)}


def triangular_weights(max_lag: int = 41, month_len: int = 20) -> np.ndarray:
    """Monthly overlap weights n_tau / 20 for tau = 2..41 (Appendix C, Figure 7).

    n_tau = tau - 1 for 2 <= tau <= 21 and 42 - tau for 22 <= tau <= 41.
    Returns (max_lag + 1,) array indexed by tau (zero for tau < 2).
    """
    w = np.zeros(max_lag + 1)
    for tau in range(2, max_lag + 1):
        n = tau - 1 if tau <= month_len + 1 else 2 * month_len + 2 - tau
        w[tau] = n / month_len
    return w


def forward_sum(x: np.ndarray, h: int) -> np.ndarray:
    """(T, N) sum of x over days t..t+h-1 (NaN when any of them is missing or beyond the sample)."""
    T = x.shape[0]
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(np.nan_to_num(x.astype("float64")), axis=0)])
    cnt = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(np.isfinite(x), axis=0)])
    out = np.full(x.shape, np.nan)
    out[: T - h + 1] = c[h:] - c[: T - h + 1]
    n = np.full(x.shape, 0.0)
    n[: T - h + 1] = cnt[h:] - cnt[: T - h + 1]
    out[n < h] = np.nan
    return out
