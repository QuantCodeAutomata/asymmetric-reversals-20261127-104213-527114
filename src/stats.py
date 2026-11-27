"""Inference tools: Newey-West means, time-series alphas, Fama-MacBeth regressions, NYSE-breakpoint sorts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config


def nw_lags(T: int) -> int:
    """Automatic Newey-West lag selection floor(4 (T/100)^(2/9)) (Newey and West 1994 rule)."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def nw_mean(x: np.ndarray | pd.Series, lags: int | None = None) -> tuple[float, float]:
    """Time-series mean and its Newey-West t-statistic (statsmodels OLS on a constant, HAC covariance)."""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan")
    lags = nw_lags(len(x)) if lags is None else lags
    res = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def ts_alpha(y: pd.Series, factors: pd.DataFrame, lags: int | None = None) -> dict:
    """Time-series regression y = a + b'F + e with Newey-West t-statistics (statsmodels OLS, HAC)."""
    df = pd.concat([y.rename("y"), factors], axis=1, join="inner").dropna()
    if len(df) < len(factors.columns) + 3:
        return {"alpha": np.nan, "t_alpha": np.nan, "betas": {}, "t_betas": {}, "r2": np.nan, "n": len(df)}
    X = sm.add_constant(df[factors.columns])
    lags = nw_lags(len(df)) if lags is None else lags
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha": float(res.params["const"]), "t_alpha": float(res.tvalues["const"]),
            "betas": {c: float(res.params[c]) for c in factors.columns},
            "t_betas": {c: float(res.tvalues[c]) for c in factors.columns},
            "r2": float(res.rsquared), "n": int(len(df))}


def wls_slopes(y: np.ndarray, X: np.ndarray, w: np.ndarray | None) -> tuple[np.ndarray, float] | None:
    """One cross-sectional (W)LS regression with intercept; returns (coefficients incl. intercept, adj. R2).

    Custom implementation of the Fama-MacBeth first stage (Fama and MacBeth 1973);
    WLS weights are normalized to sum to one.
    """
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if w is not None:
        ok &= np.isfinite(w) & (w > 0)
    n, k = int(ok.sum()), X.shape[1] + 1
    if n < k + 5:
        return None
    Z = np.column_stack([np.ones(n), X[ok]])
    yy = y[ok]
    ww = np.ones(n) / n if w is None else w[ok] / w[ok].sum()
    sw = np.sqrt(ww)
    coef, *_ = np.linalg.lstsq(Z * sw[:, None], yy * sw, rcond=None)
    e = yy - Z @ coef
    ybar = (ww * yy).sum()
    tss = (ww * (yy - ybar) ** 2).sum()
    r2 = 1 - (ww * e * e).sum() / tss if tss > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return coef, float(adj)


def fama_macbeth(y: np.ndarray, X: dict[str, np.ndarray], w: np.ndarray | None,
                 periods: np.ndarray | None = None, lags: int | None = None) -> pd.DataFrame:
    """Fama-MacBeth regression of y on the named regressors (all arrays (P, N)).

    Each period runs a cross-sectional (W)LS with intercept; slopes are averaged
    over time and t-statistics use Newey-West standard errors with automatic lag
    selection. Returns a frame indexed by regressor name (``const`` first) with
    columns ``coef``, ``t``, plus a row ``adj_r2`` (mean adjusted R2, t NaN) and
    ``n_periods``.
    """
    names = list(X)
    P = y.shape[0]
    periods = np.arange(P) if periods is None else periods
    coefs, r2s = [], []
    for p in periods:
        Xp = np.column_stack([X[k][p] for k in names]) if names else np.empty((y.shape[1], 0))
        out = wls_slopes(y[p], Xp, None if w is None else w[p])
        if out is None:
            continue
        coefs.append(out[0])
        r2s.append(out[1])
    rows = {}
    C = np.array(coefs) if coefs else np.full((0, len(names) + 1), np.nan)
    for j, k in enumerate(["const"] + names):
        rows[k] = nw_mean(C[:, j], lags) if len(C) else (np.nan, np.nan)
    df = pd.DataFrame(rows, index=["coef", "t"]).T
    df.loc["adj_r2"] = [float(np.mean(r2s)) if r2s else np.nan, np.nan]
    df.loc["n_periods"] = [float(len(C)), np.nan]
    return df


def fm_coef_series(y: np.ndarray, X: dict[str, np.ndarray], w: np.ndarray | None, periods: np.ndarray) -> np.ndarray:
    """Per-period Fama-MacBeth coefficients (len(periods), 1 + K), NaN rows where a regression is infeasible."""
    names = list(X)
    out = np.full((len(periods), len(names) + 1), np.nan)
    for i, p in enumerate(periods):
        Xp = np.column_stack([X[k][p] for k in names])
        res = wls_slopes(y[p], Xp, None if w is None else w[p])
        if res is not None:
            out[i] = res[0]
    return out


def nyse_breakpoints(x: np.ndarray, valid: np.ndarray, is_nyse: np.ndarray, n_q: int = config.N_QUANTILES
                     ) -> np.ndarray:
    """Interior quantile breakpoints of ``x`` among valid NYSE stocks (n_q - 1 values)."""
    v = x[valid & is_nyse & np.isfinite(x)]
    if len(v) < n_q:
        return np.full(n_q - 1, np.nan)
    return np.quantile(v, np.arange(1, n_q) / n_q)


def assign_quantiles(x: np.ndarray, valid: np.ndarray, is_nyse: np.ndarray, n_q: int = config.N_QUANTILES
                     ) -> np.ndarray:
    """Quantile label 0..n_q-1 using NYSE breakpoints (-1 when not sortable)."""
    bp = nyse_breakpoints(x, valid, is_nyse, n_q)
    lab = np.full(x.shape, -1, dtype=int)
    ok = valid & np.isfinite(x)
    if np.isnan(bp).any():
        return lab
    lab[ok] = np.searchsorted(bp, x[ok], side="right")
    return lab


def sort_portfolios(signal: np.ndarray, valid: np.ndarray, is_nyse: np.ndarray, w: np.ndarray,
                    ret_next: np.ndarray, n_q: int = config.N_QUANTILES) -> dict[str, np.ndarray]:
    """Monthly value-weighted quantile portfolios (all arrays (M, N), ``is_nyse`` (N,)).

    Row m holds the formation month; ``ret`` (M, n_q) is the month m+1 return of
    portfolios formed at the end of month m with weights ``w`` (winsorized ME).
    Also returns ``labels`` (M, N), ``n`` (M, n_q) stock counts and ``turnover``
    (M, n_q) one-way weight turnover (half the sum of absolute weight changes,
    weights drifted by the realized holding return).
    """
    M, N = signal.shape
    rets = np.full((M, n_q), np.nan)
    counts = np.zeros((M, n_q))
    turn = np.full((M, n_q), np.nan)
    labels = np.full((M, N), -1, dtype=int)
    prev_w = [None] * n_q
    for m in range(M):
        ok = valid[m] & np.isfinite(w[m]) & (w[m] > 0)
        lab = assign_quantiles(signal[m], ok, is_nyse, n_q)
        labels[m] = lab
        for q in range(n_q):
            sel = (lab == q) & np.isfinite(ret_next[m])
            counts[m, q] = sel.sum()
            if sel.sum() == 0:
                prev_w[q] = None
                continue
            wq = np.zeros(N)
            wq[sel] = w[m, sel] / w[m, sel].sum()
            rets[m, q] = float((wq * np.nan_to_num(ret_next[m])).sum())
            if prev_w[q] is not None:
                turn[m, q] = 0.5 * np.abs(wq - prev_w[q]).sum()
            drift = wq * (1 + np.nan_to_num(ret_next[m]))
            prev_w[q] = drift / drift.sum() if drift.sum() > 0 else None
    return {"ret": rets, "labels": labels, "n": counts, "turnover": turn}
