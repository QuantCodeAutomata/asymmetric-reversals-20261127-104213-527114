"""Short-horizon anomaly characteristics (Section 5.5, Appendix E.1) and the mispricing score (Section 5.3).

All monthly characteristics of month m use the active days D_{i,m} (trading
days of month m without its final trading day) and require at least
``config.MIN_DAILY_OBS`` valid daily observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, skew

from src import config

ZOO_NAMES = ("IVOL_FF3", "RetVol", "Skew", "ISkew_FF3", "Coskew", "MAX", "MAX5", "SMAX", "LagRet",
             "Amihud", "ZeroTrades", "Spread", "Turnover")


def ts_residuals(Y: np.ndarray, X: np.ndarray, min_obs: int) -> np.ndarray:
    """Stock-by-stock OLS residuals of Y (D, N) on [1, X] (D, k); NaN where Y is missing or too few days."""
    D, N = Y.shape
    F = np.hstack([np.ones((D, 1)), X])
    out = np.full(Y.shape, np.nan)
    valid = np.isfinite(Y)
    cnt = valid.sum(axis=0)
    full = np.flatnonzero(cnt == D)
    if full.size and D >= min_obs:
        beta, *_ = np.linalg.lstsq(F, Y[:, full], rcond=None)
        out[:, full] = Y[:, full] - F @ beta
    for j in np.flatnonzero((cnt >= min_obs) & (cnt < D)):
        v = valid[:, j]
        beta, *_ = np.linalg.lstsq(F[v], Y[v, j], rcond=None)
        out[v, j] = Y[v, j] - F[v] @ beta
    return out


def _skew(b: np.ndarray) -> np.ndarray:
    """Standardized third central moment per column (scipy.stats.skew, NaNs omitted)."""
    return np.asarray(skew(b, axis=0, nan_policy="omit"), dtype=float)


def coskew(e: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Coskewness: E[e_i (m - mbar)^2] / (sqrt(E[e_i^2]) E[(m - mbar)^2]) (Harvey and Siddique 2000; Appendix E.1).

    e: (D, N) CAPM residuals; m: (D,) market excess return. Moments over the
    days on which each stock has a residual.
    """
    out = np.full(e.shape[1], np.nan)
    for j in range(e.shape[1]):
        v = np.isfinite(e[:, j])
        if v.sum() < 3:
            continue
        mm = m[v] - m[v].mean()
        ee = e[v, j]
        den = np.sqrt((ee ** 2).mean()) * (mm ** 2).mean()
        out[j] = (ee * mm ** 2).mean() / den if den > 0 else np.nan
    return out


def corwin_schultz(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Daily Corwin and Schultz (2012) high-low spread estimates for day pairs (t, t+1), stored on row t.

    beta = sum of the two squared log high/low ranges, gamma = squared log range
    of the two-day high/low; alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2)
    - sqrt(gamma / (3 - 2 sqrt 2)); S = 2 (e^alpha - 1) / (1 + e^alpha).
    Negative estimates are set to zero. The last row is NaN.
    """
    h = np.where(high > 0, high, np.nan).astype("float64")
    lo = np.where(low > 0, low, np.nan).astype("float64")
    k = 3 - 2 * np.sqrt(2)
    out = np.full(h.shape, np.nan)
    with np.errstate(all="ignore"):
        b = np.log(h[:-1] / lo[:-1]) ** 2 + np.log(h[1:] / lo[1:]) ** 2
        g = np.log(np.fmax(h[:-1], h[1:]) / np.fmin(lo[:-1], lo[1:])) ** 2
        a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
        s = 2 * (np.exp(a) - 1) / (1 + np.exp(a))
    out[:-1] = np.where(np.isfinite(s), np.maximum(s, 0), np.nan)
    return out


def zoo_characteristics(ret: np.ndarray, exret: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close_u: np.ndarray, volume: np.ndarray, shares: np.ndarray, ff3: np.ndarray,
                        blocks: list[tuple[int, int]], min_obs: int = config.MIN_DAILY_OBS
                        ) -> dict[str, np.ndarray]:
    """The 13 one-day-skipped short-term characteristics of Table 6 (Appendix E.1), each (M, N).

    Args:
        ret, exret: (T, N) daily raw and excess returns; high, low: daily high/low;
        close_u: unadjusted close; volume: share volume; shares: shares outstanding.
        ff3: (T, 3) daily MKT-RF, SMB, HML.
        blocks: active-day (start, stop) row ranges per month (last day excluded).
    """
    T, N = ret.shape
    M = len(blocks)
    out = {k: np.full((M, N), np.nan) for k in ZOO_NAMES}
    cs = corwin_schultz(high, low)
    dvol = np.where(volume > 0, np.abs(close_u.astype("float64")) * volume, np.nan)
    with np.errstate(all="ignore"):
        ami_d = np.abs(ret) / dvol
        turn_d = np.where(shares > 0, volume / shares, np.nan)
    # trailing 252-day volatility of daily excess returns (SMAX scaling)
    vol252 = pd.DataFrame(exret.astype("float64")).rolling(config.SMAX_VOL_DAYS, min_periods=120).std().to_numpy()
    for k, (a, b) in enumerate(blocks):
        if b - a < 2:
            continue
        R, E = ret[a:b].astype("float64"), exret[a:b].astype("float64")
        nv = np.isfinite(R).sum(axis=0)
        ok = nv >= min_obs
        e3 = ts_residuals(E, ff3[a:b], min_obs)
        ec = ts_residuals(E, ff3[a:b, :1], min_obs)
        with np.errstate(all="ignore"):
            out["IVOL_FF3"][k] = np.nanstd(e3, axis=0, ddof=1)
            out["RetVol"][k] = np.nanstd(E, axis=0, ddof=1)
            out["Skew"][k] = _skew(R)
            out["ISkew_FF3"][k] = _skew(e3)
            out["Coskew"][k] = coskew(ec, ff3[a:b, 0])
            out["MAX"][k] = np.nanmax(np.where(np.isfinite(R), R, -np.inf), axis=0)
            top = -np.sort(-np.where(np.isfinite(R), R, -np.inf), axis=0)[:5]
            out["MAX5"][k] = np.where(np.isfinite(top), top, np.nan).mean(axis=0)
            out["SMAX"][k] = out["MAX5"][k] / vol252[b - 1]
            out["LagRet"][k] = np.nansum(E, axis=0)
            ami_n = np.isfinite(ami_d[a:b]).sum(axis=0)
            out["Amihud"][k] = np.where(ami_n >= min_obs, np.nanmean(ami_d[a:b], axis=0), np.nan) * config.AMIHUD_SCALE
            V = volume[a:b]
            zero = ((V == 0) & np.isfinite(V)).sum(axis=0).astype(float)
            mturn = np.nansum(turn_d[a:b], axis=0)
            # Liu (2006): zero-volume days + (1 / monthly turnover) / 480,000
            out["ZeroTrades"][k] = zero + np.where(mturn > 0, (1.0 / mturn) / config.ZERO_TRADES_DEFLATOR, np.nan)
            # spread pairs (t, t+1) both inside the active days
            out["Spread"][k] = np.nanmean(cs[a:b - 1], axis=0)
            out["Turnover"][k] = np.nanmean(turn_d[a:b], axis=0)
        for name in ZOO_NAMES:
            out[name][k, ~ok] = np.nan
            out[name][k, ~np.isfinite(out[name][k])] = np.nan
    return out


# Stambaugh, Yu and Yuan (2015) anomalies available from the characteristic set:
# name -> +1 if a high value means overpriced (low future return), -1 otherwise.
MISP_ANOMALIES = {"chcsho_12m": 1, "oaccruals_at": 1, "noa_at": 1, "at_gr1": 1, "inv_gr1": 1,
                  "ret_12_1": -1, "gp_at": -1, "niq_at": -1}
MISP_MIN_ANOMALIES = 5  # minimum available anomaly ranks per stock-month; not specified in the paper


def misp_score(raw: dict[str, pd.DataFrame], univ: np.ndarray, months: pd.DatetimeIndex, secs: pd.Index
               ) -> np.ndarray:
    """Mispricing score: average cross-sectional percentile rank over the anomalies (Stambaugh et al. 2015).

    Each anomaly is ranked within the investable universe each month (1..100,
    higher = more overpriced); MISP is the mean of the available ranks.
    """
    M, N = univ.shape
    tot = np.zeros((M, N))
    cnt = np.zeros((M, N))
    for name, sign in MISP_ANOMALIES.items():
        a = raw[name].reindex(index=months, columns=secs).to_numpy("float64") * sign
        for i in range(M):
            v = univ[i] & np.isfinite(a[i])
            if v.sum() < 10:
                continue
            r = np.zeros(N)
            r[v] = 100 * rankdata(a[i, v]) / v.sum()
            tot[i] += r
            cnt[i] += v
    with np.errstate(all="ignore"):
        out = np.where(cnt >= MISP_MIN_ANOMALIES, tot / cnt, np.nan)
    return out
