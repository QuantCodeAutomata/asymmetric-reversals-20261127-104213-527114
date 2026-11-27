"""Monthly formation-month signals from daily decomposed returns (Sections 4-5, Appendix E.1).

Every signal of month m uses the active days D_{i,m}: trading days of month m
excluding the final trading day of the month (one-day skip).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src import config
from src.panel import month_id


def month_blocks(dates: pd.DatetimeIndex, skip_last: bool = True) -> list[tuple[int, int]]:
    """(start, stop) row ranges of each calendar month; ``stop`` excludes the last day when ``skip_last``."""
    mid = month_id(dates)
    b = np.flatnonzero(np.r_[True, mid[1:] != mid[:-1], True])
    return [(int(s), int(e) - (1 if skip_last else 0)) for s, e in zip(b[:-1], b[1:])]


def block_apply(daily: np.ndarray, blocks: list[tuple[int, int]], fn: Callable[[np.ndarray], np.ndarray],
                min_obs: int = 1, valid: np.ndarray | None = None) -> np.ndarray:
    """(M, N) array of ``fn`` applied to each (days x N) block; NaN with fewer than ``min_obs`` valid days."""
    out = np.full((len(blocks), daily.shape[1]), np.nan)
    for k, (a, b) in enumerate(blocks):
        blk = daily[a:b].astype("float64")
        nv = np.isfinite(blk if valid is None else np.where(valid[a:b], blk, np.nan)).sum(axis=0)
        with np.errstate(all="ignore"):
            v = fn(blk)
        out[k] = np.where(nv >= min_obs, v, np.nan)
    return out


def _nansum(b: np.ndarray) -> np.ndarray:
    return np.nansum(b, axis=0)


def res_sys(eps_t: np.ndarray, r_t: np.ndarray, blocks: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """RES, SYS, RES+, RES- (Equations 10-13) and CONC+, CONC-, CONC (Section 5.4).

    RES = sum eps_tilde, SYS = sum r_tilde, RES+ = sum max(eps_tilde, 0),
    RES- = sum min(eps_tilde, 0); CONC+ = sqrt(sum (eps_tilde+)^2) / RES+ (0 if RES+ = 0),
    CONC- = sqrt(sum (eps_tilde-)^2) / RES- (0 if RES- = 0), CONC = CONC+ + CONC-.
    """
    pos = np.where(np.isfinite(eps_t), np.maximum(eps_t, 0), np.nan)
    neg = np.where(np.isfinite(eps_t), np.minimum(eps_t, 0), np.nan)
    out = {
        "RES": block_apply(eps_t, blocks, _nansum),
        "SYS": block_apply(r_t, blocks, _nansum),
        "RES_pos": block_apply(pos, blocks, _nansum),
        "RES_neg": block_apply(neg, blocks, _nansum),
    }
    l2p = block_apply(pos, blocks, lambda b: np.sqrt(np.nansum(b * b, axis=0)))
    l2n = block_apply(neg, blocks, lambda b: np.sqrt(np.nansum(b * b, axis=0)))
    with np.errstate(all="ignore"):
        out["CONC_pos"] = np.where(out["RES_pos"] != 0, l2p / out["RES_pos"], 0.0)
        out["CONC_neg"] = np.where(out["RES_neg"] != 0, l2n / out["RES_neg"], 0.0)
    for k in ("CONC_pos", "CONC_neg"):
        out[k][~np.isfinite(out["RES"])] = np.nan
    out["CONC"] = out["CONC_pos"] + out["CONC_neg"]
    return out


def ivol(eps: np.ndarray, blocks: list[tuple[int, int]], min_obs: int = 2) -> np.ndarray:
    """IVOL (Equation 14): sqrt(1/(n-1) sum (eps - mean eps)^2) over the active days."""
    return block_apply(eps, blocks, lambda b: np.nanstd(b, axis=0, ddof=1), min_obs=min_obs)


def ivol_horizon(eps: np.ndarray, blocks: list[tuple[int, int]], H: int) -> np.ndarray:
    """IVOL over the most recent ``H`` active residuals (Figure 2).

    Keeps the demeaning by the full formation-month mean eps_bar_{i,m} and the
    n_{i,m}-1 normalization of Equation (14), summing squared deviations of only
    the last H active days (so H = 1 is well defined).
    """
    def fn(b: np.ndarray) -> np.ndarray:
        n = np.isfinite(b).sum(axis=0)
        mu = np.nanmean(b, axis=0)
        last = b[-H:]
        ss = np.nansum((last - mu) ** 2, axis=0)
        return np.sqrt(ss / np.maximum(n - 1, 1))
    return block_apply(eps, blocks, fn, min_obs=2)


def monthly_sum(daily: np.ndarray, blocks: list[tuple[int, int]], min_obs: int = 1) -> np.ndarray:
    """Sum of daily values over each block (NaN with fewer than ``min_obs`` valid days)."""
    return block_apply(daily, blocks, _nansum, min_obs=min_obs)


def compound(daily_ret: np.ndarray, blocks: list[tuple[int, int]], min_obs: int = 1) -> np.ndarray:
    """Compounded return over each block."""
    return block_apply(daily_ret, blocks, lambda b: np.expm1(np.nansum(np.log1p(b), axis=0)), min_obs=min_obs)
