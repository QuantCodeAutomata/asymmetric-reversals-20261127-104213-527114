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


def monthly_inputs(panel: dict) -> dict:
    """Shared month-level inputs of the monthly experiments (Sections 4-5, 7).

    Returns a dict of (M, N) arrays indexed by formation month m (row of ``mends``):
        ``exret_next``: compounded month m+1 return minus the month m+1 risk-free rate;
        ``ret_form``: 1-day-skipped compounded raw return of month m (STREV signal);
        ``w``: end-of-month-m market equity winsorized at the 80th NYSE percentile;
        ``valid``: investable at the end of month m;
        RES, SYS, RES_pos, RES_neg, CONC_pos, CONC_neg, CONC from the earnings-adjusted decomposition;
        ``IVOL`` (raw residuals) and ``IVOL_noEA`` (earnings-adjusted residuals);
    plus ``blocks`` (active-day row ranges) and ``rf_m`` (M,) monthly risk-free rates.
    """
    dates = panel["dates"]
    blocks = month_blocks(dates, skip_last=True)
    full = month_blocks(dates, skip_last=False)
    rf_d = panel["ff_d"]["RF"].reindex(dates).to_numpy()
    rf_m = np.array([np.expm1(np.log1p(rf_d[a:b]).sum()) for a, b in full])
    ret_m = compound(panel["ret"], full)
    exret_m = ret_m - rf_m[:, None]
    exret_next = np.full_like(exret_m, np.nan)
    exret_next[:-1] = exret_m[1:]
    out = {"blocks": blocks, "rf_m": rf_m, "exret_m": exret_m, "exret_next": exret_next,
           "ret_form": compound(panel["ret"], blocks),
           "w": panel["wgt"].astype("float64"), "valid": panel["univ"].astype(bool)}
    out.update(res_sys(panel["eps_t"], panel["r_t"], blocks))
    out["IVOL"] = ivol(panel["eps_ci"], blocks)
    out["IVOL_noEA"] = ivol(panel["eps_t"], blocks)
    assert len(blocks) == len(panel["mends"])
    return out
