"""Assemble the aligned stock panel used by every experiment (Sections 2-3).

Outputs (all indexed by trading day or month end, columns = security keys):
returns, excess returns, unadjusted prices, volume, high/low, market equity,
investable universe, regression weights, NYSE flags, FF49 industries,
earnings-announcement windows and month-end characteristics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.panel import daily_returns, month_ends, to_wide
from src.universe import dividend_factor, investable_mask, split_factor_after, winsorized_weights

log = logging.getLogger(__name__)


def security_table(universe: pd.DataFrame) -> pd.DataFrame:
    """One row per security: tickers (latest first), CIK, primary exchange, last year-end seen."""
    u = universe.sort_values("last_seen")
    g = u.groupby("security")
    out = pd.DataFrame({
        "tickers": g["ticker"].agg(lambda s: list(dict.fromkeys(reversed(list(s))))),
        "cik": g["cik"].last(),
        "primary_exchange": g["primary_exchange"].last(),
        "last_seen": g["last_seen"].max(),
        "first_seen": g["first_seen"].min(),
    })
    out.index = out.index.astype(str)
    return out


def unadjust_dividends(close: pd.DataFrame, secs: pd.DataFrame, divs: pd.DataFrame,
                       window_end: str = config.PRICE_END) -> pd.DataFrame:
    """Split-adjusted closes without the dividend back-adjustment.

    Dividend-adjusted close = unadjusted close x F_t (``dividend_factor``); the
    provider's dividends are matched by ticker inside the security's trading life.
    """
    d = divs[divs["distribution_type"].isin(["recurring", "special", "supplemental", "irregular"])
             | divs["distribution_type"].isna()]
    d = d.assign(ex=pd.to_datetime(d["ex_dividend_date"])).dropna(subset=["historical_adjustment_factor"])
    by_ticker = {t: g for t, g in d.groupby("ticker")}
    out = close.copy()
    end = pd.Timestamp(window_end)
    for sec in close.columns:
        col = close[sec]
        valid = col.dropna()
        if valid.empty:
            continue
        lo, hi = valid.index[0], valid.index[-1]
        ev = [by_ticker[t] for t in secs.at[sec, "tickers"] if t in by_ticker]
        if not ev:
            continue
        ev = pd.concat(ev).drop_duplicates("ex")
        ev = ev[(ev["ex"] > lo) & (ev["ex"] <= hi + pd.Timedelta(days=1))]
        if ev.empty:
            continue
        f = dividend_factor(close.index, ev["ex"].values.astype("datetime64[ns]"),
                            ev["historical_adjustment_factor"].to_numpy(float), end)
        out[sec] = col / f
    return out


def shares_panel(dates: pd.DatetimeIndex, secs: pd.DataFrame, income: pd.DataFrame, splits: pd.DataFrame,
                 window_end: str = config.PRICE_END) -> pd.DataFrame:
    """Daily split-adjusted shares outstanding (adjusted to the split basis of ``window_end``).

    Uses the basic shares outstanding of the latest fiscal quarter ended on or
    before each day (records older than ``config.SHARES_STALE_MONTHS`` are dropped).
    The reported counts are restated to the latest split basis, so they are divided
    by the split ratios executed after ``window_end`` and market equity =
    split-adjusted price x shares on the same split basis.
    """
    inc = income.dropna(subset=["basic_shares_outstanding"])
    inc = inc[inc["basic_shares_outstanding"] > 0].assign(period_end=lambda x: pd.to_datetime(x["period_end"]))
    inc = inc.sort_values("period_end").drop_duplicates(["cik", "period_end"], keep="last")
    by_cik = {c: g for c, g in inc.groupby("cik")}
    sp = splits.assign(d=pd.to_datetime(splits["execution_date"]),
                       r=splits["split_to"] / splits["split_from"])
    sp = sp[np.isfinite(sp["r"]) & (sp["r"] > 0)]
    sp_by = {t: g for t, g in sp.groupby("ticker")}
    end = pd.Timestamp(window_end)
    out = np.full((len(dates), len(secs)), np.nan)
    dv = dates.values
    for j, sec in enumerate(secs.index):
        q = by_cik.get(secs.at[sec, "cik"])
        if q is None:
            continue
        pe = q["period_end"].values.astype("datetime64[ns]")
        ev = [sp_by[t] for t in secs.at[sec, "tickers"] if t in sp_by]
        if ev:
            e = pd.concat(ev).drop_duplicates("d")
            # after window_end a ticker may name another security; only forward splits are applied there
            e = e[(e["d"] <= end) | (e["r"] > 1)]
            fac = split_factor_after(e["d"].values.astype("datetime64[ns]"), e["r"].to_numpy(float), end)
        else:
            fac = 1.0
        sh = q["basic_shares_outstanding"].to_numpy(float) / fac
        sh = np.where(shares_jump_ok(sh), sh, np.nan)
        idx = np.searchsorted(pe, dv, side="right") - 1
        ok = idx >= 0
        age = np.where(ok, (dv - pe[np.maximum(idx, 0)]).astype("timedelta64[D]").astype(float), np.inf)
        ok &= age <= config.SHARES_STALE_MONTHS * 31
        out[ok, j] = sh[idx[ok]]
    return pd.DataFrame(out, index=dates, columns=secs.index)


def sic_to_ff49(sic: pd.Series, ranges: list[tuple[int, int, int]]) -> pd.Series:
    """Fama-French 49 industry number (1..49) of SIC codes; unmatched codes -> 49 (Other)."""
    s = pd.to_numeric(sic, errors="coerce")
    out = pd.Series(np.nan, index=sic.index)
    for ind, lo, hi in ranges:
        out[(s >= lo) & (s <= hi) & out.isna()] = ind
    out[s.notna() & out.isna()] = 49
    return out


def earnings_dates(income: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Quarterly earnings-announcement date proxy per (cik, period_end).

    The announcement is taken as the first 8-K filed between
    ``config.EA_8K_MIN_LAG_DAYS`` days after the fiscal quarter end and the 10-Q/10-K
    filing for that quarter (earnings releases are furnished on 8-K, Item 2.02);
    without such an 8-K the 10-Q/10-K filing date is used.
    """
    q = income[["cik", "period_end"]].dropna().drop_duplicates()
    q = q.assign(period_end=pd.to_datetime(q["period_end"])).sort_values("period_end")
    f = filings.assign(filing_date=pd.to_datetime(filings["filing_date"])).sort_values("filing_date")
    per = f[f["form_type"].isin(["10-Q", "10-K"])][["cik", "filing_date"]].rename(columns={"filing_date": "fq"})
    q = pd.merge_asof(q.assign(lo=q["period_end"] + pd.Timedelta(days=1)).sort_values("lo"), per,
                      left_on="lo", right_on="fq", by="cik", direction="forward",
                      tolerance=pd.Timedelta(days=config.EA_SEARCH_MAX_DAYS))
    k8 = f[f["form_type"] == "8-K"][["cik", "filing_date"]].rename(columns={"filing_date": "f8"})
    q = q.assign(lo8=q["period_end"] + pd.Timedelta(days=config.EA_8K_MIN_LAG_DAYS)).sort_values("lo8")
    q = pd.merge_asof(q, k8, left_on="lo8", right_on="f8", by="cik", direction="forward",
                      tolerance=pd.Timedelta(days=config.EA_SEARCH_MAX_DAYS))
    use8 = q["f8"].notna() & (q["fq"].isna() | (q["f8"] <= q["fq"]))
    q["ea_date"] = np.where(use8, q["f8"], q["fq"])
    return q.dropna(subset=["ea_date"])[["cik", "period_end", "ea_date"]].reset_index(drop=True)


def ea_window_mask(dates: pd.DatetimeIndex, secs: pd.DataFrame, ea: pd.DataFrame,
                   window: tuple[int, int] = config.EA_WINDOW) -> np.ndarray:
    """Boolean (days x securities): True within [-1, +1] trading days of an announcement (Section 3).

    An announcement on a non-trading day is assigned to the next trading day.
    """
    mask = np.zeros((len(dates), len(secs)), dtype=bool)
    by_cik = {c: g["ea_date"].values.astype("datetime64[ns]") for c, g in ea.groupby("cik")}
    for j, sec in enumerate(secs.index):
        e = by_cik.get(secs.at[sec, "cik"])
        if e is None:
            continue
        pos = np.searchsorted(dates.values, e, side="left")
        pos = pos[pos < len(dates)]
        for k in range(window[0], window[1] + 1):
            p = pos + k
            p = p[(p >= 0) & (p < len(dates))]
            mask[p, j] = True
    return mask


def shares_sane(vol_m: pd.DataFrame, shares_m: pd.DataFrame,
                min_turnover: float = config.MIN_MONTHLY_TURNOVER) -> pd.DataFrame:
    """Mask of month-end share counts consistent with traded volume.

    A share count whose trailing 12-month median monthly turnover (volume / shares)
    is below ``min_turnover`` is taken to be on a wrong scale (e.g. a different
    share class or unit) and is rejected.
    """
    turn = (vol_m / shares_m).rolling(12, min_periods=3).median()
    return turn >= min_turnover


def shares_jump_ok(sh: np.ndarray, factor: float = config.SHARES_JUMP_FACTOR) -> np.ndarray:
    """False for records more than ``factor`` away from the median of the previous four (trailing only)."""
    prev = pd.Series(sh).shift(1).rolling(4, min_periods=1).median().to_numpy()
    ratio = sh / prev
    return ~(np.isfinite(ratio) & ((ratio > factor) | (ratio < 1.0 / factor)))

