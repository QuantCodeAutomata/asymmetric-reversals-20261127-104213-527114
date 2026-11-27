"""Shared data pipeline: stock panel, characteristics and daily decomposition (Sections 2-3).

Every experiment calls :func:`load_panel`, which builds (or reads from the
cache outside the repository) one dict with:

- ``dates`` (T,) trading days; ``mends`` (M,) month-end trading days; ``secs`` (N,) security keys;
- daily (T, N) float32: ``ret``, ``exret``, ``close_u`` (split- but not dividend-adjusted),
  ``volume``, ``high``, ``low``, ``shares``; ``ea`` (T, N) bool earnings-announcement window;
- monthly (M, N): ``me`` (market equity, USD), ``univ`` (bool investable universe),
  ``wgt`` (ME winsorized at the 80th NYSE percentile, NaN outside the universe);
- (N,): ``is_nyse`` bool, ``ind`` FF49 industry (NaN if unknown), ``cik``;
- characteristics: ``chars`` (M, N, K) standardized, ``incl`` (M, K) bool, ``char_names`` (K,);
  ``ctrl_<name>`` (M, N) unstandardized Fama-French (2020) controls and ``sue`` (M, N);
- decomposition (T, N) float32: ``eps_ci`` (characteristics + industries, raw),
  ``eps_c`` (characteristics only, raw), ``eps_t`` (eps_ci zeroed in EA windows), ``r_t`` (r - eps_t);
  daily fit statistics ``fit_<model>_<stat>`` (T,);
- factors: ``ff_d`` daily DataFrame (MKT_RF, SMB, HML, RF), ``ff_m`` monthly DataFrame.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from data import data_loader as dl
from src import config
from src.build import (earnings_dates, ea_window_mask, security_table, shares_panel, sic_to_ff49,
                       unadjust_dividends)
from src.characteristics import (accounting_characteristics, price_characteristics, rank_standardize,
                                 select_characteristics)
from src.decomposition import daily_decomposition, ff3_timeseries_fit, zero_ea_residuals
from src.fundamentals import add_ttm_and_growth, merge_statements, sue, to_monthly
from src.panel import compound_monthly, daily_returns, month_ends, month_id, to_wide
from src.zoo import misp_score
from src.universe import investable_mask, winsorized_weights

log = logging.getLogger(__name__)
PANEL_DIR = config.CACHE_DIR / "panel"


def load_raw() -> dict:
    """All downloaded inputs through ``data/data_loader.py`` (cached outside the repository)."""
    universe = dl.load_reference_universe(config.REFERENCE_YEARS)
    bars, meta = dl.load_universe_bars(universe, config.PRICE_START, config.PRICE_END)
    return {
        "universe": universe, "bars": bars, "bars_meta": meta,
        "income": dl.load_fundamentals("income-statements", "quarterly", config.FUND_START, config.PRICE_END,
                                       config.FUND_INCOME_FIELDS),
        "balance": dl.load_fundamentals("balance-sheets", "quarterly", config.FUND_START, config.PRICE_END,
                                        config.FUND_BALANCE_FIELDS),
        "cashflow": dl.load_fundamentals("cash-flow-statements", "quarterly", config.FUND_START, config.PRICE_END,
                                         config.FUND_CASHFLOW_FIELDS),
        "splits": dl.load_splits(config.PRICE_START, config.SPLITS_END),
        "divs": dl.load_dividends(config.PRICE_START, config.PRICE_END),
        "filings": dl.load_filing_index(config.FILING_FORMS, config.FILINGS_START, config.FILINGS_END),
        "ff_d": dl.load_ff_daily(), "ff_m": dl.load_ff_monthly(), "ff49": dl.load_ff49_sic_map(),
    }


def _one_security_per_cik(me: pd.DataFrame, dvol: pd.DataFrame, cik: pd.Series) -> pd.DataFrame:
    """Keep, per CIK and month, only the security with the largest dollar volume.

    Shares outstanding come from company-level statements, so for multi-class
    firms every class would otherwise carry the full company market equity.
    """
    out = me.copy()
    c = cik.reindex(me.columns)
    multi = c[c.duplicated(keep=False) & (c != "")]
    for _, cols in multi.groupby(multi).groups.items():
        cols = list(cols)
        dv = dvol[cols].fillna(-1).to_numpy()
        best = dv.argmax(axis=1)
        mask = np.ones(dv.shape, dtype=bool)
        mask[np.arange(len(best)), best] = False
        sub = out[cols].to_numpy()
        sub[mask] = np.nan
        out[cols] = sub
    return out


def build_base(raw: dict) -> dict:
    """Daily/monthly panels, investable universe, weights, industries and EA windows."""
    secs_all = security_table(raw["universe"])
    ff_d = raw["ff_d"]
    dates = ff_d.index[(ff_d.index >= config.PRICE_START) & (ff_d.index <= config.PRICE_END)]
    mends = month_ends(dates)
    bars = raw["bars"]
    bars = bars[bars["date"].isin(dates)]
    secs_all = secs_all.loc[secs_all.index.isin(bars["security"].astype(str).unique())]
    me_bars = bars[bars["date"].isin(mends)].assign(security=lambda x: x["security"].astype(str))
    close_me = to_wide(me_bars, "close", mends, secs_all.index).astype("float64")
    me_bars = me_bars.assign(dv=me_bars["close"].astype("float64") * me_bars["volume"])
    dv_me = to_wide(me_bars, "dv", mends, secs_all.index)
    close_me_u = unadjust_dividends(close_me, secs_all, raw["divs"])
    shares_me = shares_panel(mends, secs_all, raw["income"], raw["splits"])
    me = _one_security_per_cik(close_me_u * shares_me, dv_me, secs_all["cik"])
    univ = investable_mask(me)
    first_m = mends[mends >= pd.Timestamp(config.DECOMP_START) - pd.offsets.MonthEnd(2)]
    keep = univ.loc[first_m].any(axis=0)
    secs = secs_all.loc[keep[keep].index]
    log.info("securities: %d with bars, %d ever investable since %s; universe size per month %d-%d",
             len(secs_all), len(secs), first_m[0].date(), univ.loc[first_m].sum(axis=1).min(),
             univ.loc[first_m].sum(axis=1).max())
    b = bars[bars["security"].isin(secs.index)].assign(security=lambda x: x["security"].astype(str))
    close = to_wide(b, "close", dates, secs.index)
    out = {"dates": dates, "mends": mends, "secs": secs.index.to_numpy(), "cik": secs["cik"].to_numpy()}
    ret = daily_returns(close)
    out["ret"] = ret.to_numpy()
    out["exret"] = (ret.astype("float64").sub(ff_d["RF"].reindex(dates), axis=0)).to_numpy("float32")
    out["close_u"] = unadjust_dividends(close.astype("float64"), secs, raw["divs"]).to_numpy("float32")
    del close
    for f in ("volume", "high", "low"):
        out[f] = to_wide(b, f, dates, secs.index).to_numpy("float32")
    del b
    out["shares"] = shares_panel(dates, secs, raw["income"], raw["splits"]).to_numpy("float32")
    out["me"] = me[secs.index].to_numpy()
    out["univ"] = univ[secs.index].to_numpy()
    is_nyse = (secs["primary_exchange"] == config.NYSE_EXCHANGE)
    out["is_nyse"] = is_nyse.to_numpy()
    # NYSE breakpoints use all NYSE stocks of the investable universe
    w_all = winsorized_weights(me, univ, (secs_all["primary_exchange"] == config.NYSE_EXCHANGE))
    out["wgt"] = w_all[secs.index].to_numpy()
    det = dl.load_ticker_details([t[0] for t in secs["tickers"]], [str(d)[:10] for d in secs["last_seen"]])
    out["ind"] = sic_to_ff49(det["sic_code"], raw["ff49"]).to_numpy()
    ea = earnings_dates(raw["income"], raw["filings"])
    out["ea"] = ea_window_mask(dates, secs, ea)
    log.info("base panel: %d days %s..%s, %d securities, %d month ends; SIC coverage %.1f%%",
             len(dates), dates[0].date(), dates[-1].date(), len(secs), len(mends),
             100 * np.isfinite(out["ind"]).mean())
    return out


def _monthly_beta(exret_m: pd.DataFrame, mkt_m: pd.Series, months: int = config.BETA_MONTHS,
                  min_months: int = config.BETA_MIN_MONTHS) -> pd.DataFrame:
    """Rolling market beta from monthly excess returns (Fama-French 2020 control)."""
    m = pd.DataFrame(np.where(exret_m.notna(), mkt_m.to_numpy()[:, None], np.nan),
                     index=exret_m.index, columns=exret_m.columns)
    roll = dict(window=months, min_periods=min_months)
    cov = (exret_m * m).rolling(**roll).mean() - exret_m.rolling(**roll).mean() * m.rolling(**roll).mean()
    var = (m * m).rolling(**roll).mean() - m.rolling(**roll).mean() ** 2
    return cov / var


def build_characteristics(base: dict, raw: dict) -> dict:
    """Standardized characteristics with inclusion flags, FF controls and SUE (Sections 3 and 5.2)."""
    dates, mends, secs = base["dates"], base["mends"], pd.Index(base["secs"])
    df = lambda a, idx=dates: pd.DataFrame(a, index=idx, columns=secs)
    me_m = df(base["me"], mends)
    mkt = raw["ff_d"]["MKT_RF"].reindex(dates)
    pc = price_characteristics(df(base["ret"]), df(base["exret"]), mkt, df(base["close_u"]),
                               df(base["volume"]), df(base["shares"]), me_m)
    q = add_ttm_and_growth(merge_statements(raw["income"], raw["balance"], raw["cashflow"]))
    q["sue"] = sue(q)
    fields = [c for c in q.columns if c not in ("cik", "period_end") and pd.api.types.is_numeric_dtype(q[c])]
    f = to_monthly(q, fields, mends, pd.Series(base["cik"], index=secs))
    ac = accounting_characteristics(f, me_m)
    raw_chars = {**pc, **ac}
    names = sorted(raw_chars)
    univ = base["univ"]
    M, N, K = len(mends), len(secs), len(names)
    chars = np.full((M, N, K), np.nan, dtype=np.float32)
    coverage = pd.DataFrame(0.0, index=mends, columns=names)
    std = {}
    for k, name in enumerate(names):
        a = raw_chars[name].reindex(index=mends, columns=secs).to_numpy("float64")
        s = np.full((M, N), np.nan)
        for i in range(M):
            u = univ[i]
            if u.sum() == 0:
                continue
            coverage.iat[i, k] = np.isfinite(a[i, u]).mean()
            s[i] = rank_standardize(a[i], u)
        std[name] = s
        chars[:, :, k] = s
    incl = select_characteristics(coverage, std, mends, univ)
    del std
    # Fama-French (2020) controls, unstandardized (Section 5.2)
    rf_m = (1 + raw["ff_d"]["RF"].reindex(dates)).groupby(dates.to_period("M")).prod() - 1
    ret_m = compound_monthly(df(base["ret"]))
    exret_m = ret_m.sub(rf_m.to_numpy(), axis=0)
    mkt_m = (1 + raw["ff_d"]["MKT_RF"] + raw["ff_d"]["RF"]).reindex(dates).groupby(dates.to_period("M")).prod() - 1 \
        - rf_m
    ctrl = {
        "beta": _monthly_beta(exret_m, pd.Series(mkt_m.to_numpy(), index=mends)),
        "log_me": pc["market_equity"], "log_bm": np.log(ac["be_me"].where(ac["be_me"] > 0)),
        "op": ac["ope_be"], "inv": ac["at_gr1"], "mom": pc["ret_12_1"],
    }
    out = {"chars": chars, "incl": incl.to_numpy(), "char_names": np.array(names),
           "coverage": coverage.to_numpy(), "sue": f["sue"].reindex(index=mends, columns=secs).to_numpy(),
           "misp": misp_score(raw_chars, univ, mends, secs)}
    for k, v in ctrl.items():
        out["ctrl_" + k] = v.reindex(index=mends, columns=secs).to_numpy("float64")
    log.info("characteristics: %d built, included per month %d-%d", K, incl.sum(axis=1).min(), incl.sum(axis=1).max())
    return out


def prev_month_index(dates: pd.DatetimeIndex, mends: pd.DatetimeIndex) -> np.ndarray:
    """For each day, the index of the previous month end in ``mends`` (-1 if none)."""
    return np.searchsorted(mends.values, dates.values, side="left") - 1


def daily_weights(panel: dict) -> np.ndarray:
    """(T, N) WLS weights of day t: beginning-of-month winsorized ME of investable stocks (Section 3)."""
    pm = prev_month_index(panel["dates"], panel["mends"])
    wgt = np.where(panel["univ"], panel["wgt"], np.nan).astype("float32")
    w_day = np.where(pm[:, None] >= 0, wgt[np.maximum(pm, 0)], np.nan)
    w_day[panel["dates"] < pd.Timestamp(config.DECOMP_START)] = np.nan
    return w_day


def build_decomposition(base: dict, ch: dict, ff_d: pd.DataFrame) -> dict:
    """Daily decompositions: characteristics + industries (baseline) and characteristics only (Section 3)."""
    dates, mends = base["dates"], base["mends"]
    pm = prev_month_index(dates, mends)
    days = range(int(np.searchsorted(dates.values, np.datetime64(config.DECOMP_START))), len(dates))
    wgt = np.where(base["univ"], base["wgt"], np.nan)
    ind_m = np.broadcast_to(base["ind"], wgt.shape)
    out = {}
    t0 = time.time()
    for key, ind in (("ci", ind_m), ("c", None)):
        res = daily_decomposition(base["exret"], pm, wgt, ch["chars"], ch["incl"], ind, days)
        out["eps_" + key] = res.pop("eps")
        res.pop("fitted")
        for s, v in res.items():
            out[f"fit_{key}_{s}"] = v
        log.info("decomposition %s done in %.0fs", key, time.time() - t0)
    out["eps_t"], out["r_t"] = zero_ea_residuals(base["exret"], out["eps_ci"], base["ea"])
    # stock-by-stock monthly FF3 time-series benchmark (Section 3.1, Figure 1)
    w_day = np.where(pm[:, None] >= 0, wgt[np.maximum(pm, 0)], np.nan)
    w_day[: days.start] = np.nan
    ff = ff_d[["MKT_RF", "SMB", "HML"]].reindex(dates).to_numpy()
    res = ff3_timeseries_fit(base["exret"], ff, month_id(dates), w_day)
    for s, v in res.items():
        out[f"fit_ff3_{s}"] = v
    log.info("FF3 benchmark done in %.0fs", time.time() - t0)
    return out


def load_panel(rebuild: bool = False) -> dict:
    """The shared panel (see module docstring), built once and cached as .npy files outside the repository."""
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    done = PANEL_DIR / "DONE"
    if done.exists() and not rebuild:
        out = {}
        for p in PANEL_DIR.glob("*.npy"):
            out[p.stem] = np.load(p, allow_pickle=False, mmap_mode="r")  # memory-mapped, read-only
        out["dates"], out["mends"] = pd.DatetimeIndex(out["dates"]), pd.DatetimeIndex(out["mends"])
        out["ff_d"] = pd.read_parquet(PANEL_DIR / "ff_d.parquet")
        out["ff_m"] = pd.read_parquet(PANEL_DIR / "ff_m.parquet")
        return out
    raw = load_raw()
    base = build_base(raw)
    ch = build_characteristics(base, raw)
    dec = build_decomposition(base, ch, raw["ff_d"])
    out = {**base, **ch, **dec, "ff_d": raw["ff_d"], "ff_m": raw["ff_m"]}
    for k, v in out.items():
        if isinstance(v, pd.DataFrame):
            v.to_parquet(PANEL_DIR / f"{k}.parquet")
        elif isinstance(v, pd.DatetimeIndex):
            np.save(PANEL_DIR / f"{k}.npy", v.values)
        else:
            np.save(PANEL_DIR / f"{k}.npy", np.asarray(v))
    done.write_text("ok")
    return load_panel()
