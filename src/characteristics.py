"""Firm characteristics, their rank standardization and inclusion filters (Section 3).

The paper uses the 151 JKP characteristics (Jensen, Kelly and Pedersen 2023).
Here a set of JKP-style characteristics is built from daily bars and quarterly
statements with the JKP names and definitions where the inputs allow it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config
from src.panel import month_ends


def _sample_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of a daily panel on the last trading day of each month."""
    return df.loc[month_ends(df.index)]


def _rolling_beta_ivol(exret: pd.DataFrame, mkt: pd.Series, window: int, min_periods: int
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling CAPM beta and residual volatility over ``window`` trading days."""
    valid = exret.notna()
    m = pd.DataFrame(np.where(valid, mkt.to_numpy()[:, None], np.nan), index=exret.index, columns=exret.columns)
    r = exret.astype("float64")
    roll = dict(window=window, min_periods=min_periods)
    mr, mm = r.rolling(**roll).mean(), m.rolling(**roll).mean()
    cov = (r * m).rolling(**roll).mean() - mr * mm
    var_m = (m * m).rolling(**roll).mean() - mm * mm
    var_r = (r * r).rolling(**roll).mean() - mr * mr
    beta = cov / var_m
    ivol = np.sqrt((var_r - cov * cov / var_m).clip(lower=0))
    return beta, ivol


def price_characteristics(ret: pd.DataFrame, exret: pd.DataFrame, mkt: pd.Series, close_u: pd.DataFrame,
                          volume: pd.DataFrame, shares: pd.DataFrame, me_m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Month-end market-based characteristics (JKP names) from daily panels.

    Args:
        ret, exret: daily raw and excess returns (dates x securities).
        mkt: daily market excess return.
        close_u: daily unadjusted close; volume: daily share volume;
        shares: daily shares outstanding; me_m: month-end market equity.
    """
    out: dict[str, pd.DataFrame] = {}
    me_idx = me_m.index
    out["market_equity"] = np.log(me_m.where(me_m > 0))
    logr = np.log1p(ret.astype("float64"))
    cum = logr.fillna(0).cumsum()
    cnt = ret.notna().astype("float64").cumsum()
    cm, nm = _sample_month_end(cum), _sample_month_end(cnt)

    def mret(a: int, b: int) -> pd.DataFrame:
        r = np.expm1(cm.shift(b) - cm.shift(a))
        n = nm.shift(b) - nm.shift(a)
        return r.where(n >= 0.6 * 21 * (a - b))
    for name, (a, b) in {"ret_1_0": (1, 0), "ret_3_1": (3, 1), "ret_6_1": (6, 1), "ret_12_1": (12, 1),
                         "ret_12_7": (12, 7), "ret_36_12": (36, 12)}.items():
        out[name] = mret(a, b)
    beta, ivol252 = _rolling_beta_ivol(exret, mkt, 252, 120)
    out["beta_252d"] = _sample_month_end(beta)
    out["ivol_capm_252d"] = _sample_month_end(ivol252)
    _, ivol21 = _rolling_beta_ivol(exret, mkt, 21, 15)
    out["ivol_capm_21d"] = _sample_month_end(ivol21)
    out["rvol_21d"] = _sample_month_end(exret.rolling(21, min_periods=15).std())
    out["rvol_252d"] = _sample_month_end(exret.rolling(252, min_periods=120).std())
    out["rmax1_21d"] = _sample_month_end(ret.rolling(21, min_periods=15).max())
    out["rskew_21d"] = _sample_month_end(ret.rolling(21, min_periods=15).skew())
    out["rmax5_21d"] = monthly_block_stat(ret, top_k_mean, min_obs=15)
    turn = volume / shares
    out["turnover_126d"] = _sample_month_end(turn.rolling(126, min_periods=60).mean())
    dvol = volume * close_u
    out["dolvol_126d"] = np.log(_sample_month_end(dvol.rolling(126, min_periods=60).mean()).where(lambda x: x > 0))
    ami = (ret.abs() / dvol.where(dvol > 0))
    out["ami_126d"] = _sample_month_end(ami.rolling(126, min_periods=60).mean()) * 1e6
    out["zero_trades_126d"] = _sample_month_end((volume == 0).astype("float64").where(volume.notna())
                                                .rolling(126, min_periods=60).sum())
    out["prc"] = np.log(_sample_month_end(close_u).where(lambda x: x > 0))
    for k in out:
        out[k] = out[k].reindex(me_idx)
    return out


def top_k_mean(block: np.ndarray, k: int = 5) -> np.ndarray:
    """Column-wise mean of the ``k`` largest non-NaN values of a (days x stocks) block."""
    a = np.where(np.isnan(block), -np.inf, block)
    top = -np.sort(-a, axis=0)[:k]
    top = np.where(np.isfinite(top), top, np.nan)
    return np.nanmean(top, axis=0) if top.size else np.full(block.shape[1], np.nan)


def monthly_block_stat(daily: pd.DataFrame, fn, min_obs: int) -> pd.DataFrame:
    """Apply ``fn`` to each calendar-month block of a daily panel; NaN when fewer than ``min_obs`` valid days."""
    per = daily.index.to_period("M")
    arr = daily.to_numpy(dtype="float64")
    bounds = np.flatnonzero(np.r_[True, per[1:] != per[:-1], True])
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        blk = arr[a:b]
        with np.errstate(all="ignore"):
            v = fn(blk)
        v = np.where(np.isfinite(blk).sum(axis=0) >= min_obs, v, np.nan)
        rows.append(v)
    return pd.DataFrame(np.vstack(rows), index=month_ends(daily.index), columns=daily.columns)


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a / b.where(b.abs() > 0)).replace([np.inf, -np.inf], np.nan)


def accounting_characteristics(f: dict[str, pd.DataFrame], me_m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Month-end accounting characteristics (JKP names) from aligned statement panels ``f``."""
    ni = f["net_income_loss_attributable_common_shareholders_ttm"].fillna(f["consolidated_net_income_loss_ttm"])
    at, be = f["total_assets"], f["total_equity_attributable_to_parent"].fillna(f["total_equity"])
    sale, me = f["revenue_ttm"], me_m * 1.0
    ocf = f["net_cash_from_operating_activities_ttm"]
    debt = f["long_term_debt_and_capital_lease_obligations"].fillna(0) + f["debt_current"].fillna(0)
    out = {
        "be_me": _safe_div(be.where(be > 0), me), "sale_me": _safe_div(sale, me), "ni_me": _safe_div(ni, me),
        "ocf_me": _safe_div(ocf, me), "div12m_me": _safe_div(-f["dividends_ttm"], me), "at_me": _safe_div(at, me),
        "debt_me": _safe_div(debt, me),
        "gp_at": _safe_div(f["gross_profit_ttm"], at), "ope_be": _safe_div(f["operating_income_ttm"], be.where(be > 0)),
        "ni_be": _safe_div(ni, be.where(be > 0)), "niq_at": _safe_div(ni, at), "ocf_at": _safe_div(ocf, at),
        "ebitda_at": _safe_div(f["ebitda_ttm"], at),
        "at_gr1": _safe_div(at, f["total_assets_l4"]) - 1, "sale_gr1": _safe_div(sale, f["revenue_ttm_l4"]) - 1,
        "inv_gr1": _safe_div(f["inventories"] - f["inventories_l4"], f["total_assets_l4"]),
        "be_gr1a": _safe_div(be - f["total_equity_l4"], f["total_assets_l4"]),
        "debt_at": _safe_div(debt, at), "cash_at": _safe_div(f["cash_and_equivalents"], at),
        "ppe_at": _safe_div(f["property_plant_equipment_net"], at),
        "oaccruals_at": _safe_div(ni - ocf, at), "capex_at": _safe_div(-f["purchase_of_property_plant_and_equipment_ttm"], at),
        "rd_sale": _safe_div(f["research_development_ttm"], sale.where(sale > 0)),
        "sga_sale": _safe_div(f["selling_general_administrative_ttm"], sale.where(sale > 0)),
        "gp_sale": _safe_div(f["gross_profit_ttm"], sale.where(sale > 0)),
        "ca_cl": _safe_div(f["total_current_assets"], f["total_current_liabilities"]),
        "chcsho_12m": np.log(_safe_div(f["basic_shares_outstanding"], f["basic_shares_outstanding_l4"]).where(lambda x: x > 0)),
        "niq_su": f["sue"], "lti_gr1a": _safe_div(f["total_liabilities"] - f["total_liabilities_l4"], f["total_assets_l4"]),
        "tax_pi": _safe_div(f["income_taxes_ttm"], f["income_before_income_taxes_ttm"].where(lambda x: x > 0)),
    }
    noa = (at - f["cash_and_equivalents"].fillna(0)) - (f["total_liabilities"] - debt)
    out["noa_at"] = _safe_div(noa, f["total_assets_l4"])
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def rank_standardize(c: np.ndarray, in_universe: np.ndarray) -> np.ndarray:
    """Rank standardization of one characteristic at one month end (Section 3).

    rC = rank(c)/(n+1) - 0.5 among valid universe stocks (average ranks for ties),
    missing values set to 0, then x = rC / mean(|rC|) over the universe.
    Returns an array with NaN outside the universe.
    """
    out = np.full(c.shape, np.nan)
    valid = in_universe & np.isfinite(c)
    n = valid.sum()
    rc = np.zeros(c.shape)
    if n > 0:
        rc[valid] = rankdata(c[valid], method="average") / (n + 1) - 0.5
    scale = np.abs(rc[in_universe]).mean()
    out[in_universe] = rc[in_universe] / scale if scale > 0 else 0.0
    return out


def select_characteristics(coverage: pd.DataFrame, std_chars: dict[str, np.ndarray], months: pd.DatetimeIndex,
                           in_universe: np.ndarray) -> pd.DataFrame:
    """Monthly inclusion flags (months x characteristics) by the Section 3 filters.

    Coverage hysteresis: include when coverage >= 40%, remove when < 20%.
    Collinearity: if |corr| >= 99% between two included characteristics drop the
    one with the shorter tenor (months included so far); reintroduce it when
    |corr| <= 97.5%.

    Args:
        coverage: months x characteristics share of universe stocks with valid values.
        std_chars: name -> (months x securities) standardized arrays.
        in_universe: months x securities boolean.
    """
    names = list(coverage.columns)
    incl = pd.DataFrame(False, index=months, columns=names)
    prev = {k: False for k in names}
    tenor = {k: 0 for k in names}
    dropped_for: dict[str, str] = {}
    for i, m in enumerate(months):
        cov_ok = {}
        for k in names:
            cv = coverage.at[m, k]
            cov_ok[k] = cv >= config.CHAR_MIN_COVERAGE_IN or ((prev[k] or k in dropped_for) and cv >= config.CHAR_MIN_COVERAGE_OUT)
        active = [k for k in names if cov_ok[k]]
        u = in_universe[i]
        if u.sum() < 3 or not active:
            for k in names:
                prev[k] = False
            continue
        mat = np.column_stack([std_chars[k][i, u] for k in active])
        corr = np.corrcoef(mat, rowvar=False) if mat.shape[1] > 1 else np.ones((1, 1))
        pos = {k: j for j, k in enumerate(active)}
        keep = set(active)
        # previously dropped characteristics stay out until |corr| <= 97.5% with their partner
        for k, partner in list(dropped_for.items()):
            if k in pos and partner in pos and partner in keep and abs(corr[pos[k], pos[partner]]) > config.CHAR_CORR_READD:
                keep.discard(k)
            else:
                dropped_for.pop(k)
        order = sorted(keep, key=lambda k: -tenor[k])
        for a_i, a in enumerate(order):
            if a not in keep:
                continue
            for b in order[a_i + 1:]:
                if b in keep and abs(corr[pos[a], pos[b]]) >= config.CHAR_CORR_DROP:
                    keep.discard(b)
                    dropped_for[b] = a
        for k in names:
            flag = k in keep
            incl.iat[i, names.index(k)] = flag
            tenor[k] = tenor[k] + 1 if flag else 0
            prev[k] = flag
    return incl
