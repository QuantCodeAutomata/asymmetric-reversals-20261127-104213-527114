"""Quarterly accounting data aligned to month ends (inputs of the firm characteristics).

Statements are keyed by CIK; a quarter becomes usable ``config.FUND_LAG_MONTHS``
after its fiscal period end (the paper uses the JKP characteristics, which
assume accounting data are public with a lag; Section 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

FLOW_FIELDS = ("revenue", "cost_of_revenue", "gross_profit", "selling_general_administrative",
               "research_development", "operating_income", "interest_expense", "income_before_income_taxes",
               "income_taxes", "consolidated_net_income_loss", "net_income_loss_attributable_common_shareholders",
               "ebitda", "net_cash_from_operating_activities", "purchase_of_property_plant_and_equipment",
               "dividends", "depreciation_depletion_and_amortization")


def merge_statements(income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, period_end) with income, balance-sheet and cash-flow fields."""
    keys = ["cik", "period_end"]
    out = None
    for df in (income, balance, cashflow):
        d = df.drop(columns=[c for c in ("tickers", "filing_date", "fiscal_quarter", "fiscal_year", "timeframe")
                             if c in df.columns])
        d = d.assign(period_end=pd.to_datetime(d["period_end"]))
        d = d.groupby(keys, as_index=False).last()
        out = d if out is None else out.merge(d, on=keys, how="outer", suffixes=("", "_dup"))
    out = out[[c for c in out.columns if not c.endswith("_dup")]]
    return out.sort_values(keys, ignore_index=True)


def add_ttm_and_growth(q: pd.DataFrame) -> pd.DataFrame:
    """Trailing-four-quarter sums of flow items and year-over-year lags (per CIK).

    A TTM value requires four consecutive quarters (period ends within ~13 months).
    Adds ``<f>_ttm``, ``<f>_ttm_l4`` (TTM four quarters earlier) and ``<b>_l4`` for
    balance items, plus ``eps_q`` (quarterly earnings per basic share).
    """
    q = q.sort_values(["cik", "period_end"]).copy()
    g = q.groupby("cik", sort=False)
    span = (q["period_end"] - g["period_end"].shift(3)).dt.days
    complete = span.between(250, 400)
    for f in FLOW_FIELDS:
        if f not in q.columns:
            continue
        ttm = g[f].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        q[f + "_ttm"] = ttm.where(complete)
    g = q.groupby("cik", sort=False)
    gap4 = (q["period_end"] - g["period_end"].shift(4)).dt.days.between(330, 400)
    for c in [c for c in q.columns if c.endswith("_ttm")] + ["total_assets", "inventories", "total_equity",
                                                              "basic_shares_outstanding", "receivables",
                                                              "total_liabilities"]:
        if c in q.columns:
            q[c + "_l4"] = g[c].shift(4).where(gap4)
    ni = q["net_income_loss_attributable_common_shareholders"].fillna(q["consolidated_net_income_loss"])
    q["eps_q"] = ni / q["basic_shares_outstanding"]
    return q


def sue(q: pd.DataFrame) -> pd.Series:
    """Standardized unexpected earnings (Table 1): four-quarter change in quarterly
    EPS divided by the standard deviation of that change over the previous eight
    quarters (at least six required; minimum not specified in the paper)."""
    g = q.groupby("cik", sort=False)["eps_q"]
    chg = q["eps_q"] - g.shift(4)
    sd = chg.groupby(q["cik"], sort=False).transform(lambda s: s.shift(1).rolling(8, min_periods=6).std())
    out = chg / sd
    return out.where(np.isfinite(out))


def to_monthly(q: pd.DataFrame, fields: list[str], months: pd.DatetimeIndex, cik_of: pd.Series,
               lag_months: int = config.FUND_LAG_MONTHS, stale_months: int = 12) -> dict[str, pd.DataFrame]:
    """Month-end panels (months x securities) of quarterly ``fields``.

    At month end m a security uses the latest quarter of its CIK with
    period_end + ``lag_months`` <= m and period_end older than ``stale_months``
    + ``lag_months`` months is discarded.

    Args:
        q: quarterly table with ``cik``, ``period_end`` and ``fields``.
        months: month-end dates.
        cik_of: Series security -> CIK.
    """
    q = q.dropna(subset=["cik"]).copy()
    q["avail"] = (q["period_end"] + pd.offsets.MonthEnd(lag_months)).values.astype("datetime64[ns]")
    q = q.sort_values("avail")
    cal = pd.DataFrame({"month": months}).sort_values("month")
    out = {}
    sec_by_cik = pd.DataFrame({"security": cik_of.index, "cik": cik_of.values}).dropna()
    sec_by_cik = sec_by_cik[sec_by_cik["cik"] != ""]
    m = pd.merge_asof(cal.assign(key=1).merge(sec_by_cik.assign(key=1), on="key").sort_values("month"),
                      q[["cik", "avail", "period_end", *fields]], left_on="month", right_on="avail",
                      by="cik", direction="backward")
    fresh = (m["month"] - m["period_end"]).dt.days <= (stale_months + lag_months) * 31
    for f in fields:
        out[f] = m.loc[fresh].pivot_table(index="month", columns="security", values=f, aggfunc="last") \
            .reindex(index=months, columns=cik_of.index).astype("float64")
    return out
