"""Shared helpers of the experiment scripts: factor alignment, portfolio summaries, FM controls, output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.stats import nw_mean, ts_alpha

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FF6 = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "UMD"]
CONTROLS = ["beta", "log_me", "log_bm", "op", "inv", "mom"]


def factors_next(ff_m: pd.DataFrame, mends: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly factor returns of the holding month m+1 for each formation month-end row m."""
    f = ff_m.copy()
    f.index = f.index.to_period("M")
    nxt = (mends.to_period("M") + 1)
    return f.reindex(nxt).set_axis(mends)


def formation_rows(panel: dict, start: str = config.DECOMP_START, end: str | None = None) -> np.ndarray:
    """Formation-month rows m whose month lies inside the decomposition sample and has a month m+1."""
    mends = panel["mends"]
    ok = (mends >= pd.Timestamp(start)) & (np.arange(len(mends)) < len(mends) - 1)
    if end is not None:
        ok &= mends <= pd.Timestamp(end)
    return np.flatnonzero(ok)


def long_short_summary(ret: pd.Series, ff_next: pd.DataFrame, models: dict[str, list[str]]) -> dict:
    """Mean monthly return, NW t-stat, annualized volatility and Sharpe ratio, and factor alphas."""
    r = ret.dropna()
    mean, t = nw_mean(r.to_numpy())
    out = {"mean": mean, "t": t, "vol_ann": r.std(ddof=1) * np.sqrt(12),
           "sharpe_ann": mean / r.std(ddof=1) * np.sqrt(12), "n_months": len(r)}
    for name, cols in models.items():
        a = ts_alpha(r, ff_next.loc[r.index, cols])
        out[f"alpha_{name}"], out[f"t_alpha_{name}"] = a["alpha"], a["t_alpha"]
    return out


def fm_controls(panel: dict) -> dict[str, np.ndarray]:
    """The six Fama-French (2020) controls (M, N): beta, log ME, log B/M, OP, INV, MOM (Section 5.2)."""
    return {c: panel["ctrl_" + c] for c in CONTROLS}


def save_csv(df: pd.DataFrame, name: str) -> Path:
    """Write an aggregate table to results/<name>.csv."""
    RESULTS_DIR.mkdir(exist_ok=True)
    p = RESULTS_DIR / f"{name}.csv"
    df.to_csv(p, float_format="%.6g")
    return p


def save_json(obj: dict, name: str) -> Path:
    """Write aggregate metrics to results/<name>.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    p = RESULTS_DIR / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x)))
    return p
