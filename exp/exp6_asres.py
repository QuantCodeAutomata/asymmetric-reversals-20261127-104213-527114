"""Out-of-sample asymmetric residual return factor ASRES (Section 7, Tables 7-9).

Daily spline Fama-MacBeth regressions of 21-day returns on lags 2..252 of
eps~+, eps~- and r~ (Equation 25); expanding-window average coefficients
(Equation 26); ASRES forecast (Equations 27-28); quintile long-short factors
for STREV, three RES versions and ASRES (Table 7); ASRES loadings (Table 8);
size groups and decimalization subsamples (Table 9).

Run: python -m exp.exp6_asres
"""

from __future__ import annotations

import argparse
import logging
import resource
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.asres import asres_signal, expanding_lambda, first_day_next_month, forecast_coefficients, size_groups
from src.pipeline import daily_weights, load_panel
from src.reporting import FF6, RESULTS_DIR, factors_next, formation_rows, long_short_summary, save_csv, save_json
from src.signals import monthly_inputs, monthly_sum
from src.stats import sort_portfolios, ts_alpha

log = logging.getLogger(__name__)

ALPHA_MODELS = {"CAPM": ["MKT_RF"], "FF3": ["MKT_RF", "SMB", "HML"], "FF6": FF6, "FF6+REV": FF6 + ["REV"]}


def long_short(signal: np.ndarray, valid: np.ndarray, is_nyse: np.ndarray, w: np.ndarray, ret_next: np.ndarray,
               rows: np.ndarray, idx: pd.DatetimeIndex, long_high: bool) -> tuple[pd.Series, pd.DataFrame]:
    """Quintile long-short factor (High-Low if ``long_high`` else Low-High) and quintile returns."""
    res = sort_portfolios(signal, valid, is_nyse, w, ret_next)
    q = pd.DataFrame(res["ret"][rows], index=idx, columns=[f"Q{i + 1}" for i in range(config.N_QUANTILES)])
    ls = q.iloc[:, -1] - q.iloc[:, 0]
    return (ls if long_high else -ls), q


def main() -> None:
    """Estimate the ASRES forecasts and evaluate the STREV, RES and ASRES factors."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    t0 = time.time()
    panel = load_panel()
    dates, mends = panel["dates"], panel["mends"]
    mi = monthly_inputs(panel)
    w_day = daily_weights(panel)

    # Equation 25: daily spline regressions with a full 252-day lag history
    start = int(np.searchsorted(dates.values, np.datetime64(config.DECOMP_START)))
    reg_days = np.arange(start + config.N_LAGS, len(dates) - config.FORECAST_HORIZON + 1)
    log.info("Equation 25 regressions: %d days %s..%s", len(reg_days), dates[reg_days[0]].date(),
             dates[reg_days[-1]].date())
    lam = forecast_coefficients(panel["exret"], w_day, panel["eps_t"], panel["r_t"], reg_days)
    log.info("spline forecast regressions done %.0fs", time.time() - t0)
    lam_bar = expanding_lambda(lam, reg_days, dates, mends)
    t_m = first_day_next_month(dates, mends)
    sig = asres_signal(lam_bar, panel["eps_t"], t_m)

    # evaluation starts after a two-year burn-in from the first decomposition month (Section 7)
    eval_start = (pd.Timestamp(config.DECOMP_START) + pd.DateOffset(years=config.ASRES_BURN_IN_YEARS)
                  - pd.DateOffset(months=1))
    rows = formation_rows(panel, start=str(eval_start.date()))
    rows = rows[np.isfinite(sig["ASRES"][rows]).any(axis=1)]
    idx = mends[rows]
    ff_next = factors_next(panel["ff_m"], mends)
    log.info("evaluation formation months %s..%s (%d)", idx[0].date(), idx[-1].date(), len(rows))

    blocks = mi["blocks"]
    signals = {"STREV": (mi["ret_form"], False),
               "RES Char": (monthly_sum(panel["eps_c"], blocks), False),
               "RES Char+Ind": (monthly_sum(panel["eps_ci"], blocks), False),
               "RES Char+Ind+ZeroEAD": (mi["RES"], False),
               "ASRES": (sig["ASRES"], True)}
    valid, w, ret_next, is_nyse = mi["valid"], mi["w"], mi["exret_next"], panel["is_nyse"]
    table7, factors = {}, {}
    for name, (s, long_high) in signals.items():
        ls, _ = long_short(s, valid & np.isfinite(sig["ASRES"]), is_nyse, w, ret_next, rows, idx, long_high)
        factors[name] = ls
        table7[name] = long_short_summary(ls, ff_next, ALPHA_MODELS)
    t7 = pd.DataFrame(table7).T
    save_csv(t7, "exp6_table7_factors")

    # Table 8: ASRES loadings on FF6 + REV
    a = ts_alpha(factors["ASRES"], ff_next.loc[idx, FF6 + ["REV"]])
    t8 = pd.DataFrame({"coef": {"alpha": a["alpha"], **a["betas"]}, "t": {"alpha": a["t_alpha"], **a["t_betas"]}})
    t8.loc["R2", "coef"] = a["r2"]
    save_csv(t8, "exp6_table8_loadings")

    # Table 9: NYSE size groups and decimalization subsamples
    groups = size_groups(panel["me"], valid, is_nyse)
    t9 = {}
    for g, mask in groups.items():
        ls, _ = long_short(sig["ASRES"], valid & mask & np.isfinite(sig["ASRES"]), is_nyse, w, ret_next, rows, idx,
                           True)
        t9[g] = long_short_summary(ls, ff_next, {"FF6": FF6})
    dec = pd.Timestamp(config.DECIMALIZATION_DATE)
    for lab, sel in (("Pre-decimalization", idx < dec), ("Post-decimalization", idx >= dec)):
        if sel.sum() >= 12:
            t9[lab] = long_short_summary(factors["ASRES"][sel], ff_next, {"FF6": FF6})
    t9 = pd.DataFrame(t9).T
    save_csv(t9, "exp6_table9_robustness")

    # average implied coefficients used by the final forecast (aggregate over lags, not dates)
    lags = np.arange(config.ASRES_FIRST_LAG, config.N_LAGS + 1)
    last = lam_bar[rows[-1]]
    coef = pd.DataFrame({"lag": lags, "lambda_bar_eps_pos": last[0], "lambda_bar_eps_neg": last[1]}).set_index("lag")
    save_csv(coef, "exp6_lambda_bar_final")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lags, coef["lambda_bar_eps_pos"], label="eps~+")
    ax.plot(lags, coef["lambda_bar_eps_neg"], label="eps~-")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(config.ASRES_SHORT_LAG, color="grey", ls="--", lw=0.8, label="lag 22")
    ax.set_xlabel("lag tau (trading days)")
    ax.set_ylabel("expanding-window coefficient")
    ax.set_title(f"Spline coefficients of the 21-day forecast (Equation 25), window ending {idx[-1].date()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_lambda_bar.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, ls in factors.items():
        ax.plot(idx, (1 + ls.fillna(0)).cumprod(), label=name)
    ax.set_yscale("log")
    ax.set_xlabel("month")
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title("Long-short quintile factors (Table 7)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_factors_cumulative.png", dpi=130)
    plt.close(fig)

    meta = {"n_regressions": int(np.isfinite(lam[:, 0, 0]).sum()), "first_regression": str(dates[reg_days[0]].date()),
            "eval_months": [str(idx[0].date()), str(idx[-1].date())], "n_eval_months": int(len(rows)),
            "runtime_s": time.time() - t0, "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    save_json(meta, "exp6_meta")
    log.info("Table 7\n%s", t7.to_string())
    log.info("Table 8\n%s", t8.to_string())
    log.info("Table 9\n%s", t9.to_string())
    log.info("meta %s", meta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
