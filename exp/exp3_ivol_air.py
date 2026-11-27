"""Spanning IVOL and mispricing with asymmetric idiosyncratic reversal (Sections 5.1-5.4).

Equation (15) fit of IVOL on RES+ and RES-, Table 2 spanning regressions
(WLS and OLS, with and without the six Fama-French controls), Figure 2 horizon
dependence of IVOL, Table 3 regressions within MISP quintiles and Table 4
concavity regressions.

Run: python -m exp.exp3_ivol_air
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.asres import first_day_next_month
from src.irf import forward_sum
from src.pipeline import load_panel
from src.reporting import RESULTS_DIR, fm_controls, formation_rows, save_csv, save_json
from src.signals import ivol_horizon, month_blocks, monthly_inputs
from src.stats import assign_quantiles, fama_macbeth, fm_coef_series, nw_mean

log = logging.getLogger(__name__)

TABLE2_SPECS = {"(1) IVOL": ["IVOL"], "(2) RES+": ["RES_pos"], "(3) RES-": ["RES_neg"],
                "(4) RES+ RES-": ["RES_pos", "RES_neg"], "(5) IVOL RES+ RES-": ["IVOL", "RES_pos", "RES_neg"]}
TABLE4_SPECS = {"(1)": ["RES_pos", "RES_neg"], "(2)": ["RES_pos", "RES_neg", "CONC_pos", "CONC_neg"],
                "(3)": ["RES_pos", "RES_neg", "CONC_pos", "CONC_neg", "SYS"],
                "(4)": ["RES_pos", "RES_neg", "SYS", "CONC"]}
TABLE3_SPECS = {"IVOL": ["IVOL"], "RES+ RES-": ["RES_pos", "RES_neg"], "IVOL RES+ RES-": ["IVOL", "RES_pos", "RES_neg"]}


def masked(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Set observations outside the investable universe to NaN."""
    return np.where(valid, x, np.nan)


def fm_table(y: np.ndarray, mi: dict, specs: dict[str, list[str]], ctrl: dict[str, np.ndarray] | None,
             w: np.ndarray | None, rows: np.ndarray) -> pd.DataFrame:
    """Fama-MacBeth slopes and NW t-stats for each specification (columns), long format rows."""
    out = []
    for name, regs in specs.items():
        X = {k: mi[k] for k in regs}
        if ctrl is not None:
            X.update(ctrl)
        res = fama_macbeth(y, X, w, periods=rows)
        df = res.reset_index().rename(columns={"index": "regressor"})
        df.insert(0, "spec", name)
        out.append(df)
        if "RES_pos" in regs and "RES_neg" in regs:
            Xn = list(X)
            C = fm_coef_series(y, X, w, rows)
            d = C[:, 1 + Xn.index("RES_pos")] - C[:, 1 + Xn.index("RES_neg")]
            m, t = nw_mean(d)
            out.append(pd.DataFrame({"spec": [name], "regressor": ["RES+ minus RES-"], "coef": [m], "t": [t]}))
    return pd.concat(out, ignore_index=True)


def ivol_horizon_figure(panel: dict, mi: dict, ctrl: dict, rows: np.ndarray) -> pd.DataFrame:
    """Figure 2: WLS FM t-statistics of IVOL(H) (with controls) for 1-, 5- and 21-day holding returns."""
    t0 = first_day_next_month(panel["dates"], panel["mends"])
    valid = mi["valid"]
    ys = {}
    for h in config.HOLD_HORIZONS:
        fs = forward_sum(panel["exret"], h)
        y = np.full(mi["w"].shape, np.nan)
        ok = t0 >= 0
        y[ok] = fs[t0[ok]]
        ys[h] = masked(y, valid)
    # Figure 2 caption: the lookback covers the 1..21 days immediately before the
    # return measurement window, so the last trading day of the month is not skipped.
    blocks_full = month_blocks(panel["dates"], skip_last=False)
    assert len(blocks_full) == len(mi["blocks"])
    recs = []
    for H in config.IVOL_H_GRID:
        iv = ivol_horizon(panel["eps_ci"], blocks_full, H)
        for h, y in ys.items():
            res = fama_macbeth(y, {"IVOL_H": iv, **ctrl}, mi["w"], periods=rows)
            recs.append({"H": H, "hold_days": h, "coef": res.loc["IVOL_H", "coef"], "t": res.loc["IVOL_H", "t"]})
    return pd.DataFrame(recs)


def misp_table(panel: dict, mi: dict, y: np.ndarray, ctrl: dict) -> pd.DataFrame:
    """Table 3: WLS FM regressions within MISP quintiles (all-stock breakpoints of the screened sample)."""
    rows = formation_rows(panel, end=config.MISP_END)
    misp = panel["misp"]
    M, N = misp.shape
    lab = np.full((M, N), -1)
    allstocks = np.ones(N, dtype=bool)
    for m in rows:
        lab[m] = assign_quantiles(misp[m], mi["valid"][m], allstocks)
    out = []
    for q, qname in enumerate(["Underpriced", "2", "3", "4", "Overpriced"]):
        yq = np.where(lab == q, y, np.nan)
        t = fm_table(yq, mi, TABLE3_SPECS, ctrl, mi["w"], rows)
        out.append(t.assign(misp_quintile=qname))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    """Run all Section 5.1-5.4 regressions and save the tables, summary and Figure 2."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    panel = load_panel()
    mi = monthly_inputs(panel)
    rows = formation_rows(panel)
    valid = mi["valid"]
    for k in ("IVOL", "IVOL_noEA", "RES_pos", "RES_neg", "CONC_pos", "CONC_neg", "CONC", "SYS"):
        mi[k] = masked(mi[k], valid)
    ctrl = {k: masked(v, valid) for k, v in fm_controls(panel).items()}
    y = masked(mi["exret_next"], valid)
    summary = {}

    eq15 = fama_macbeth(mi["IVOL"], {"RES_pos": mi["RES_pos"], "RES_neg": mi["RES_neg"]}, mi["w"], periods=rows)
    save_csv(eq15, "exp3_eq15_ivol_on_res")
    summary["eq15"] = {"RES_pos": eq15.loc["RES_pos", "coef"], "RES_neg": eq15.loc["RES_neg", "coef"],
                       "adj_r2": eq15.loc["adj_r2", "coef"]}
    log.info("Eq 15:\n%s", eq15)

    t2 = []
    for est, w in (("WLS", mi["w"]), ("OLS", None)):
        for ctl_name, c in (("controls", ctrl), ("no_controls", None)):
            t2.append(fm_table(y, mi, TABLE2_SPECS, c, w, rows).assign(estimator=est, controls=ctl_name))
    t2 = pd.concat(t2, ignore_index=True)
    save_csv(t2.set_index(["estimator", "controls", "spec", "regressor"]), "exp3_table2_spanning_ivol")
    sel = t2[(t2.estimator == "WLS") & (t2.controls == "controls")].set_index(["spec", "regressor"])
    summary["table2_wls_controls"] = {f"{s}|{r}": {"coef": sel.loc[(s, r), "coef"], "t": sel.loc[(s, r), "t"]}
                                      for s, r in sel.index if r in ("IVOL", "RES_pos", "RES_neg", "RES+ minus RES-")}
    log.info("Table 2 (WLS, controls):\n%s", sel)

    t4 = fm_table(y, mi, TABLE4_SPECS, ctrl, mi["w"], rows)
    save_csv(t4.set_index(["spec", "regressor"]), "exp3_table4_concavity")
    s4 = t4.set_index(["spec", "regressor"])
    summary["table4"] = {f"{s}|{r}": {"coef": s4.loc[(s, r), "coef"], "t": s4.loc[(s, r), "t"]}
                         for s, r in s4.index if r.startswith("CONC") or r in ("RES_pos", "RES_neg", "SYS")}

    t3 = misp_table(panel, mi, y, ctrl)
    save_csv(t3.set_index(["misp_quintile", "spec", "regressor"]), "exp3_table3_misp")
    s3 = t3[t3.regressor == "IVOL"].set_index(["misp_quintile", "spec"])
    summary["table3_ivol"] = {f"{q}|{s}": {"coef": s3.loc[(q, s), "coef"], "t": s3.loc[(q, s), "t"]}
                              for q, s in s3.index}

    fig2 = ivol_horizon_figure(panel, mi, ctrl, rows)
    save_csv(fig2.set_index(["hold_days", "H"]), "exp3_figure2_ivol_horizon")
    summary["figure2_t_H1"] = {int(h): float(fig2[(fig2.H == 1) & (fig2.hold_days == h)]["t"].iloc[0])
                               for h in config.HOLD_HORIZONS}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for h, g in fig2.groupby("hold_days"):
        ax.plot(g["H"], g["t"], marker="o", label=f"{h}-day holding return")
    ax.axhline(0, color="k", lw=0.8)
    for lvl in (-1.96, 1.96):
        ax.axhline(lvl, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("IVOL lookback H (trading days)")
    ax.set_ylabel("Fama-MacBeth t-statistic of IVOL")
    ax.set_title("Horizon dependence of the IVOL slope (Figure 2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_figure2_ivol_horizon.png", dpi=130)
    plt.close(fig)

    summary["months"] = [str(panel["mends"][rows[0]].date()), str(panel["mends"][rows[-1]].date())]
    save_json(summary, "exp3_summary")
    log.info("summary: %s", summary)


if __name__ == "__main__":
    main()
