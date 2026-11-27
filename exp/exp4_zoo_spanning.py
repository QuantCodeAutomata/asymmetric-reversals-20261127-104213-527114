"""Spanning the short-term anomaly zoo with AIR, concentration and systematic returns (Section 5.5, Table 6).

Panel A: WLS Fama-MacBeth fit of each of the 13 one-day-skipped characteristics
on RES+ and RES-. Panels B-E: WLS Fama-MacBeth regressions of month m+1 excess
returns on the characteristic (B), plus RES+ and RES- (C), plus CONC (D), plus
SYS (E, Lag Ret omitted), all with the six Fama-French controls.

Run: python -m exp.exp4_zoo_spanning
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.pipeline import load_panel
from src.reporting import RESULTS_DIR, fm_controls, formation_rows, save_csv, save_json
from src.signals import monthly_inputs
from src.stats import fama_macbeth
from src.zoo import ZOO_NAMES, zoo_characteristics

log = logging.getLogger(__name__)
SCALED = ("Skew", "ISkew_FF3", "Coskew")  # divided by 100 in the regressions (Table 6 notes)
PANELS = {"B": [], "C": ["RES_pos", "RES_neg"], "D": ["RES_pos", "RES_neg", "CONC"],
          "E": ["RES_pos", "RES_neg", "CONC", "SYS"]}


def main() -> None:
    """Compute the 13 characteristics, run Table 6 Panels A-E and save the results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    panel = load_panel()
    mi = monthly_inputs(panel)
    rows = formation_rows(panel)
    valid = mi["valid"]
    mask = lambda x: np.where(valid, x, np.nan)
    ff3 = panel["ff_d"][["MKT_RF", "SMB", "HML"]].reindex(panel["dates"]).to_numpy()
    zoo = zoo_characteristics(panel["ret"], panel["exret"], panel["high"], panel["low"], panel["close_u"],
                              panel["volume"], panel["shares"], ff3, mi["blocks"])
    for k in SCALED:
        zoo[k] = zoo[k] / config.SKEW_SCALE
    zoo = {k: mask(v) for k, v in zoo.items()}
    sig = {k: mask(mi[k]) for k in ("RES_pos", "RES_neg", "CONC", "SYS")}
    ctrl = {k: mask(v) for k, v in fm_controls(panel).items()}
    y = mask(mi["exret_next"])

    recs = []
    for name in ZOO_NAMES:
        cov = np.isfinite(zoo[name][rows]).sum(axis=1).mean()
        a = fama_macbeth(zoo[name], {"RES_pos": sig["RES_pos"], "RES_neg": sig["RES_neg"]}, mi["w"], periods=rows)
        for reg in ("RES_pos", "RES_neg"):
            recs.append({"panel": "A", "characteristic": name, "regressor": reg,
                         "coef": a.loc[reg, "coef"], "t": a.loc[reg, "t"]})
        recs.append({"panel": "A", "characteristic": name, "regressor": "adj_r2", "coef": a.loc["adj_r2", "coef"],
                     "t": np.nan})
        recs.append({"panel": "A", "characteristic": name, "regressor": "avg_n_stocks", "coef": cov, "t": np.nan})
        for p, regs in PANELS.items():
            if p == "E" and name == "LagRet":
                continue  # Lag Ret = RES+ + RES- + SYS
            X = {"char": zoo[name], **{k: sig[k] for k in regs}, **ctrl}
            res = fama_macbeth(y, X, mi["w"], periods=rows)
            for reg in ["char"] + regs:
                recs.append({"panel": p, "characteristic": name, "regressor": reg,
                             "coef": res.loc[reg, "coef"], "t": res.loc[reg, "t"]})
            recs.append({"panel": p, "characteristic": name, "regressor": "adj_r2",
                         "coef": res.loc["adj_r2", "coef"], "t": np.nan})
        log.info("done %s", name)
    tab = pd.DataFrame(recs)
    save_csv(tab.set_index(["panel", "characteristic", "regressor"]), "exp4_table6_spanning")

    ch = tab[tab.regressor.isin(["char", "adj_r2"])]
    summary = {}
    for p in ("A", "B", "C", "D", "E"):
        sub = tab[(tab.panel == p)]
        if p == "A":
            summary["A_adj_r2"] = sub[sub.regressor == "adj_r2"].set_index("characteristic")["coef"].to_dict()
        else:
            c = sub[sub.regressor == "char"].set_index("characteristic")
            summary[f"{p}_char"] = {k: {"coef": c.loc[k, "coef"], "t": c.loc[k, "t"]} for k in c.index}
            summary[f"{p}_n_significant"] = int((c["t"].abs() > 1.96).sum())
    summary["months"] = [str(panel["mends"][rows[0]].date()), str(panel["mends"][rows[-1]].date())]
    save_json(summary, "exp4_summary")
    del ch

    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(ZOO_NAMES))
    for k, p in enumerate(("B", "C", "D", "E")):
        c = tab[(tab.panel == p) & (tab.regressor == "char")].set_index("characteristic")["t"].reindex(ZOO_NAMES)
        ax.bar(xs + 0.2 * k - 0.3, c.to_numpy(), width=0.2, label=f"Panel {p}")
    ax.axhline(1.96, color="grey", ls="--", lw=0.8)
    ax.axhline(-1.96, color="grey", ls="--", lw=0.8)
    ax.set_xticks(xs, ZOO_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Newey-West t-statistic of the characteristic")
    ax.set_title("Spanning the short-term zoo (Table 6, Panels B-E)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_table6_tstats.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
