"""Short-term reversal in residual returns vs. momentum in systematic returns (Section 4, Table 1).

Monthly RES and SYS (sums of daily earnings-adjusted residual / systematic
returns, last trading day skipped) are sorted into NYSE-breakpoint quintiles;
portfolios are weighted by ME winsorized at the 80th NYSE percentile.

Run: python -m exp.exp2_res_sys_sorts
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
from src.reporting import FF6, RESULTS_DIR, factors_next, formation_rows, save_csv, save_json
from src.signals import month_blocks, monthly_inputs, monthly_sum
from src.stats import nw_mean, sort_portfolios, ts_alpha
from src.zoo import zoo_characteristics

log = logging.getLogger(__name__)
LABELS = ["Low", "2", "3", "4", "High"]


def portfolio_medians(labels: np.ndarray, x: np.ndarray, rows: np.ndarray, n_q: int) -> np.ndarray:
    """Time-series average of the monthly cross-sectional median of ``x`` within each portfolio."""
    out = np.full((len(rows), n_q), np.nan)
    for i, m in enumerate(rows):
        for q in range(n_q):
            v = x[m, labels[m] == q]
            v = v[np.isfinite(v)]
            if v.size:
                out[i, q] = np.median(v)
    return np.nanmean(out, axis=0)


def sort_table(signal: np.ndarray, mi: dict, panel: dict, chars: dict[str, np.ndarray], rows: np.ndarray,
               ff_next: pd.DataFrame) -> pd.DataFrame:
    """Table 1 panel for one sorting signal: returns, FF6 alphas and portfolio characteristics."""
    n_q = config.N_QUANTILES
    post = sort_portfolios(signal, mi["valid"], panel["is_nyse"], mi["w"], mi["exret_next"])
    pre = sort_portfolios(signal, mi["valid"], panel["is_nyse"], mi["w"], mi["exret_m"])
    idx = panel["mends"][rows]
    ret_post = pd.DataFrame(post["ret"][rows], index=idx, columns=LABELS)
    ret_pre = pd.DataFrame(pre["ret"][rows], index=idx, columns=LABELS)
    ret_post["H-L"] = ret_post["High"] - ret_post["Low"]
    ret_pre["H-L"] = ret_pre["High"] - ret_pre["Low"]
    tab = {}
    for c in ret_post.columns:
        m_pre, t_pre = nw_mean(ret_pre[c].to_numpy())
        m_post, t_post = nw_mean(ret_post[c].to_numpy())
        a = ts_alpha(ret_post[c].dropna(), ff_next.loc[idx, FF6])
        tab[c] = {"RET_m": m_pre, "t_RET_m": t_pre, "RET_m+1": m_post, "t_RET_m+1": t_post,
                  "FF6_alpha": a["alpha"], "t_FF6_alpha": a["t_alpha"]}
    for name, x in chars.items():
        med = portfolio_medians(post["labels"], x, rows, n_q)
        for q, c in enumerate(LABELS):
            tab[c][name] = med[q]
    n = post["n"][rows].mean(axis=0)
    turn = np.nanmean(post["turnover"][rows], axis=0)
    for q, c in enumerate(LABELS):
        tab[c]["N_stocks"] = n[q]
        tab[c]["Weight_turnover"] = turn[q]
    return pd.DataFrame(tab).T


def main() -> None:
    """Build Table 1 (RES and SYS quintile sorts) and the post-formation return bar chart."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    panel = load_panel()
    mi = monthly_inputs(panel)
    rows = formation_rows(panel)
    ff_next = factors_next(panel["ff_m"], panel["mends"])
    ff3 = panel["ff_d"][["MKT_RF", "SMB", "HML"]].reindex(panel["dates"]).to_numpy()
    zoo = zoo_characteristics(panel["ret"], panel["exret"], panel["high"], panel["low"], panel["close_u"],
                              panel["volume"], panel["shares"], ff3, mi["blocks"])
    full_blocks = month_blocks(panel["dates"], skip_last=False)
    with np.errstate(all="ignore"):
        turn_m = monthly_sum(np.where(panel["shares"] > 0, panel["volume"] / panel["shares"], np.nan), full_blocks)
    chars = {"IVOL_ann": mi["IVOL"] * np.sqrt(252), "Size_bn": panel["me"] / 1e9, "SUE": panel["sue"],
             "Amihud": zoo["Amihud"], "Turnover_pct": 100 * turn_m}
    out = []
    summary = {}
    for sig in ("RES", "SYS"):
        t = sort_table(mi[sig], mi, panel, chars, rows, ff_next)
        log.info("%s sorts:\n%s", sig, t.round(4))
        out.append(t.assign(signal=sig).set_index("signal", append=True).swaplevel())
        hl = t.loc["H-L"]
        summary[sig] = {"HL_ret": hl["RET_m+1"], "t_HL_ret": hl["t_RET_m+1"], "HL_FF6_alpha": hl["FF6_alpha"],
                        "t_HL_FF6_alpha": hl["t_FF6_alpha"], "SUE_low": t.loc["Low", "SUE"],
                        "SUE_high": t.loc["High", "SUE"]}
    tab = pd.concat(out)
    tab.index.names = ["signal", "portfolio"]
    save_csv(tab, "exp2_table1_sorts")
    summary["months"] = [str(panel["mends"][rows[0]].date()), str(panel["mends"][rows[-1]].date())]
    summary["n_months"] = len(rows)
    save_json(summary, "exp2_summary")
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(LABELS))
    for k, sig in enumerate(("RES", "SYS")):
        ax.bar(xs + 0.4 * k - 0.2, 100 * tab.loc[sig].loc[LABELS, "RET_m+1"], width=0.4, label=sig)
    ax.set_xticks(xs, LABELS)
    ax.set_xlabel("Quintile of the formation-month signal")
    ax.set_ylabel("Post-formation excess return (% per month)")
    ax.set_title("Quintile sorts on RES and SYS (Table 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_quintile_returns.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
