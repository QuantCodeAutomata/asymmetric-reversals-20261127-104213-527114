"""Daily cross-sectional return decomposition and goodness-of-fit evaluation (Section 3, Figure 1).

Annual adjusted R^2 (Equation 7) of the daily WLS cross-sectional models
(characteristics only; characteristics + FF49 industries) against stock-by-stock
monthly Fama-French three-factor time-series regressions, weighted and equal-weighted.

Run: python -m exp.exp1_decomposition_fit
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.decomposition import annual_adj_r2
from src.pipeline import load_panel
from src.reporting import RESULTS_DIR, save_csv, save_json

log = logging.getLogger(__name__)
MODELS = {"char_ind": "ci", "char_only": "c", "ff3_ts": "ff3"}


def adj_r2_table(panel: dict) -> pd.DataFrame:
    """Annual adjusted R^2 per model, weighted (WLS) and equal-weighted (OLS) versions."""
    cols = {}
    for name, key in MODELS.items():
        g = lambda s: panel[f"fit_{key}_{s}"]
        cols[f"{name}_wls"] = annual_adj_r2(panel["dates"], g("rss"), g("tss"), g("nobs"), g("npar"))
        cols[f"{name}_ols"] = annual_adj_r2(panel["dates"], g("rss_eq"), g("tss_eq"), g("nobs"), g("npar"))
    return pd.DataFrame(cols).rename_axis("year")


def plot(tab: pd.DataFrame) -> None:
    """Figure 1: annual adjusted R^2 of the three models (weighted and equal-weighted)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, suf, title in zip(axes, ("wls", "ols"), ("Weighted (WLS)", "Equal-weighted (OLS)")):
        for name, lab in (("char_ind", "Characteristics + industries"), ("char_only", "Characteristics only"),
                          ("ff3_ts", "FF3 stock-by-stock time series")):
            ax.plot(tab.index, 100 * tab[f"{name}_{suf}"], marker="o", label=lab)
        ax.set_title(f"Annual adjusted R$^2$, {title}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Adjusted R$^2$ (%)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_adj_r2.png", dpi=130)
    plt.close(fig)


def main() -> None:
    """Compute the annual adjusted R^2 table, its averages and Figure 1."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    panel = load_panel()
    tab = adj_r2_table(panel)
    log.info("annual adjusted R2:\n%s", tab.round(4))
    save_csv(tab, "exp1_annual_adj_r2")
    avg = tab.mean()
    summary = {f"avg_{k}": float(v) for k, v in avg.items()}
    for suf in ("wls", "ols"):
        summary[f"gap_char_ind_vs_ff3_{suf}"] = float(avg[f"char_ind_{suf}"] - avg[f"ff3_ts_{suf}"])
        summary[f"gap_char_only_vs_ff3_{suf}"] = float(avg[f"char_only_{suf}"] - avg[f"ff3_ts_{suf}"])
    summary["max_char_ind_wls"] = float(tab["char_ind_wls"].max())
    summary["year_max_char_ind_wls"] = int(tab["char_ind_wls"].idxmax())
    summary["years"] = [int(tab.index.min()), int(tab.index.max())]
    save_json(summary, "exp1_summary")
    plot(tab)
    log.info("summary: %s", summary)


if __name__ == "__main__":
    main()
