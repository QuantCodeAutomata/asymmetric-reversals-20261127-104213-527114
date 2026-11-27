"""Daily impulse responses of signed residual and systematic shocks (Section 6, Figures 3-7).

Dense daily Fama-MacBeth regression of r_{i,t} on 252 lags of eps~+, eps~-,
r~+ and r~- (Equation 19, 1,008 slopes), impulse responses of +/-1% shocks
(Equations 20-21), the 27-knot degree-1 spline version (Equations 22-24, 81
slopes) and the monthly triangular aggregation weights (Appendix C, Figure 7).

Run: python -m exp.exp5_irf
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
from src.irf import daily_dl_regression, impulse_response, signed_shocks, spline_basis, triangular_weights
from src.pipeline import daily_weights, load_panel
from src.reporting import RESULTS_DIR, save_csv, save_json

log = logging.getLogger(__name__)

SHOCKS = {"eps_pos": +1.0, "eps_neg": -1.0, "r_pos": +1.0, "r_neg": -1.0}
LABELS = {"eps_pos": "+1% residual (eps~+)", "eps_neg": "-1% residual (eps~-)",
          "r_pos": "+1% systematic (r~+)", "r_neg": "-1% systematic (r~-)"}
REPORT_TAUS = (1, 5, 10, 21, 32, 42, 53, 63, 126, 189, 252)


def regression_days(dates: pd.DatetimeIndex, n_lags: int) -> np.ndarray:
    """Days with a full 252-day lag history inside the decomposition sample."""
    start = int(np.searchsorted(dates.values, np.datetime64(config.DECOMP_START)))
    return np.arange(start + n_lags, len(dates))


def irf_summary(ir: np.ndarray, sign: float, horizon_max: int = 63) -> dict:
    """Reversal size, timing and day-1 share of an impulse response path (Section 6.1)."""
    base = ir[0]
    # reversal = movement of the price path back toward zero, in the direction opposite to the shock
    rev = -sign * (ir[1:horizon_max + 1] - base)
    k = int(np.argmax(rev))
    peak = float(rev[k])
    day1 = float(-sign * (ir[1] - base))
    out = {"ir_0": base, "max_reversal": peak, "tau_max_reversal": k + 1,
           "day1_share_of_reversal": day1 / peak if peak > 0 else np.nan,
           "week1_reversal": float(-sign * (ir[5] - base)), "ir_252": float(ir[-1]),
           "continuation_after_peak": float(sign * (ir[-1] - ir[k + 1]))}
    for tau in REPORT_TAUS:
        out[f"ir_{tau}"] = float(ir[tau])
    return out


def run_irfs(lam: np.ndarray, names: list[str], tag: str) -> tuple[dict, dict]:
    """Impulse responses (with NW 95% bands) for every shock in ``names``; lam (T, S, 252)."""
    irs, rows = {}, {}
    for s, name in enumerate(names):
        res = impulse_response(lam[:, s, :], SHOCKS[name])
        irs[name] = res
        rows[f"{tag}:{name}"] = {**irf_summary(res["ir"], SHOCKS[name]), "n_starts": res["n"],
                                 "band_width_21": float(res["hi"][21] - res["lo"][21])}
    if "eps_pos" in irs and "eps_neg" in irs:
        asym = irs["eps_pos"]["ir"] + irs["eps_neg"]["ir"]
        rows[f"{tag}:asymmetry_eps"] = {f"ir_{t}": float(asym[t]) for t in REPORT_TAUS}
    return irs, rows


def plot_irfs(dense: dict, spline: dict, path) -> None:
    """Figure 3/6 style panels: dense IRF with bands and the spline IRF."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, name in zip(axes.ravel(), SHOCKS):
        tau = np.arange(config.N_LAGS + 1)
        d = dense[name]
        ax.plot(tau, 100 * d["ir"], color="C0", label="dense (1,008 slopes)")
        ax.fill_between(tau, 100 * d["lo"], 100 * d["hi"], color="C0", alpha=0.2)
        if name in spline:
            s = spline[name]
            ax.plot(tau, 100 * s["ir"], color="C3", label="spline (81 slopes)")
            ax.fill_between(tau, 100 * s["lo"], 100 * s["hi"], color="C3", alpha=0.2)
        ax.axhline(100 * SHOCKS[name] * config.SHOCK_SIZE, color="k", lw=0.6, ls="--")
        ax.set_title(LABELS[name])
        ax.set_xlabel("trading days after the shock")
        ax.set_ylabel("cumulative return (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Impulse responses to 1% shocks (Equations 20-21), 95% Newey-West bands")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_aggregation(lam_s: np.ndarray, path) -> pd.DataFrame:
    """Figure 7: spline marginal responses at lags 1..41 against the triangular monthly weights."""
    w = triangular_weights()
    taus = np.arange(1, 42)
    mean_lam = np.nanmean(lam_s, axis=0)  # (3, 252) average implied lag coefficients
    df = pd.DataFrame({"tau": taus, "weight_n_tau_over_20": w[taus],
                       "lambda_eps_pos": mean_lam[0, taus - 1], "lambda_eps_neg": mean_lam[1, taus - 1],
                       "lambda_r": mean_lam[2, taus - 1]}).set_index("tau")
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(taus, df["lambda_eps_pos"], marker="o", ms=3, label="lambda eps~+ (spline)")
    ax1.plot(taus, df["lambda_eps_neg"], marker="o", ms=3, label="lambda eps~- (spline)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("lag tau (trading days)")
    ax1.set_ylabel("daily lag coefficient")
    ax2 = ax1.twinx()
    ax2.bar(taus, df["weight_n_tau_over_20"], color="grey", alpha=0.25, label="n_tau / 20")
    ax2.set_ylabel("monthly overlap weight")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8)
    ax1.set_title("Spline marginal responses and monthly triangular aggregation weights (Appendix C)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return df


def main() -> None:
    """Run the dense and spline daily distributed-lag regressions and save IRF summaries and figures."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    t0 = time.time()
    panel = load_panel()
    dates = panel["dates"]
    w_day = daily_weights(panel)
    sh = signed_shocks(panel["eps_t"], panel["r_t"])
    y = panel["exret"]
    days = regression_days(dates, config.N_LAGS)
    log.info("IRF regressions on %d days %s..%s", len(days), dates[days[0]].date(), dates[days[-1]].date())
    dense_names = ["eps_pos", "eps_neg", "r_pos", "r_neg"]
    lam_d = daily_dl_regression(y, w_day, [sh[k] for k in dense_names], days, 1, config.N_LAGS)
    log.info("dense regressions done %.0fs", time.time() - t0)
    B = spline_basis()
    assert B.shape == (config.N_LAGS, len(config.SPLINE_KNOTS))
    lam_s = daily_dl_regression(y, w_day, [sh["eps_pos"], sh["eps_neg"], sh["r"]], days, 1, config.N_LAGS, basis=B)
    log.info("spline regressions done %.0fs", time.time() - t0)

    dense_irs, rows = run_irfs(lam_d, dense_names, "dense")
    # pooled r~ in the spline: the same coefficients drive the +1% and -1% systematic responses (Equation 24)
    lam_s4 = np.stack([lam_s[:, 0], lam_s[:, 1], lam_s[:, 2], lam_s[:, 2]], axis=1)
    spline_irs, rows_s = run_irfs(lam_s4, dense_names, "spline")
    rows.update(rows_s)
    summary = pd.DataFrame(rows).T
    save_csv(summary, "exp5_irf_summary")
    plot_irfs(dense_irs, spline_irs, RESULTS_DIR / "exp5_irf.png")
    agg = plot_aggregation(lam_s, RESULTS_DIR / "exp5_monthly_aggregation.png")
    save_csv(agg, "exp5_monthly_aggregation")
    w = triangular_weights()
    taus = np.arange(2, 42)
    implied = {k: float((w[taus] * agg.reindex(taus)[f"lambda_{k}"]).sum()) for k in ("eps_pos", "eps_neg")}
    meta = {"n_days": int(len(days)), "first_day": str(dates[days[0]].date()), "last_day": str(dates[days[-1]].date()),
            "n_dense_slopes": 4 * config.N_LAGS, "n_spline_slopes": 3 * B.shape[1],
            "implied_monthly_slope_from_daily_spline": implied,
            "peak_weight_lags": [int(t) for t in taus[w[taus] == w[taus].max()]],
            "runtime_s": time.time() - t0,
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    save_json(meta, "exp5_meta")
    log.info("%s", summary.to_string())
    log.info("meta %s", meta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
