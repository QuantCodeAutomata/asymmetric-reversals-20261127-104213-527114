"""Unit tests for src.stats: Newey-West inference, Fama-MacBeth and NYSE-breakpoint sorts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stats import (assign_quantiles, fama_macbeth, fm_coef_series, nw_lags, nw_mean, nyse_breakpoints,
                       sort_portfolios, ts_alpha, wls_slopes)


# ------------------------------------------------------------------ Newey-West
@pytest.mark.parametrize("T", [1, 50, 100, 250, 500, 612, 3000])
def test_nw_lags_rule(T: int) -> None:
    assert nw_lags(T) == int(np.floor(4 * (T / 100) ** (2 / 9)))


def test_nw_lags_known_values() -> None:
    assert nw_lags(100) == 4
    assert nw_lags(500) == 5
    assert nw_lags(612) == 5


def test_nw_mean_close_to_iid_t_for_iid_data(rng: np.random.Generator) -> None:
    x = rng.normal(0.1, 1.0, 2000)
    m, t = nw_mean(x)
    t_iid = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert m == pytest.approx(x.mean())
    assert t == pytest.approx(t_iid, rel=0.1)


def test_nw_mean_zero_lags_is_white_t(rng: np.random.Generator) -> None:
    x = rng.normal(0.0, 1.0, 300)
    m, t = nw_mean(x, lags=0)
    assert t == pytest.approx(x.mean() / (x.std(ddof=0) / np.sqrt(len(x))), rel=1e-8)


def test_nw_mean_positive_autocorrelation_lowers_t(rng: np.random.Generator) -> None:
    e = rng.standard_normal(3000)
    x = 0.05 + np.convolve(e, np.ones(10) / 10, mode="valid")  # MA(9): strongly autocorrelated
    _, t_nw = nw_mean(x, lags=20)
    _, t_white = nw_mean(x, lags=0)
    assert abs(t_nw) < 0.6 * abs(t_white)


def test_nw_mean_drops_nan_and_short_samples() -> None:
    m, t = nw_mean(np.array([1.0, np.nan, 2.0, 3.0, np.nan]))
    assert m == pytest.approx(2.0) and np.isfinite(t)
    for x in (np.array([]), np.array([1.0]), np.array([np.nan, 1.0, 2.0])):
        m, t = nw_mean(x)
        assert np.isnan(m) and np.isnan(t)


def test_ts_alpha_recovers_alpha_and_betas(rng: np.random.Generator) -> None:
    T = 400
    idx = pd.date_range("1990-01-31", periods=T, freq="ME")
    F = pd.DataFrame(rng.standard_normal((T, 2)) * 0.04, index=idx, columns=["MKT", "SMB"])
    y = pd.Series(0.004 + 1.2 * F["MKT"] - 0.5 * F["SMB"] + rng.standard_normal(T) * 1e-4, index=idx)
    out = ts_alpha(y, F)
    assert out["alpha"] == pytest.approx(0.004, abs=3e-5)
    assert out["betas"]["MKT"] == pytest.approx(1.2, abs=1e-3)
    assert out["betas"]["SMB"] == pytest.approx(-0.5, abs=1e-3)
    assert out["t_alpha"] > 10 and out["n"] == T and out["r2"] > 0.99
    short = ts_alpha(y.iloc[:4], F)
    assert np.isnan(short["alpha"]) and short["n"] == 4


# ------------------------------------------------------------------ Fama-MacBeth
def test_wls_slopes_exact_and_too_few_obs(rng: np.random.Generator) -> None:
    X = rng.standard_normal((40, 2))
    y = 0.1 + X @ np.array([2.0, -1.0])
    coef, adj = wls_slopes(y, X, rng.uniform(1, 2, 40))
    np.testing.assert_allclose(coef, [0.1, 2.0, -1.0], atol=1e-10)
    assert adj == pytest.approx(1.0)
    assert wls_slopes(y[:7], X[:7], None) is None  # n < k + 5 with k = 3


def test_fama_macbeth_recovers_known_slope(rng: np.random.Generator) -> None:
    P, N = 120, 200
    x = rng.standard_normal((P, N))
    z = rng.standard_normal((P, N))
    lam = 0.5 + rng.standard_normal(P) * 0.05  # time-varying slope with mean 0.5
    y = 0.01 + lam[:, None] * x + 0.0 * z + rng.standard_normal((P, N)) * 0.1
    y[0, :] = np.nan  # an infeasible period is skipped
    res = fama_macbeth(y, {"x": x, "z": z}, None)
    assert list(res.index) == ["const", "x", "z", "adj_r2", "n_periods"]
    assert res.at["x", "coef"] == pytest.approx(lam[1:].mean(), abs=0.01)
    assert res.at["const", "coef"] == pytest.approx(0.01, abs=0.005)
    assert res.at["x", "t"] > 30 and abs(res.at["z", "t"]) < 4
    assert res.at["n_periods", "coef"] == P - 1
    assert 0.5 < res.at["adj_r2", "coef"] < 1
    w = rng.uniform(0.5, 3.0, (P, N))
    res_w = fama_macbeth(y, {"x": x, "z": z}, w)
    assert res_w.at["x", "coef"] == pytest.approx(0.5, abs=0.03)


def test_fm_coef_series_rows_match_cross_sections(rng: np.random.Generator) -> None:
    P, N = 4, 30
    x = rng.standard_normal((P, N))
    y = np.array([0.1 * p + (p + 1) * x[p] for p in range(P)])
    y[2] = np.nan
    out = fm_coef_series(y, {"x": x}, None, np.arange(P))
    np.testing.assert_allclose(out[[0, 1, 3]], [[0.0, 1.0], [0.1, 2.0], [0.3, 4.0]], atol=1e-10)
    assert np.isnan(out[2]).all()


# ------------------------------------------------------------------ NYSE breakpoints and sorts
def test_nyse_breakpoints_use_only_nyse_stocks() -> None:
    x = np.r_[np.arange(1.0, 11.0), np.full(10, 1000.0)]
    is_nyse = np.r_[np.ones(10, bool), np.zeros(10, bool)]
    valid = np.ones(20, bool)
    bp = nyse_breakpoints(x, valid, is_nyse, n_q=5)
    np.testing.assert_allclose(bp, np.quantile(np.arange(1.0, 11.0), [0.2, 0.4, 0.6, 0.8]))
    lab = assign_quantiles(x, valid, is_nyse, n_q=5)
    assert (lab[10:] == 4).all()  # large non-NYSE stocks all land in the top quintile
    np.testing.assert_array_equal(np.bincount(lab[:10], minlength=5), [2, 2, 2, 2, 2])


def test_assign_quantiles_invalid_and_too_few_nyse() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    valid = np.array([True, True, True, False, True, True])
    is_nyse = np.ones(6, bool)
    lab = assign_quantiles(x, valid, is_nyse, n_q=2)
    assert lab[2] == -1 and lab[3] == -1  # missing signal / not valid
    np.testing.assert_array_equal(lab[[0, 1, 4, 5]], [0, 0, 1, 1])
    assert np.isnan(nyse_breakpoints(x, valid, np.zeros(6, bool), n_q=2)).all()
    assert (assign_quantiles(x, valid, np.zeros(6, bool), n_q=2) == -1).all()


def test_sort_portfolios_value_weights_and_high_minus_low(rng: np.random.Generator) -> None:
    M, N, nq = 3, 50, 5
    signal = rng.standard_normal((M, N))
    w = rng.uniform(1.0, 10.0, (M, N))
    valid = np.ones((M, N), bool)
    is_nyse = rng.random(N) < 0.6
    ret_next = 0.01 * signal + rng.standard_normal((M, N)) * 0.001  # high signal -> high return
    out = sort_portfolios(signal, valid, is_nyse, w, ret_next, n_q=nq)
    for m in range(M):
        for q in range(nq):
            sel = out["labels"][m] == q
            assert out["n"][m, q] == sel.sum() > 0
            np.testing.assert_allclose(out["ret"][m, q], (w[m, sel] * ret_next[m, sel]).sum() / w[m, sel].sum())
    ones = sort_portfolios(signal, valid, is_nyse, w, np.ones((M, N)), n_q=nq)
    np.testing.assert_allclose(ones["ret"], 1.0)  # weights sum to one in every portfolio
    hl = out["ret"][:, -1] - out["ret"][:, 0]
    assert (hl > 0).all()
    assert (np.diff(out["ret"], axis=1) > 0).all()


def test_sort_portfolios_turnover_zero_for_unchanged_holdings_and_one_for_replacement() -> None:
    M, N = 3, 20
    signal = np.tile(np.arange(N, dtype=float), (M, 1))
    signal[2] = signal[2][::-1]  # month 2: ranks fully reversed
    w = np.ones((M, N))
    out = sort_portfolios(signal, np.ones((M, N), bool), np.ones(N, bool), w, np.zeros((M, N)), n_q=2)
    assert np.isnan(out["turnover"][0]).all()
    np.testing.assert_allclose(out["turnover"][1], 0.0)
    np.testing.assert_allclose(out["turnover"][2], 1.0)


def test_sort_portfolios_excludes_invalid_and_nonpositive_weights(rng: np.random.Generator) -> None:
    M, N = 1, 30
    signal = rng.standard_normal((M, N))
    w = np.ones((M, N))
    w[0, 0] = 0.0
    w[0, 1] = np.nan
    valid = np.ones((M, N), bool)
    valid[0, 2] = False
    out = sort_portfolios(signal, valid, np.ones(N, bool), w, np.zeros((M, N)), n_q=3)
    assert (out["labels"][0, :3] == -1).all()
    assert out["n"].sum() == N - 3
