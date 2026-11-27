"""Unit tests for src.decomposition (Section 3, Equations 3-9) on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.decomposition import (annual_adj_r2, daily_decomposition, industry_dummies, wls_cross_section,
                               zero_ea_residuals)
from src.pipeline import prev_month_index


def _design(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    return np.column_stack([np.ones(n), rng.standard_normal((n, k))])


# ------------------------------------------------------------------ wls_cross_section
def test_wls_recovers_known_coefficients_noiseless(rng: np.random.Generator) -> None:
    Z = _design(rng, 50, 3)
    f_true = np.array([0.01, -0.5, 0.25, 2.0])
    w = rng.uniform(0.1, 2.0, 50)
    f, fitted, rank = wls_cross_section(Z @ f_true, Z, w)
    np.testing.assert_allclose(f, f_true, atol=1e-10)
    np.testing.assert_allclose(fitted, Z @ f_true, atol=1e-10)
    assert rank == 4


def test_wls_residuals_weighted_orthogonal_to_regressors(rng: np.random.Generator) -> None:
    Z = _design(rng, 80, 2)
    w = rng.uniform(0.1, 3.0, 80)
    r = Z @ np.array([0.1, 1.0, -1.0]) + rng.standard_normal(80)
    f, fitted, _ = wls_cross_section(r, Z, w)
    e = r - fitted
    np.testing.assert_allclose(Z.T @ (w * e), 0.0, atol=1e-10)
    # matches the closed-form WLS estimator (Z'WZ)^{-1} Z'Wr
    f_closed = np.linalg.solve(Z.T @ (w[:, None] * Z), Z.T @ (w * r))
    np.testing.assert_allclose(f, f_closed, atol=1e-10)


def test_wls_weights_matter(rng: np.random.Generator) -> None:
    """Zero weight on an outlier removes its influence on the coefficients."""
    Z = _design(rng, 30, 1)
    r = Z @ np.array([0.0, 1.0])
    r_out = r.copy()
    r_out[0] += 100.0
    w = np.ones(30)
    w[0] = 0.0
    f, _, _ = wls_cross_section(r_out, Z, w)
    np.testing.assert_allclose(f, [0.0, 1.0], atol=1e-10)


def test_wls_rank_deficient_empty_industry(rng: np.random.Generator) -> None:
    """An all-zero dummy column (empty industry) lowers the rank but the fit is still exact."""
    n = 20
    D = np.zeros((n, 3))
    D[:10, 0] = 1
    D[10:, 1] = 1
    x = rng.standard_normal(n)
    Z = np.column_stack([D, x])
    r = 0.5 * D[:, 0] - 0.3 * D[:, 1] + 2.0 * x
    f, fitted, rank = wls_cross_section(r, Z, np.ones(n))
    assert rank == 3
    np.testing.assert_allclose(fitted, r, atol=1e-10)
    assert abs(f[2]) < 1e-10  # minimum-norm solution puts zero on the empty column


def test_wls_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        wls_cross_section(np.zeros(5), np.zeros((4, 2)), np.ones(5))
    with pytest.raises(ValueError):
        wls_cross_section(np.zeros(5), np.zeros((5, 2)), np.ones(4))


# ------------------------------------------------------------------ industry_dummies
def test_industry_dummies_rows_sum_to_one() -> None:
    ind = np.array([1, 49, 3, 3, 17], dtype=float)
    D = industry_dummies(ind)
    assert D.shape == (5, config.N_INDUSTRIES)
    np.testing.assert_array_equal(D.sum(axis=1), 1.0)
    assert D[0, 0] == 1 and D[1, 48] == 1 and D[2, 2] == 1 and D[4, 16] == 1
    np.testing.assert_array_equal(D.sum(axis=0)[[0, 2, 16, 48]], [1, 2, 1, 1])


def test_industry_dummies_custom_count_and_empty() -> None:
    D = industry_dummies(np.array([2, 1]), n_ind=3)
    np.testing.assert_array_equal(D, [[0, 1, 0], [1, 0, 0]])
    assert industry_dummies(np.array([]), n_ind=3).shape == (0, 3)


# ------------------------------------------------------------------ daily_decomposition
def _decomp_inputs(rng: np.random.Generator, T: int = 6, N: int = 30, K: int = 2
                   ) -> tuple[np.ndarray, ...]:
    month_of_day = np.array([-1, -1, 0, 0, 1, 1])[:T]
    chars_m = rng.standard_normal((2, N, K))
    weights_m = rng.uniform(1.0, 5.0, (2, N))
    incl_m = np.ones((2, K), dtype=bool)
    return month_of_day, chars_m, weights_m, incl_m


def test_daily_decomposition_uses_previous_month_characteristics(rng: np.random.Generator) -> None:
    month_of_day, chars_m, weights_m, incl_m = _decomp_inputs(rng)
    T, N = 6, chars_m.shape[1]
    beta = np.array([0.02, -0.01])
    exret = np.full((T, N), 0.001)
    for t in range(T):
        m = max(month_of_day[t], 0)
        exret[t] = 0.003 + chars_m[m] @ beta  # exact factor model with the information-set month
    out = daily_decomposition(exret, month_of_day, weights_m, chars_m, incl_m, None, range(T))
    assert np.isnan(out["eps"][:2]).all() and np.isnan(out["fitted"][:2]).all()
    np.testing.assert_allclose(out["eps"][2:], 0.0, atol=1e-6)
    np.testing.assert_allclose(out["fitted"][2:], exret[2:], atol=1e-6)
    np.testing.assert_allclose(out["rss"][2:], 0.0, atol=1e-12)
    np.testing.assert_array_equal(out["nobs"][2:], N)
    np.testing.assert_array_equal(out["npar"][2:], 3)
    # the same returns regressed on the *other* month's characteristics would not fit exactly
    shifted = month_of_day.copy()
    shifted[2:4] = 1
    bad = daily_decomposition(exret, shifted, weights_m, chars_m, incl_m, None, range(T))
    assert np.nanmax(np.abs(bad["eps"][2:4])) > 1e-3


def test_daily_decomposition_matches_manual_wls_with_normalized_weights(rng: np.random.Generator) -> None:
    month_of_day, chars_m, weights_m, incl_m = _decomp_inputs(rng)
    T, N = 6, chars_m.shape[1]
    exret = rng.standard_normal((T, N)) * 0.02
    out = daily_decomposition(exret, month_of_day, weights_m, chars_m, incl_m, None, range(T))
    t, m = 4, 1
    Z = np.column_stack([np.ones(N), chars_m[m]])
    wn = weights_m[m] / weights_m[m].sum()
    f = np.linalg.solve(Z.T @ (wn[:, None] * Z), Z.T @ (wn * exret[t]))
    e = exret[t] - Z @ f
    np.testing.assert_allclose(out["eps"][t], e, atol=1e-6)
    np.testing.assert_allclose(out["rss"][t], (wn * e * e).sum(), rtol=1e-8)
    rbar = (wn * exret[t]).sum()
    np.testing.assert_allclose(out["tss"][t], (wn * (exret[t] - rbar) ** 2).sum(), rtol=1e-8)
    e0 = exret[t] - Z @ np.linalg.lstsq(Z, exret[t], rcond=None)[0]
    np.testing.assert_allclose(out["rss_eq"][t], (e0 ** 2).mean(), rtol=1e-8)
    np.testing.assert_allclose(out["tss_eq"][t], exret[t].var(), rtol=1e-8)
    assert out["eps"].dtype == np.float32 and out["fitted"].dtype == np.float32


def test_daily_decomposition_nan_outside_universe_and_day_range(rng: np.random.Generator) -> None:
    month_of_day, chars_m, weights_m, incl_m = _decomp_inputs(rng)
    T, N = 6, chars_m.shape[1]
    weights_m[0, :3] = np.nan  # not in the month-0 universe
    weights_m[1, 5] = 0.0
    exret = rng.standard_normal((T, N)) * 0.02
    exret[3, 7] = np.nan  # missing return
    out = daily_decomposition(exret, month_of_day, weights_m, chars_m, incl_m, None, range(0, 5))
    assert np.isnan(out["eps"][2:4, :3]).all()
    assert np.isfinite(out["eps"][4:5, :3]).all()  # back in the universe in month 1
    assert np.isnan(out["eps"][4, 5])
    assert np.isnan(out["eps"][3, 7]) and np.isfinite(out["eps"][2, 7])
    assert np.isnan(out["eps"][5]).all() and np.isnan(out["rss"][5])  # day 5 not in day_range
    assert out["nobs"][2] == N - 3 and out["nobs"][3] == N - 4


def test_daily_decomposition_respects_inclusion_flags(rng: np.random.Generator) -> None:
    """An excluded characteristic (incl=False) is not a regressor, even if it holds NaN."""
    month_of_day, chars_m, weights_m, incl_m = _decomp_inputs(rng, K=3)
    T, N = 6, chars_m.shape[1]
    chars_m[:, :, 2] = np.nan
    incl_m[:, 2] = False
    exret = np.vstack([0.001 + chars_m[max(m, 0), :, 0] * 0.01 for m in month_of_day])
    out = daily_decomposition(exret, month_of_day, weights_m, chars_m, incl_m, None, range(T))
    np.testing.assert_allclose(out["eps"][2:], 0.0, atol=1e-6)
    np.testing.assert_array_equal(out["npar"][2:], 3)


def test_daily_decomposition_with_industries_absorbs_industry_means(rng: np.random.Generator) -> None:
    T, N, K = 4, 100, 1
    month_of_day = np.array([-1, 0, 0, 0])
    chars_m = rng.standard_normal((1, N, K))
    weights_m = rng.uniform(1.0, 2.0, (1, N))
    ind_m = rng.integers(1, 5, (1, N)).astype(float)
    ind_m[0, 0] = np.nan  # unknown industry -> excluded
    incl_m = np.ones((1, K), dtype=bool)
    ind_effect = np.array([0.0, 0.01, -0.02, 0.03, 0.005])
    exret = np.vstack([ind_effect[np.nan_to_num(ind_m[0]).astype(int)] + 0.02 * chars_m[0, :, 0]] * T)
    out = daily_decomposition(exret, month_of_day, weights_m, chars_m, incl_m, ind_m, range(T))
    assert np.isnan(out["eps"][1:, 0]).all()
    np.testing.assert_allclose(out["eps"][1:, 1:], 0.0, atol=1e-6)
    np.testing.assert_array_equal(out["npar"][1:], 4 + K)  # only non-empty industries count


def test_daily_decomposition_too_few_stocks_skipped(rng: np.random.Generator) -> None:
    N, K = 4, 2  # minimum is 3*(K+1)//2 = 4 valid stocks; only 3 are valid here
    month_of_day = np.array([0])
    chars_m = rng.standard_normal((1, N, K))
    weights_m = np.ones((1, N))
    weights_m[0, 0] = np.nan
    out = daily_decomposition(rng.standard_normal((1, N)), month_of_day, weights_m, chars_m,
                              np.ones((1, K), dtype=bool), None, range(1))
    assert np.isnan(out["eps"]).all() and np.isnan(out["rss"]).all()


# ------------------------------------------------------------------ zero_ea_residuals
def test_zero_ea_residuals_zeroes_window_and_returns_systematic_part() -> None:
    exret = np.array([[0.01, 0.02, np.nan], [0.03, -0.01, 0.00]])
    eps = np.array([[0.004, 0.015, np.nan], [-0.002, 0.005, np.nan]])
    ea = np.array([[False, True, True], [True, False, False]])
    eps_t, r_t = zero_ea_residuals(exret, eps, ea)
    np.testing.assert_allclose(eps_t, [[0.004, 0.0, np.nan], [0.0, 0.005, np.nan]], atol=1e-7)
    np.testing.assert_allclose(r_t, [[0.006, 0.02, np.nan], [0.03, -0.015, np.nan]], atol=1e-7)
    finite = np.isfinite(eps_t)
    np.testing.assert_allclose(eps_t[finite] + r_t[finite], exret[finite], atol=1e-7)
    assert eps_t.dtype == np.float32 and r_t.dtype == np.float32


def test_zero_ea_residuals_inside_window_all_return_is_systematic(rng: np.random.Generator) -> None:
    exret = rng.standard_normal((5, 4))
    eps = rng.standard_normal((5, 4))
    eps_t, r_t = zero_ea_residuals(exret, eps, np.ones((5, 4), dtype=bool))
    assert (eps_t == 0).all()
    np.testing.assert_allclose(r_t, exret, atol=1e-6)


# ------------------------------------------------------------------ annual_adj_r2
def test_annual_adj_r2_hand_computed() -> None:
    dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-06-01", "2021-03-01", "2021-03-02"])
    rss = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    tss = np.array([2.0, 4.0, 5.0, 10.0, 1.0])
    nobs = np.array([100.0, 110.0, 120.0, 200.0, 200.0])
    npar = np.array([10.0, 10.0, 10.0, 20.0, 20.0])
    out = annual_adj_r2(dates, rss, tss, nobs, npar)
    # 2020: RSS=6, N=330, K=30, TSS=11, D=3 ; 2021: last day dropped (NaN), RSS=4, N=200, K=20, TSS=10, D=1
    exp2020 = 1 - (6 / (330 - 30)) / (11 / (330 - 3))
    exp2021 = 1 - (4 / (200 - 20)) / (10 / (200 - 1))
    assert list(out.index) == [2020, 2021]
    np.testing.assert_allclose(out.to_numpy(), [exp2020, exp2021], rtol=1e-12)
    assert out.name == "adj_r2"


def test_annual_adj_r2_perfect_fit_is_one() -> None:
    dates = pd.bdate_range("2019-01-01", periods=5)
    out = annual_adj_r2(dates, np.zeros(5), np.ones(5), np.full(5, 50.0), np.full(5, 3.0))
    np.testing.assert_allclose(out.to_numpy(), [1.0])


def test_prev_month_index_points_to_previous_month_end() -> None:
    """month_of_day for daily_decomposition: each day uses the month end strictly before it."""
    dates = pd.bdate_range("2021-01-01", "2021-03-05")
    mends = pd.DatetimeIndex(["2021-01-29", "2021-02-26"])
    idx = prev_month_index(dates, mends)
    assert (idx[dates <= "2021-01-29"] == -1).all()  # the month-end day itself still uses the earlier info set
    assert (idx[(dates > "2021-01-29") & (dates <= "2021-02-26")] == 0).all()
    assert (idx[dates > "2021-02-26"] == 1).all()
