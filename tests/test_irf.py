"""Unit tests for src.irf: spline basis, distributed-lag regressions and impulse responses (Sections 6-7)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.irf import (daily_dl_regression, forward_sum, impulse_response, lag_window, signed_shocks,
                     spline_basis, triangular_weights)


# ------------------------------------------------------------------ spline_basis
def test_spline_basis_default_shape_partition_and_peaks() -> None:
    B = spline_basis(config.SPLINE_KNOTS)
    knots = np.asarray(config.SPLINE_KNOTS)
    assert B.shape == (252, 27)
    np.testing.assert_allclose(B.sum(axis=1), 1.0)
    assert (B >= 0).all()
    for j, k in enumerate(knots):
        assert B[k - 1, j] == pytest.approx(1.0)  # row index = lag - 1
        assert np.argmax(B[:, j]) == k - 1


def test_spline_basis_is_piecewise_linear_tent_interpolation() -> None:
    knots = np.asarray(config.SPLINE_KNOTS, dtype=float)
    lags = np.arange(1, 253, dtype=float)
    B = spline_basis()
    for j in range(len(knots)):
        tent = np.interp(lags, knots, np.eye(len(knots))[j])
        np.testing.assert_allclose(B[:, j], tent, atol=1e-12)


def test_spline_basis_custom_lags_and_knots() -> None:
    B = spline_basis((2, 4, 8), lags=np.array([2, 3, 6, 8]))
    np.testing.assert_allclose(B, [[1, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0, 0, 1]], atol=1e-12)


# ------------------------------------------------------------------ lag_window / signed_shocks
def test_lag_window_rows_are_lagged_values_with_zero_padding() -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    win = lag_window(x, t=6, first_lag=2, last_lag=4)
    np.testing.assert_array_equal(win, x[[4, 3, 2]])
    early = lag_window(x, t=2, first_lag=1, last_lag=4)
    np.testing.assert_array_equal(early[:2], x[[1, 0]])
    np.testing.assert_array_equal(early[2:], 0.0)
    assert (lag_window(x, t=0, first_lag=1, last_lag=3) == 0).all()
    assert (lag_window(x, t=1, first_lag=2, last_lag=3) == 0).all()


def test_signed_shocks_split_and_zero_fill() -> None:
    eps = np.array([[0.01, -0.02, np.nan]])
    r = np.array([[-0.03, 0.04, 0.05]])
    s = signed_shocks(eps, r)
    np.testing.assert_allclose(s["eps_pos"], [[0.01, 0, 0]])
    np.testing.assert_allclose(s["eps_neg"], [[0, -0.02, 0]])
    np.testing.assert_allclose(s["eps_pos"] + s["eps_neg"], np.nan_to_num(eps))
    np.testing.assert_allclose(s["r_pos"] + s["r_neg"], s["r"])
    assert all(v.dtype == np.float32 for v in s.values())


# ------------------------------------------------------------------ daily_dl_regression
def _dl_panel(rng: np.random.Generator, lam: dict[int, float], T: int = 40, N: int = 150, noise: float = 0.0
              ) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal((T, N))
    y = np.full((T, N), 0.001) + noise * rng.standard_normal((T, N))
    for tau, c in lam.items():
        y[tau:] += c * x[:-tau]
    return x, y


def test_daily_dl_regression_dense_recovers_lag_coefficients(rng: np.random.Generator) -> None:
    x, y = _dl_panel(rng, {1: 0.5, 3: -0.2})
    w = rng.uniform(0.5, 2.0, y.shape)
    days = np.arange(5, 40)
    lam = daily_dl_regression(y, w, [x], days, first_lag=1, last_lag=5)
    assert lam.shape == (len(days), 1, 5)
    np.testing.assert_allclose(lam[:, 0], np.tile([0.5, 0.0, -0.2, 0.0, 0.0], (len(days), 1)), atol=1e-8)


def test_daily_dl_regression_dense_with_noise_averages_to_truth(rng: np.random.Generator) -> None:
    x, y = _dl_panel(rng, {1: 0.5, 3: -0.2}, T=80, N=200, noise=1.0)
    lam = daily_dl_regression(y, np.ones_like(y), [x], np.arange(5, 80), first_lag=1, last_lag=4)
    np.testing.assert_allclose(lam[:, 0].mean(axis=0), [0.5, 0.0, -0.2, 0.0], atol=0.03)


def test_daily_dl_regression_spline_basis_recovers_representable_lags(rng: np.random.Generator) -> None:
    lam_pos = {1: 0.5, 2: 0.15, 3: -0.2, 4: -0.1, 5: 0.0}  # linear between knots 1, 3, 5
    lam_neg = {1: -0.4, 2: -0.2, 3: 0.0, 4: 0.1, 5: 0.2}
    T, N = 40, 150
    xp = rng.standard_normal((T, N))
    xn = rng.standard_normal((T, N))
    y = np.full((T, N), -0.002)
    for tau in range(1, 6):
        y[tau:] += lam_pos[tau] * xp[:-tau] + lam_neg[tau] * xn[:-tau]
    B = spline_basis((1, 3, 5))
    assert B.shape == (5, 3)
    lam = daily_dl_regression(y, np.ones((T, N)), [xp, xn], np.arange(6, T), 1, 5, basis=B)
    assert lam.shape == (T - 6, 2, 5)
    np.testing.assert_allclose(lam[:, 0], np.tile(list(lam_pos.values()), (T - 6, 1)), atol=1e-8)
    np.testing.assert_allclose(lam[:, 1], np.tile(list(lam_neg.values()), (T - 6, 1)), atol=1e-8)


def test_daily_dl_regression_skips_infeasible_days(rng: np.random.Generator) -> None:
    x, y = _dl_panel(rng, {1: 0.5}, T=12, N=40)
    w = np.ones_like(y)
    w[5, :] = 0.0  # nobody in the cross-section
    w[6, 15:] = np.nan  # 15 stocks < 6 regressors + 10 required
    y[7, :] = np.nan
    lam = daily_dl_regression(y, w, [x], np.array([4, 5, 6, 7, 8]), 1, 5)
    assert np.isnan(lam[1:4]).all()
    assert np.isfinite(lam[[0, 4]]).all()


# ------------------------------------------------------------------ impulse_response
def test_impulse_response_closed_form_constant_lambda() -> None:
    H, T = 6, 30
    lam_h = np.array([-0.3, 0.2, -0.1, 0.05, 0.0, 0.4])
    lam = np.tile(lam_h, (T, 1))
    for s in (1.0, -1.0):
        with np.errstate(all="ignore"):
            out = impulse_response(lam, s)
        expected = (1 + s * 0.01) * np.r_[1.0, np.cumprod(1 + s * 0.01 * lam_h)] - 1
        np.testing.assert_allclose(out["ir"], expected, atol=1e-12)
        np.testing.assert_allclose(out["paths_mean"], expected, atol=1e-12)
        assert out["ir"][0] == pytest.approx(s * config.SHOCK_SIZE)
        assert out["n"] == T - H
        np.testing.assert_allclose(out["lo"], expected, atol=1e-8)
        np.testing.assert_allclose(out["hi"], expected, atol=1e-8)


def test_impulse_response_uses_coefficient_estimated_on_day_t_plus_h(rng: np.random.Generator) -> None:
    T, H = 40, 3
    lam = rng.standard_normal((T, H))
    lam[10, 1] = np.nan  # invalidates exactly the start t = 8 (needs lam[t + 2, 1])
    out = impulse_response(lam, 1.0)
    paths = []
    for t in range(T - H):
        steps = [1 + 0.01 * lam[t + h, h - 1] for h in range(1, H + 1)]
        p = 1.01 * np.r_[1.0, np.cumprod(steps)] - 1
        if np.isfinite(p).all():
            paths.append(p)
    paths = np.array(paths)
    assert out["n"] == len(paths) == T - H - 1
    np.testing.assert_allclose(out["ir"], paths.mean(axis=0), atol=1e-12)
    assert (out["lo"][1:] < out["ir"][1:]).all() and (out["hi"][1:] > out["ir"][1:]).all()


def test_impulse_response_too_short_raises() -> None:
    with pytest.raises(ValueError):
        impulse_response(np.zeros((5, 3)), 1.0)


# ------------------------------------------------------------------ triangular_weights / forward_sum
def test_triangular_weights_shape_peak_and_support() -> None:
    w = triangular_weights()
    assert w.shape == (42,)
    np.testing.assert_array_equal(w[:2], 0.0)
    assert w[21] == w[22] == 1.0 and w.max() == 1.0
    assert w[2] == pytest.approx(1 / 20) and w[41] == pytest.approx(1 / 20)
    np.testing.assert_allclose(w[2:22], np.arange(1, 21) / 20)
    np.testing.assert_allclose(w[22:], np.arange(20, 0, -1) / 20)
    assert w.sum() == pytest.approx(21.0)  # 20 shock days x 21 holding days / 20


def test_forward_sum_matches_explicit_sum_and_nan_at_end(rng: np.random.Generator) -> None:
    x = rng.standard_normal((15, 3))
    x[7, 2] = np.nan
    h = 4
    out = forward_sum(x, h)
    for t in range(15):
        for j in range(3):
            seg = x[t:t + h, j]
            if t + h > 15 or not np.isfinite(seg).all():
                assert np.isnan(out[t, j])
            else:
                assert out[t, j] == pytest.approx(seg.sum())
    assert np.isnan(out[-(h - 1):]).all()
    np.testing.assert_allclose(forward_sum(x[:, :2], 1), x[:, :2])
