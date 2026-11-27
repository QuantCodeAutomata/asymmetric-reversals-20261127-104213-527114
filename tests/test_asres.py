"""Unit tests for src.asres: out-of-sample ASRES forecast (Section 7, Equations 25-28)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.asres import asres_basis, asres_signal, expanding_lambda, first_day_next_month, size_groups
from src.panel import month_ends


# ------------------------------------------------------------------ asres_basis
def test_asres_basis_shape_partition_and_first_knot() -> None:
    lags, B = asres_basis()
    np.testing.assert_array_equal(lags, np.arange(2, 253))
    assert B.shape == (251, 26)
    np.testing.assert_allclose(B.sum(axis=1), 1.0)
    assert B[0, 0] == pytest.approx(1.0)  # first tent peaks at lag 2
    assert B[-1, -1] == pytest.approx(1.0)  # last tent peaks at lag 252
    knots = [k for k in config.SPLINE_KNOTS if k >= 2]
    for j, k in enumerate(knots):
        assert B[k - 2, j] == pytest.approx(1.0)


# ------------------------------------------------------------------ expanding_lambda
def _calendar() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = pd.bdate_range("2021-01-01", "2021-04-30")
    return dates, month_ends(dates)


def test_expanding_lambda_excludes_regressions_ending_after_previous_month_end() -> None:
    dates, mends = _calendar()
    L, h = 3, 5
    jan_end = dates.get_loc(mends[0])
    feb_end = dates.get_loc(mends[1])
    # day 0: holding period ends in January; jan_end - 1: formed in January but holding ends in February
    reg_days = np.array([0, jan_end - 1, jan_end + 1])
    lam = np.zeros((3, 3, L))
    lam[0, :2] = 1.0
    lam[1, :2] = 1e6  # huge coefficient that must not leak into February's forecast
    lam[2, :2] = 3.0
    out = expanding_lambda(lam, reg_days, dates, mends, horizon=h)
    assert out.shape == (len(mends), 2, L)
    assert np.isnan(out[0]).all()  # nothing is known before the first month end
    np.testing.assert_allclose(out[1], 1.0)  # formation month Feb: only the January-complete regression
    # formation month Mar (cutoff end of Feb): all three regressions ended by then
    assert dates[jan_end + 1 + h - 1] <= mends[1] and feb_end > jan_end + h
    np.testing.assert_allclose(out[2], (1.0 + 1e6 + 3.0) / 3)
    np.testing.assert_allclose(out[3], out[2])


def test_expanding_lambda_no_lookahead_against_brute_force(rng: np.random.Generator) -> None:
    dates, mends = _calendar()
    h, L = config.FORECAST_HORIZON, 4
    reg_days = np.arange(0, len(dates))
    lam = rng.standard_normal((len(reg_days), 3, L))
    lam[5, 0, 1] = np.nan  # an infeasible regression is skipped
    lam[:, 2] = np.nan  # the pooled r~ coefficients are not used
    out = expanding_lambda(lam, reg_days, dates, mends, horizon=h)
    for m in range(1, len(mends)):
        end = reg_days + h - 1
        use = (end < len(dates)) & np.isfinite(lam[:, :2]).all(axis=(1, 2))
        use &= dates[np.minimum(end, len(dates) - 1)] <= mends[m - 1]
        if use.any():
            np.testing.assert_allclose(out[m], lam[use, :2].mean(axis=0))
        else:
            assert np.isnan(out[m]).all()
    # January has 21 business days: only the regression of Jan 1 has its 21-day holding period done by Jan 29
    np.testing.assert_allclose(out[1], lam[0, :2])


# ------------------------------------------------------------------ asres_signal
def test_asres_signal_hand_example() -> None:
    T, N, L = 40, 3, 25  # lags 2..26; the 1-month component covers lags 2..22
    t = 30
    eps = np.zeros((T, N))
    eps[t - 1] = 1.0  # lag 1: skipped by the one-day implementation gap
    eps[t - 27] = 1.0  # lag 27: beyond the lag window
    eps[t - 2, 0] = 0.02  # lag 2, positive
    eps[t - 23, 0] = -0.05  # lag 23, negative, long component
    eps[t - 22, 1] = -0.01  # lag 22, negative, still in the 1-month component
    eps[t - 5, 2] = np.nan  # missing -> 0
    lam = np.zeros((2, 2, L))
    lam[:, 0, :21] = -0.1
    lam[:, 1, :21] = -0.2
    lam[:, :, 21:] = -0.05
    lam[1] = np.nan  # no coefficients available -> NaN
    out = asres_signal(lam, eps, np.array([t, t]))
    mu21 = np.array([-0.1 * 0.02, -0.2 * -0.01, 0.0])
    full = mu21 + np.array([-0.05 * -0.05, 0.0, 0.0])
    np.testing.assert_allclose(out["mu_21"][0], mu21, atol=1e-15)
    np.testing.assert_allclose(out["mu_full"][0], full, atol=1e-15)
    np.testing.assert_allclose(out["ASRES"][0], mu21 + 0.4 * (full - mu21), atol=1e-15)
    np.testing.assert_allclose(out["ASRES"][0], [-0.001, 0.002, 0.0], atol=1e-15)
    assert np.isnan(out["ASRES"][1]).all()


def test_asres_signal_missing_holding_month_and_custom_weight(rng: np.random.Generator) -> None:
    lam = rng.standard_normal((2, 2, 30))
    eps = rng.standard_normal((50, 4)) * 0.01
    out = asres_signal(lam, eps, np.array([-1, 45]), short_lag=10, long_weight=1.0)
    assert np.isnan(out["ASRES"][0]).all()
    np.testing.assert_allclose(out["ASRES"][1], out["mu_full"][1])  # weight 1 -> full-horizon forecast


# ------------------------------------------------------------------ first_day_next_month / size_groups
def test_first_day_next_month() -> None:
    dates, mends = _calendar()
    idx = first_day_next_month(dates, mends)
    assert idx[-1] == -1
    for k in range(len(mends) - 1):
        assert dates[idx[k]] > mends[k] and dates[idx[k] - 1] == mends[k]
        assert dates[idx[k]].month == mends[k + 1].month


def test_size_groups_partition_nyse_percentiles(rng: np.random.Generator) -> None:
    M, N = 2, 400
    me = np.exp(rng.normal(20, 2, (M, N)))
    univ = rng.random((M, N)) < 0.9
    is_nyse = rng.random(N) < 0.4
    out = size_groups(me, univ, is_nyse)
    assert list(out) == list(config.SIZE_GROUPS)
    total = sum(g.astype(int) for g in out.values())
    assert total.max() == 1  # groups are disjoint
    for m in range(M):
        v = univ[m] & is_nyse
        q = {p: np.quantile(me[m, v], p) for p in (0.01, 0.2, 0.5, 0.8)}
        np.testing.assert_array_equal(total[m] == 1, univ[m] & (me[m] > q[0.01]))
        np.testing.assert_array_equal(out["Mega"][m], univ[m] & (me[m] > q[0.8]))
        np.testing.assert_array_equal(out["Small"][m], univ[m] & (me[m] > q[0.2]) & (me[m] <= q[0.5]))
        # NYSE stocks split by the NYSE percentiles: about 30% are Small
        assert out["Small"][m, v].mean() == pytest.approx(0.30, abs=0.02)


def test_size_groups_too_few_nyse_stocks() -> None:
    me = np.arange(1.0, 31.0)[None, :]
    is_nyse = np.zeros(30, bool)
    is_nyse[:19] = True
    out = size_groups(me, np.ones((1, 30), bool), is_nyse)
    assert not any(g.any() for g in out.values())
