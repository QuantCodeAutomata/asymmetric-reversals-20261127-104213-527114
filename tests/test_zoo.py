"""Unit tests for src.zoo: short-horizon anomaly characteristics and the mispricing score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew

from src import config
from src.signals import month_blocks
from src.zoo import (MISP_ANOMALIES, MISP_MIN_ANOMALIES, ZOO_NAMES, _skew, corwin_schultz, coskew, misp_score,
                     ts_residuals, zoo_characteristics)


# ------------------------------------------------------------------ corwin_schultz
def test_corwin_schultz_zero_when_high_equals_low() -> None:
    p = np.full((5, 2), 10.0)
    s = corwin_schultz(p, p)
    np.testing.assert_allclose(s[:-1], 0.0, atol=1e-12)
    assert np.isnan(s[-1]).all()


def test_corwin_schultz_closed_form_constant_range() -> None:
    """Equal one-day ranges H/L = 1.1 on both days give alpha = log 1.1 and S = 2 * 0.1 / 2.1."""
    high = np.array([[11.0], [11.0]])
    low = np.array([[10.0], [10.0]])
    s = corwin_schultz(high, low)
    assert s[0, 0] == pytest.approx(2 * 0.1 / 2.1, rel=1e-10)


def test_corwin_schultz_nonnegative_and_invalid_prices(rng: np.random.Generator) -> None:
    mid = 50 * np.exp(np.cumsum(rng.standard_normal((60, 4)) * 0.02, axis=0))
    high = mid * (1 + rng.uniform(0, 0.03, mid.shape))
    low = mid * (1 - rng.uniform(0, 0.03, mid.shape))
    high[10, 0] = 0.0  # invalid price -> NaN for the two pairs that use it
    s = corwin_schultz(high, low)
    finite = np.isfinite(s)
    assert (s[finite] >= 0).all()
    assert (s[finite] > 0).any()
    assert np.isnan(s[[9, 10], 0]).all() and np.isfinite(s[[8, 11], 0]).all()


# ------------------------------------------------------------------ skewness and coskewness
def test_skew_matches_scipy_and_ignores_nan(rng: np.random.Generator) -> None:
    b = rng.exponential(1.0, (40, 3))
    b[[3, 7], 1] = np.nan
    out = _skew(b)
    assert out[0] == pytest.approx(skew(b[:, 0]))
    assert out[1] == pytest.approx(skew(b[np.isfinite(b[:, 1]), 1]))
    assert (out > 0).all()  # exponential data are right-skewed
    sym = np.r_[-np.arange(1.0, 6.0), np.arange(1.0, 6.0)][:, None]
    assert _skew(sym)[0] == pytest.approx(0.0, abs=1e-12)


def test_coskew_hand_formula_sign_and_scale(rng: np.random.Generator) -> None:
    m = rng.standard_normal(50)
    dm2 = (m - m.mean()) ** 2
    e = np.column_stack([dm2 - dm2.mean(), -(dm2 - dm2.mean()), rng.standard_normal(50)])
    out = coskew(e, m)
    assert out[0] > 0 and out[1] < 0
    assert out[0] == pytest.approx(-out[1])
    j = 2
    expected = (e[:, j] * dm2).mean() / (np.sqrt((e[:, j] ** 2).mean()) * dm2.mean())
    assert out[j] == pytest.approx(expected)
    np.testing.assert_allclose(coskew(5.0 * e, m), out)  # invariant to the scale of the residual


def test_coskew_uses_stock_days_and_requires_three() -> None:
    m = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    e = np.array([[0.1, np.nan], [0.2, np.nan], [-0.1, 0.1], [0.0, np.nan], [0.3, 0.2]])
    out = coskew(e, m)
    assert np.isfinite(out[0]) and np.isnan(out[1])


# ------------------------------------------------------------------ ts_residuals
def test_ts_residuals_orthogonal_to_factors_and_missing_handling(rng: np.random.Generator) -> None:
    D = 25
    X = rng.standard_normal((D, 2))
    Y = 0.001 + X @ np.array([[1.0, 0.5, 0.2], [-0.3, 0.0, 1.0]]) + rng.standard_normal((D, 3)) * 0.01
    Y[[1, 4], 1] = np.nan
    Y[5:, 2] = np.nan  # only 5 days < min_obs
    out = ts_residuals(Y, X, min_obs=10)
    F = np.column_stack([np.ones(D), X])
    np.testing.assert_allclose(F.T @ out[:, 0], 0.0, atol=1e-12)
    v = np.isfinite(Y[:, 1])
    np.testing.assert_allclose(F[v].T @ out[v, 1], 0.0, atol=1e-12)
    assert np.isnan(out[~v, 1]).all()
    assert np.isnan(out[:, 2]).all()
    exact = ts_residuals(F[:, 1:] @ np.array([[2.0], [3.0]]) + 0.5, X, min_obs=10)
    np.testing.assert_allclose(exact, 0.0, atol=1e-12)


# ------------------------------------------------------------------ zoo_characteristics
@pytest.mark.filterwarnings("ignore::RuntimeWarning", "ignore:After omitting NaNs")
def test_zoo_characteristics_block_values(rng: np.random.Generator) -> None:
    dates = pd.bdate_range("2021-01-01", "2021-02-26")
    T, N = len(dates), 3
    ret = rng.standard_normal((T, N)) * 0.02
    ff3 = rng.standard_normal((T, 3)) * 0.01
    exret = ret - 0.0001
    close = 20 + np.abs(rng.standard_normal((T, N)))
    high, low = close * 1.01, close * 0.99
    volume = rng.integers(1000, 5000, (T, N)).astype(float)
    volume[3, 0] = 0.0
    shares = np.full((T, N), 1e6)
    ret[5:, 2] = np.nan  # too few observations for stock 2
    exret[5:, 2] = np.nan
    blocks = month_blocks(dates, skip_last=True)
    out = zoo_characteristics(ret, exret, high, low, close, volume, shares, ff3, blocks)
    assert set(out) == set(ZOO_NAMES)
    a, b = blocks[0]
    assert all(np.isnan(out[k][:, 2]).all() for k in ZOO_NAMES)
    np.testing.assert_allclose(out["LagRet"][0, :2], exret[a:b, :2].sum(axis=0))
    np.testing.assert_allclose(out["RetVol"][0, :2], exret[a:b, :2].std(axis=0, ddof=1))
    np.testing.assert_allclose(out["MAX"][0, :2], ret[a:b, :2].max(axis=0))
    np.testing.assert_allclose(out["MAX5"][0, :2], np.sort(ret[a:b, :2], axis=0)[-5:].mean(axis=0))
    np.testing.assert_allclose(out["Turnover"][0, :2], (volume[a:b, :2] / 1e6).mean(axis=0))
    e3 = ts_residuals(exret[a:b], ff3[a:b], config.MIN_DAILY_OBS)
    np.testing.assert_allclose(out["IVOL_FF3"][0, :2], e3[:, :2].std(axis=0, ddof=1))
    assert (out["Spread"][0, :2] > 0).all()
    ami = np.abs(ret[a:b, :2]) / (close[a:b, :2] * volume[a:b, :2])
    np.testing.assert_allclose(out["Amihud"][0, 1], ami[:, 1].mean() * config.AMIHUD_SCALE)
    np.testing.assert_allclose(out["Amihud"][0, 0], np.delete(ami[:, 0], 3).mean() * config.AMIHUD_SCALE)
    zt = (volume[a:b, 0] == 0).sum() + (1 / (volume[a:b, 0] / 1e6).sum()) / config.ZERO_TRADES_DEFLATOR
    assert out["ZeroTrades"][0, 0] == pytest.approx(zt)
    assert np.isnan(out["SMAX"]).all()  # fewer than 120 days for the 252-day volatility


# ------------------------------------------------------------------ misp_score
def test_misp_score_average_percentile_rank() -> None:
    M, N = 2, 12
    months = pd.date_range("2020-01-31", periods=M, freq="ME")
    secs = pd.Index([f"s{i}" for i in range(N)])
    base = np.tile(np.arange(1.0, N + 1), (M, 1))
    raw = {k: pd.DataFrame(sign * base, index=months, columns=secs) for k, sign in MISP_ANOMALIES.items()}
    univ = np.ones((M, N), bool)
    univ[:, 0] = False
    names = list(MISP_ANOMALIES)
    for k in names[: len(names) - MISP_MIN_ANOMALIES + 1]:
        raw[k].loc[:, "s1"] = np.nan  # leaves MISP_MIN_ANOMALIES - 1 anomalies for s1
    out = misp_score(raw, univ, months, secs)
    assert np.isnan(out[:, :2]).all()
    # every anomaly ranks s2..s11 identically (higher = more overpriced); s1 is missing in some
    np.testing.assert_allclose(out[0, -1], 100.0)
    assert (np.diff(out[0, 2:]) > 0).all()
