"""Unit tests for src.signals: active days, RES/SYS split, concentration and IVOL (Sections 4-5)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.signals import block_apply, compound, ivol, ivol_horizon, month_blocks, monthly_sum, res_sys


# ------------------------------------------------------------------ month_blocks
def test_month_blocks_skip_last_excludes_last_trading_day(bdays: pd.DatetimeIndex) -> None:
    full = month_blocks(bdays, skip_last=False)
    skip = month_blocks(bdays, skip_last=True)
    assert len(full) == len(skip) == 3
    assert full[0][0] == 0 and full[-1][1] == len(bdays)
    for (a, b), (a2, b2) in zip(full[:-1], full[1:]):
        assert b == a2  # contiguous, non-overlapping
    for (a, b), (sa, sb) in zip(full, skip):
        assert sa == a and sb == b - 1
        month = bdays[a].month
        assert (bdays[a:b].month == month).all()
        assert bdays[sb] == bdays[bdays.month == month].max()  # the excluded day is the month's last day
        assert bdays[sb - 1] < bdays[sb]


def test_month_blocks_single_day_month() -> None:
    dates = pd.DatetimeIndex(["2021-01-29", "2021-02-01", "2021-02-02"])
    assert month_blocks(dates, skip_last=True) == [(0, 0), (1, 2)]
    assert month_blocks(dates, skip_last=False) == [(0, 1), (1, 3)]


# ------------------------------------------------------------------ block_apply
def test_block_apply_min_obs_and_valid_mask() -> None:
    daily = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, np.nan]])
    out = block_apply(daily, [(0, 3)], lambda b: np.nansum(b, axis=0), min_obs=2)
    np.testing.assert_array_equal(out, [[7.0, np.nan]])
    valid = np.array([[True, True], [False, True], [False, True]])
    out_v = block_apply(daily, [(0, 3)], lambda b: np.nansum(b, axis=0), min_obs=2, valid=valid)
    assert np.isnan(out_v).all()


# ------------------------------------------------------------------ res_sys
def _one_block_panel() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    eps = np.array([
        [0.01, 0.02, np.nan, -0.01],
        [-0.02, 0.00, np.nan, -0.03],
        [0.03, 0.00, np.nan, 0.00],
        [np.nan, 0.00, np.nan, 0.00],
        [9.00, 9.00, 9.00, 9.00],  # last trading day of the month: skipped
    ])
    r = np.array([[0.001, 0.002, 0.003, 0.004]] * 5)
    blocks = [(0, 4)]
    return eps, r, blocks


def test_res_sys_hand_example() -> None:
    eps, r, blocks = _one_block_panel()
    out = res_sys(eps, r, blocks)
    np.testing.assert_allclose(out["RES"][0, [0, 1, 3]], [0.02, 0.02, -0.04])
    np.testing.assert_allclose(out["RES_pos"][0, [0, 1, 3]], [0.04, 0.02, 0.0])
    np.testing.assert_allclose(out["RES_neg"][0, [0, 1, 3]], [-0.02, 0.0, -0.04])
    np.testing.assert_allclose(out["SYS"][0], [0.004, 0.008, 0.012, 0.016])
    # CONC+ = sqrt(sum pos^2) / RES+
    np.testing.assert_allclose(out["CONC_pos"][0, 0], np.sqrt(0.01 ** 2 + 0.03 ** 2) / 0.04)
    assert out["CONC_pos"][0, 1] == pytest.approx(1.0)  # one positive day: fully concentrated
    assert out["CONC_pos"][0, 3] == 0.0  # RES+ = 0 -> 0
    assert out["CONC_neg"][0, 1] == 0.0  # RES- = 0 -> 0
    # CONC- = sqrt(sum neg^2) / RES- carries the sign of RES- (single negative day -> -1)
    assert out["CONC_neg"][0, 0] == pytest.approx(-1.0)
    np.testing.assert_allclose(out["CONC_neg"][0, 3], np.sqrt(0.01 ** 2 + 0.03 ** 2) / -0.04)
    np.testing.assert_allclose(out["CONC"], out["CONC_pos"] + out["CONC_neg"])
    for k in ("RES", "RES_pos", "RES_neg", "CONC_pos", "CONC_neg", "CONC"):
        assert np.isnan(out[k][0, 2]), k  # no valid residual in the block


def test_res_sys_identities_on_random_panel(rng: np.random.Generator) -> None:
    T, N = 63, 40
    eps = rng.standard_normal((T, N)) * 0.02
    eps[rng.random((T, N)) < 0.1] = np.nan
    r = rng.standard_normal((T, N)) * 0.01
    blocks = [(0, 20), (21, 41), (42, 62)]
    out = res_sys(eps, r, blocks)
    np.testing.assert_allclose(out["RES"], out["RES_pos"] + out["RES_neg"], atol=1e-12)
    assert (out["RES_pos"] >= 0).all() and (out["RES_neg"] <= 0).all()
    assert ((out["CONC_pos"] > 0) & (out["CONC_pos"] <= 1 + 1e-12)).all()
    assert ((out["CONC_neg"] < 0) & (out["CONC_neg"] >= -1 - 1e-12)).all()
    for k, (a, b) in enumerate(blocks):
        np.testing.assert_allclose(out["RES"][k], np.nansum(eps[a:b], axis=0))
        np.testing.assert_allclose(out["SYS"][k], r[a:b].sum(axis=0))
        n_pos = (eps[a:b] > 0).sum(axis=0)
        assert (out["CONC_pos"][k] >= 1 / np.sqrt(n_pos) - 1e-12).all()  # bounded by equal spreading


def test_res_sys_concentration_ordering() -> None:
    """The same RES+ spread over more days is less concentrated."""
    eps = np.zeros((5, 2))
    eps[0, 0] = 0.04
    eps[:4, 1] = 0.01
    out = res_sys(eps, np.zeros((5, 2)), [(0, 4)])
    np.testing.assert_allclose(out["RES_pos"][0], [0.04, 0.04])
    np.testing.assert_allclose(out["CONC_pos"][0], [1.0, 0.5])


# ------------------------------------------------------------------ ivol and aggregation
@pytest.mark.filterwarnings("ignore:Degrees of freedom:RuntimeWarning")
def test_ivol_equals_sample_std_over_active_days(rng: np.random.Generator) -> None:
    eps = rng.standard_normal((21, 3))
    eps[20] = 100.0  # last day of the month, outside the active block
    eps[[2, 5], 1] = np.nan
    eps[1:, 2] = np.nan
    out = ivol(eps, [(0, 20)])
    np.testing.assert_allclose(out[0, 0], np.std(eps[:20, 0], ddof=1))
    v = eps[:20, 1]
    np.testing.assert_allclose(out[0, 1], np.std(v[np.isfinite(v)], ddof=1))
    assert np.isnan(out[0, 2])  # a single observation is below min_obs = 2


def test_ivol_horizon_full_window_equals_ivol(rng: np.random.Generator) -> None:
    eps = rng.standard_normal((20, 4))
    blocks = [(0, 10), (10, 19)]
    np.testing.assert_allclose(ivol_horizon(eps, blocks, H=10), ivol(eps, blocks))
    h1 = ivol_horizon(eps, blocks, H=1)
    b = eps[0:10]
    np.testing.assert_allclose(h1[0], np.abs(b[-1] - b.mean(axis=0)) / 3.0)  # sqrt(n - 1) = 3


def test_monthly_sum_and_compound(rng: np.random.Generator) -> None:
    ret = rng.uniform(-0.05, 0.05, (10, 2))
    ret[3, 1] = np.nan
    blocks = [(0, 5), (5, 10)]
    np.testing.assert_allclose(monthly_sum(ret, blocks), [np.nansum(ret[:5], 0), np.nansum(ret[5:], 0)])
    np.testing.assert_allclose(compound(ret, blocks)[0, 0], np.prod(1 + ret[:5, 0]) - 1)
    np.testing.assert_allclose(compound(ret, blocks)[0, 1], np.prod(1 + np.delete(ret[:5, 1], 3)) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert np.isnan(monthly_sum(np.full((4, 1), np.nan), [(0, 4)])).all()
