"""Unit tests for src.characteristics: rank standardization and inclusion filters (Section 3)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import config
from src.characteristics import (_rolling_beta_ivol, monthly_block_stat, rank_standardize,
                                 select_characteristics, top_k_mean)


# ------------------------------------------------------------------ rank_standardize
def test_rank_standardize_hand_example_with_missing_and_outside() -> None:
    c = np.array([3.0, 1.0, 2.0, np.nan, 5.0])
    univ = np.array([True, True, True, True, False])
    # valid n = 3: ranks 3,1,2 -> rC = r/4 - 0.5 = 0.25,-0.25,0 ; missing -> 0 ; mean|rC| = 0.5/4
    x = rank_standardize(c, univ)
    np.testing.assert_allclose(x[:4], [2.0, -2.0, 0.0, 0.0])
    assert np.isnan(x[4])


def test_rank_standardize_average_ranks_for_ties() -> None:
    x = rank_standardize(np.array([1.0, 1.0, 2.0, 3.0]), np.ones(4, dtype=bool))
    # ranks 1.5,1.5,3,4 -> rC = -0.2,-0.2,0.1,0.3 ; mean|rC| = 0.2
    np.testing.assert_allclose(x, [-1.0, -1.0, 0.5, 1.5])


def test_rank_standardize_unit_mean_abs_and_monotone(rng: np.random.Generator) -> None:
    N = 200
    c = rng.standard_normal(N) ** 3
    c[rng.random(N) < 0.2] = np.nan
    univ = rng.random(N) < 0.8
    x = rank_standardize(c, univ)
    assert np.isnan(x[~univ]).all()
    assert np.isfinite(x[univ]).all()
    np.testing.assert_allclose(np.abs(x[univ]).mean(), 1.0)
    v = univ & np.isfinite(c)
    order = np.argsort(c[v])
    assert (np.diff(x[v][order]) > 0).all()  # strictly monotone without ties
    assert (x[univ & ~np.isfinite(c)] == 0).all()
    # invariant to monotone transformations of the input
    np.testing.assert_allclose(rank_standardize(np.exp(c), univ)[univ], x[univ])


def test_rank_standardize_symmetric_around_zero(rng: np.random.Generator) -> None:
    x = rank_standardize(rng.standard_normal(51), np.ones(51, dtype=bool))
    np.testing.assert_allclose(x.sum(), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sort(x), -np.sort(x)[::-1])


def test_rank_standardize_degenerate_inputs() -> None:
    univ = np.array([True, True, False])
    all_nan = rank_standardize(np.full(3, np.nan), univ)
    np.testing.assert_array_equal(all_nan[:2], 0.0)
    assert np.isnan(all_nan[2])
    single = rank_standardize(np.array([5.0, np.nan, 1.0]), univ)  # one valid stock -> rC = 0.5 - 0.5
    np.testing.assert_array_equal(single[:2], 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # mean of an empty universe
        empty = rank_standardize(np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=bool))
    assert np.isnan(empty).all()


# ------------------------------------------------------------------ select_characteristics
def _corr_pair(rng: np.random.Generator, n: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Two vectors with sample correlation exactly ``rho``."""
    a = rng.standard_normal(n)
    a -= a.mean()
    z = rng.standard_normal(n)
    z -= z.mean()
    z -= (z @ a) / (a @ a) * a
    a_u, z_u = a / np.linalg.norm(a), z / np.linalg.norm(z)
    return a_u, rho * a_u + np.sqrt(1 - rho ** 2) * z_u


def _run_select(cov: dict[str, list[float]], chars: dict[str, np.ndarray]) -> pd.DataFrame:
    M = len(next(iter(cov.values())))
    months = pd.date_range("2020-01-31", periods=M, freq="ME")
    coverage = pd.DataFrame(cov, index=months)
    N = next(iter(chars.values())).shape[1]
    return select_characteristics(coverage, chars, months, np.ones((M, N), dtype=bool))


def test_select_coverage_hysteresis_40_in_20_out(rng: np.random.Generator) -> None:
    cov = [0.30, 0.45, 0.30, 0.20, 0.19, 0.30, 0.40]
    chars = {"a": rng.standard_normal((len(cov), 20))}
    incl = _run_select({"a": cov}, chars)
    assert incl["a"].tolist() == [False, True, True, True, False, False, True]
    assert config.CHAR_MIN_COVERAGE_IN == 0.40 and config.CHAR_MIN_COVERAGE_OUT == 0.20


def test_select_drops_shorter_tenor_when_corr_at_least_99(rng: np.random.Generator) -> None:
    n, M = 60, 5
    rhos = [0.5, 0.995, 0.98, 0.97, 0.995]
    a = np.zeros((M, n))
    b = np.zeros((M, n))
    for m, rho in enumerate(rhos):
        a[m], b[m] = _corr_pair(rng, n, rho)
    cov = {"a": [0.5] * M, "b": [0.1, 0.5, 0.5, 0.5, 0.5]}
    incl = _run_select(cov, {"a": a, "b": b})
    assert incl["a"].all()  # longer tenor always kept
    # m0: b lacks coverage; m1: corr 99.5% -> b (shorter tenor) dropped;
    # m2: 98% > 97.5% -> stays out; m3: 97% <= 97.5% -> reintroduced; m4: 99.5% -> dropped again
    assert incl["b"].tolist() == [False, False, False, True, False]


def test_select_keeps_both_when_corr_between_thresholds_without_prior_drop(rng: np.random.Generator) -> None:
    a, b = _corr_pair(rng, 60, 0.98)
    incl = _run_select({"a": [0.5, 0.5], "b": [0.5, 0.5]}, {"a": np.vstack([a, a]), "b": np.vstack([b, b])})
    assert incl.to_numpy().all()


def test_select_negative_correlation_also_drops(rng: np.random.Generator) -> None:
    a, b = _corr_pair(rng, 60, -0.995)
    incl = _run_select({"a": [0.5, 0.5], "b": [0.1, 0.5]}, {"a": np.vstack([a, a]), "b": np.vstack([b, b])})
    assert incl["a"].tolist() == [True, True] and incl["b"].tolist() == [False, False]


def test_select_tenor_decides_which_one_is_dropped(rng: np.random.Generator) -> None:
    """The characteristic included for more months survives, regardless of column order."""
    a, b = _corr_pair(rng, 60, 0.999)
    cov = {"a": [0.1, 0.1, 0.5], "b": [0.5, 0.5, 0.5]}
    incl = _run_select(cov, {"a": np.vstack([a] * 3), "b": np.vstack([b] * 3)})
    assert incl["b"].all()
    assert not incl["a"].any()


def test_select_empty_universe_month_excludes_all(rng: np.random.Generator) -> None:
    months = pd.date_range("2020-01-31", periods=2, freq="ME")
    coverage = pd.DataFrame({"a": [0.5, 0.5]}, index=months)
    univ = np.array([[False] * 10, [True] * 10])
    incl = select_characteristics(coverage, {"a": rng.standard_normal((2, 10))}, months, univ)
    assert incl["a"].tolist() == [False, True]


# ------------------------------------------------------------------ helpers of the price characteristics
def test_top_k_mean_ignores_nan_and_handles_short_columns() -> None:
    block = np.array([[1.0, np.nan], [5.0, 2.0], [3.0, np.nan], [4.0, np.nan]])
    np.testing.assert_allclose(top_k_mean(block, k=2), [4.5, 2.0])
    np.testing.assert_allclose(top_k_mean(block, k=10), [3.25, 2.0])


def test_monthly_block_stat_applies_min_obs(bdays: pd.DatetimeIndex) -> None:
    df = pd.DataFrame({"x": np.arange(len(bdays), dtype=float), "y": np.nan}, index=bdays)
    df.loc[bdays[:3], "y"] = 1.0
    out = monthly_block_stat(df, lambda b: np.nansum(b, axis=0), min_obs=3)
    assert len(out) == 3 and out.index[0] == bdays[bdays.month == 1].max()
    jan = bdays.month == 1
    assert out["x"].iloc[0] == df["x"][jan].sum()
    assert out["y"].iloc[0] == 3.0 and out["y"].iloc[1:].isna().all()


def test_rolling_beta_ivol_exact_capm(rng: np.random.Generator) -> None:
    T = 60
    idx = pd.bdate_range("2021-01-01", periods=T)
    mkt = pd.Series(rng.standard_normal(T) * 0.01, index=idx)
    resid = rng.standard_normal(T) * 0.02
    exret = pd.DataFrame({"a": 0.0005 + 1.5 * mkt.to_numpy(), "b": -0.5 * mkt.to_numpy() + resid}, index=idx)
    beta, ivol = _rolling_beta_ivol(exret, mkt, window=T, min_periods=T)
    assert np.isnan(beta.iloc[-2, 0])
    np.testing.assert_allclose(beta.iloc[-1, 0], 1.5, rtol=1e-6)
    assert ivol.iloc[-1, 0] < 1e-6
    m = mkt.to_numpy()
    b_ols = np.cov(exret["b"], m, ddof=0)[0, 1] / m.var()
    np.testing.assert_allclose(beta.iloc[-1, 1], b_ols, rtol=1e-6)
    e = exret["b"].to_numpy() - b_ols * m
    np.testing.assert_allclose(ivol.iloc[-1, 1], e.std(), rtol=1e-5)
