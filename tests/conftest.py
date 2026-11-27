"""Shared pytest configuration and small synthetic fixtures (no network, no market data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "market_data: tests that load a reduced market-data slice through data/data_loader.py "
        "(skipped without DATALOADER_API_KEY)",
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator so every synthetic test is deterministic."""
    return np.random.default_rng(12345)


@pytest.fixture
def bdays() -> pd.DatetimeIndex:
    """Business days spanning three calendar months (Jan-Mar 2021)."""
    return pd.bdate_range("2021-01-01", "2021-03-31")
