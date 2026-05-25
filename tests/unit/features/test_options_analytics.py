from __future__ import annotations

import pandas as pd

from trading_engine.features.options_analytics import (
    compute_max_pain,
    compute_oi_imbalance,
    compute_put_call_ratio,
    compute_realized_volatility,
)


def _sample_chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:15:00"] * 4),
            "expiry": ["2026-01-29"] * 4,
            "strike": [23000, 23100, 23000, 23100],
            "option_type": ["CE", "CE", "PE", "PE"],
            "oi": [1000, 800, 1200, 900],
            "iv": [0.15, 0.16, 0.18, 0.19],
        }
    )


def test_put_call_ratio_and_oi_imbalance() -> None:
    chain = _sample_chain()

    pcr = compute_put_call_ratio(chain)
    imbalance = compute_oi_imbalance(chain)

    assert len(pcr) == 1
    assert len(imbalance) == 1
    assert pcr["put_call_ratio"].iloc[0] > 0


def test_max_pain_output_shape() -> None:
    chain = _sample_chain()
    result = compute_max_pain(chain)

    assert {"timestamp", "expiry", "max_pain", "max_pain_total_payout"}.issubset(result.columns)
    assert len(result) == 1


def test_realized_volatility_returns_series() -> None:
    close = pd.Series([100, 101, 100, 102, 103, 104], dtype=float)
    rv = compute_realized_volatility(close, window=3)

    assert isinstance(rv, pd.Series)
    assert rv.isna().sum() >= 2
