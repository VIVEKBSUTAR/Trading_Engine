from __future__ import annotations

import pandas as pd

from trading_engine.data.validators import DataValidator


def test_validator_detects_invalid_timestamp_and_non_numeric() -> None:
    validator = DataValidator()
    frame = pd.DataFrame(
        {
            "timestamp": ["invalid", "2026-01-01 09:15:00"],
            "close": [100.0, "bad"],
        }
    )

    report = validator.validate(
        frame,
        required_columns={"timestamp", "close"},
        timestamp_col="timestamp",
        numeric_columns={"close"},
    )

    assert report.is_valid is False
    assert report.rows_corrupted >= 1
    assert any(item.code == "invalid_timestamp" for item in report.issues)
