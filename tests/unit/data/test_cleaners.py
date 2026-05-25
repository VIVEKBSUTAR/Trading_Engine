from __future__ import annotations

import pandas as pd

from trading_engine.data.cleaners import DataCleaner


def test_normalize_timestamps_localizes_to_kolkata() -> None:
    cleaner = DataCleaner()
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 09:15:00", "2026-01-01 09:16:00"],
            "close": [100.0, 101.0],
        }
    )

    output = cleaner.normalize_timestamps(
        frame,
        timestamp_col="timestamp",
        target_timezone="Asia/Kolkata",
    )

    assert str(output["timestamp"].dt.tz) == "Asia/Kolkata"


def test_fill_missing_timestamps_reindexes_range() -> None:
    cleaner = DataCleaner()
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 09:15:00+05:30", "2026-01-01 09:17:00+05:30"]
            ),
            "close": [100.0, 102.0],
        }
    )

    output = cleaner.fill_missing_timestamps(
        frame,
        timestamp_col="timestamp",
        frequency="1min",
        fill_value=None,
    )

    assert len(output) == 3
