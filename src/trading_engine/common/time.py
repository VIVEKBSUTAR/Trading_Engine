"""Timezone and timestamp helper functions."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


def now_in_timezone(timezone: str) -> datetime:
    """Return current time in the configured timezone."""
    return datetime.now(tz=ZoneInfo(timezone))


def ensure_datetime_series(
    series: pd.Series,
    *,
    target_timezone: str,
    source_timezone: str | None = None,
) -> pd.Series:
    """Normalize a pandas series into timezone-aware datetimes.

    Naive timestamps are localized to source_timezone if provided, otherwise to
    target_timezone. Aware timestamps are converted to target_timezone.
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=False)

    if parsed.dt.tz is None:
        local_tz = source_timezone or target_timezone
        localized = parsed.dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="shift_forward")
        return localized.dt.tz_convert(target_timezone)

    return parsed.dt.tz_convert(target_timezone)
