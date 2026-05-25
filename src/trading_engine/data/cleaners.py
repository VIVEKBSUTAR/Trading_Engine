"""Cleaning and normalization utilities for feed data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_engine.common.time import ensure_datetime_series


class DataCleaner:
    """Normalize feed data to canonical timestamped frames."""

    def normalize_timestamps(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_col: str,
        target_timezone: str,
        source_timezone: str | None = None,
    ) -> pd.DataFrame:
        """Normalize timestamp column and sort ascending."""
        output = frame.copy()
        output[timestamp_col] = ensure_datetime_series(
            output[timestamp_col],
            target_timezone=target_timezone,
            source_timezone=source_timezone,
        )
        output = output.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
        return output.reset_index(drop=True)

    def drop_duplicate_rows(
        self,
        frame: pd.DataFrame,
        *,
        subset: list[str] | None = None,
        keep: str | bool = "last",
    ) -> pd.DataFrame:
        """Drop duplicated rows while preserving deterministic ordering."""
        return frame.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    def fill_missing_timestamps(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_col: str,
        frequency: str,
        fill_value: float | None,
    ) -> pd.DataFrame:
        """Insert missing timestamps over a regular frequency grid."""
        output = frame.copy().sort_values(timestamp_col)
        output = output.set_index(timestamp_col)

        if output.empty:
            return output.reset_index()

        complete_index = pd.date_range(
            start=output.index.min(),
            end=output.index.max(),
            freq=frequency,
            tz=output.index.tz,
        )
        output = output.reindex(complete_index)

        if fill_value is not None:
            output = output.fillna(fill_value)

        output.index.name = timestamp_col
        return output.reset_index()

    def resample(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_col: str,
        frequency: str,
        aggregation: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Resample a frame into minute/daily bars or custom frequencies."""
        if frame.empty:
            return frame.copy()

        output = frame.copy().set_index(timestamp_col)

        if aggregation is None:
            aggregation = self._default_aggregation(output.columns.tolist())

        resampled = output.resample(frequency).agg(aggregation)
        resampled = resampled.dropna(how="all")
        resampled.index.name = timestamp_col
        return resampled.reset_index()

    @staticmethod
    def _default_aggregation(columns: list[str]) -> dict[str, str]:
        aggregation: dict[str, str] = {}
        for column in columns:
            lower = column.lower()
            if lower.endswith("open") or lower == "open":
                aggregation[column] = "first"
            elif lower.endswith("high") or lower == "high":
                aggregation[column] = "max"
            elif lower.endswith("low") or lower == "low":
                aggregation[column] = "min"
            elif lower.endswith("close") or lower == "close":
                aggregation[column] = "last"
            elif "volume" in lower or lower in {"oi", "open_interest"}:
                aggregation[column] = "sum"
            else:
                aggregation[column] = "last"
        return aggregation
