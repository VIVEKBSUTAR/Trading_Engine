"""End-to-end ingestion orchestration for a single feed."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_engine.common.exceptions import DataValidationError
from trading_engine.config.settings import RuntimeSettings
from trading_engine.data.cleaners import DataCleaner
from trading_engine.data.storage import ParquetDuckDBStore
from trading_engine.data.validators import DataValidator


@dataclass(slots=True)
class IngestionResult:
    """Output of one ingestion pipeline run."""

    cleaned_frame: pd.DataFrame
    dataset_path: str


class IngestionPipeline:
    """Composable pipeline for validation-cleaning-persistence."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._validator = DataValidator()
        self._cleaner = DataCleaner()
        self._store = ParquetDuckDBStore(settings)

    def run(
        self,
        frame: pd.DataFrame,
        *,
        dataset: str,
        feed: str,
        timestamp_col: str,
        required_columns: set[str],
        numeric_columns: set[str],
        frequency: str,
    ) -> IngestionResult:
        """Validate, clean, and persist a feed dataframe."""
        report = self._validator.validate(
            frame,
            required_columns=required_columns,
            timestamp_col=timestamp_col,
            numeric_columns=numeric_columns,
        )

        if report.corruption_ratio > self._settings.app.ingestion.corruption_error_threshold:
            raise DataValidationError(
                "Corruption ratio exceeds threshold: "
                f"{report.corruption_ratio:.4f} > {self._settings.app.ingestion.corruption_error_threshold:.4f}"
            )

        cleaned = self._validator.drop_corrupted_rows(frame, report, timestamp_col=timestamp_col)
        cleaned = self._cleaner.normalize_timestamps(
            cleaned,
            timestamp_col=timestamp_col,
            target_timezone=self._settings.app.ingestion.default_timezone,
        )
        cleaned = self._cleaner.drop_duplicate_rows(
            cleaned,
            subset=[timestamp_col],
            keep=self._settings.app.ingestion.drop_duplicates_keep,
        )
        cleaned = self._cleaner.fill_missing_timestamps(
            cleaned,
            timestamp_col=timestamp_col,
            frequency=frequency,
            fill_value=self._settings.app.ingestion.missing_timestamp_fill,
        )

        dataset_path = self._store.write_dataset(
            cleaned,
            dataset=dataset,
            feed=feed,
            timestamp_col=timestamp_col,
            mode="append",
        )

        return IngestionResult(cleaned_frame=cleaned, dataset_path=str(dataset_path))
