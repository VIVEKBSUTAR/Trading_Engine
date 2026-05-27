"""Parquet and raw snapshot persistence for NSE option-chain ingestion."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from loguru import logger

from config.settings import StorageSettings


class ParquetWriter:
    """Write raw JSON snapshots and processed partitioned parquet datasets."""

    def __init__(self, settings: StorageSettings) -> None:
        self._settings = settings

    def write_raw_snapshot(self, raw_text: str, fetched_at: datetime, source: str = "nse") -> Path:
        """Persist unmodified raw API response to JSON file by date partitions."""
        year = fetched_at.strftime("%Y")
        month = fetched_at.strftime("%m")
        day = fetched_at.strftime("%d")

        dest_dir = self._settings.raw_dir / source / f"year={year}" / f"month={month}" / f"day={day}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        file_path = dest_dir / f"snapshot_{fetched_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        file_path.write_text(raw_text, encoding="utf-8")
        logger.info("Raw snapshot persisted", path=str(file_path))
        return file_path

    def write_processed(self, frame: pd.DataFrame, dataset_name: str) -> Path:
        """Persist cleaned option-chain records as partitioned parquet dataset."""
        if frame.empty:
            raise ValueError("Cannot write empty processed frame")

        working = frame.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
        working["year"] = working["timestamp"].dt.strftime("%Y")
        working["month"] = working["timestamp"].dt.strftime("%m")
        working["day"] = working["timestamp"].dt.strftime("%d")

        dataset_root = self._settings.processed_dir / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(working, preserve_index=False)
        ds.write_dataset(
            data=table,
            base_dir=str(dataset_root),
            format="parquet",
            partitioning=["year", "month", "day"],
            existing_data_behavior="overwrite_or_ignore",
            basename_template="part-{i}.parquet",
        )

        logger.info("Processed parquet write complete", dataset=str(dataset_root), rows=len(working))
        return dataset_root

    def write_metadata_blob(self, payload: dict, file_name: str) -> Path:
        """Persist metadata JSON blobs for diagnostics and audit trails."""
        self._settings.metadata_dir.mkdir(parents=True, exist_ok=True)
        path = self._settings.metadata_dir / file_name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
