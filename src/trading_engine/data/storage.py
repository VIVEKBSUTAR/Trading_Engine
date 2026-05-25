"""DuckDB + partitioned Parquet storage integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa

from trading_engine.common.exceptions import StorageError
from trading_engine.common.logging import get_logger
from trading_engine.config.settings import RuntimeSettings


class ParquetDuckDBStore:
    """Persist datasets in Parquet and index them in DuckDB."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._logger = get_logger("ingestion")
        self._con = duckdb.connect(str(settings.duckdb_path))
        self._bootstrap_catalog()

    def close(self) -> None:
        """Close underlying DuckDB connection."""
        self._con.close()

    def _bootstrap_catalog(self) -> None:
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_files (
                dataset VARCHAR NOT NULL,
                feed VARCHAR NOT NULL,
                file_path VARCHAR PRIMARY KEY,
                partition_key VARCHAR,
                min_timestamp TIMESTAMPTZ,
                max_timestamp TIMESTAMPTZ,
                row_count BIGINT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )

        self._con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_dataset_feed_ts
            ON dataset_files(dataset, feed, min_timestamp, max_timestamp)
            """
        )

    def write_dataset(
        self,
        frame: pd.DataFrame,
        *,
        dataset: str,
        feed: str,
        timestamp_col: str,
        mode: str = "append",
    ) -> Path:
        """Write dataframe into a partitioned Parquet dataset."""
        if frame.empty:
            raise StorageError("Cannot persist an empty frame")

        if timestamp_col not in frame.columns:
            raise StorageError(f"Timestamp column '{timestamp_col}' not present")

        working = frame.copy()
        working["feed"] = feed
        working["date"] = pd.to_datetime(working[timestamp_col]).dt.strftime("%Y-%m-%d")

        dataset_root = self._settings.processed_dir / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(working, preserve_index=False)
        format_ = ds.ParquetFileFormat()
        file_options = format_.make_write_options(
            compression=self._settings.app.storage.parquet_compression,
        )

        existing_data_behavior = "overwrite_or_ignore" if mode == "append" else "delete_matching"

        try:
            ds.write_dataset(
                data=table,
                base_dir=str(dataset_root),
                format=format_,
                file_options=file_options,
                partitioning=self._settings.app.storage.partition_columns,
                max_rows_per_group=self._settings.app.storage.parquet_row_group_size,
                existing_data_behavior=existing_data_behavior,
                basename_template="part-{i}.parquet",
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Failed to write dataset '{dataset}': {exc}") from exc

        self._refresh_metadata_index(dataset=dataset, feed=feed, timestamp_col=timestamp_col)
        self._logger.info("Dataset written", dataset=dataset, feed=feed, path=str(dataset_root))
        return dataset_root

    def _refresh_metadata_index(self, *, dataset: str, feed: str, timestamp_col: str) -> None:
        dataset_glob = self._settings.processed_dir / dataset / "**" / "*.parquet"

        self._con.execute(
            "DELETE FROM dataset_files WHERE dataset = ? AND feed = ?",
            [dataset, feed],
        )

        query = f"""
            INSERT INTO dataset_files (
                dataset,
                feed,
                file_path,
                partition_key,
                min_timestamp,
                max_timestamp,
                row_count
            )
            SELECT
                ?,
                ?,
                filename,
                regexp_extract(filename, '.*/(feed=.*?/date=.*?)/.*', 1) AS partition_key,
                MIN({timestamp_col}) AS min_timestamp,
                MAX({timestamp_col}) AS max_timestamp,
                COUNT(*) AS row_count
            FROM read_parquet(?, filename = true)
            WHERE feed = ?
            GROUP BY filename, partition_key
        """

        self._con.execute(query, [dataset, feed, str(dataset_glob), feed])

    def load_dataset(
        self,
        *,
        dataset: str,
        columns: list[str] | None = None,
        where: str | None = None,
    ) -> pd.DataFrame:
        """Load dataset slices through DuckDB SQL pushdown."""
        dataset_glob = self._settings.processed_dir / dataset / "**" / "*.parquet"
        select_clause = ", ".join(columns) if columns else "*"
        where_clause = f"WHERE {where}" if where else ""

        query = f"""
            SELECT {select_clause}
            FROM read_parquet('{dataset_glob}')
            {where_clause}
        """

        return self._con.execute(query).df()

    def query_metadata(self, *, dataset: str | None = None, feed: str | None = None) -> pd.DataFrame:
        """Retrieve dataset file metadata for fast diagnostics."""
        predicates: list[str] = []
        params: list[Any] = []

        if dataset:
            predicates.append("dataset = ?")
            params.append(dataset)
        if feed:
            predicates.append("feed = ?")
            params.append(feed)

        where_clause = f"WHERE {' AND '.join(predicates)}" if predicates else ""

        return self._con.execute(
            f"SELECT * FROM dataset_files {where_clause} ORDER BY created_at DESC",
            params,
        ).df()
