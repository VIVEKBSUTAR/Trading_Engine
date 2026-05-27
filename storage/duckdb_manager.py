"""DuckDB metadata and analytics interface for NSE ingestion datasets."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from loguru import logger


class DuckDBManager:
    """Manage metadata tables and analytical queries over parquet datasets."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._con = duckdb.connect(str(db_path))
        self._bootstrap()

    def _bootstrap(self) -> None:
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_snapshots (
                snapshot_id BIGINT PRIMARY KEY,
                snapshot_ts TIMESTAMPTZ NOT NULL,
                source VARCHAR NOT NULL,
                raw_path VARCHAR NOT NULL,
                processed_dataset VARCHAR,
                rows_ingested BIGINT,
                min_timestamp TIMESTAMPTZ,
                max_timestamp TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )

        self._con.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS ingestion_snapshot_seq START 1
            """
        )

        self._con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_snapshot_ts
            ON ingestion_snapshots(snapshot_ts)
            """
        )

    def register_snapshot(
        self,
        *,
        source: str,
        raw_path: Path,
        processed_dataset: str | None,
        rows_ingested: int,
        min_timestamp: datetime | None,
        max_timestamp: datetime | None,
        snapshot_ts: datetime | None = None,
    ) -> None:
        """Register one ingestion run in metadata table."""
        snap_ts = snapshot_ts or datetime.now(UTC)
        self._con.execute(
            """
            INSERT INTO ingestion_snapshots (
                snapshot_id,
                snapshot_ts,
                source,
                raw_path,
                processed_dataset,
                rows_ingested,
                min_timestamp,
                max_timestamp
            )
            VALUES (
                nextval('ingestion_snapshot_seq'),
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                snap_ts,
                source,
                str(raw_path),
                processed_dataset,
                rows_ingested,
                min_timestamp,
                max_timestamp,
            ],
        )
        logger.info("Registered ingestion snapshot", raw_path=str(raw_path), rows=rows_ingested)

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Run ad-hoc analytical SQL query and return DataFrame."""
        return self._con.execute(sql, params or []).df()

    def query_processed_dataset(self, dataset_root: Path, where_clause: str | None = None) -> pd.DataFrame:
        """Read parquet dataset through DuckDB with optional filter clause."""
        where_sql = f"WHERE {where_clause}" if where_clause else ""
        sql = f"SELECT * FROM read_parquet('{dataset_root}/**/*.parquet') {where_sql}"
        return self._con.execute(sql).df()

    def close(self) -> None:
        """Close DuckDB connection."""
        self._con.close()
