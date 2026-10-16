"""DuckDB metadata and analytics interface for NSE ingestion datasets."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from loguru import logger


class DuckDBManager:
    """Manage metadata tables, signals, trade outcomes, and analytical queries over parquet datasets."""

    def __init__(self, db_path: Path | str | None = None, *, read_only: bool = False) -> None:
        if db_path is None:
            db_path = Path("data/cache/metadata.duckdb")
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only

        if read_only and not self._db_path.exists():
            try:
                init_con = duckdb.connect(str(self._db_path))
                self._con = init_con
                self._bootstrap()
                self._con.close()
            except Exception as exc:
                logger.warning("Could not bootstrap initial DuckDB file: {exc}", exc=str(exc))

        try:
            self._con = duckdb.connect(str(self._db_path), read_only=read_only)
        except Exception as exc:
            logger.warning(
                "Failed to open DuckDB with read_only={read_only}, falling back to default connection: {exc}",
                read_only=read_only,
                exc=str(exc),
            )
            self._con = duckdb.connect(str(self._db_path))

        if not read_only:
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

        # Step 1: Signals relational table
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                signal_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                spot_price DOUBLE,
                strike DOUBLE,
                expiry VARCHAR,
                option_type VARCHAR,
                action VARCHAR NOT NULL,
                confidence DOUBLE,
                trap_score DOUBLE,
                spread_percentage DOUBLE,
                iv_percentile DOUBLE,
                trade_grade VARCHAR,
                filter_passed BOOLEAN NOT NULL,
                filter_rejection_reason VARCHAR
            )
            """
        )

        self._con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp
            ON signals(timestamp)
            """
        )

        # Step 1: Trade outcomes relational table
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_outcomes (
                trade_id VARCHAR PRIMARY KEY,
                signal_id VARCHAR,
                instrument_token BIGINT NOT NULL,
                entry_timestamp TIMESTAMPTZ NOT NULL,
                exit_timestamp TIMESTAMPTZ NOT NULL,
                entry_ask_price DOUBLE NOT NULL,
                exit_bid_price DOUBLE NOT NULL,
                quantity BIGINT NOT NULL,
                holding_time_minutes DOUBLE NOT NULL,
                exit_reason VARCHAR NOT NULL,
                realized_pnl DOUBLE NOT NULL
            )
            """
        )

        self._con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trade_outcomes_exit_ts
            ON trade_outcomes(exit_timestamp)
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

    def insert_signal(
        self,
        *,
        signal_id: str,
        timestamp: datetime,
        spot_price: float | None,
        strike: float | None,
        expiry: str | None,
        option_type: str | None,
        action: str,
        confidence: float,
        trap_score: float,
        spread_percentage: float | None,
        iv_percentile: float | None,
        trade_grade: str,
        filter_passed: bool,
        filter_rejection_reason: str | None = None,
    ) -> None:
        """Insert one signal record into DuckDB using a prepared statement."""
        stmt = """
        INSERT OR REPLACE INTO signals (
            signal_id,
            timestamp,
            spot_price,
            strike,
            expiry,
            option_type,
            action,
            confidence,
            trap_score,
            spread_percentage,
            iv_percentile,
            trade_grade,
            filter_passed,
            filter_rejection_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._con.execute(
            stmt,
            [
                str(signal_id),
                timestamp,
                float(spot_price) if spot_price is not None else None,
                float(strike) if strike is not None else None,
                str(expiry) if expiry is not None else None,
                str(option_type) if option_type is not None else None,
                str(action),
                float(confidence) if confidence is not None else 0.0,
                float(trap_score) if trap_score is not None else 0.0,
                float(spread_percentage) if spread_percentage is not None else None,
                float(iv_percentile) if iv_percentile is not None else None,
                str(trade_grade),
                bool(filter_passed),
                str(filter_rejection_reason) if filter_rejection_reason is not None else None,
            ],
        )
        logger.debug(
            "Inserted signal record into DuckDB",
            signal_id=signal_id,
            action=action,
            filter_passed=filter_passed,
        )

    def insert_trade_outcome(
        self,
        *,
        trade_id: str,
        signal_id: str | None,
        instrument_token: int,
        entry_timestamp: datetime,
        exit_timestamp: datetime,
        entry_ask_price: float,
        exit_bid_price: float,
        quantity: int,
        holding_time_minutes: float,
        exit_reason: str,
        realized_pnl: float,
    ) -> None:
        """Insert one completed trade outcome record into DuckDB using a prepared statement."""
        stmt = """
        INSERT OR REPLACE INTO trade_outcomes (
            trade_id,
            signal_id,
            instrument_token,
            entry_timestamp,
            exit_timestamp,
            entry_ask_price,
            exit_bid_price,
            quantity,
            holding_time_minutes,
            exit_reason,
            realized_pnl
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._con.execute(
            stmt,
            [
                str(trade_id),
                str(signal_id) if signal_id is not None else None,
                int(instrument_token),
                entry_timestamp,
                exit_timestamp,
                float(entry_ask_price),
                float(exit_bid_price),
                int(quantity),
                float(holding_time_minutes),
                str(exit_reason),
                float(realized_pnl),
            ],
        )
        logger.debug(
            "Inserted trade outcome record into DuckDB",
            trade_id=trade_id,
            exit_reason=exit_reason,
            realized_pnl=realized_pnl,
        )

    def query(self, sql: str, params: list[Any] | None = None, *, max_retries: int = 3) -> pd.DataFrame:
        """Run ad-hoc analytical SQL query and return DataFrame with lock retry."""
        import time

        for attempt in range(max_retries):
            try:
                return self._con.execute(sql, params or []).df()
            except Exception as exc:
                err_msg = str(exc).lower()
                if "does not exist" in err_msg or "not found" in err_msg:
                    logger.warning("DuckDB table does not exist: {exc}", exc=str(exc))
                    return pd.DataFrame()
                if attempt < max_retries - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                logger.warning("DuckDB query failed after retries: {exc}", exc=str(exc))
                return pd.DataFrame()
        return pd.DataFrame()

    def get_signals(self, limit: int | None = None) -> pd.DataFrame:
        """Retrieve recent signal records."""
        limit_sql = f"LIMIT {limit}" if limit else ""
        return self.query(f"SELECT * FROM signals ORDER BY timestamp DESC {limit_sql}")

    def get_trade_outcomes(self, limit: int | None = None) -> pd.DataFrame:
        """Retrieve recent trade outcome records."""
        limit_sql = f"LIMIT {limit}" if limit else ""
        return self.query(f"SELECT * FROM trade_outcomes ORDER BY exit_timestamp DESC {limit_sql}")

    def query_processed_dataset(self, dataset_root: Path, where_clause: str | None = None) -> pd.DataFrame:
        """Read parquet dataset through DuckDB with optional filter clause."""
        where_sql = f"WHERE {where_clause}" if where_clause else ""
        sql = f"SELECT * FROM read_parquet('{dataset_root}/**/*.parquet') {where_sql}"
        return self._con.execute(sql).df()

    def close(self) -> None:
        """Close DuckDB connection."""
        self._con.close()
