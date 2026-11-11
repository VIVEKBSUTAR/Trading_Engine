"""DuckDB metadata and analytics interface for NSE ingestion datasets."""

from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime, timedelta
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
        self._tmp_copy: Path | None = None  # tracks temp copy used for dashboard reads

        if read_only and not self._db_path.exists():
            # Bootstrap a fresh file so the read-only open below always has a target.
            try:
                init_con = duckdb.connect(str(self._db_path))
                self._con = init_con
                self._bootstrap()
                self._con.close()
            except Exception as exc:
                logger.warning("Could not bootstrap initial DuckDB file: {exc}", exc=str(exc))

        if read_only:
            self._con = self._open_readonly()
        else:
            try:
                self._con = duckdb.connect(str(self._db_path))
            except Exception as exc:
                raise RuntimeError(f"Cannot open DuckDB for writing at {self._db_path}: {exc}") from exc
            self._bootstrap()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_readonly(self) -> duckdb.DuckDBPyConnection:
        """Open a read-only connection, falling back to a temp-file copy when
        another process holds an exclusive write lock on the original file.

        DuckDB does not allow *any* new connections (even read-only) while a
        writer holds the file lock.  The safe workaround is to copy the file
        to a temporary location and open the copy read-only — the copy is
        always consistent because DuckDB flushes fully before every checkpoint.
        """
        # 1. Try a direct read-only connection (works if no writer is active).
        try:
            con = duckdb.connect(str(self._db_path), read_only=True)
            logger.debug("Opened DuckDB read-only (direct)", path=str(self._db_path))
            return con
        except Exception as exc:
            err_lower = str(exc).lower()
            # Catch both cross-process lock conflicts and in-process config
            # conflicts (DuckDB raises different exception types for each).
            is_lock_conflict = any(
                kw in err_lower
                for kw in ("lock", "conflict", "different configuration", "can't open", "cannot open")
            )
            if not is_lock_conflict:
                raise  # surface unexpected errors immediately
            logger.warning(
                "DuckDB conflict detected — opening a temp-file copy for read-only access",
                path=str(self._db_path),
                error=str(exc),
            )

        # 2. Copy the file to a temp path and open the copy read-only.
        self._cleanup_tmp()
        import os
        tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".duckdb", prefix="te_dashboard_")
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)
        wal_path = Path(str(self._db_path) + ".wal")
        tmp_wal_path = Path(str(tmp_path) + ".wal")
        try:
            shutil.copy2(str(self._db_path), str(tmp_path))
            if wal_path.exists():
                shutil.copy2(str(wal_path), str(tmp_wal_path))
            con = duckdb.connect(str(tmp_path), read_only=True)
            self._tmp_copy = tmp_path
            logger.info(
                "Opened DuckDB read-only via temp copy (intelligence runtime is active)",
                tmp_path=str(tmp_path),
            )
            return con
        except Exception as exc2:
            tmp_path.unlink(missing_ok=True)
            tmp_wal_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Cannot open DuckDB for reading at {self._db_path} "
                f"(lock conflict and temp-copy also failed): {exc2}"
            ) from exc2

    def _cleanup_tmp(self) -> None:
        """Remove any leftover temp copy from a previous session."""
        if self._tmp_copy is not None:
            try:
                self._tmp_copy.unlink(missing_ok=True)
                Path(str(self._tmp_copy) + ".wal").unlink(missing_ok=True)
            except OSError:
                pass
            self._tmp_copy = None

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
        # Idempotent additive migration for databases created before these filters.
        for column, kind in (("max_holding_minutes", "INTEGER"), ("days_to_expiry", "INTEGER"), ("strike_oi_ratio", "DOUBLE"), ("entry_price", "DOUBLE")):
            self._con.execute(f"ALTER TABLE signals ADD COLUMN IF NOT EXISTS {column} {kind}")

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
        max_holding_minutes: int | None = None,
        days_to_expiry: int | None = None,
        strike_oi_ratio: float | None = None,
        entry_price: float | None = None,
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
            filter_rejection_reason,
            max_holding_minutes,
            days_to_expiry,
            strike_oi_ratio,
            entry_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                max_holding_minutes,
                days_to_expiry,
                strike_oi_ratio,
                entry_price,
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

    def get_live_signals(self, *, now: datetime | None = None) -> pd.DataFrame:
        """Passed directional alerts, independently of the 50-row audit window."""
        now = now or datetime.now(UTC)
        return self.query(
            """SELECT signal_id, timestamp, action, strike, expiry, option_type,
                      entry_price, filter_passed FROM signals
               WHERE filter_passed = TRUE AND UPPER(action) IN ('BUY_CE', 'BUY_PE')
                 AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp DESC""",
            [now - timedelta(seconds=60), now],
        )

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
        """Close DuckDB connection and remove any temp copy used for read-only access."""
        try:
            self._con.close()
        except Exception:
            pass
        self._cleanup_tmp()
