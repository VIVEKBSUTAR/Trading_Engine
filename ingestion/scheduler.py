"""Minute-level ingestion scheduler with graceful shutdown and metrics."""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event

from loguru import logger

from config.settings import AppSettings
from ingestion.nse_fetcher import NSEOptionChainFetcher
from ingestion.parser import OptionChainParser
from ingestion.validators import OptionChainValidator
from storage.duckdb_manager import DuckDBManager
from storage.parquet_writer import ParquetWriter


@dataclass(slots=True)
class RunMetrics:
    """Collection timing and row-count metrics for each polling cycle."""

    started_at_utc: datetime
    finished_at_utc: datetime
    elapsed_ms: float
    input_rows: int
    output_rows: int


class IngestionScheduler:
    """Orchestrate fetch -> parse -> validate -> persist loop at fixed interval."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        fetcher: NSEOptionChainFetcher,
        parser: OptionChainParser,
        validator: OptionChainValidator,
        parquet_writer: ParquetWriter,
        duckdb_manager: DuckDBManager,
    ) -> None:
        self._settings = settings
        self._fetcher = fetcher
        self._parser = parser
        self._validator = validator
        self._parquet_writer = parquet_writer
        self._duckdb = duckdb_manager
        self._stop_event = Event()

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful termination."""

        def _shutdown_handler(signum: int, _frame: object) -> None:
            logger.warning("Shutdown signal received", signal=signum)
            self._stop_event.set()

        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)

    def run_forever(self) -> None:
        """Run ingestion cycles until shutdown signal is received."""
        logger.info(
            "Starting NSE option-chain scheduler",
            interval_seconds=self._settings.scheduler.interval_seconds,
        )

        while not self._stop_event.is_set():
            cycle_started = time.perf_counter()

            try:
                metrics = self.run_once()
                logger.info(
                    "Collection cycle finished",
                    elapsed_ms=round(metrics.elapsed_ms, 2),
                    input_rows=metrics.input_rows,
                    output_rows=metrics.output_rows,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Collection cycle failed", error=str(exc))

            elapsed = time.perf_counter() - cycle_started
            sleep_seconds = max(0.0, self._settings.scheduler.interval_seconds - elapsed)
            if sleep_seconds > 0 and not self._stop_event.is_set():
                self._stop_event.wait(timeout=sleep_seconds)

        logger.info("Scheduler stopped gracefully")

    def run_once(self) -> RunMetrics:
        """Execute one ingestion cycle and return run metrics."""
        started_at = datetime.now(UTC)
        started_clock = time.perf_counter()

        fetch_result = self._fetcher.fetch_option_chain()
        raw_path = self._parquet_writer.write_raw_snapshot(
            raw_text=fetch_result.raw_text,
            fetched_at=fetch_result.fetched_at_utc,
            source="nse_option_chain",
        )

        parsed = self._parser.parse(fetch_result.payload, fetch_result.fetched_at_utc)
        validation = self._validator.validate(parsed)

        processed_dataset_name = None
        min_timestamp = None
        max_timestamp = None

        if not validation.frame.empty:
            processed_dataset_name = self._settings.storage.processed_dataset_name
            self._parquet_writer.write_processed(validation.frame, dataset_name=processed_dataset_name)
            min_timestamp = validation.frame["timestamp"].min().to_pydatetime()
            max_timestamp = validation.frame["timestamp"].max().to_pydatetime()
        else:
            logger.warning("Validated frame empty; skipping processed parquet write")

        self._duckdb.register_snapshot(
            source="NSE",
            raw_path=raw_path,
            processed_dataset=processed_dataset_name,
            rows_ingested=validation.output_rows,
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
            snapshot_ts=fetch_result.fetched_at_utc,
        )

        finished_at = datetime.now(UTC)
        elapsed_ms = (time.perf_counter() - started_clock) * 1000.0
        return RunMetrics(
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            elapsed_ms=elapsed_ms,
            input_rows=validation.input_rows,
            output_rows=validation.output_rows,
        )

    def stop(self) -> None:
        """Request scheduler shutdown from external caller."""
        self._stop_event.set()
