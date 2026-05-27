"""Entrypoint for production NSE option-chain ingestion pipeline."""

from __future__ import annotations

import sys

from loguru import logger

from config.settings import AppSettings
from ingestion.nse_fetcher import NSEOptionChainFetcher
from ingestion.parser import OptionChainParser
from ingestion.scheduler import IngestionScheduler
from ingestion.validators import OptionChainValidator
from storage.duckdb_manager import DuckDBManager
from storage.parquet_writer import ParquetWriter


def configure_logging(settings: AppSettings) -> None:
    """Configure structured loguru sinks for console and file."""
    logger.remove()
    logger.add(sys.stdout, level=settings.logging.level, enqueue=True, backtrace=False, diagnose=False)
    logger.add(
        settings.logging.log_dir / settings.logging.file_name,
        level=settings.logging.level,
        enqueue=True,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        backtrace=False,
        diagnose=False,
    )


def build_scheduler(settings: AppSettings) -> tuple[IngestionScheduler, DuckDBManager]:
    """Build all pipeline components and return scheduler + DB manager."""
    fetcher = NSEOptionChainFetcher(settings.api)
    parser = OptionChainParser()
    validator = OptionChainValidator()
    parquet_writer = ParquetWriter(settings.storage)
    duckdb_manager = DuckDBManager(settings.storage.duckdb_path)

    scheduler = IngestionScheduler(
        settings=settings,
        fetcher=fetcher,
        parser=parser,
        validator=validator,
        parquet_writer=parquet_writer,
        duckdb_manager=duckdb_manager,
    )
    return scheduler, duckdb_manager


def main() -> int:
    """Start the NSE option-chain scheduler until termination signal."""
    settings = AppSettings.from_env()
    configure_logging(settings)
    logger.info("Initializing NSE ingestion pipeline", project_root=str(settings.storage.project_root))

    scheduler, duckdb_manager = build_scheduler(settings)
    scheduler.install_signal_handlers()

    try:
        scheduler.run_forever()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error in ingestion runtime", error=str(exc))
        return 2
    finally:
        duckdb_manager.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
