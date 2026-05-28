"""Centralized configuration for NSE option-chain ingestion pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class APISettings:
    """HTTP endpoint and header configuration for NSE API calls."""

    option_chain_url: str = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    bootstrap_url: str = "https://www.nseindia.com"
    bootstrap_urls: tuple[str, ...] = (
        "https://www.nseindia.com",
        "https://www.nseindia.com/option-chain",
        "https://www.nseindia.com/market-data/live-equity-market",
    )
    timeout_seconds: int = 15
    max_retries: int = 4
    backoff_seconds: float = 1.5
    headers: dict[str, str] = field(
        default_factory=lambda: {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest",
        }
    )


@dataclass(slots=True)
class StorageSettings:
    """Filesystem and DuckDB paths for raw, processed, and metadata datasets."""

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    metadata_dir: Path
    duckdb_path: Path
    processed_dataset_name: str = "nse_option_chain"


@dataclass(slots=True)
class SchedulerSettings:
    """Collection loop behavior and runtime controls."""

    interval_seconds: int = 60
    symbol: str = "NIFTY"
    timezone: str = "Asia/Kolkata"


@dataclass(slots=True)
class LoggingSettings:
    """Log sink settings for loguru output."""

    log_dir: Path
    file_name: str = "ingestion.log"
    broker_file_name: str = "broker.log"
    level: str = "INFO"
    rotation: str = "20 MB"
    retention: str = "14 days"


@dataclass(slots=True)
class KiteSettings:
    """Kite Connect authentication and streaming configuration."""

    api_key: str | None = None
    api_secret: str | None = None
    access_token: str | None = None
    request_token: str | None = None
    token_store_path: Path | None = None
    reconnect: bool = True
    reconnect_max_tries: int = 50
    reconnect_max_delay: int = 60
    connect_timeout: int = 30
    stream_tokens: tuple[int, ...] = ()
    stream_mode: str = "full"


@dataclass(slots=True)
class AppSettings:
    """Aggregate runtime settings used by pipeline components."""

    api: APISettings
    storage: StorageSettings
    scheduler: SchedulerSettings
    logging: LoggingSettings
    kite: KiteSettings

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> AppSettings:
        """Build settings from environment variables with safe defaults."""
        load_dotenv()
        root = project_root or Path(__file__).resolve().parents[1]

        raw_dir = Path(os.getenv("TE_RAW_DIR", str(root / "data" / "raw"))).resolve()
        processed_dir = Path(os.getenv("TE_PROCESSED_DIR", str(root / "data" / "processed"))).resolve()
        metadata_dir = Path(os.getenv("TE_METADATA_DIR", str(root / "data" / "metadata"))).resolve()
        duckdb_path = Path(os.getenv("TE_DUCKDB_PATH", str(metadata_dir / "nse_ingestion.duckdb"))).resolve()
        log_dir = Path(os.getenv("TE_LOG_DIR", str(root / "logs"))).resolve()

        interval_seconds = int(os.getenv("TE_FETCH_INTERVAL_SECONDS", "60"))
        timeout_seconds = int(os.getenv("TE_API_TIMEOUT_SECONDS", "15"))
        max_retries = int(os.getenv("TE_API_MAX_RETRIES", "4"))

        token_store_path = Path(
            os.getenv("KITE_TOKEN_STORE_PATH", str(metadata_dir / "kite_access_token.txt"))
        ).resolve()

        stream_tokens_raw = os.getenv("KITE_STREAM_TOKENS", "").strip()
        stream_tokens = tuple(
            int(token.strip())
            for token in stream_tokens_raw.split(",")
            if token.strip()
        )

        settings = cls(
            api=APISettings(timeout_seconds=timeout_seconds, max_retries=max_retries),
            storage=StorageSettings(
                project_root=root.resolve(),
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                metadata_dir=metadata_dir,
                duckdb_path=duckdb_path,
            ),
            scheduler=SchedulerSettings(
                interval_seconds=interval_seconds,
                symbol=os.getenv("TE_SYMBOL", "NIFTY"),
                timezone=os.getenv("TE_TIMEZONE", "Asia/Kolkata"),
            ),
            logging=LoggingSettings(
                log_dir=log_dir,
                level=os.getenv("TE_LOG_LEVEL", "INFO"),
            ),
            kite=KiteSettings(
                api_key=os.getenv("KITE_API_KEY") or None,
                api_secret=os.getenv("KITE_API_SECRET") or None,
                access_token=os.getenv("KITE_ACCESS_TOKEN") or None,
                request_token=os.getenv("KITE_REQUEST_TOKEN") or None,
                token_store_path=token_store_path,
                reconnect=os.getenv("KITE_RECONNECT", "true").lower() in {"1", "true", "yes"},
                reconnect_max_tries=int(os.getenv("KITE_RECONNECT_MAX_TRIES", "50")),
                reconnect_max_delay=int(os.getenv("KITE_RECONNECT_MAX_DELAY", "60")),
                connect_timeout=int(os.getenv("KITE_CONNECT_TIMEOUT", "30")),
                stream_tokens=stream_tokens,
                stream_mode=os.getenv("KITE_STREAM_MODE", "full").lower(),
            ),
        )

        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        """Create all required runtime directories."""
        self.storage.raw_dir.mkdir(parents=True, exist_ok=True)
        self.storage.processed_dir.mkdir(parents=True, exist_ok=True)
        self.storage.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.storage.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)
        if self.kite.token_store_path is not None:
            self.kite.token_store_path.parent.mkdir(parents=True, exist_ok=True)
