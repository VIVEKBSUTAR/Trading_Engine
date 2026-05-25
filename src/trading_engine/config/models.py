"""Pydantic configuration models for all runtime layers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    """Filesystem paths used by the application."""

    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))
    data_cache_dir: Path = Field(default=Path("data/cache"))
    duckdb_path: Path = Field(default=Path("data/cache/metadata.duckdb"))
    log_dir: Path = Field(default=Path("logs"))

    @field_validator("data_raw_dir", "data_processed_dir", "data_cache_dir", "duckdb_path", "log_dir", mode="before")
    @classmethod
    def _parse_path(cls, value: str | Path) -> Path:
        return Path(value)


class IngestionConfig(BaseModel):
    """Data ingestion and cleaning behavior."""

    default_timezone: str = "Asia/Kolkata"
    default_resolution: Literal["1min", "5min", "15min", "1h", "1d"] = "1min"
    allowed_resolutions: list[str] = Field(default_factory=lambda: ["1min", "5min", "15min", "1h", "1d"])
    drop_duplicates_keep: Literal["first", "last", False] = "last"
    missing_timestamp_fill: float | None = None
    corruption_error_threshold: float = 0.02


class StorageConfig(BaseModel):
    """Parquet and metadata catalog settings."""

    parquet_compression: str = "zstd"
    parquet_row_group_size: int = 200_000
    partition_columns: list[str] = Field(default_factory=lambda: ["feed", "date"])


class LoggingConfig(BaseModel):
    """Global logging controls."""

    level: str = "INFO"
    json: bool = False


class AppConfig(BaseModel):
    """Top-level static application configuration."""

    environment: str = "dev"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
