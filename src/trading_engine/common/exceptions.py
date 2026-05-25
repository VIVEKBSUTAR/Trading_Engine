"""Custom exceptions for trading engine modules."""

from __future__ import annotations


class TradingEngineError(Exception):
    """Base class for domain-specific errors."""


class ConfigurationError(TradingEngineError):
    """Raised when runtime configuration is invalid."""


class DataValidationError(TradingEngineError):
    """Raised when data quality validation fails."""


class DataIngestionError(TradingEngineError):
    """Raised when feed ingestion cannot be completed."""


class StorageError(TradingEngineError):
    """Raised for persistence and catalog integration failures."""
