"""Structured logging setup using loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from loguru import logger

LogChannel = Literal["ingestion", "features", "backtests", "live"]


def configure_logging(*, log_dir: Path, level: str = "INFO", serialize: bool = False) -> None:
    """Configure console and per-channel file sinks."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        serialize=serialize,
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    for channel in ("ingestion", "features", "backtests", "live"):
        channel_dir = log_dir / channel
        channel_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            channel_dir / f"{channel}.log",
            level=level,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            serialize=serialize,
            enqueue=True,
            backtrace=True,
            diagnose=True,
            filter=lambda record, ch=channel: record["extra"].get("channel") == ch,
        )


def get_logger(channel: LogChannel):
    """Bind a channel-scoped logger for routing to dedicated files."""
    return logger.bind(channel=channel)
