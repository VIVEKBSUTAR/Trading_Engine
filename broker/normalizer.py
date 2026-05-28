"""Tick normalization utilities for Kite Connect streaming payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

import pandas as pd
from loguru import logger


@dataclass(slots=True)
class NormalizedTickBatch:
    """Normalized ticks ready for DataFrame-based analytics."""

    frame: pd.DataFrame
    row_count: int
    source: str = "kite"


class KiteTickNormalizer:
    """Convert raw Kite ticks into a standardized tabular schema."""

    BASE_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "exchange_timestamp",
        "last_trade_time",
        "instrument_token",
        "tradingsymbol",
        "exchange",
        "mode",
        "tradable",
        "last_price",
        "average_traded_price",
        "last_traded_quantity",
        "volume_traded",
        "total_buy_quantity",
        "total_sell_quantity",
        "change",
        "oi",
        "oi_day_low",
        "oi_day_high",
        "ohlc_open",
        "ohlc_high",
        "ohlc_low",
        "ohlc_close",
    )

    def normalize_tick(self, tick: Mapping[str, Any]) -> dict[str, Any]:
        """Flatten one raw tick into a standardized dictionary."""
        ohlc = tick.get("ohlc") if isinstance(tick.get("ohlc"), Mapping) else {}

        exchange_ts = self._normalize_timestamp(tick.get("exchange_timestamp"))
        trade_ts = self._normalize_timestamp(tick.get("last_trade_time"))
        timestamp = exchange_ts or trade_ts or datetime.now(UTC)

        normalized: dict[str, Any] = {
            "timestamp": timestamp,
            "exchange_timestamp": exchange_ts,
            "last_trade_time": trade_ts,
            "instrument_token": _safe_int(tick.get("instrument_token")),
            "tradingsymbol": tick.get("tradingsymbol"),
            "exchange": tick.get("exchange"),
            "mode": tick.get("mode"),
            "tradable": _safe_bool(tick.get("tradable")),
            "last_price": _safe_float(tick.get("last_price")),
            "average_traded_price": _safe_float(tick.get("average_traded_price")),
            "last_traded_quantity": _safe_float(tick.get("last_traded_quantity")),
            "volume_traded": _safe_float(tick.get("volume_traded")),
            "total_buy_quantity": _safe_float(tick.get("total_buy_quantity")),
            "total_sell_quantity": _safe_float(tick.get("total_sell_quantity")),
            "change": _safe_float(tick.get("change")),
            "oi": _safe_float(tick.get("oi")),
            "oi_day_low": _safe_float(tick.get("oi_day_low")),
            "oi_day_high": _safe_float(tick.get("oi_day_high")),
            "ohlc_open": _safe_float(ohlc.get("open")),
            "ohlc_high": _safe_float(ohlc.get("high")),
            "ohlc_low": _safe_float(ohlc.get("low")),
            "ohlc_close": _safe_float(ohlc.get("close")),
        }

        return normalized

    def normalize_ticks(self, ticks: Sequence[Mapping[str, Any]], source: str = "kite") -> NormalizedTickBatch:
        """Normalize a list of raw ticks into a cleaned DataFrame."""
        records = [self.normalize_tick(tick) for tick in ticks if isinstance(tick, Mapping)]
        frame = pd.DataFrame.from_records(records)

        if frame.empty:
            logger.warning("Normalized tick batch is empty", source=source)
            return NormalizedTickBatch(frame=frame, row_count=0, source=source)

        for column in ("timestamp", "exchange_timestamp", "last_trade_time"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)

        numeric_columns = [column for column in self.BASE_COLUMNS if column not in {"timestamp", "exchange_timestamp", "last_trade_time", "tradingsymbol", "exchange", "mode", "tradable"}]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "tradable" in frame.columns:
            frame["tradable"] = frame["tradable"].astype("boolean")

        frame = frame.sort_values(["timestamp", "instrument_token"], na_position="last").reset_index(drop=True)
        frame = frame.loc[:, [column for column in self.BASE_COLUMNS if column in frame.columns]]

        logger.info("Normalized tick batch", source=source, rows=len(frame))
        return NormalizedTickBatch(frame=frame, row_count=len(frame), source=source)

    @staticmethod
    def _normalize_timestamp(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)
