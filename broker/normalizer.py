"""Tick normalization utilities for Fyers and legacy Kite payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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


class FyersTickNormalizer:
    """Convert raw Fyers DataSocket ticks into a standardized tabular schema."""

    BASE_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "instrument_token",
        "tradingsymbol",
        "last_price",
        "bid_price",
        "ask_price",
        "volume_traded",
        "total_buy_quantity",
        "total_sell_quantity",
        "average_traded_price",
        "change",
        "oi",
        "ohlc_open",
        "ohlc_high",
        "ohlc_low",
        "ohlc_close",
    )

    def __init__(self, symbol_token_map: dict[str, int] | None = None) -> None:
        import re
        self._re = re
        self._symbol_to_token: dict[str, int] = dict(symbol_token_map or {})
        self._token_to_symbol: dict[int, str] = {v: k for k, v in self._symbol_to_token.items()}

    def register_symbol_token(self, symbol: str, token: int) -> None:
        """Register or update symbol-to-token mapping."""
        self._symbol_to_token[symbol] = int(token)
        self._token_to_symbol[int(token)] = symbol

    def normalize_tick(self, tick: Mapping[str, Any]) -> dict[str, Any]:
        """Flatten one raw Fyers websocket tick dictionary."""
        sym = str(tick.get("symbol") or "")
        token = self._resolve_token(tick, sym)

        # Quote time is the exchange feed time, not the last executed trade.
        # An unchanged/illiquid contract can have fresh quotes without a new trade.
        ts = self._normalize_timestamp(tick.get("exch_feed_time")) or datetime.now(UTC)

        def first(*keys: str) -> Any:
            return next((tick[key] for key in keys if key in tick and tick[key] is not None), None)

        lp = _safe_float(first("ltp", "last_price", "lp"))
        bid = _safe_float(first("bid_price", "bid", "bp"))
        ask = _safe_float(first("ask_price", "ask", "ap"))
        # Never invent executable prices from LTP when bid/ask are absent.

        return {
            "timestamp": ts,
            "instrument_token": token,
            "tradingsymbol": sym,
            "last_price": lp,
            "bid_price": bid,
            "ask_price": ask,
            "volume_traded": _safe_float(first("vol_traded_today", "volume", "vol", "volume_traded")),
            "total_buy_quantity": _safe_float(tick.get("tot_buy_qty") or tick.get("bid_size")),
            "total_sell_quantity": _safe_float(tick.get("tot_sell_qty") or tick.get("ask_size")),
            "average_traded_price": _safe_float(tick.get("avg_trade_price") or tick.get("atp")),
            "change": _safe_float(tick.get("ch") or tick.get("change")),
            "oi": _safe_float(tick.get("OI") or tick.get("oi")),
            "ohlc_open": _safe_float(tick.get("open_price")),
            "ohlc_high": _safe_float(tick.get("high_price")),
            "ohlc_low": _safe_float(tick.get("low_price")),
            "ohlc_close": _safe_float(tick.get("prev_close_price")),
        }

    def normalize_ticks(
        self,
        ticks: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        source: str = "fyers",
    ) -> NormalizedTickBatch:
        """Normalize a tick or sequence of ticks into NormalizedTickBatch."""
        raw_list = [ticks] if isinstance(ticks, Mapping) else list(ticks)

        records = [self.normalize_tick(t) for t in raw_list if isinstance(t, Mapping)]
        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return NormalizedTickBatch(frame=frame, row_count=0, source=source)

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        numeric_cols = [
            "instrument_token", "last_price", "bid_price", "ask_price",
            "volume_traded", "total_buy_quantity", "total_sell_quantity",
            "average_traded_price", "change", "oi",
            "ohlc_open", "ohlc_high", "ohlc_low", "ohlc_close",
        ]
        for col in numeric_cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        return NormalizedTickBatch(frame=frame, row_count=len(frame), source=source)

    def _resolve_token(self, tick: Mapping[str, Any], symbol: str) -> int | None:
        if symbol in self._symbol_to_token:
            return self._symbol_to_token[symbol]

        tok = tick.get("instrument_token") or tick.get("token") or tick.get("fyToken") or tick.get("exchange_token")
        if tok is not None:
            clean = self._re.sub(r"\D", "", str(tok))
            if clean:
                return int(clean[-10:])

        if symbol:
            match = self._re.search(r"(\d{5})(CE|PE)", symbol.upper())
            if match:
                strike_val = int(match.group(1))
                side_mult = 1 if match.group(2) == "CE" else 2
                return strike_val * 10 + side_mult

        return None

    @staticmethod
    def _normalize_timestamp(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().replace(".", "", 1).isdigit()):
            number = float(value)
            ts = number / 1000.0 if number > 1e11 else number
            return pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts
