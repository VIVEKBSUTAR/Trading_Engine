"""Parser for NSE option-chain JSON payloads into normalized records."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from loguru import logger


class OptionChainParser:
    """Transform NSE nested option-chain JSON into a structured DataFrame."""

    REQUIRED_OUTPUT_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "expiry",
        "strike",
        "option_type",
        "open_interest",
        "change_in_oi",
        "implied_volatility",
        "traded_volume",
        "last_price",
        "bid_qty",
        "ask_qty",
        "underlying_price",
    )

    def parse(self, payload: dict, fetched_at_utc: datetime | None = None) -> pd.DataFrame:
        """Parse CE and PE legs into one normalized DataFrame."""
        records_node = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records_node, dict):
            raise ValueError("Invalid payload: missing 'records' node")

        entries = records_node.get("data", [])
        if not isinstance(entries, list):
            raise ValueError("Invalid payload: 'records.data' must be a list")

        timestamp = self._resolve_snapshot_timestamp(records_node, fetched_at_utc)
        default_underlying = records_node.get("underlyingValue")

        normalized_rows: list[dict] = []
        for row in entries:
            if not isinstance(row, dict):
                continue

            strike = row.get("strikePrice")
            expiry = row.get("expiryDate")
            if strike is None or expiry is None:
                continue

            for option_type in ("CE", "PE"):
                leg = row.get(option_type)
                if not isinstance(leg, dict):
                    continue

                normalized_rows.append(
                    {
                        "timestamp": timestamp,
                        "expiry": expiry,
                        "strike": strike,
                        "option_type": option_type,
                        "open_interest": leg.get("openInterest"),
                        "change_in_oi": leg.get("changeinOpenInterest"),
                        "implied_volatility": leg.get("impliedVolatility"),
                        "traded_volume": leg.get("totalTradedVolume"),
                        "last_price": leg.get("lastPrice"),
                        "bid_qty": leg.get("bidQty"),
                        "ask_qty": leg.get("askQty"),
                        "underlying_price": leg.get("underlyingValue", default_underlying),
                    }
                )

        frame = pd.DataFrame(normalized_rows)
        if frame.empty:
            logger.warning("Parser produced empty DataFrame")
            return pd.DataFrame(columns=self.REQUIRED_OUTPUT_COLUMNS)

        missing_cols = [col for col in self.REQUIRED_OUTPUT_COLUMNS if col not in frame.columns]
        for missing in missing_cols:
            frame[missing] = pd.NA

        frame = frame.loc[:, list(self.REQUIRED_OUTPUT_COLUMNS)]
        logger.info("Parsed option chain payload", rows=len(frame))
        return frame

    @staticmethod
    def _resolve_snapshot_timestamp(records_node: dict, fetched_at_utc: datetime | None) -> datetime:
        raw_ts = records_node.get("timestamp")
        if isinstance(raw_ts, str):
            # NSE usually emits timestamps like: "27-May-2026 15:30:00"
            for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
                try:
                    return datetime.strptime(raw_ts, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue

        fallback = fetched_at_utc or datetime.now(UTC)
        logger.warning("Unable to parse NSE timestamp; using fallback", fallback=str(fallback))
        return fallback
