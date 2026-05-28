"""Live market state aggregation for probabilistic options intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from loguru import logger

from broker.kite_client import LiveMarketSnapshot, MarketQuote
from broker.normalizer import NormalizedTickBatch
from trading_engine.intelligence.models import MarketSnapshotBundle, utcnow


@dataclass(slots=True)
class StateConfig:
    """Limits for in-memory live history buffers."""

    max_spot_rows: int = 2000
    max_vix_rows: int = 2000
    max_option_rows: int = 20000
    max_tick_rows: int = 50000
    candle_frequency: str = "1min"


class MarketStateAggregator:
    """Maintain rolling live state from broker and NSE market feeds."""

    def __init__(self, config: StateConfig | None = None) -> None:
        self._config = config or StateConfig()
        self._spot_history = pd.DataFrame()
        self._vix_history = pd.DataFrame()
        self._option_chain = pd.DataFrame()
        self._ticks = pd.DataFrame()
        self._last_snapshot: MarketSnapshotBundle | None = None

    def update_spot(self, quote: MarketQuote) -> None:
        """Append a new spot quote snapshot."""
        row = pd.DataFrame(
            [
                {
                    "timestamp": utcnow(),
                    "instrument": quote.instrument,
                    "last_price": quote.last_price,
                    "change": quote.change,
                    "open": quote.ohlc.get("open"),
                    "high": quote.ohlc.get("high"),
                    "low": quote.ohlc.get("low"),
                    "close": quote.ohlc.get("close"),
                }
            ]
        )
        self._spot_history = self._append_and_trim(self._spot_history, row, self._config.max_spot_rows)
        logger.debug("Updated spot history", rows=len(self._spot_history))

    def update_vix(self, quote: MarketQuote) -> None:
        """Append a new India VIX snapshot."""
        row = pd.DataFrame(
            [
                {
                    "timestamp": utcnow(),
                    "instrument": quote.instrument,
                    "last_price": quote.last_price,
                    "change": quote.change,
                    "close": quote.last_price,
                }
            ]
        )
        self._vix_history = self._append_and_trim(self._vix_history, row, self._config.max_vix_rows)
        logger.debug("Updated VIX history", rows=len(self._vix_history))

    def update_option_chain(self, frame: pd.DataFrame) -> None:
        """Replace the current option-chain snapshot with the latest validated frame."""
        if frame is None or frame.empty:
            logger.warning("Ignoring empty option-chain update")
            return

        working = frame.copy()
        if "timestamp" in working.columns:
            working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        self._option_chain = self._append_and_trim(self._option_chain, working, self._config.max_option_rows)
        logger.debug("Updated option chain history", rows=len(self._option_chain))

    def update_ticks(self, batch: NormalizedTickBatch) -> None:
        """Append normalized websocket ticks and keep a bounded history."""
        if batch.frame.empty:
            return

        working = batch.frame.copy()
        self._ticks = self._append_and_trim(self._ticks, working, self._config.max_tick_rows)
        logger.debug("Updated tick history", rows=len(self._ticks))

    def update_from_snapshot(self, snapshot: LiveMarketSnapshot, option_chain: pd.DataFrame | None = None) -> None:
        """Convenience method to ingest a fresh REST market snapshot."""
        self.update_spot(snapshot.nifty_spot)
        self.update_vix(snapshot.india_vix)
        if option_chain is not None:
            self.update_option_chain(option_chain)

    def build_snapshot(self) -> MarketSnapshotBundle:
        """Construct a normalized live market bundle for the intelligence engine."""
        spot_frame = self._spot_history.copy()
        vix_frame = self._vix_history.copy()
        option_frame = self._option_chain.copy()
        tick_frame = self._ticks.copy()
        candles_1m = self._build_candles(tick_frame, "1min")
        candles_5m = self._build_candles(tick_frame, "5min")

        spot_price = self._latest_numeric(spot_frame, "last_price")
        vix_value = self._latest_numeric(vix_frame, "last_price")

        bundle = MarketSnapshotBundle(
            timestamp=utcnow(),
            spot_price=spot_price,
            vix_value=vix_value,
            spot_frame=spot_frame,
            vix_frame=vix_frame,
            option_chain_frame=option_frame,
            ticks_frame=tick_frame,
            candles_1m=candles_1m,
            candles_5m=candles_5m,
        )
        self._last_snapshot = bundle
        return bundle

    @property
    def last_snapshot(self) -> MarketSnapshotBundle | None:
        """Return the most recently built snapshot if available."""
        return self._last_snapshot

    @staticmethod
    def _append_and_trim(existing: pd.DataFrame, new_rows: pd.DataFrame, limit: int) -> pd.DataFrame:
        combined = pd.concat([existing, new_rows], ignore_index=True)
        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp")
        if limit > 0 and len(combined) > limit:
            combined = combined.tail(limit)
        return combined.reset_index(drop=True)

    @staticmethod
    def _build_candles(ticks: pd.DataFrame, frequency: str) -> pd.DataFrame:
        if ticks.empty or "timestamp" not in ticks.columns or "last_price" not in ticks.columns:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        working = ticks.copy().dropna(subset=["timestamp", "last_price"])
        if working.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        working = working.dropna(subset=["timestamp"])
        working = working.set_index("timestamp").sort_index()

        price_col = "last_price"
        volume_col = "volume_traded" if "volume_traded" in working.columns else None

        agg: dict[str, Any] = {
            price_col: ["first", "max", "min", "last"],
        }
        if volume_col:
            agg[volume_col] = "last"

        resampled = working.resample(frequency).agg(agg)
        resampled.columns = ["open", "high", "low", "close"] + (["volume"] if volume_col else [])
        if "volume" not in resampled.columns:
            resampled["volume"] = 0.0

        return resampled.reset_index().rename(columns={"timestamp": "timestamp"})

    @staticmethod
    def _latest_numeric(frame: pd.DataFrame, column: str) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])
