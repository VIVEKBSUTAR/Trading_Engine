"""Live runtime wiring for Kite Connect, NSE option chain, and intelligence engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from loguru import logger

from broker.kite_auth import KiteAuthManager
from broker.kite_client import KiteMarketClient
from broker.kite_stream import KiteStream
from broker.normalizer import NormalizedTickBatch
from config.settings import AppSettings
from ingestion.nse_fetcher import NSEOptionChainFetcher
from ingestion.parser import OptionChainParser
from ingestion.validators import OptionChainValidator
from trading_engine.intelligence.engine import LiveIntelligenceEngine
from trading_engine.intelligence.state import MarketStateAggregator
from trading_engine.intelligence.models import IntelligenceReport, MarketState


@dataclass(slots=True)
class RuntimeCallbacks:
    """Optional callbacks for live report publishing."""

    on_report: Callable[[IntelligenceReport], None] | None = None
    on_error: Callable[[Exception], None] | None = None


class LiveIntelligenceRuntime:
    """Wire live market sources into the intelligence engine and monitor."""

    def __init__(self, settings: AppSettings, callbacks: RuntimeCallbacks | None = None) -> None:
        self._settings = settings
        self._callbacks = callbacks or RuntimeCallbacks()
        self._auth = KiteAuthManager(settings.kite)
        self._client = KiteMarketClient(settings.kite, auth_manager=self._auth)
        self._stream = KiteStream(settings.kite, auth_manager=self._auth)
        self._fetcher = NSEOptionChainFetcher(settings.api)
        self._parser = OptionChainParser()
        self._validator = OptionChainValidator()
        self._aggregator = MarketStateAggregator()
        self._engine = LiveIntelligenceEngine(settings)
        self._latest_report: IntelligenceReport | None = None
        self._latest_state: MarketState | None = None

        self._stream.register_tick_handler(self._on_tick_frame)
        self._stream.register_connect_handler(self._on_connect)

    @property
    def latest_report(self) -> IntelligenceReport | None:
        """Return the latest computed intelligence report."""
        return self._latest_report

    @property
    def latest_state(self) -> MarketState | None:
        """Return the latest computed market state."""
        return self._latest_state

    @property
    def stream(self) -> KiteStream:
        """Expose the underlying websocket stream wrapper."""
        return self._stream

    def refresh_once(self) -> IntelligenceReport:
        """Poll REST sources once and compute a fresh intelligence report."""
        snapshot = self._client.get_live_snapshot()
        self._aggregator.update_from_snapshot(snapshot)

        try:
            chain_result = self._fetcher.fetch_option_chain()
            chain_frame = self._parser.parse(chain_result.payload, chain_result.fetched_at_utc)
            validated = self._validator.validate(chain_frame)
            self._aggregator.update_option_chain(validated.frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Option-chain refresh failed; continuing with cached state", error=str(exc))
            if self._callbacks.on_error:
                self._callbacks.on_error(exc)

        state = self._aggregator.build_market_state(snapshot)
        self._latest_state = state
        report = self._engine.process(snapshot, state=state)
        self._latest_report = report
        if self._callbacks.on_report:
            self._callbacks.on_report(report)
        return report

    def start_streaming(self) -> None:
        """Start websocket streaming when tokens are configured."""
        if self._settings.kite.stream_tokens:
            self._stream.subscribe(list(self._settings.kite.stream_tokens))
        self._stream.connect(threaded=True)

    def stop(self) -> None:
        """Stop websocket streaming cleanly."""
        self._stream.stop()

    def _on_connect(self) -> None:
        try:
            self.refresh_once()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Initial runtime refresh failed", error=str(exc))
            if self._callbacks.on_error:
                self._callbacks.on_error(exc)

    def _on_tick_frame(self, frame: pd.DataFrame) -> None:
        self._aggregator.update_ticks(NormalizedTickBatch(frame=frame, row_count=len(frame), source="kite"))
        try:
            snapshot = self._aggregator.build_snapshot()
            self._latest_state = self._aggregator.build_market_state(snapshot)
            self._latest_report = self._engine.process(snapshot, state=self._latest_state)
            if self._callbacks.on_report:
                self._callbacks.on_report(self._latest_report)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process live tick frame", error=str(exc))
            if self._callbacks.on_error:
                self._callbacks.on_error(exc)
