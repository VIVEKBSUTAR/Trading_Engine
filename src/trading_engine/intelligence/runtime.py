"""Live runtime wiring for Fyers API V3, NSE option chain, and intelligence engines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from broker.api_security import APISecurityError, APISecurityGuard
from broker.fyers_auth import FyersAuthManager
from broker.fyers_client import FyersMarketClient
from broker.fyers_stream import FyersStream
from broker.normalizer import NormalizedTickBatch
from config.settings import AppSettings
from ingestion.nse_fetcher import NSEOptionChainFetcher
from ingestion.parser import OptionChainParser
from ingestion.validators import OptionChainValidator
from trading_engine.intelligence.engine import LiveIntelligenceEngine
from trading_engine.intelligence.models import IntelligenceReport, MarketState
from trading_engine.intelligence.state import MarketStateAggregator


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
        self._security = APISecurityGuard(settings.security)
        self._auth = FyersAuthManager(settings.fyers, security_guard=self._security)
        self._client = FyersMarketClient(settings.fyers, auth_manager=self._auth, security_guard=self._security)
        self._aggregator = MarketStateAggregator()
        self._stream = FyersStream(
            settings.fyers,
            auth_manager=self._auth,
            security_guard=self._security,
            aggregator=self._aggregator,
        )
        self._fetcher = NSEOptionChainFetcher(settings.api)
        self._parser = OptionChainParser()
        self._validator = OptionChainValidator()
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
    def stream(self) -> FyersStream:
        """Expose the underlying websocket stream wrapper."""
        return self._stream

    @property
    def api_health(self):
        """Expose API health for UI and supervisory consumers."""
        return self._security.health_snapshot()

    @property
    def engine(self) -> LiveIntelligenceEngine:
        """Expose the live intelligence engine."""
        return self._engine

    @property
    def monitor(self):
        """Expose the live trade monitor."""
        return self._engine.monitor

    def refresh_once(self) -> IntelligenceReport:
        """Poll REST sources once and compute a fresh intelligence report."""
        self._security.validate_environment()
        snapshot = self._client.get_live_snapshot()
        self._aggregator.update_from_snapshot(snapshot)

        try:
            chain_result = self._fetcher.fetch_option_chain()
            self._security.validate_fresh_timestamp(
                chain_result.fetched_at_utc,
                max_age_seconds=self._settings.security.stale_option_chain_seconds,
                label="option_chain",
            )
            chain_frame = self._parser.parse(chain_result.payload, chain_result.fetched_at_utc)
            validated = self._validator.validate(chain_frame)
            self._aggregator.update_option_chain(validated.frame)
            self._security.register_rest_success()
        except Exception as exc:  # noqa: BLE001
            self._security.register_rest_failure(f"option_chain_refresh_failed:{exc}")
            logger.warning("Option-chain refresh failed; continuing with cached state", error=str(exc))
            if self._callbacks.on_error:
                self._callbacks.on_error(exc)

        snapshot_bundle = self._aggregator.build_snapshot()
        state = self._aggregator.build_market_state(snapshot_bundle)
        self._latest_state = state
        report = self._engine.process(snapshot_bundle, state=state)
        self._latest_report = report

        # Dynamically subscribe to nearby option tokens using the fresh snapshot
        try:
            self._update_dynamic_subscriptions(snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update dynamic subscriptions", error=str(exc))

        if self._callbacks.on_report:
            self._callbacks.on_report(report)
        return report

    def start_streaming(self) -> None:
        """Start websocket streaming when tokens or symbols are configured."""
        if self._security.safe_mode:
            raise APISecurityError("Safe mode is active; streaming is blocked")
        if self._settings.fyers.stream_symbols:
            self._stream.subscribe(list(self._settings.fyers.stream_symbols))
        elif self._settings.fyers.stream_tokens:
            self._stream.subscribe(list(self._settings.fyers.stream_tokens))
        self._stream.connect(threaded=True)

    def stop(self) -> None:
        """Stop websocket streaming cleanly."""
        self._stream.stop()

    def _update_dynamic_subscriptions(self, snapshot) -> None:
        if snapshot.nifty_spot.last_price is None or snapshot.option_instruments.empty:
            return

        spot = snapshot.nifty_spot.last_price
        atm_strike = round(spot / 50) * 50
        target_strikes = [atm_strike + offset for offset in range(-150, 200, 50)]

        df = snapshot.option_instruments.copy()
        if "expiry" not in df.columns or "strike" not in df.columns or "instrument_token" not in df.columns:
            return

        df["expiry_date"] = pd.to_datetime(df["expiry"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        future_expiries = df[df["expiry_date"] >= now.normalize()]
        if future_expiries.empty:
            future_expiries = df

        nearest_expiry = future_expiries["expiry_date"].min()
        relevant = future_expiries[
            (future_expiries["expiry_date"] == nearest_expiry)
            & (future_expiries["strike"].isin(target_strikes))
        ]
        if relevant.empty:
            relevant = future_expiries.head(10)

        # Register symbol-token mapping with stream
        for _, r in relevant.iterrows():
            sym = r.get("symbol") or r.get("tradingsymbol")
            tok = r.get("instrument_token")
            if pd.notna(sym) and pd.notna(tok):
                self._stream.register_symbol_token(str(sym), int(tok))

        tokens = set(relevant["instrument_token"].astype(int).tolist())
        tokens.update(self._settings.fyers.stream_tokens)

        symbols = [str(s) for s in relevant["symbol"].dropna().tolist()] if "symbol" in relevant.columns else []

        current_subs = self._stream.state.subscribed_tokens
        if tokens != current_subs:
            logger.info("Updating dynamic websocket subscriptions", new_count=len(tokens), added=len(tokens - current_subs))
            self._stream.subscribe(symbols or list(tokens))

    def _on_connect(self) -> None:
        try:
            self.refresh_once()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Initial runtime refresh failed", error=str(exc))
            if self._callbacks.on_error:
                self._callbacks.on_error(exc)

    def _on_tick_frame(self, frame: pd.DataFrame) -> None:
        data_received_at = datetime.now(UTC)
        self._aggregator.update_ticks(NormalizedTickBatch(frame=frame, row_count=len(frame), source="fyers"))
        try:
            snapshot = self._aggregator.build_snapshot()
            self._latest_state = self._aggregator.build_market_state(snapshot)
            self._latest_report = self._engine.process(snapshot, state=self._latest_state, data_received_at=data_received_at)
            if self._callbacks.on_report:
                self._callbacks.on_report(self._latest_report)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process live tick frame", error=str(exc))
            if self._callbacks.on_error:
                self._callbacks.on_error(exc)
