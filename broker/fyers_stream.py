"""Fyers DataSocket websocket streaming with subscriptions, normalization, and aggregator routing."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure project root and src/ are on sys.path when this module is loaded
# directly rather than via the installed package entry-point.
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Event, Thread
from typing import Any

import pandas as pd
from fyers_apiv3.FyersWebsocket import data_ws
from loguru import logger

from broker.api_security import APISecurityError, APISecurityGuard
from broker.fyers_auth import FyersAuthError, FyersAuthManager
from broker.normalizer import FyersTickNormalizer, NormalizedTickBatch
from config.settings import FyersSettings

try:
    from trading_engine.intelligence.state import MarketStateAggregator
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from trading_engine.intelligence.state import MarketStateAggregator

TickFrameHandler = Callable[[pd.DataFrame], None]
RawTickHandler = Callable[[Any], None]


@dataclass(slots=True)
class StreamState:
    """Current Fyers DataSocket websocket stream state and subscriptions."""

    connected: bool = False
    subscribed_tokens: set[int] = field(default_factory=set)
    subscribed_symbols: set[str] = field(default_factory=set)
    reconnect_attempts: int = 0
    last_tick_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    duplicate_ticks: int = 0
    out_of_order_ticks: int = 0
    safe_mode: bool = False


class FyersStream:
    """Threaded websocket client wrapping Fyers DataSocket."""

    def __init__(
        self,
        settings: FyersSettings,
        auth_manager: FyersAuthManager | None = None,
        security_guard: APISecurityGuard | None = None,
        aggregator: MarketStateAggregator | None = None,
        normalizer: FyersTickNormalizer | None = None,
    ) -> None:
        self._settings = settings
        self._security_guard = security_guard
        self._aggregator = aggregator
        self._auth_manager = auth_manager or FyersAuthManager(settings, security_guard=security_guard)

        access_token = self._auth_manager.resolve_access_token()
        if not access_token:
            access_token = self._settings.access_token
        if not access_token:
            raise FyersAuthError("Fyers access token is required for streaming")

        self._normalizer = normalizer or FyersTickNormalizer()
        self._token_to_symbol: dict[int, str] = {}
        self._symbol_to_token: dict[str, int] = {}

        initial_tokens = set(settings.stream_tokens)
        initial_symbols = set(settings.stream_symbols)
        self._state = StreamState(
            subscribed_tokens=initial_tokens,
            subscribed_symbols=initial_symbols,
        )

        self._connected_event = Event()
        self._stop_event = Event()
        self._thread: Thread | None = None

        self._tick_handlers: list[TickFrameHandler] = []
        self._raw_tick_handlers: list[RawTickHandler] = []
        self._connect_handlers: list[Callable[[], None]] = []
        self._close_handlers: list[Callable[[int | None, str | None], None]] = []

        log_path = str(settings.token_store_path.parent) if settings.token_store_path else None
        self._socket = data_ws.FyersDataSocket(
            access_token=access_token,
            log_path=log_path,
            litemode=settings.litemode,
            reconnect=settings.reconnect,
            reconnect_retry=settings.reconnect_max_tries,
            on_connect=self._on_connect,
            on_close=self._on_close,
            on_error=self._on_error,
            on_message=self._on_message,
        )

    @property
    def state(self) -> StreamState:
        """Expose current websocket stream state."""
        return self._state

    @property
    def aggregator(self) -> MarketStateAggregator | None:
        """Expose the market state aggregator."""
        return self._aggregator

    @aggregator.setter
    def aggregator(self, agg: MarketStateAggregator | None) -> None:
        self._aggregator = agg

    @property
    def normalizer(self) -> FyersTickNormalizer:
        """Expose the tick normalizer."""
        return self._normalizer

    def register_symbol_token(self, symbol: str, token: int) -> None:
        """Register bidirectional mapping between symbol and instrument token."""
        t_int = int(token)
        s_clean = str(symbol).strip()
        self._symbol_to_token[s_clean] = t_int
        self._token_to_symbol[t_int] = s_clean
        self._normalizer.register_symbol_token(s_clean, t_int)

    def register_tick_handler(self, handler: TickFrameHandler) -> None:
        """Register consumer for normalized tick batches."""
        self._tick_handlers.append(handler)

    def register_raw_tick_handler(self, handler: RawTickHandler) -> None:
        """Register consumer for raw websocket messages."""
        self._raw_tick_handlers.append(handler)

    def register_connect_handler(self, handler: Callable[[], None]) -> None:
        """Register consumer for websocket connect events."""
        self._connect_handlers.append(handler)

    def register_close_handler(self, handler: Callable[[int | None, str | None], None]) -> None:
        """Register consumer for websocket close events."""
        self._close_handlers.append(handler)

    def subscribe(self, symbols_or_tokens: list[str | int] | str | int) -> None:
        """Subscribe to specific instrument tokens or symbols when selected by the engine."""
        items = [symbols_or_tokens] if isinstance(symbols_or_tokens, (str, int)) else list(symbols_or_tokens)
        if not items:
            return

        symbols_to_sub: list[str] = []
        for item in items:
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                token_int = int(item)
                self._state.subscribed_tokens.add(token_int)
                # Map token to known symbol if available
                sym = self._token_to_symbol.get(token_int)
                if sym:
                    self._state.subscribed_symbols.add(sym)
                    symbols_to_sub.append(sym)
            else:
                sym_str = str(item).strip()
                if sym_str:
                    self._state.subscribed_symbols.add(sym_str)
                    symbols_to_sub.append(sym_str)
                    tok = self._symbol_to_token.get(sym_str)
                    if tok:
                        self._state.subscribed_tokens.add(tok)

        if symbols_to_sub and self._state.connected:
            try:
                self._socket.subscribe(symbols=symbols_to_sub, data_type="SymbolUpdate")
                logger.info("Subscribed to Fyers symbols", symbols=symbols_to_sub)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fyers socket subscribe call failed", symbols=symbols_to_sub, error=str(exc))

    def unsubscribe(self, symbols_or_tokens: list[str | int] | str | int) -> None:
        """Unsubscribe from specific instrument tokens or symbols."""
        items = [symbols_or_tokens] if isinstance(symbols_or_tokens, (str, int)) else list(symbols_or_tokens)
        if not items:
            return

        symbols_to_unsub: list[str] = []
        for item in items:
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                token_int = int(item)
                self._state.subscribed_tokens.discard(token_int)
                sym = self._token_to_symbol.get(token_int)
                if sym:
                    self._state.subscribed_symbols.discard(sym)
                    symbols_to_unsub.append(sym)
            else:
                sym_str = str(item).strip()
                self._state.subscribed_symbols.discard(sym_str)
                symbols_to_unsub.append(sym_str)
                tok = self._symbol_to_token.get(sym_str)
                if tok:
                    self._state.subscribed_tokens.discard(tok)

        if symbols_to_unsub and self._state.connected:
            try:
                self._socket.unsubscribe(symbols=symbols_to_unsub)
                logger.info("Unsubscribed from Fyers symbols", symbols=symbols_to_unsub)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fyers socket unsubscribe failed", error=str(exc))

    def connect(self, *, threaded: bool = True) -> None:
        """Connect to Fyers DataSocket. If threaded is True, launch connection in background thread."""
        if self._security_guard is not None and self._security_guard.safe_mode:
            raise FyersAuthError("Safe mode is active; websocket connection is blocked")

        logger.info(
            "Connecting Fyers DataSocket",
            reconnect=self._settings.reconnect,
            thread_mode=threaded,
            subscriptions=len(self._state.subscribed_symbols),
        )

        if threaded:
            self._thread = Thread(target=self._run_socket, daemon=True, name="FyersDataSocketThread")
            self._thread.start()
        else:
            self._run_socket()

    def _run_socket(self) -> None:
        try:
            self._socket.connect()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fyers DataSocket connection error", error=str(exc))
            self._on_error(exc)

    def stop(self) -> None:
        """Gracefully close the websocket connection."""
        self._stop_event.set()
        try:
            self._socket.close_connection()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fyers socket close failed or already closed", error=str(exc))
        self._state.connected = False

    def wait_until_stopped(self) -> None:
        """Block until stop is requested."""
        self._stop_event.wait()

    def is_connected(self) -> bool:
        """Return websocket connection status."""
        return self._state.connected

    def _on_connect(self) -> None:
        self._state.connected = True
        self._state.safe_mode = bool(self._security_guard.safe_mode) if self._security_guard is not None else False
        self._connected_event.set()
        logger.info("Fyers DataSocket connected successfully")

        # Apply subscriptions if present
        if self._state.subscribed_symbols:
            symbols = sorted(self._state.subscribed_symbols)
            try:
                self._socket.subscribe(symbols=symbols, data_type="SymbolUpdate")
                logger.info("Applied initial subscription set", symbols=symbols)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to apply initial subscriptions", error=str(exc))

        for handler in self._connect_handlers:
            try:
                handler()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Connect handler failed", error=str(exc))

    def _on_message(self, message: Any) -> None:
        """Process incoming websocket ticks, normalize, and push to aggregator buffer."""
        if self._security_guard is not None:
            self._security_guard.register_heartbeat()
            self._state.last_heartbeat_at = datetime.now(UTC)

        for raw_handler in self._raw_tick_handlers:
            try:
                raw_handler(message)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Raw tick handler failed", error=str(exc))

        normalized: NormalizedTickBatch = self._normalizer.normalize_ticks(message, source="fyers")
        if normalized.row_count == 0:
            return

        self._state.last_tick_at = datetime.now(UTC)

        # Push bid and ask prices directly to the market state aggregator's option token buffer
        if self._aggregator is not None:
            for _, row in normalized.frame.iterrows():
                tok = row.get("instrument_token")
                bid = row.get("bid_price")
                ask = row.get("ask_price")
                lp = row.get("last_price")
                ts = row.get("timestamp")

                if pd.notna(tok) and pd.notna(bid) and pd.notna(ask):
                    try:
                        token_int = int(tok)
                        bid_float = float(bid)
                        ask_float = float(ask)
                        lp_float = float(lp) if pd.notna(lp) else None
                        ts_val = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else None
                        self._aggregator.push_option_tick(
                            instrument_token=token_int,
                            bid=bid_float,
                            ask=ask_float,
                            timestamp=ts_val,
                            last_price=lp_float,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Failed pushing tick to aggregator buffer", error=str(exc))

        if self._security_guard is not None:
            try:
                notes = self._security_guard.validate_stream_frame(normalized.frame)
                self._state.duplicate_ticks = self._security_guard.health_snapshot().duplicate_ticks
                self._state.out_of_order_ticks = self._security_guard.health_snapshot().out_of_order_ticks
                self._state.safe_mode = self._security_guard.safe_mode
                logger.debug("Stream integrity notes", notes=notes)
            except APISecurityError as exc:
                self._state.safe_mode = True
                logger.error("Stream integrity validation failed", error=str(exc))
                self.stop()
                return

        for tick_handler in self._tick_handlers:
            try:
                tick_handler(normalized.frame)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tick handler failed", error=str(exc))

    def _on_close(self, code: int | None = None, reason: str | None = None) -> None:
        self._state.connected = False
        logger.warning("Fyers DataSocket closed", code=code, reason=reason)
        for handler in self._close_handlers:
            try:
                handler(code, reason)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Close handler failed", error=str(exc))

    def _on_error(self, error: Any) -> None:
        if self._security_guard is not None:
            self._security_guard.register_reconnect()
        logger.error("Fyers DataSocket error", error=str(error))
