"""KiteTicker websocket streaming with reconnect, subscriptions, and tick dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Event
from typing import Any, Callable

import pandas as pd
from kiteconnect import KiteTicker
from loguru import logger

from broker.api_security import APISecurityError, APISecurityGuard
from broker.kite_auth import KiteAuthError, KiteAuthManager
from broker.normalizer import KiteTickNormalizer
from config.settings import KiteSettings


TickFrameHandler = Callable[[pd.DataFrame], None]
RawTickHandler = Callable[[list[dict[str, Any]]], None]


@dataclass(slots=True)
class StreamState:
    """Current websocket stream state and subscriptions."""

    connected: bool = False
    subscribed_tokens: set[int] = field(default_factory=set)
    reconnect_attempts: int = 0
    last_tick_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    duplicate_ticks: int = 0
    out_of_order_ticks: int = 0
    safe_mode: bool = False


class KiteStream:
    """Wrap KiteTicker with structured callbacks and subscription management."""

    def __init__(self, settings: KiteSettings, auth_manager: KiteAuthManager | None = None, security_guard: APISecurityGuard | None = None) -> None:
        self._settings = settings
        self._security_guard = security_guard
        self._auth_manager = auth_manager or KiteAuthManager(settings, security_guard=security_guard)
        access_token = self._auth_manager.ensure_access_token()

        if not settings.api_key:
            raise KiteAuthError("KITE_API_KEY is required")

        self._ticker = KiteTicker(
            settings.api_key,
            access_token,
            reconnect=settings.reconnect,
            reconnect_max_tries=settings.reconnect_max_tries,
            reconnect_max_delay=settings.reconnect_max_delay,
            connect_timeout=settings.connect_timeout,
        )
        self._normalizer = KiteTickNormalizer()
        self._state = StreamState(subscribed_tokens=set(settings.stream_tokens))
        self._connected_event = Event()
        self._stop_event = Event()

        self._tick_handlers: list[TickFrameHandler] = []
        self._raw_tick_handlers: list[RawTickHandler] = []
        self._connect_handlers: list[Callable[[], None]] = []
        self._close_handlers: list[Callable[[int | None, str | None], None]] = []

        self._ticker.on_connect = self._on_connect
        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_reconnect = self._on_reconnect
        self._ticker.on_noreconnect = self._on_noreconnect
        self._ticker.on_message = self._on_message
        self._ticker.on_order_update = self._on_order_update

    @property
    def state(self) -> StreamState:
        """Expose the current stream state."""
        return self._state

    def register_tick_handler(self, handler: TickFrameHandler) -> None:
        """Register a consumer for normalized tick frames."""
        self._tick_handlers.append(handler)

    def register_raw_tick_handler(self, handler: RawTickHandler) -> None:
        """Register a consumer for raw tick lists."""
        self._raw_tick_handlers.append(handler)

    def register_connect_handler(self, handler: Callable[[], None]) -> None:
        """Register a consumer for websocket connect events."""
        self._connect_handlers.append(handler)

    def register_close_handler(self, handler: Callable[[int | None, str | None], None]) -> None:
        """Register a consumer for websocket close events."""
        self._close_handlers.append(handler)

    def subscribe(self, instrument_tokens: list[int]) -> None:
        """Subscribe to instrument tokens and update local state."""
        if not instrument_tokens:
            return

        tokens = [int(token) for token in instrument_tokens]
        self._state.subscribed_tokens.update(tokens)
        if self._state.connected:
            self._ticker.subscribe(tokens)
            self._ticker.set_mode(self._resolve_mode(), tokens)
        logger.info("Updated subscription set", tokens=tokens)

    def unsubscribe(self, instrument_tokens: list[int]) -> None:
        """Unsubscribe from instrument tokens and update local state."""
        if not instrument_tokens:
            return

        tokens = [int(token) for token in instrument_tokens]
        self._state.subscribed_tokens.difference_update(tokens)
        if self._state.connected:
            self._ticker.unsubscribe(tokens)
        logger.info("Removed subscription set", tokens=tokens)

    def set_mode(self, instrument_tokens: list[int], mode: str | None = None) -> None:
        """Set websocket mode for subscribed tokens."""
        tokens = [int(token) for token in instrument_tokens]
        if not tokens:
            return

        resolved_mode = self._resolve_mode(mode)
        if self._state.connected:
            self._ticker.set_mode(resolved_mode, tokens)
        logger.info("Set tick mode", mode=resolved_mode, tokens=tokens)

    def connect(self, *, threaded: bool = True) -> None:
        """Open websocket connection using the official KiteTicker client."""
        if self._security_guard is not None and self._security_guard.safe_mode:
            raise KiteAuthError("Safe mode is active; websocket connection is blocked")
        logger.info(
            "Connecting KiteTicker",
            reconnect=self._settings.reconnect,
            thread_mode=threaded,
            subscriptions=len(self._state.subscribed_tokens),
        )
        self._ticker.connect(threaded=threaded, disable_ssl_verification=False)

    def stop(self) -> None:
        """Gracefully close the websocket connection and stop retries."""
        self._stop_event.set()
        try:
            self._ticker.stop_retry()
        except Exception as exc:  # noqa: BLE001
            logger.debug("stop_retry failed or was unnecessary", error=str(exc))
        try:
            self._ticker.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("close failed or was unnecessary", error=str(exc))

    def wait_until_stopped(self) -> None:
        """Block until stop is requested by the caller or callbacks."""
        self._stop_event.wait()

    def is_connected(self) -> bool:
        """Return websocket connection state."""
        return bool(self._ticker.is_connected())

    def resubscribe(self) -> None:
        """Re-subscribe to all current instrument tokens."""
        if self._state.subscribed_tokens:
            self._ticker.resubscribe()

    def _on_connect(self, ws: KiteTicker, _response: Any) -> None:
        self._state.connected = True
        self._state.safe_mode = bool(self._security_guard.safe_mode) if self._security_guard is not None else False
        self._connected_event.set()
        logger.info("KiteTicker connected")

        if self._state.subscribed_tokens:
            tokens = sorted(self._state.subscribed_tokens)
            ws.subscribe(tokens)
            ws.set_mode(self._resolve_mode(), tokens)
            logger.info("Applied initial subscription set", tokens=tokens)

        for handler in self._connect_handlers:
            try:
                handler()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Connect handler failed", error=str(exc))

    def _on_ticks(self, _ws: KiteTicker, ticks: list[dict[str, Any]]) -> None:
        logger.debug("Received tick batch", rows=len(ticks))
        if self._security_guard is not None:
            self._security_guard.register_heartbeat()

        for handler in self._raw_tick_handlers:
            try:
                handler(ticks)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Raw tick handler failed", error=str(exc))

        normalized = self._normalizer.normalize_ticks(ticks)
        if normalized.row_count == 0:
            return

        self._state.last_tick_at = datetime.now(UTC)

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

        for handler in self._tick_handlers:
            try:
                handler(normalized.frame)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tick handler failed", error=str(exc))

    def _on_close(self, _ws: KiteTicker, code: int | None, reason: str | None) -> None:
        self._state.connected = False
        self._state.safe_mode = bool(self._security_guard.safe_mode) if self._security_guard is not None else self._state.safe_mode
        logger.warning("KiteTicker closed", code=code, reason=reason)
        for handler in self._close_handlers:
            try:
                handler(code, reason)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Close handler failed", error=str(exc))

    def _on_error(self, _ws: KiteTicker, code: int | None, reason: str | None) -> None:
        if self._security_guard is not None:
            self._security_guard.register_reconnect()
        logger.error("KiteTicker error", code=code, reason=reason)

    def _on_reconnect(self, _ws: KiteTicker, attempts_count: int) -> None:
        self._state.reconnect_attempts = attempts_count
        if self._security_guard is not None:
            self._security_guard.register_reconnect()
        logger.warning("KiteTicker reconnect attempt", attempts=attempts_count)

    def _on_noreconnect(self, _ws: KiteTicker) -> None:
        if self._security_guard is not None:
            self._security_guard.mark_safe_mode("Websocket exhausted reconnect attempts")
            self._state.safe_mode = True
        logger.error("KiteTicker exhausted reconnect attempts")

    def _on_message(self, _ws: KiteTicker, _payload: Any, is_binary: bool) -> None:
        if self._security_guard is not None:
            self._security_guard.register_heartbeat()
            self._state.last_heartbeat_at = datetime.now(UTC)
        logger.debug("KiteTicker message received", is_binary=is_binary)

    def _on_order_update(self, _ws: KiteTicker, data: dict[str, Any]) -> None:
        logger.info("Kite order update received", keys=list(data.keys()))

    def _resolve_mode(self, mode: str | None = None) -> Any:
        """Resolve a human-readable mode into the official KiteTicker constant."""
        resolved = (mode or self._settings.stream_mode or "full").strip().lower()
        if resolved == "ltp":
            return self._ticker.MODE_LTP
        if resolved == "quote":
            return self._ticker.MODE_QUOTE
        return self._ticker.MODE_FULL
