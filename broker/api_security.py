"""Shared API security and health monitoring helpers for live market connectivity."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import re
from typing import Any

import pandas as pd

from config.settings import SecuritySettings


class APISecurityError(RuntimeError):
    """Raised when API validation, freshness, or safety checks fail."""


@dataclass(slots=True)
class APIHealthSnapshot:
    """Current health and safety status for the API layer."""

    environment: str
    safe_mode: bool
    execution_allowed: bool
    rest_requests: int
    rest_failures: int
    rejected_requests: int
    reconnect_attempts: int
    duplicate_ticks: int
    out_of_order_ticks: int
    stale_rest_count: int
    stale_stream_count: int
    rate_limit_hits: int
    last_request_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    last_stream_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    notes: list[str] = field(default_factory=list)


class APISecurityGuard:
    """Track request rate, payload freshness, and safe-mode conditions."""

    _INSTRUMENT_PATTERN = re.compile(r"^[A-Z]{2,10}:[A-Z0-9 .&_\-/]+$")

    def __init__(self, settings: SecuritySettings) -> None:
        self._settings = settings
        self._request_timestamps: deque[datetime] = deque()
        self._reconnect_timestamps: deque[datetime] = deque()
        self._safe_mode = False
        self._rest_requests = 0
        self._rest_failures = 0
        self._rejected_requests = 0
        self._rate_limit_hits = 0
        self._duplicate_ticks = 0
        self._out_of_order_ticks = 0
        self._stale_rest_count = 0
        self._stale_stream_count = 0
        self._reconnect_attempts = 0
        self._last_request_at: datetime | None = None
        self._last_success_at: datetime | None = None
        self._last_error_at: datetime | None = None
        self._last_stream_at: datetime | None = None
        self._last_heartbeat_at: datetime | None = None
        self._notes: list[str] = []

    @property
    def safe_mode(self) -> bool:
        return self._safe_mode

    @property
    def settings(self) -> SecuritySettings:
        return self._settings

    def mark_safe_mode(self, reason: str) -> None:
        self._safe_mode = True
        self._notes.append(reason)

    def clear_safe_mode(self) -> None:
        if self._settings.environment in {"live", "paper"} and self._settings.allow_live_execution:
            self._safe_mode = False

    def should_allow_execution(self, *, state_quality: float | None = None, tradeable: bool = True) -> bool:
        if self._safe_mode:
            return False
        if not self._settings.allow_live_execution:
            return False
        if self._settings.require_manual_execution_approval:
            return False
        if not tradeable:
            return False
        if state_quality is not None and state_quality < 0.55:
            return False
        return True

    def validate_environment(self) -> None:
        if self._settings.environment not in {"development", "paper", "live"}:
            raise APISecurityError(f"Unsupported execution environment: {self._settings.environment}")

    def validate_instrument_identifier(self, instrument: str) -> None:
        if not instrument or not self._INSTRUMENT_PATTERN.match(instrument.strip().upper()):
            raise APISecurityError(f"Invalid instrument identifier: {instrument!r}")

    def validate_instrument_tokens(self, tokens: list[int], *, max_count: int | None = None) -> list[int]:
        if not tokens:
            return []
        unique_tokens = sorted({int(token) for token in tokens if int(token) > 0})
        if len(unique_tokens) != len(tokens):
            self._rejected_requests += 1
            raise APISecurityError("Instrument tokens contained duplicates or invalid values")
        if max_count is not None and len(unique_tokens) > max_count:
            self._rejected_requests += 1
            raise APISecurityError("Instrument token batch exceeds the configured limit")
        return unique_tokens

    def register_rest_request(self, api_name: str, endpoint: str) -> None:
        now = datetime.now(UTC)
        self._last_request_at = now
        self._rest_requests += 1
        self._request_timestamps.append(now)
        self._trim_old(self._request_timestamps, timedelta(minutes=1))

        if self._safe_mode:
            self._rejected_requests += 1
            raise APISecurityError(f"Safe mode active; blocked API request: {api_name}:{endpoint}")

        if len(self._request_timestamps) > max(self._settings.api_requests_per_minute, 1):
            self._rate_limit_hits += 1
            self.mark_safe_mode("API request rate limit exceeded")
            raise APISecurityError("API request rate limit exceeded")

    def register_rest_success(self) -> None:
        self._last_success_at = datetime.now(UTC)

    def register_rest_failure(self, reason: str) -> None:
        self._rest_failures += 1
        self._last_error_at = datetime.now(UTC)
        self._notes.append(reason)
        if self._rest_failures >= 3:
            self.mark_safe_mode("Repeated API failures")

    def validate_fresh_timestamp(self, timestamp: datetime | None, *, max_age_seconds: int, label: str) -> None:
        if timestamp is None:
            self._stale_rest_count += 1
            self.mark_safe_mode(f"Missing timestamp for {label}")
            raise APISecurityError(f"Missing timestamp for {label}")

        now = datetime.now(UTC)
        normalized = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)
        age = now - normalized.astimezone(UTC)
        if age.total_seconds() > max_age_seconds:
            self._stale_rest_count += 1
            self.mark_safe_mode(f"Stale {label}: age={age.total_seconds():.1f}s")
            raise APISecurityError(f"Stale {label}: {age.total_seconds():.1f}s old")

    def validate_quote_payload(self, quote: dict[str, Any], *, instrument: str) -> None:
        if not isinstance(quote, dict):
            raise APISecurityError(f"Invalid quote payload for {instrument}")
        self.validate_instrument_identifier(instrument)
        if quote.get("last_price") is None:
            self.mark_safe_mode(f"Missing last price for {instrument}")
            raise APISecurityError(f"Missing last price for {instrument}")

    def validate_stream_frame(self, frame: pd.DataFrame) -> list[str]:
        if frame is None or frame.empty:
            self._stale_stream_count += 1
            self.mark_safe_mode("Empty websocket tick frame")
            raise APISecurityError("Empty websocket tick frame")

        if "timestamp" not in frame.columns:
            self._stale_stream_count += 1
            self.mark_safe_mode("Tick frame missing timestamp column")
            raise APISecurityError("Tick frame missing timestamp column")

        working = frame.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        working = working.dropna(subset=["timestamp"])
        if working.empty:
            self._stale_stream_count += 1
            self.mark_safe_mode("Tick frame timestamps could not be parsed")
            raise APISecurityError("Tick frame timestamps could not be parsed")

        ts = working["timestamp"]
        if not ts.is_monotonic_increasing:
            out_of_order = int((ts.diff().dropna() < pd.Timedelta(0)).sum())
            self._out_of_order_ticks += out_of_order
            self._notes.append(f"out_of_order_ticks={out_of_order}")
            if out_of_order >= self._settings.max_out_of_order_ticks:
                self.mark_safe_mode("Out-of-order tick threshold exceeded")

        subset = [column for column in ("instrument_token", "timestamp", "last_price") if column in working.columns]
        duplicates = int(working.duplicated(subset=subset, keep="last").sum()) if subset else 0
        self._duplicate_ticks += duplicates
        if duplicates >= self._settings.max_duplicate_ticks:
            self.mark_safe_mode("Duplicate tick threshold exceeded")

        latest_tick = ts.max().to_pydatetime()
        self._last_stream_at = latest_tick
        self._last_heartbeat_at = datetime.now(UTC)
        stale_age = (datetime.now(UTC) - latest_tick.astimezone(UTC)).total_seconds()
        if stale_age > self._settings.stale_stream_seconds:
            self._stale_stream_count += 1
            self.mark_safe_mode(f"Stale websocket frame: age={stale_age:.1f}s")
            raise APISecurityError(f"Stale websocket frame: {stale_age:.1f}s old")

        return [f"duplicate_ticks={duplicates}", f"out_of_order_ticks={self._out_of_order_ticks}"]

    def register_heartbeat(self) -> None:
        self._last_heartbeat_at = datetime.now(UTC)

    def register_reconnect(self) -> None:
        now = datetime.now(UTC)
        self._reconnect_attempts += 1
        self._reconnect_timestamps.append(now)
        self._trim_old(self._reconnect_timestamps, timedelta(seconds=max(self._settings.reconnect_cooldown_seconds, 1)))
        if len(self._reconnect_timestamps) > self._settings.max_reconnect_attempts:
            self.mark_safe_mode("Reconnect storm detected")

    def health_snapshot(self) -> APIHealthSnapshot:
        return APIHealthSnapshot(
            environment=self._settings.environment,
            safe_mode=self._safe_mode,
            execution_allowed=self.should_allow_execution(),
            rest_requests=self._rest_requests,
            rest_failures=self._rest_failures,
            rejected_requests=self._rejected_requests,
            reconnect_attempts=self._reconnect_attempts,
            duplicate_ticks=self._duplicate_ticks,
            out_of_order_ticks=self._out_of_order_ticks,
            stale_rest_count=self._stale_rest_count,
            stale_stream_count=self._stale_stream_count,
            rate_limit_hits=self._rate_limit_hits,
            last_request_at=self._last_request_at,
            last_success_at=self._last_success_at,
            last_error_at=self._last_error_at,
            last_stream_at=self._last_stream_at,
            last_heartbeat_at=self._last_heartbeat_at,
            notes=list(self._notes[-10:]),
        )

    @staticmethod
    def _trim_old(queue: deque[datetime], window: timedelta) -> None:
        cutoff = datetime.now(UTC) - window
        while queue and queue[0] < cutoff:
            queue.popleft()