"""Kite Connect authentication helpers and secure token persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException
from loguru import logger

from broker.api_security import APISecurityError, APISecurityGuard
from config.settings import KiteSettings


class KiteAuthError(RuntimeError):
    """Raised when Kite authentication or token persistence fails."""


@dataclass(slots=True)
class KiteSessionResult:
    """Session exchange response returned by the official Kite API."""

    access_token: str
    raw_response: dict[str, Any]


class KiteAuthManager:
    """Owns KiteConnect auth lifecycle and access-token persistence."""

    def __init__(self, settings: KiteSettings, security_guard: APISecurityGuard | None = None) -> None:
        if not settings.api_key:
            raise KiteAuthError("KITE_API_KEY is required")

        self._settings = settings
        self._security_guard = security_guard
        self._kite = KiteConnect(api_key=settings.api_key)
        self._kite.set_session_expiry_hook(self._on_session_expired)

        resolved_token = self.resolve_access_token()
        if resolved_token:
            self._kite.set_access_token(resolved_token)

    @property
    def kite(self) -> KiteConnect:
        """Expose the underlying KiteConnect client."""
        return self._kite

    def login_url(self) -> str:
        """Return the official Kite login URL."""
        return self._kite.login_url()

    def resolve_access_token(self) -> str | None:
        """Resolve access token from env or secure token store."""
        if self._settings.access_token:
            return self._settings.access_token.strip()

        stored = self.load_access_token()
        if stored:
            self._settings.access_token = stored
        return stored

    def load_access_token(self) -> str | None:
        """Load access token from the secure token store if present."""
        token_path = self._settings.token_store_path
        if token_path is None or not token_path.exists():
            return None

        token = token_path.read_text(encoding="utf-8").strip()
        return token or None

    def persist_access_token(self, access_token: str) -> Path:
        """Persist access token with restrictive filesystem permissions."""
        token_path = self._settings.token_store_path
        if token_path is None:
            raise KiteAuthError("Token store path is not configured")

        token_path.parent.mkdir(parents=True, exist_ok=True)

        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        try:
            fd = os.open(token_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(access_token.strip() + "\n")
        except OSError as exc:
            raise KiteAuthError(f"Failed to persist access token: {exc}") from exc

        try:
            token_path.chmod(0o600)
        except OSError:
            logger.warning("Unable to enforce 0600 permissions on access-token file", path=str(token_path))

        logger.info("Persisted Kite access token securely", path=str(token_path))
        return token_path

    def exchange_request_token(self, request_token: str | None = None) -> KiteSessionResult:
        """Exchange a request token for an access token using official Kite API."""
        token = (request_token or self._settings.request_token or "").strip()
        if not token:
            raise KiteAuthError("A request token is required for session exchange")
        if not self._settings.api_secret:
            raise KiteAuthError("KITE_API_SECRET is required for request-token exchange")

        try:
            if self._security_guard is not None:
                self._security_guard.register_rest_request("kite", "generate_session")
            session = self._kite.generate_session(token, self._settings.api_secret)
            if self._security_guard is not None:
                self._security_guard.register_rest_success()
        except Exception as exc:  # noqa: BLE001 - official client raises its own exceptions
            if self._security_guard is not None:
                self._security_guard.register_rest_failure(f"session_exchange_failed:{exc}")
            raise KiteAuthError(f"Kite session exchange failed: {exc}") from exc

        access_token = session.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise KiteAuthError("Kite session exchange did not return a valid access token")

        self._settings.access_token = access_token
        self._kite.set_access_token(access_token)
        self.persist_access_token(access_token)

        logger.info("Kite session exchange complete")
        return KiteSessionResult(access_token=access_token, raw_response=session)

    def ensure_access_token(self) -> str:
        """Return a usable access token or raise a clear configuration error."""
        access_token = self.resolve_access_token()
        if not access_token:
            raise KiteAuthError("No Kite access token available")

        self._kite.set_access_token(access_token)
        return access_token

    def _on_session_expired(self, *_args: Any, **_kwargs: Any) -> None:
        """Callback invoked by KiteConnect on token expiry / session errors."""
        logger.warning("Kite session expired or became invalid")

    def validate_session(self) -> dict[str, Any]:
        """Validate the currently authenticated session using the official profile API."""
        try:
            if self._security_guard is not None:
                self._security_guard.register_rest_request("kite", "profile")
            profile = self._kite.profile()
            if self._security_guard is not None:
                self._security_guard.register_rest_success()
        except TokenException as exc:
            if self._security_guard is not None:
                self._security_guard.register_rest_failure(f"token_validation_failed:{exc}")
            raise KiteAuthError(f"Kite session validation failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            if self._security_guard is not None:
                self._security_guard.register_rest_failure(f"profile_request_failed:{exc}")
            raise KiteAuthError(f"Kite profile request failed: {exc}") from exc

        if not isinstance(profile, dict):
            raise KiteAuthError("Unexpected profile response type")

        logger.info("Kite session validated", user_id=profile.get("user_id"))
        return profile
