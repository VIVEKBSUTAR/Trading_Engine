"""Fyers API V3 authentication helpers and secure token persistence."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fyers_apiv3 import fyersModel
from loguru import logger

from broker.api_security import APISecurityError, APISecurityGuard
from config.settings import AppSettings, FyersSettings


class FyersAuthError(RuntimeError):
    """Raised when Fyers authentication or token persistence fails."""


@dataclass(slots=True)
class FyersSessionResult:
    """Session exchange response returned by the official Fyers API."""

    access_token: str
    raw_response: dict[str, Any]


class FyersAuthManager:
    """Owns Fyers API V3 auth lifecycle and daily access-token persistence."""

    def __init__(self, settings: FyersSettings, security_guard: APISecurityGuard | None = None) -> None:
        if not settings.client_id:
            raise FyersAuthError("FYERS_CLIENT_ID is required")

        self._settings = settings
        self._security_guard = security_guard
        self._session: fyersModel.SessionModel | None = None

        resolved_token = self.resolve_access_token()
        if resolved_token:
            self._settings.access_token = resolved_token

    @property
    def settings(self) -> FyersSettings:
        """Expose current Fyers settings."""
        return self._settings

    def _get_session_model(self) -> fyersModel.SessionModel:
        """Build or return the underlying Fyers SessionModel."""
        if self._session is None:
            if not self._settings.secret_key:
                raise FyersAuthError("FYERS_SECRET_KEY is required to generate auth URL or exchange token")
            self._session = fyersModel.SessionModel(
                client_id=self._settings.client_id,
                secret_key=self._settings.secret_key,
                redirect_uri=self._settings.redirect_uri or "https://127.0.0.1",
                response_type="code",
                grant_type="authorization_code",
            )
        return self._session

    def generate_login_url(self) -> str:
        """Generate the official Fyers authorization URL using client_id and redirect_uri."""
        if self._security_guard is not None:
            self._security_guard.validate_environment()

        session = self._get_session_model()
        url = session.generate_authcode()
        logger.info("Generated Fyers login URL", redirect_uri=self._settings.redirect_uri)
        return url

    def login_url(self) -> str:
        """Alias for generate_login_url."""
        return self.generate_login_url()

    def exchange_auth_code(self, auth_code: str | None = None, *, persist: bool = True) -> str:
        """Exchange authorization code copied from browser for a daily access_token."""
        code = (auth_code or self._settings.auth_code or "").strip()
        if not code:
            raise FyersAuthError("Authorization code is required for exchange")

        if self._security_guard is not None:
            self._security_guard.validate_environment()
            if not self._security_guard.should_allow_execution():
                raise APISecurityError("Token exchange blocked by security policy")

        session = self._get_session_model()
        session.set_token(code)

        try:
            response = session.generate_token()
        except Exception as exc:  # noqa: BLE001
            raise FyersAuthError(f"Failed to generate token from Fyers: {exc}") from exc

        if not isinstance(response, dict):
            raise FyersAuthError(f"Invalid token response from Fyers: {response!r}")

        # Check response status
        status = str(response.get("s", "")).lower()
        if status != "ok" and "access_token" not in response:
            message = response.get("message") or response.get("errMsg") or str(response)
            code_num = response.get("code")
            raise FyersAuthError(f"Fyers token exchange failed (code={code_num}): {message}")

        access_token = response.get("access_token")
        if not access_token or not isinstance(access_token, str):
            raise FyersAuthError(f"Access token missing in Fyers response: {response}")

        cleaned_token = access_token.strip()
        self._settings.access_token = cleaned_token

        if persist:
            self.persist_access_token(cleaned_token)

        logger.info("Exchanged auth code for Fyers access token successfully")
        return cleaned_token

    def exchange_request_token(self, request_token: str | None = None) -> FyersSessionResult:
        """Backward-compatibility wrapper returning FyersSessionResult."""
        token = self.exchange_auth_code(request_token, persist=True)
        return FyersSessionResult(access_token=token, raw_response={"access_token": token, "s": "ok"})

    def resolve_access_token(self) -> str | None:
        """Resolve access token from settings, env, or secure token store."""
        if self._settings.access_token and self._settings.access_token.strip():
            return self._settings.access_token.strip()

        stored = self.load_access_token()
        if stored:
            self._settings.access_token = stored
        return stored

    def load_access_token(self) -> str | None:
        """Load access token from the secure token store file if present."""
        token_path = self._settings.token_store_path
        if token_path is None or not token_path.exists():
            return None

        token = token_path.read_text(encoding="utf-8").strip()
        return token or None

    def persist_access_token(self, access_token: str) -> Path:
        """Persist access token with restrictive filesystem permissions (0600)."""
        token_path = self._settings.token_store_path
        if token_path is None:
            raise FyersAuthError("Token store path is not configured")

        token_path.parent.mkdir(parents=True, exist_ok=True)

        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        try:
            fd = os.open(token_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(access_token.strip() + "\n")
        except OSError as exc:
            raise FyersAuthError(f"Failed to persist access token: {exc}") from exc

        try:
            token_path.chmod(0o600)
        except OSError:
            logger.warning("Unable to enforce 0600 permissions on access-token file", path=str(token_path))

        logger.info("Persisted Fyers access token securely", path=str(token_path))
        return token_path

    def ensure_access_token(self) -> str:
        """Return the active access token or raise FyersAuthError."""
        token = self.resolve_access_token()
        if not token:
            raise FyersAuthError(
                "No Fyers access token available. Please generate login URL, authorize in browser, "
                "and exchange the authorization code."
            )
        return token

    def validate_session(self) -> dict[str, Any]:
        """Validate session state using the official profile endpoint."""
        token = self.ensure_access_token()
        fyers = fyersModel.FyersModel(
            client_id=self._settings.client_id or "",
            token=token,
            is_async=False,
            log_path=str(self._settings.token_store_path.parent) if self._settings.token_store_path else None,
        )
        try:
            response = fyers.get_profile()
        except Exception as exc:  # noqa: BLE001
            raise FyersAuthError(f"Fyers profile check failed: {exc}") from exc

        if not isinstance(response, dict) or response.get("s") != "ok":
            msg = response.get("message") if isinstance(response, dict) else str(response)
            raise FyersAuthError(f"Session validation failed: {msg}")

        data = response.get("data")
        return data if isinstance(data, dict) else response


def main() -> None:
    """CLI script to generate login URL or exchange authorization code."""
    parser = argparse.ArgumentParser(description="Fyers API V3 Authentication CLI")
    parser.add_argument("--url-only", action="store_true", help="Only generate and print login URL")
    parser.add_argument("--code", type=str, default=None, help="Authorization code copied from browser redirect")
    args = parser.parse_args()

    app_settings = AppSettings.from_env()
    manager = FyersAuthManager(app_settings.fyers)

    login_url = manager.generate_login_url()
    print("\n" + "=" * 70)
    print("FYERS API V3 LOGIN URL:")
    print("=" * 70)
    print(login_url)
    print("=" * 70 + "\n")

    if args.url_only:
        return

    code = args.code
    if not code:
        try:
            code = input("Paste the authorization code from browser redirect: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return

    if not code:
        print("No authorization code provided.")
        return

    token = manager.exchange_auth_code(code)
    print("\nAccess Token exchanged and persisted successfully!")
    print(f"Token: {token[:10]}...{token[-5:]}")


if __name__ == "__main__":
    main()
