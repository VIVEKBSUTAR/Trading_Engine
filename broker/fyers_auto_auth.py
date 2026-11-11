"""Headless daily FYERS SSO login; never print credentials or response bodies.

The SSO endpoints are not a documented stable public API. Fail closed on any
unexpected response; validate with the account before relying on unattended use.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import os
import secrets
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pyotp
import requests
from dotenv import dotenv_values, load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import FyersSettings

SSO = "https://api-t2.fyers.in/vagator/v2"
API = "https://api-t1.fyers.in/api/v3"


class AutoAuthError(RuntimeError):
    """Redacted authentication failure safe for operational logs."""


def post(session: requests.Session, url: str, payload: dict, *, headers=None, redirect=False) -> dict:
    """Never follow SSO redirects or include sensitive broker errors in logs."""
    response = session.post(url, json=payload, headers=headers, timeout=20, allow_redirects=False)
    allowed = {200, 308} if redirect else {200}
    if response.status_code not in allowed:
        raise AutoAuthError(f"Authentication step failed (HTTP {response.status_code})")
    data = response.json()
    if not isinstance(data, dict) or str(data.get("s", "")).lower() == "error":
        raise AutoAuthError("Authentication step rejected by Fyers")
    return data


def required(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AutoAuthError(f"Authentication response missing {key}")
    return value.strip()


def authenticate(settings: FyersSettings, session: requests.Session) -> str:
    settings.validate_headless_auth()
    def encoded(value: str) -> str:
        return base64.b64encode(value.encode()).decode()
    login = post(session, f"{SSO}/send_login_otp_v2", {
        "fy_id": encoded(settings.user_id), "app_id": "2",
    })
    totp = pyotp.TOTP(settings.totp_secret)
    # Avoid sending an OTP immediately before its validity boundary.
    remaining = totp.interval - (time.time() % totp.interval)
    if remaining < 5:
        time.sleep(remaining + 0.1)
    verified = post(session, f"{SSO}/verify_otp", {
        "request_key": required(login, "request_key"), "otp": totp.now(),
    })
    pin = post(session, f"{SSO}/verify_pin_v2", {
        "request_key": required(verified, "request_key"),
        "identity_type": "pin", "identifier": encoded(settings.pin),
    })
    sso_token = required(pin.get("data", {}), "access_token")
    app_id, app_type = settings.client_id.rsplit("-", 1)
    state = secrets.token_urlsafe(24)
    redirect = post(session, f"{API}/token", {
        "fyers_id": settings.user_id, "app_id": app_id, "appType": app_type,
        "redirect_uri": settings.redirect_uri, "response_type": "code",
        "state": state, "code_challenge": "", "scope": "", "nonce": "",
        "create_cookie": True,
    }, headers={"Authorization": f"Bearer {sso_token}"}, redirect=True)
    url = urlsplit(required(redirect, "Url"))
    expected = urlsplit(settings.redirect_uri)
    if (url.scheme, url.netloc, url.path) != (expected.scheme, expected.netloc, expected.path):
        raise AutoAuthError("Authentication redirect URI mismatch")
    query = parse_qs(url.query)
    if query.get("state") != [state] or len(query.get("auth_code", [])) != 1:
        raise AutoAuthError("Authentication redirect state or auth_code invalid")
    result = post(session, f"{API}/validate-authcode", {
        "grant_type": "authorization_code",
        "appIdHash": hashlib.sha256(f"{settings.client_id}:{settings.secret_key}".encode()).hexdigest(),
        "code": query["auth_code"][0],
    })
    return required(result, "access_token")


def persist_token(path: Path, token: str) -> None:
    """Update the mounted .env inode under a lock, preserving other settings.

    In-place writes support Docker file bind mounts. No reader should consume
    .env until this command succeeds. fsync completes before releasing the lock.
    """
    if not token or any(char in token for char in "\r\n'\\"):
        raise AutoAuthError("Invalid access token")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        os.fchmod(handle.fileno(), 0o600)
        handle.seek(0)
        lines = handle.read().splitlines()
        replacements = {
            "FYERS_ACCESS_TOKEN": token,
            "FYERS_TOKEN_GENERATED_AT": datetime.now(UTC).isoformat(),
        }
        kept = [line for line in lines if line.split("=", 1)[0].strip().removeprefix("export ") not in replacements]
        kept.extend(f"{key}='{value}'" for key, value in replacements.items())
        handle.seek(0)
        handle.write("\n".join(kept) + "\n")
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
    if dotenv_values(path).get("FYERS_ACCESS_TOKEN") != token:
        raise AutoAuthError("Could not verify persisted access token")


def main() -> int:
    path = Path(os.getenv("FYERS_ENV_FILE", str(Path(__file__).resolve().parents[1] / ".env")))
    load_dotenv(path)
    settings = FyersSettings(
        client_id=os.getenv("FYERS_CLIENT_ID"), secret_key=os.getenv("FYERS_SECRET_KEY"),
        redirect_uri=os.getenv("FYERS_REDIRECT_URI", ""), user_id=os.getenv("FYERS_USER_ID"),
        pin=os.getenv("FYERS_PIN"), totp_secret=os.getenv("FYERS_TOTP_SECRET"),
    )
    try:
        with requests.Session() as session:
            token = authenticate(settings, session)
        persist_token(path, token)
    except Exception as exc:
        message = str(exc) if isinstance(exc, AutoAuthError) else type(exc).__name__
        print(f"Headless authentication failed: {message}", file=sys.stderr)
        return 1
    print("Daily Fyers token generated and verified in .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
