"""Headless authentication contracts without contacting Fyers."""
import base64
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlencode

import pytest
from dotenv import dotenv_values

from broker.fyers_auto_auth import API, AutoAuthError, authenticate, persist_token
from config.settings import FyersSettings


def settings():
    return FyersSettings(client_id="APP-100", secret_key="app-secret", user_id="USER",
                         pin="0123", totp_secret="JBSWY3DPEHPK3PXP", redirect_uri="https://localhost/callback")


def session_for(settings, *, bad_state=False):
    session = Mock()
    def respond(url, **kwargs):
        response = Mock(status_code=200)
        if url.endswith("send_login_otp_v2"):
            data = {"s": "ok", "request_key": "first"}
        elif url.endswith("verify_otp"):
            data = {"s": "ok", "request_key": "second"}
        elif url.endswith("verify_pin_v2"):
            data = {"s": "ok", "data": {"access_token": "sso-token"}}
        elif url.endswith("/token"):
            response.status_code = 308
            data = {"Url": settings.redirect_uri + "?" + urlencode({
                "auth_code": "auth-code", "state": "wrong" if bad_state else kwargs["json"]["state"]})}
        else:
            data = {"s": "ok", "access_token": "daily-token"}
        response.json.return_value = data
        return response
    session.post.side_effect = respond
    return session


def test_sso_totp_pin_redirect_and_hash_exchange():
    config = settings()
    session = session_for(config)
    with patch("broker.fyers_auto_auth.time.time", return_value=1):
        assert authenticate(config, session) == "daily-token"
    calls = session.post.call_args_list
    assert len(calls) == 5
    assert calls[0].kwargs["json"]["fy_id"] == base64.b64encode(b"USER").decode()
    assert len(calls[1].kwargs["json"]["otp"]) == 6
    assert calls[2].kwargs["json"]["identifier"] == base64.b64encode(b"0123").decode()
    assert calls[-1].args[0] == f"{API}/validate-authcode"
    assert calls[-1].kwargs["json"]["appIdHash"] == hashlib.sha256(b"APP-100:app-secret").hexdigest()
    assert all(call.kwargs["allow_redirects"] is False and call.kwargs["timeout"] == 20 for call in calls)


def test_redirect_state_failure_never_exchanges_code():
    config = settings()
    session = session_for(config, bad_state=True)
    with patch("broker.fyers_auto_auth.time.time", return_value=1), pytest.raises(AutoAuthError, match="state"):
        authenticate(config, session)
    assert session.post.call_count == 4


def test_http_failure_does_not_expose_body():
    session = Mock()
    session.post.return_value.status_code = 403
    session.post.return_value.text = "secret-provider-body"
    with pytest.raises(AutoAuthError, match="HTTP 403") as error:
        authenticate(settings(), session)
    assert "secret-provider-body" not in str(error.value)
    assert session.post.call_count == 1


def test_required_credentials_before_network():
    config = settings()
    config.pin = None
    session = Mock()
    with pytest.raises(ValueError, match="FYERS_PIN"):
        authenticate(config, session)
    session.post.assert_not_called()


def test_token_injection_replaces_duplicates_preserves_inode_and_other_values(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text("# preserved\nFYERS_PIN='0123'\nFYERS_ACCESS_TOKEN=old\nexport FYERS_ACCESS_TOKEN=older\n")
    inode = path.stat().st_ino
    persist_token(path, "daily-token")
    values = dotenv_values(path)
    assert values["FYERS_ACCESS_TOKEN"] == "daily-token"
    assert values["FYERS_PIN"] == "0123"
    assert path.read_text().count("FYERS_ACCESS_TOKEN=") == 1
    assert path.stat().st_ino == inode
    assert path.stat().st_mode & 0o777 == 0o600
    assert (datetime.now(UTC) - datetime.fromisoformat(values["FYERS_TOKEN_GENERATED_AT"])).total_seconds() < 10
