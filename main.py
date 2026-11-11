"""Entrypoint for authenticated Fyers API V3 live connectivity."""

from __future__ import annotations

import signal
import sys
from pathlib import Path as _Path
from threading import Event

# Ensure project root and src/ are on sys.path when run directly:
# python main.py
_PROJECT_ROOT = _Path(__file__).resolve().parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loguru import logger

from broker.api_security import APISecurityError, APISecurityGuard
from broker.fyers_auth import FyersAuthError, FyersAuthManager
from broker.fyers_client import FyersMarketClient
from broker.fyers_stream import FyersStream
from config.settings import AppSettings


def configure_logging(settings: AppSettings) -> None:
    """Configure structured loguru sinks for console and broker/runtime files."""
    logger.remove()
    logger.add(sys.stdout, level=settings.logging.level, enqueue=True, backtrace=False, diagnose=False)
    logger.add(
        settings.logging.log_dir / settings.logging.file_name,
        level=settings.logging.level,
        enqueue=True,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        settings.logging.log_dir / settings.logging.broker_file_name,
        level=settings.logging.level,
        enqueue=True,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        backtrace=False,
        diagnose=False,
    )


def main() -> int:
    """Start authenticated Fyers connectivity and websocket streaming."""
    settings = AppSettings.from_env()
    configure_logging(settings)
    logger.info("Initializing Fyers API V3 runtime", project_root=str(settings.storage.project_root))

    security = APISecurityGuard(settings.security)
    try:
        security.validate_environment()
    except APISecurityError as exc:
        logger.exception("Invalid API execution environment", error=str(exc))
        return 2

    if not security.should_allow_execution():
        logger.error(
            "Live execution is blocked by security policy",
            environment=settings.security.environment,
            allow_live_execution=settings.security.allow_live_execution,
            manual_approval_required=settings.security.require_manual_execution_approval,
        )
        return 1

    try:
        auth = FyersAuthManager(settings.fyers, security_guard=security)
    except FyersAuthError as exc:
        logger.exception("Fyers auth manager initialization failed", error=str(exc))
        return 2

    access_token = auth.resolve_access_token()
    if not access_token and settings.fyers.auth_code:
        try:
            access_token = auth.exchange_auth_code(settings.fyers.auth_code)
        except FyersAuthError as exc:
            logger.exception("Authorization code exchange failed", error=str(exc))
            return 2

    if not access_token:
        login_url = auth.login_url()
        logger.warning("No Fyers access token available. Complete the login flow first.", login_url=login_url)
        print(login_url)
        return 1

    client = FyersMarketClient(settings.fyers, auth_manager=auth, security_guard=security)
    try:
        profile = client.validate_session()
        snapshot = client.get_live_snapshot()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unable to initialize live market client", error=str(exc))
        return 2

    logger.info(
        "Live market snapshot acquired",
        user_id=profile.get("user_id") or profile.get("fy_id"),
        nifty_last_price=snapshot.nifty_spot.last_price,
        vix_last_price=snapshot.india_vix.last_price,
        option_rows=len(snapshot.option_instruments),
    )

    stream = FyersStream(settings.fyers, auth_manager=auth, security_guard=security)

    def _log_normalized_ticks(frame):
        logger.info("Normalized live tick batch received", rows=len(frame))

    stream.register_tick_handler(_log_normalized_ticks)

    if settings.fyers.stream_symbols:
        stream.subscribe(list(settings.fyers.stream_symbols))
        logger.info("Pre-subscribed configured symbols", symbols=list(settings.fyers.stream_symbols))
    elif settings.fyers.stream_tokens:
        stream.subscribe(list(settings.fyers.stream_tokens))
        logger.info("Pre-subscribed configured tokens", tokens=list(settings.fyers.stream_tokens))
    else:
        logger.warning("No stream symbols/tokens configured. Websocket will connect without subscriptions.")

    shutdown_event = Event()

    def _handle_shutdown(_signum: int, _frame: object) -> None:
        logger.warning("Shutdown signal received")
        shutdown_event.set()
        stream.stop()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        if security.safe_mode:
            raise APISecurityError("Safe mode is active; refusing to start websocket execution")
        stream.connect(threaded=True)
        logger.info("Fyers websocket started")
        shutdown_event.wait()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error in Fyers runtime", error=str(exc))
        return 2
    finally:
        stream.stop()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
