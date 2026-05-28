"""Run the live NIFTY options intelligence service."""

from __future__ import annotations

import signal
import sys
import time
from threading import Event

from loguru import logger

from config.settings import AppSettings
from trading_engine.intelligence.models import IntelligenceReport
from trading_engine.intelligence.runtime import LiveIntelligenceRuntime, RuntimeCallbacks


def main() -> int:
    """Start the continuous live intelligence runtime."""
    settings = AppSettings.from_env()
    _configure_logging(settings)

    shutdown_event = Event()

    def _on_report(report: IntelligenceReport) -> None:
        logger.info(
            "Intelligence update",
            regime=report.regime.regime.value,
            action=report.signal.action.value,
            confidence=round(report.signal.confidence, 3),
            strike=report.strike.strike,
            quantity=report.signal.quantity,
        )

    def _on_error(exc: Exception) -> None:
        logger.exception("Live intelligence runtime error", error=str(exc))

    callbacks = RuntimeCallbacks(on_report=_on_report, on_error=_on_error)
    runtime = LiveIntelligenceRuntime(settings, callbacks=callbacks)

    def _handle_shutdown(_signum: int, _frame: object) -> None:
        logger.warning("Shutdown requested for intelligence runtime")
        shutdown_event.set()
        runtime.stop()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        runtime.refresh_once()
        if settings.kite.stream_tokens:
            runtime.start_streaming()
            logger.info("Streaming live intelligence service started")
        else:
            logger.warning("No stream tokens configured. Running in polling mode only.")

        while not shutdown_event.is_set():
            if not settings.kite.stream_tokens:
                try:
                    runtime.refresh_once()
                except Exception as exc:  # noqa: BLE001
                    _on_error(exc)
            time.sleep(max(settings.intelligence.dashboard_refresh_seconds, 1))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal live intelligence service error", error=str(exc))
        return 2
    finally:
        runtime.stop()

    return 0


def _configure_logging(settings: AppSettings) -> None:
    logger.remove()
    logger.add(sys.stdout, level=settings.logging.level, enqueue=True, backtrace=False, diagnose=False)
    logger.add(
        settings.logging.log_dir / "intelligence.log",
        level=settings.logging.level,
        enqueue=True,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        backtrace=False,
        diagnose=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
