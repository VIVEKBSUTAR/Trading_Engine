"""Run the live NIFTY options intelligence service."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
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
    last_publish = 0.0

    def _on_report(report: IntelligenceReport) -> None:
        nonlocal last_publish
        logger.info(
            "Intelligence update",
            regime=report.regime.regime.value,
            action=report.signal.action.value,
            confidence=round(report.signal.confidence, 3),
            strike=report.strike.strike,
            quantity=report.signal.quantity,
        )
        snapshot_dir = os.getenv("TE_DASHBOARD_SNAPSHOT_DIR")
        if snapshot_dir and time.monotonic() - last_publish >= settings.intelligence.dashboard_refresh_seconds:
            publish_dashboard_snapshot(Path(snapshot_dir), report, runtime)
            last_publish = time.monotonic()

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
        streaming = bool(settings.fyers.stream_tokens or settings.fyers.stream_symbols
                         or runtime.stream.state.subscribed_symbols)
        if streaming:
            runtime.start_streaming()
            logger.info("Streaming live intelligence service started")
        else:
            logger.warning("No stream tokens configured. Running in polling mode only.")

        last_chain_refresh = time.monotonic()
        while not shutdown_event.is_set():
            if not streaming or time.monotonic() - last_chain_refresh >= 15:
                try:
                    runtime.refresh_once()
                    last_chain_refresh = time.monotonic()
                except Exception as exc:  # noqa: BLE001
                    _on_error(exc)
            time.sleep(max(settings.intelligence.dashboard_refresh_seconds, 1))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal live intelligence service error", error=str(exc))
        return 2
    finally:
        runtime.stop()

    return 0


def publish_dashboard_snapshot(directory: Path, report: IntelligenceReport, runtime: LiveIntelligenceRuntime) -> None:
    """Publish one atomic JSON view; the dashboard never opens the writer's DB."""
    import tempfile

    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": report.timestamp.isoformat(),
        "regime": report.regime.regime.value,
        "action": report.signal.action.value,
        "confidence": report.signal.confidence,
        "reasoning": report.signal.reasoning,
        "spot_price": report.snapshot.spot_price,
        "vix_value": report.snapshot.vix_value,
        "websocket": {
            "connected": runtime.stream.state.connected,
            "last_tick_at": runtime.stream.state.last_tick_at,
            "safe_mode": runtime.stream.state.safe_mode,
        },
        "lifecycle_alerts": runtime.engine.duckdb_manager.get_lifecycle_alerts().to_dict("records"),
        "live_signals": runtime.engine.duckdb_manager.get_live_signals().to_dict("records"),
        "signals": runtime.engine.duckdb_manager.get_signals(limit=50).to_dict("records"),
        "outcomes": runtime.engine.duckdb_manager.get_trade_outcomes(limit=1000).to_dict("records"),
        "positions": [{
            "trade_id": trade.trade_id, "instrument_token": trade.instrument_token,
            "strike": trade.strike, "option_type": trade.option_type,
            "quantity": trade.quantity,
            "unrealized_pnl": runtime.monitor.calculate_profit(trade),
            "entry_ask": trade.entry_ask_price, "current_bid": trade.current_bid_price,
            "stop_loss": trade.signal.stop_loss, "is_frozen": trade.is_frozen,
            "live_iv": trade.live_iv, "delta": trade.delta,
            "time_in_trade": max(0, (report.timestamp - trade.opened_at).total_seconds() / 60),
            "opened_at": trade.opened_at.isoformat(), "signal_timestamp": trade.signal_timestamp.isoformat(),
        } for trade in runtime.monitor.open_trades],
    }
    path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, suffix=".tmp", delete=False) as handle:
            path = Path(handle.name)
            json.dump(payload, handle, default=str)
        path.replace(directory / "latest.json")
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


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
