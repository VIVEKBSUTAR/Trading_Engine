"""Live trade monitor for open-position tracking and trailing management."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from trading_engine.intelligence.models import OpenTrade, SignalAction, SignalResult, TradeUpdate


@dataclass(slots=True)
class MonitorConfig:
    """Trailing and exit controls for live trade tracking."""

    trail_profit_threshold_pct: float = 0.004
    trail_distance_pct: float = 0.003
    max_holding_minutes: int = 12


class LiveTradeMonitor:
    """Track open signals, mark-to-market, and surface exit alerts."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self._config = config or MonitorConfig()
        self._trade_counter = itertools.count(1)
        self._open_trades: dict[str, OpenTrade] = {}
        self._closed_trades: list[OpenTrade] = []
        self._alerts: list[str] = []

    def open_trade(self, signal: SignalResult, entry_price: float, quantity: int) -> OpenTrade:
        """Register a new live trade."""
        trade_id = f"LT-{next(self._trade_counter):05d}"
        trade = OpenTrade(
            trade_id=trade_id,
            signal=signal,
            entry_price=float(entry_price),
            quantity=int(quantity),
            opened_at=datetime.now(UTC),
            last_price=float(entry_price),
        )
        self._open_trades[trade_id] = trade
        logger.info("Opened live trade", trade_id=trade_id, action=signal.action.value, qty=quantity)
        return trade

    def update(self, current_price: float | None, *, latest_features: pd.DataFrame | None = None) -> TradeUpdate:
        """Update live trade marks, trailing stops, and exit conditions."""
        now = datetime.now(UTC)
        closed_ids: list[str] = []
        if current_price is None:
            return self.summary()

        for trade_id, trade in list(self._open_trades.items()):
            trade.last_price = float(current_price)
            pnl_pct = self._pnl_pct(trade, current_price)
            trade.peak_pnl = max(trade.peak_pnl, pnl_pct)

            if trade.signal.stop_loss is not None and self._should_trail(trade, pnl_pct):
                trade.signal.stop_loss = self._trail_stop(trade, current_price)
                self._alerts.append(f"{trade_id}: trailing stop adjusted")

            if self._should_close(trade, current_price, now, latest_features):
                trade.status = "closed"
                trade.realized_pnl = self._realized_pnl(trade, current_price)
                closed_ids.append(trade_id)
                self._closed_trades.append(trade)

        for trade_id in closed_ids:
            self._open_trades.pop(trade_id, None)

        return self.summary()

    def summary(self) -> TradeUpdate:
        """Return a snapshot of open/closed trades and PnL."""
        total_pnl = sum(trade.realized_pnl for trade in self._closed_trades)
        return TradeUpdate(
            open_trades=list(self._open_trades.values()),
            closed_trades=list(self._closed_trades),
            total_pnl=float(total_pnl),
            alert_messages=list(self._alerts[-20:]),
        )

    @property
    def open_trades(self) -> list[OpenTrade]:
        return list(self._open_trades.values())

    @property
    def closed_trades(self) -> list[OpenTrade]:
        return list(self._closed_trades)

    def _should_trail(self, trade: OpenTrade, pnl_pct: float) -> bool:
        return pnl_pct >= self._config.trail_profit_threshold_pct

    def _trail_stop(self, trade: OpenTrade, current_price: float) -> float:
        if trade.signal.action == SignalAction.BUY_CE:
            return float(current_price * (1.0 - self._config.trail_distance_pct))
        if trade.signal.action == SignalAction.BUY_PE:
            return float(current_price * (1.0 + self._config.trail_distance_pct))
        return float(trade.signal.stop_loss or current_price)

    def _should_close(
        self,
        trade: OpenTrade,
        current_price: float,
        now: datetime,
        latest_features: pd.DataFrame | None,
    ) -> bool:
        if trade.signal.stop_loss is not None:
            if trade.signal.action == SignalAction.BUY_CE and current_price <= trade.signal.stop_loss:
                self._alerts.append(f"{trade.trade_id}: stop loss hit")
                return True
            if trade.signal.action == SignalAction.BUY_PE and current_price >= trade.signal.stop_loss:
                self._alerts.append(f"{trade.trade_id}: stop loss hit")
                return True

        minutes_open = (now - trade.opened_at).total_seconds() / 60.0
        if minutes_open >= self._config.max_holding_minutes:
            self._alerts.append(f"{trade.trade_id}: max holding time reached")
            return True

        if latest_features is not None and not latest_features.empty:
            latest = latest_features.iloc[-1]
            trap_score = float(latest.get("trap_score", 0.0) or 0.0)
            if trap_score > 0.7:
                self._alerts.append(f"{trade.trade_id}: trap score exited")
                return True

        return False

    def _pnl_pct(self, trade: OpenTrade, current_price: float) -> float:
        if trade.signal.action == SignalAction.BUY_CE:
            return (current_price - trade.entry_price) / max(trade.entry_price, 1e-9)
        if trade.signal.action == SignalAction.BUY_PE:
            return (trade.entry_price - current_price) / max(trade.entry_price, 1e-9)
        return 0.0

    def _realized_pnl(self, trade: OpenTrade, exit_price: float) -> float:
        delta = (exit_price - trade.entry_price) if trade.signal.action == SignalAction.BUY_CE else (trade.entry_price - exit_price)
        return float(delta * trade.quantity)
