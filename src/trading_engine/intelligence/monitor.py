"""Live trade monitor for open-position tracking and trailing management."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from trading_engine.intelligence.models import MarketRegime, MarketState, OpenTrade, SignalAction, SignalResult, TradeGrade, TradeUpdate


@dataclass(slots=True)
class MonitorConfig:
    """Trailing and exit controls for live trade tracking."""

    trail_profit_threshold_pct: float = 0.004
    trail_distance_pct: float = 0.003
    max_holding_minutes: int = 12
    partial_exit_rr: float = 1.0


class LiveTradeMonitor:
    """Track open signals, mark-to-market, and surface exit alerts."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self._config = config or MonitorConfig()
        self._trade_counter = itertools.count(1)
        self._open_trades: dict[str, OpenTrade] = {}
        self._closed_trades: list[OpenTrade] = []
        self._expired_trades: list[OpenTrade] = []
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
            expires_at=signal.expires_at,
            last_price=float(entry_price),
            grade=signal.trade_grade,
        )
        self._open_trades[trade_id] = trade
        logger.info("Opened live trade", trade_id=trade_id, action=signal.action.value, qty=quantity)
        return trade

    def open_trade_from_signal(self, signal: SignalResult, quantity: int | None = None) -> OpenTrade | None:
        """Convenience helper to open a trade directly from a signal."""
        if signal.action == SignalAction.NO_TRADE or signal.trade_grade == TradeGrade.AVOID:
            return None
        if signal.expires_at is not None and datetime.now(UTC) >= signal.expires_at:
            return None
        entry_price = signal.entry_reference
        if entry_price is None:
            return None
        qty = int(quantity or signal.quantity or 0)
        if qty <= 0:
            return None
        return self.open_trade(signal, entry_price, qty)

    def update(self, current_price: float | None, *, latest_state: MarketState | None = None) -> TradeUpdate:
        """Update live trade marks, trailing stops, and exit conditions."""
        now = datetime.now(UTC)
        if current_price is None:
            return self.summary()

        closed_ids: list[str] = []
        expired_ids: list[str] = []
        for trade_id, trade in list(self._open_trades.items()):
            trade.last_price = float(current_price)
            pnl_pct = self._pnl_pct(trade, current_price)
            trade.peak_pnl = max(trade.peak_pnl, pnl_pct)

            if trade.signal.stop_loss is not None and self._should_trail(trade, pnl_pct):
                trade.signal.stop_loss = self._trail_stop(trade, current_price)
                self._alerts.append(f"{trade_id}: trailing stop adjusted")

            if trade.expires_at is not None and now >= trade.expires_at:
                trade.status = "expired"
                trade.exit_reason = "signal_expired"
                trade.realized_pnl = self._realized_pnl(trade, current_price)
                expired_ids.append(trade_id)
                self._expired_trades.append(trade)
                self._alerts.append(f"{trade_id}: signal expired")
                continue

            if self._should_partial_exit(trade, pnl_pct, latest_state):
                trade.partial_exit_done = True
                self._alerts.append(f"{trade_id}: partial exit triggered")

            if self._should_close(trade, current_price, now, latest_state):
                trade.status = "closed"
                trade.realized_pnl = self._realized_pnl(trade, current_price)
                closed_ids.append(trade_id)
                self._closed_trades.append(trade)

        for trade_id in closed_ids:
            self._open_trades.pop(trade_id, None)
        for trade_id in expired_ids:
            self._open_trades.pop(trade_id, None)

        return self.summary()

    def summary(self) -> TradeUpdate:
        """Return a snapshot of open/closed trades and PnL."""
        total_pnl = sum(trade.realized_pnl for trade in self._closed_trades)
        return TradeUpdate(
            open_trades=list(self._open_trades.values()),
            closed_trades=list(self._closed_trades),
            expired_trades=list(self._expired_trades),
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

    def _should_partial_exit(self, trade: OpenTrade, pnl_pct: float, latest_state: MarketState | None) -> bool:
        if trade.partial_exit_done:
            return False
        if pnl_pct < self._config.partial_exit_rr:
            return False
        if latest_state is not None and latest_state.trap_probability >= 0.55:
            return True
        return pnl_pct >= self._config.partial_exit_rr

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
        latest_state: MarketState | None,
    ) -> bool:
        if trade.signal.stop_loss is not None:
            if trade.signal.action == SignalAction.BUY_CE and current_price <= trade.signal.stop_loss:
                trade.exit_reason = "stop_loss"
                self._alerts.append(f"{trade.trade_id}: stop loss hit")
                return True
            if trade.signal.action == SignalAction.BUY_PE and current_price >= trade.signal.stop_loss:
                trade.exit_reason = "stop_loss"
                self._alerts.append(f"{trade.trade_id}: stop loss hit")
                return True

        minutes_open = (now - trade.opened_at).total_seconds() / 60.0
        if minutes_open >= self._config.max_holding_minutes:
            trade.exit_reason = "time_stop"
            self._alerts.append(f"{trade.trade_id}: max holding time reached")
            return True

        if latest_state is not None:
            if latest_state.trap_probability >= 0.70:
                trade.exit_reason = "trap_exit"
                self._alerts.append(f"{trade.trade_id}: trap score exited")
                return True
            if latest_state.quality_score < 0.35:
                trade.exit_reason = "quality_exit"
                self._alerts.append(f"{trade.trade_id}: market quality deteriorated")
                return True
            if latest_state.regime in {MarketRegime.SIDEWAYS_COMPRESSION, MarketRegime.HIGH_NOISE_ENVIRONMENT} and trade.grade != TradeGrade.A_PLUS:
                trade.exit_reason = "regime_downgrade"
                self._alerts.append(f"{trade.trade_id}: regime downgraded")
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
