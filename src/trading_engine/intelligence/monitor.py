"""Live trade monitor for open-position tracking and trailing management."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from loguru import logger

from storage.duckdb_manager import DuckDBManager
from trading_engine.intelligence.models import (
    MarketRegime,
    MarketState,
    OpenTrade,
    SignalAction,
    SignalResult,
    TradeGrade,
    TradeUpdate,
)

NIFTY_LOT_SIZE: int = 65


@dataclass(slots=True)
class MonitorConfig:
    """Trailing and exit controls for live trade tracking."""

    trail_profit_threshold_pct: float = 0.004
    trail_distance_pct: float = 0.003
    max_holding_minutes: int = 12
    partial_exit_rr: float = 1.0


class LiveTradeMonitor:
    """Track open signals, mark-to-market option premiums, and surface exit alerts."""

    NIFTY_LOT_SIZE: int = NIFTY_LOT_SIZE

    def __init__(
        self,
        config: MonitorConfig | None = None,
        duckdb_manager: DuckDBManager | None = None,
    ) -> None:
        self._config = config or MonitorConfig()
        self._duckdb_manager = duckdb_manager
        self._trade_counter = itertools.count(1)
        self._open_trades: dict[str, OpenTrade] = {}
        self._closed_trades: list[OpenTrade] = []
        self._expired_trades: list[OpenTrade] = []
        self._alerts: list[str] = []

    @property
    def duckdb_manager(self) -> DuckDBManager | None:
        """Expose the DuckDB manager."""
        return self._duckdb_manager

    @duckdb_manager.setter
    def duckdb_manager(self, manager: DuckDBManager | None) -> None:
        self._duckdb_manager = manager

    def open_trade(
        self,
        signal: SignalResult,
        entry_price: float,
        quantity: int,
        *,
        instrument_token: int = 0,
        strike: float = 0.0,
        expiry: str = "",
        option_type: str = "",
        entry_ask_price: float = 0.0,
        signal_timestamp: datetime | None = None,
    ) -> OpenTrade:
        """Register a new live trade."""
        trade_id = f"LT-{next(self._trade_counter):05d}"
        now = datetime.now(UTC)
        ask = float(entry_ask_price if entry_ask_price > 0 else entry_price)
        trade = OpenTrade(
            trade_id=trade_id,
            signal=signal,
            entry_price=float(entry_price),
            quantity=int(quantity),
            opened_at=now,
            expires_at=signal.expires_at,
            last_price=ask,
            grade=signal.trade_grade,
            instrument_token=int(instrument_token),
            strike=float(strike),
            expiry=str(expiry),
            option_type=str(option_type),
            entry_ask_price=ask,
            current_bid_price=ask,
            signal_timestamp=signal_timestamp or now,
        )
        self._open_trades[trade_id] = trade
        logger.info(
            "Opened live trade",
            trade_id=trade_id,
            action=signal.action.value,
            token=trade.instrument_token,
            strike=trade.strike,
            entry_ask=trade.entry_ask_price,
            qty=quantity,
        )
        return trade

    def open_trade_from_signal(
        self,
        signal: SignalResult,
        quantity: int | None = None,
        *,
        instrument_token: int | None = None,
        entry_ask_price: float | None = None,
        strike: float | None = None,
        expiry: str | None = None,
        option_type: str | None = None,
    ) -> OpenTrade | None:
        """Convenience helper to open a trade directly from a signal."""
        if signal.action == SignalAction.NO_TRADE or signal.trade_grade == TradeGrade.AVOID:
            return None
        if signal.expires_at is not None and datetime.now(UTC) >= signal.expires_at:
            return None

        st_obj = signal.strike
        resolved_strike = float(strike if strike is not None else (st_obj.strike if st_obj and st_obj.strike else 0.0))
        resolved_expiry = str(expiry if expiry is not None else (st_obj.expiry if st_obj and st_obj.expiry else ""))
        resolved_type = str(
            option_type if option_type is not None else (
                st_obj.option_side.value if st_obj and hasattr(st_obj, "option_side")
                else ("CE" if signal.action == SignalAction.BUY_CE else "PE")
            )
        )

        token = int(instrument_token or 0)
        ask_price = float(entry_ask_price if entry_ask_price is not None and entry_ask_price > 0 else (signal.entry_reference or 0.0))
        entry_price = ask_price if ask_price > 0 else (signal.entry_reference or 0.0)

        qty = int(quantity or signal.quantity or 0)
        if qty <= 0:
            return None

        sig_ts = signal.state.timestamp if signal.state else datetime.now(UTC)
        return self.open_trade(
            signal,
            entry_price,
            qty,
            instrument_token=token,
            strike=resolved_strike,
            expiry=resolved_expiry,
            option_type=resolved_type,
            entry_ask_price=ask_price,
            signal_timestamp=sig_ts,
        )

    def update(
        self,
        option_ticks: dict[int, Any] | pd.DataFrame | None = None,
        *,
        latest_state: MarketState | None = None,
    ) -> TradeUpdate:
        """Update live trade marks, trailing stops, and exit conditions using current option tick data."""
        now = datetime.now(UTC)
        if not self._open_trades:
            return self.summary()

        closed_ids: list[str] = []
        expired_ids: list[str] = []

        for trade_id, trade in list(self._open_trades.items()):
            # Extract current option tick data for this trade's instrument token
            bid, ask, tick_ts = _extract_token_tick(option_ticks, trade.instrument_token)

            if bid is None or tick_ts is None:
                trade.is_frozen = True
                logger.warning(
                    "Missing option tick data for tracked token; freezing trade evaluation",
                    trade_id=trade_id,
                    instrument_token=trade.instrument_token,
                )
                self._alerts.append(f"{trade_id}: no tick data for token {trade.instrument_token} - evaluation frozen")
                continue

            # Step 5: Stale data circuit breaker (gap > 2 seconds)
            if tick_ts.tzinfo is None:
                tick_ts = tick_ts.replace(tzinfo=UTC)
            gap_seconds = (now - tick_ts).total_seconds()
            if gap_seconds > 2.0:
                trade.is_frozen = True
                logger.warning(
                    "Stale option tick data detected; freezing trade evaluation",
                    trade_id=trade_id,
                    instrument_token=trade.instrument_token,
                    gap_seconds=round(gap_seconds, 3),
                    threshold_seconds=2.0,
                )
                self._alerts.append(
                    f"{trade_id}: stale tick gap {gap_seconds:.2f}s > 2.0s - evaluation frozen"
                )
                # Freeze trade evaluation: do not calculate profit or trigger stops on stale data
                continue

            trade.is_frozen = False

            # Update trade marks with fresh option quotes
            trade.current_bid_price = float(bid)
            trade.last_price = float(bid)

            # Step 4: Profit calculation: (current_bid_price - entry_ask_price) * 65
            pnl_amount = self.calculate_profit(trade)
            pnl_pct = self._pnl_pct(trade)
            trade.peak_pnl = max(trade.peak_pnl, pnl_pct)

            # Trailing stop adjustments
            if trade.signal.stop_loss is not None and self._should_trail(trade, pnl_pct):
                trade.signal.stop_loss = self._trail_stop(trade, trade.current_bid_price)
                self._alerts.append(f"{trade_id}: trailing stop adjusted to {trade.signal.stop_loss:.2f}")

            # Check signal expiry
            if trade.expires_at is not None and now >= trade.expires_at:
                trade.status = "expired"
                trade.exit_reason = "signal_expired"
                trade.realized_pnl = pnl_amount
                expired_ids.append(trade_id)
                self._expired_trades.append(trade)
                self._alerts.append(f"{trade_id}: signal expired (pnl={pnl_amount:.2f})")
                self.record_closed_trade_outcome(trade, exit_timestamp=now)
                continue

            # Partial exit evaluation
            if self._should_partial_exit(trade, pnl_pct, latest_state):
                trade.partial_exit_done = True
                self._alerts.append(f"{trade_id}: partial exit triggered")

            # Exit condition checks (stop loss, time stop, trap exit, regime downgrade)
            if self._should_close(trade, trade.current_bid_price, now, latest_state):
                trade.status = "closed"
                trade.realized_pnl = pnl_amount
                closed_ids.append(trade_id)
                self._closed_trades.append(trade)
                self._alerts.append(f"{trade_id}: closed ({trade.exit_reason}) pnl={pnl_amount:.2f}")
                self.record_closed_trade_outcome(trade, exit_timestamp=now)

        for trade_id in closed_ids:
            self._open_trades.pop(trade_id, None)
        for trade_id in expired_ids:
            self._open_trades.pop(trade_id, None)

        return self.summary()

    def record_closed_trade_outcome(
        self,
        trade: OpenTrade,
        exit_timestamp: datetime | None = None,
    ) -> None:
        """Write completed trade outcome to DuckDB trade_outcomes table."""
        if self._duckdb_manager is None:
            return

        exit_ts = exit_timestamp or datetime.now(UTC)
        holding_mins = max(0.0, (exit_ts - trade.opened_at).total_seconds() / 60.0)
        sig_id = getattr(trade.signal, "signal_id", None) or f"SIG-{trade.trade_id}"
        exit_bid = float(trade.current_bid_price if trade.current_bid_price > 0 else trade.last_price)

        try:
            self._duckdb_manager.insert_trade_outcome(
                trade_id=str(trade.trade_id),
                signal_id=str(sig_id),
                instrument_token=int(trade.instrument_token),
                entry_timestamp=trade.opened_at,
                exit_timestamp=exit_ts,
                entry_ask_price=float(trade.entry_ask_price),
                exit_bid_price=exit_bid,
                quantity=int(trade.quantity),
                holding_time_minutes=float(holding_mins),
                exit_reason=str(trade.exit_reason or "closed"),
                realized_pnl=float(trade.realized_pnl),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to persist trade outcome to DuckDB", trade_id=trade.trade_id, error=str(exc))

    def calculate_profit(self, trade: OpenTrade) -> float:
        """Calculate profit: (current_bid_price - entry_ask_price) * 65 * quantity."""
        qty = max(1, trade.quantity)
        return float((trade.current_bid_price - trade.entry_ask_price) * self.NIFTY_LOT_SIZE * qty)

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

    def _trail_stop(self, trade: OpenTrade, current_bid_price: float) -> float:
        """Trail stop-loss below current bid price for long option position."""
        trail_level = current_bid_price * (1.0 - self._config.trail_distance_pct)
        if trade.signal.stop_loss is not None:
            return float(max(trade.signal.stop_loss, trail_level))
        return float(trail_level)

    def _should_close(
        self,
        trade: OpenTrade,
        current_bid_price: float,
        now: datetime,
        latest_state: MarketState | None,
    ) -> bool:
        if trade.signal.stop_loss is not None and current_bid_price <= trade.signal.stop_loss:
            is_trailing = (
                trade.peak_pnl >= self._config.trail_profit_threshold_pct
                or (trade.entry_ask_price > 0 and trade.signal.stop_loss > trade.entry_ask_price * 0.995)
            )
            trade.exit_reason = "trailing_stop" if is_trailing else "stop_loss"
            self._alerts.append(f"{trade.trade_id}: {trade.exit_reason} hit at {current_bid_price:.2f}")
            return True

        if trade.signal.target is not None and current_bid_price >= trade.signal.target:
            trade.exit_reason = "target_hit"
            self._alerts.append(f"{trade.trade_id}: target reached at {current_bid_price:.2f}")
            return True

        minutes_open = (now - trade.opened_at).total_seconds() / 60.0
        if minutes_open >= self._config.max_holding_minutes:
            trade.exit_reason = "time_stop"
            self._alerts.append(f"{trade.trade_id}: max holding time reached ({minutes_open:.1f}m)")
            return True

        if latest_state is not None:
            if latest_state.trap_probability >= 0.70:
                trade.exit_reason = "trap_exit"
                self._alerts.append(f"{trade.trade_id}: trap score exited ({latest_state.trap_probability:.2f})")
                return True
            if latest_state.quality_score < 0.35:
                trade.exit_reason = "quality_exit"
                self._alerts.append(f"{trade.trade_id}: market quality deteriorated ({latest_state.quality_score:.2f})")
                return True
            if latest_state.regime in {MarketRegime.SIDEWAYS_COMPRESSION, MarketRegime.HIGH_NOISE_ENVIRONMENT} and trade.grade != TradeGrade.A_PLUS:
                trade.exit_reason = "regime_downgrade"
                self._alerts.append(f"{trade.trade_id}: regime downgraded to {latest_state.regime.value}")
                return True

        return False

    def _pnl_pct(self, trade: OpenTrade) -> float:
        """Percentage return based on option premium change."""
        ref = trade.entry_ask_price if trade.entry_ask_price > 0 else trade.entry_price
        if ref <= 0.0:
            return 0.0
        return (trade.current_bid_price - ref) / ref

    def _realized_pnl(self, trade: OpenTrade, exit_price: float) -> float:
        """Backward-compatible helper redirecting to calculate_profit."""
        return self.calculate_profit(trade)


def _extract_token_tick(option_ticks: Any, token: int) -> tuple[float | None, float | None, datetime | None]:
    """Extract (bid_price, ask_price, timestamp) for a given instrument token from various tick sources."""
    if option_ticks is None:
        return None, None, None

    if isinstance(option_ticks, dict):
        tick_data = option_ticks.get(token)
        if tick_data is None:
            tick_data = option_ticks.get(str(token))
        if tick_data is None:
            return None, None, None

        if isinstance(tick_data, pd.DataFrame):
            if tick_data.empty:
                return None, None, None
            row = tick_data.iloc[-1]
            bid = _safe_float(row.get("bid_price") or row.get("bid") or row.get("last_price"))
            ask = _safe_float(row.get("ask_price") or row.get("ask") or row.get("last_price"))
            ts = _safe_datetime(row.get("timestamp") or row.get("exchange_timestamp"))
            return bid, ask, ts

        if isinstance(tick_data, dict):
            bid = _safe_float(tick_data.get("bid_price") or tick_data.get("bid") or tick_data.get("last_price"))
            ask = _safe_float(tick_data.get("ask_price") or tick_data.get("ask") or tick_data.get("last_price"))
            ts = _safe_datetime(tick_data.get("timestamp") or tick_data.get("exchange_timestamp"))
            return bid, ask, ts

        if isinstance(tick_data, (tuple, list)) and len(tick_data) >= 2:
            bid = _safe_float(tick_data[0])
            ask = _safe_float(tick_data[1])
            ts = _safe_datetime(tick_data[2]) if len(tick_data) > 2 else None
            return bid, ask, ts

    elif isinstance(option_ticks, pd.DataFrame):
        if option_ticks.empty:
            return None, None, None
        if "instrument_token" in option_ticks.columns:
            rows = option_ticks.loc[option_ticks["instrument_token"] == token]
            if rows.empty:
                return None, None, None
            last_row = rows.iloc[-1]
            bid = _safe_float(last_row.get("bid_price") or last_row.get("bid") or last_row.get("last_price"))
            ask = _safe_float(last_row.get("ask_price") or last_row.get("ask") or last_row.get("last_price"))
            ts = _safe_datetime(last_row.get("timestamp") or last_row.get("exchange_timestamp"))
            return bid, ask, ts

    return None, None, None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_datetime(value: Any) -> datetime | None:
    try:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        if hasattr(parsed, "to_pydatetime"):
            dt: Any = parsed.to_pydatetime()
            if isinstance(dt, datetime):
                return dt
        return None
    except Exception:  # noqa: BLE001
        return None
