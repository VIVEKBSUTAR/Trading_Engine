"""Live trade monitor for open-position tracking and trailing management."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from storage.duckdb_manager import DuckDBManager
from trading_engine.intelligence.models import (
    MarketState,
    OpenTrade,
    SignalAction,
    SignalResult,
    TradeGrade,
    TradeUpdate,
)
from trading_engine.intelligence.options_math import number

NIFTY_LOT_SIZE: int = 65


@dataclass(slots=True)
class MonitorConfig:
    """Trailing and exit controls for live trade tracking."""

    trail_profit_threshold_pct: float = 0.004
    trail_distance_pct: float = 0.003
    max_holding_minutes: int = 12
    partial_exit_rr: float = 1.0
    pcr_drop_fraction: float = 0.20
    decay_minutes: float = 20.0
    decay_fraction: float = 0.02
    iv_drop_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not 0 < self.pcr_drop_fraction < 1:
            raise ValueError("PCR drop fraction must be between 0 and 1")


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
        self.state_aggregator: Any = None
        self._events: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}
        self.contract_context: dict[tuple[str, float, str], dict[str, Any]] = {}

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
        option_ticks: pd.DataFrame | None = None,
    ) -> OpenTrade:
        """Register a new live trade."""
        trade_id = f"LT-{next(self._trade_counter):05d}"
        now = datetime.now(UTC)
        ask = float(entry_ask_price if entry_ask_price > 0 else entry_price)
        delta = number(signal.strike.delta) if signal.strike else None
        atr = number(signal.underlying_atr)
        distance = atr * abs(delta) if atr and delta is not None else None
        if distance is None or not 0 < abs(delta) <= 1 or distance <= 0 or not math.isfinite(ask) or ask <= distance:
            raise ValueError("NIFTY ATR/Delta unavailable or invalid premium stop; entry suppressed")
        if not signal.entry_iv or not signal.entry_pcr:
            raise ValueError("Entry IV/PCR unavailable; entry suppressed")
        signal.stop_loss = ask - distance
        signal.target = None
        holding_limit = signal.max_holding_minutes or self._config.max_holding_minutes
        trade = OpenTrade(
            trade_id=trade_id,
            signal=signal,
            entry_price=float(entry_price),
            quantity=int(quantity),
            opened_at=now,
            # Signal TTL controls admission. Holding lifetime starts at entry.
            expires_at=now + timedelta(minutes=holding_limit),
            last_price=ask,
            grade=signal.trade_grade,
            instrument_token=int(instrument_token),
            strike=float(strike),
            expiry=str(expiry),
            option_type=str(option_type),
            entry_ask_price=ask,
            current_bid_price=ask,
            signal_timestamp=signal_timestamp or now,
            option_atr=distance,
            initial_risk=distance,
            delta=delta, entry_iv=signal.entry_iv, live_iv=signal.entry_iv,
            entry_pcr=signal.entry_pcr, high_premium=ask, last_high_at=now,
            previous_spot=number(self.context.get("spot")),
            stop_loss_price=signal.stop_loss,
        )
        self._open_trades[trade_id] = trade
        self._emit(trade, "ENTRY_BUY", now)
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
        distance = signal.underlying_atr * abs(signal.strike.delta) if signal.underlying_atr and signal.strike and signal.strike.delta else None
        if not distance or ask_price <= distance or not signal.entry_iv or not signal.entry_pcr:
            self._alerts.append(f"Token {token}: NIFTY ATR/Greeks context warming up; entry suppressed")
            return None
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
            if gap_seconds > 2.0 or gap_seconds < -1.0 or not math.isfinite(bid) or bid <= 0:
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

            context = self.contract_context.get((trade.expiry, trade.strike, trade.option_type), {})
            context_time = pd.to_datetime(context.get("timestamp"), utc=True, errors="coerce")
            fresh_context = not pd.isna(context_time) and 0 <= (pd.Timestamp(now) - context_time).total_seconds() <= 45
            trade.live_iv = number(context.get("iv")) if fresh_context else None
            delta = number(context.get("delta")) if fresh_context else None
            if delta is not None and 0 < abs(delta) <= 1:
                trade.delta = delta
            atr = number(self.context.get("atr"))
            if fresh_context and atr and trade.delta and atr > 0:
                trade.option_atr = atr * abs(trade.delta)
                trade.signal.stop_loss = max(trade.stop_loss_price or 0, trade.current_bid_price - trade.option_atr)
                trade.stop_loss_price = trade.signal.stop_loss
                trade.trailing_active = trade.stop_loss_price > trade.entry_ask_price - (trade.initial_risk or 0)
            if trade.current_bid_price > trade.high_premium:
                trade.high_premium = trade.current_bid_price
                trade.last_high_at = now

            # Exit condition checks (stop loss, time stop, trap exit, regime downgrade)
            if self._should_close(trade, trade.current_bid_price, now, latest_state):
                trade.status = "closed"
                trade.realized_pnl = pnl_amount
                closed_ids.append(trade_id)
                self._closed_trades.append(trade)
                self._alerts.append(f"{trade_id}: closed ({trade.exit_reason}) pnl={pnl_amount:.2f}")
                self._emit(trade, trade.exit_reason or "EXIT_STOP_LOSS", now)
                self.record_closed_trade_outcome(trade, exit_timestamp=now)

        for trade_id in closed_ids:
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
        self._alerts = self._alerts[-200:]
        total_pnl = sum(trade.realized_pnl for trade in self._closed_trades)
        return TradeUpdate(
            open_trades=list(self._open_trades.values()),
            closed_trades=list(self._closed_trades),
            expired_trades=list(self._expired_trades),
            total_pnl=float(total_pnl),
            alert_messages=list(self._alerts[-20:]),
            lifecycle_alerts=list(self._events[-200:]),
        )

    @property
    def open_trades(self) -> list[OpenTrade]:
        return list(self._open_trades.values())

    @property
    def closed_trades(self) -> list[OpenTrade]:
        return list(self._closed_trades)

    def _should_close(
        self,
        trade: OpenTrade,
        current_bid_price: float,
        now: datetime,
        latest_state: MarketState | None,
    ) -> bool:
        reason = None
        context = self.contract_context.get((trade.expiry, trade.strike, trade.option_type), {})
        timestamp = pd.to_datetime(context.get("timestamp"), errors="coerce", utc=True)
        fresh = not pd.isna(timestamp) and 0 <= (pd.Timestamp(now) - timestamp).total_seconds() <= 45
        pcr, pain = (number(context.get("pcr")), number(context.get("max_pain"))) if fresh else (None, None)
        spot = number(self.context.get("spot"))
        if trade.stop_loss_price is not None and current_bid_price <= trade.stop_loss_price:
            reason = "EXIT_STOP_LOSS"
        elif trade.live_iv and trade.entry_iv and trade.live_iv < trade.entry_iv * (1 - self._config.iv_drop_fraction):
            reason = "EXIT_VOLATILITY_CRUSH"
        elif ((now - (trade.last_high_at or trade.opened_at)).total_seconds() > self._config.decay_minutes * 60
              and current_bid_price < trade.entry_ask_price * (1 - self._config.decay_fraction)):
            # A time/price heuristic, not a causal attribution of loss to Theta.
            reason = "EXIT_TIME_DECAY"
        elif (pain is not None and spot is not None and trade.previous_spot is not None
              and min(spot, trade.previous_spot) <= pain <= max(spot, trade.previous_spot)):
            reason = "EXIT_TAKE_PROFIT"
        elif pcr is not None and trade.entry_pcr and pcr <= trade.entry_pcr * (1 - self._config.pcr_drop_fraction):
            reason = "EXIT_PCR_DROP"
        elif (now - trade.opened_at).total_seconds() >= (trade.signal.max_holding_minutes or self._config.max_holding_minutes) * 60:
            reason = "EXIT_TIME_LIMIT"
        trade.previous_spot = spot
        trade.exit_reason = reason
        return reason is not None

    def _emit(self, trade: OpenTrade, alert_type: str, now: datetime) -> None:
        event = {"event_id": f"{trade.trade_id}:{trade.opened_at.isoformat()}:{alert_type}",
                 "timestamp": now.isoformat(), "alert_type": alert_type, "trade_id": trade.trade_id,
                 "action": trade.signal.action.value, "strike": trade.strike,
                 "premium": trade.entry_ask_price if alert_type == "ENTRY_BUY" else trade.current_bid_price}
        self._events.append(event)
        self._events = self._events[-200:]
        self._alerts.append(f"{trade.trade_id}: {alert_type}")
        if self._duckdb_manager is not None:
            try:
                self._duckdb_manager.insert_lifecycle_alert(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lifecycle alert persistence failed: {}", type(exc).__name__)

    def _pnl_pct(self, trade: OpenTrade) -> float:
        """Percentage return based on option premium change."""
        ref = trade.entry_ask_price if trade.entry_ask_price > 0 else trade.entry_price
        if ref <= 0.0:
            return 0.0
        return (trade.current_bid_price - ref) / ref

    def _realized_pnl(self, trade: OpenTrade, exit_price: float) -> float:
        """Backward-compatible helper redirecting to calculate_profit."""
        return self.calculate_profit(trade)


def calculate_option_atr(ticks: pd.DataFrame | None, *, now: datetime, period: int = 14) -> float | None:
    """SMA of 14 completed 1-minute midpoint true ranges, including close gaps.

    Only contiguous bars from the current market date are used. Missing/warm-up
    data suppresses entry; the forming minute and any future ticks are excluded.
    """
    if ticks is None or ticks.empty or not {"timestamp", "bid_price", "ask_price"}.issubset(ticks):
        return None
    frame = ticks.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["bid_price"] = pd.to_numeric(frame["bid_price"], errors="coerce")
    frame["ask_price"] = pd.to_numeric(frame["ask_price"], errors="coerce")
    stamp = pd.Timestamp(now)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    frame = frame.dropna(subset=["timestamp", "bid_price", "ask_price"])
    frame = frame.loc[(frame["timestamp"] < stamp.floor("min")) & (frame["bid_price"] > 0)
                      & (frame["ask_price"] >= frame["bid_price"])
                      & (frame["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date == stamp.tz_convert("Asia/Kolkata").date())]
    frame = frame.sort_values("timestamp").set_index("timestamp")
    frame["mid"] = (frame["bid_price"] + frame["ask_price"]) / 2
    bars = frame["mid"].resample("1min").ohlc().tail(period)
    if len(bars) < period or bars.isna().any().any():
        return None
    if bars.index[-1] != stamp.floor("min") - pd.Timedelta(minutes=1):
        return None
    previous_close = frame["mid"].resample("1min").last().shift(1).reindex(bars.index)
    true_range = pd.concat([bars["high"] - bars["low"], (bars["high"] - previous_close).abs(),
                            (bars["low"] - previous_close).abs()], axis=1).max(axis=1)
    atr = float(true_range.mean())
    return atr if math.isfinite(atr) and atr > 0 else None


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
            bid = _safe_float(row.get("bid_price") or row.get("bid"))
            ask = _safe_float(row.get("ask_price") or row.get("ask"))
            ts = _safe_datetime(row.get("timestamp") or row.get("exchange_timestamp"))
            return bid, ask, ts

        if isinstance(tick_data, dict):
            bid = _safe_float(tick_data.get("bid_price") or tick_data.get("bid"))
            ask = _safe_float(tick_data.get("ask_price") or tick_data.get("ask"))
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
            bid = _safe_float(last_row.get("bid_price") or last_row.get("bid"))
            ask = _safe_float(last_row.get("ask_price") or last_row.get("ask"))
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
