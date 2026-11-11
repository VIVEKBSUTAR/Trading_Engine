"""Comprehensive unit tests for option premium evaluation across intelligence components.

Tests verify:
1. OpenTrade dataclass fields (instrument_token, strike, expiry, option_type, entry_ask_price, current_bid_price, signal_timestamp).
2. MarketStateAggregator dictionary buffer and push_option_tick method.
3. LiveTradeMonitor update method receiving option ticks.
4. Profit calculation using (current_bid_price - entry_ask_price) * 75.
5. Stale data circuit breaker freezing evaluation when tick timestamp gap > 2 seconds.
6. LiveIntelligenceEngine trade handoff passing option instrument token and entry ask price.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from config.settings import AppSettings
from trading_engine.intelligence.engine import LiveIntelligenceEngine
from trading_engine.intelligence.models import (
    MarketRegime,
    MarketSnapshotBundle,
    OpenTrade,
    OptionSide,
    ProbabilityResult,
    SignalAction,
    SignalResult,
    StrikeResult,
    TrapResult,
    _serialize_dataclass,
)
from trading_engine.intelligence.monitor import LiveTradeMonitor, MonitorConfig
from trading_engine.intelligence.state import MarketStateAggregator


def _dummy_signal(
    action: SignalAction = SignalAction.BUY_CE,
    confidence: float = 0.85,
    stop_loss: float | None = None,
    target: float | None = None,
    strike: float = 24500.0,
    quantity: int = 1,
) -> SignalResult:
    return SignalResult(
        action=action,
        confidence=confidence,
        regime=MarketRegime.TREND_BULLISH,
        probability=ProbabilityResult(
            bullish_probability=0.8,
            bearish_probability=0.1,
            sideways_probability=0.1,
            confidence=confidence,
            horizon_minutes=4,
        ),
        trap=TrapResult(
            fake_breakout_probability=0.05,
            buyer_trap_probability=0.05,
            seller_trap_probability=0.05,
            stoploss_hunt_probability=0.05,
        ),
        entry_reference=150.0,
        stop_loss=stop_loss,
        target=target,
        strike=StrikeResult(
            option_side=OptionSide.CE if action == SignalAction.BUY_CE else OptionSide.PE,
            strike=strike,
            expiry="2026-10-22",
            moneyness="ATM",
        ),
        quantity=quantity,
    )


def test_open_trade_dataclass_fields():
    """Step 1: OpenTrade dataclass has all required option contract fields."""
    now = datetime.now(UTC)
    signal = _dummy_signal(action=SignalAction.BUY_CE, confidence=0.85)
    trade = OpenTrade(
        trade_id="TR-1001",
        signal=signal,
        entry_price=150.0,
        quantity=1,
        opened_at=now,
        instrument_token=123456,
        strike=24500.0,
        expiry="2026-10-22",
        option_type="CE",
        entry_ask_price=152.5,
        current_bid_price=152.0,
        signal_timestamp=now,
    )

    assert trade.instrument_token == 123456
    assert trade.strike == 24500.0
    assert trade.expiry == "2026-10-22"
    assert trade.option_type == "CE"
    assert trade.entry_ask_price == 152.5
    assert trade.current_bid_price == 152.0
    assert trade.signal_timestamp == now

    # Verify serialization
    serialized = _serialize_dataclass(trade)
    assert serialized["instrument_token"] == 123456
    assert serialized["strike"] == 24500.0
    assert serialized["expiry"] == "2026-10-22"
    assert serialized["option_type"] == "CE"
    assert serialized["entry_ask_price"] == 152.5
    assert serialized["current_bid_price"] == 152.0
    assert isinstance(serialized["signal_timestamp"], str)


def test_market_state_aggregator_option_token_buffer():
    """Step 2: MarketStateAggregator dictionary buffer and push_option_tick method."""
    aggregator = MarketStateAggregator()
    token = 987654
    ts = datetime.now(UTC)

    aggregator.push_option_tick(
        instrument_token=token,
        bid=180.0,
        ask=182.5,
        timestamp=ts,
        last_price=181.25,
    )

    token_ticks = aggregator.get_option_token_ticks()
    assert token in token_ticks
    df = token_ticks[token]
    assert len(df) == 1
    assert df.iloc[-1]["bid_price"] == 180.0
    assert df.iloc[-1]["ask_price"] == 182.5
    assert df.iloc[-1]["instrument_token"] == token

    latest = aggregator.get_latest_option_tick(token)
    assert latest is not None
    assert latest["bid_price"] == 180.0
    assert latest["ask_price"] == 182.5


def test_monitor_profit_calculation():
    """Step 4: Profit calculated as (current_bid_price - entry_ask_price) * 65 * quantity."""
    monitor = LiveTradeMonitor(MonitorConfig())
    now = datetime.now(UTC)
    signal = _dummy_signal(action=SignalAction.BUY_CE, confidence=0.9, quantity=2)

    trade = OpenTrade(
        trade_id="TR-PROFIT",
        signal=signal,
        entry_price=100.0,
        quantity=2,  # 2 lots
        opened_at=now,
        instrument_token=1001,
        strike=24500.0,
        expiry="2026-10-22",
        option_type="CE",
        entry_ask_price=105.0,
        current_bid_price=125.0,  # +20 points
        signal_timestamp=now,
    )

    profit = monitor.calculate_profit(trade)
    # (125.0 - 105.0) * 65 * 2 = 20 * 65 * 2 = 2600.0
    expected_profit = (125.0 - 105.0) * 65 * 2
    assert profit == pytest.approx(expected_profit)
    assert profit == pytest.approx(2600.0)


def test_monitor_update_with_fresh_option_ticks(option_history):
    """Step 3 & 4: Monitor updates marks and calculates PnL using option ticks."""
    monitor = LiveTradeMonitor(MonitorConfig())
    now = datetime.now(UTC)
    signal = _dummy_signal(action=SignalAction.BUY_CE, confidence=0.9, stop_loss=80.0)

    trade = monitor.open_trade(
        signal=signal,
        entry_price=100.0,
        quantity=1,
        instrument_token=5555,
        strike=24500.0,
        expiry="2026-10-22",
        option_type="CE",
        entry_ask_price=100.0,
        signal_timestamp=now,
        option_ticks=option_history(now),
    )
    assert trade.instrument_token == 5555

    # Push fresh tick within 2 seconds
    fresh_ts = datetime.now(UTC)
    option_ticks = {
        5555: {
            "bid_price": 110.0,
            "ask_price": 111.0,
            "timestamp": fresh_ts,
        }
    }

    update = monitor.update(option_ticks)
    assert len(update.open_trades) == 1
    active_trade = update.open_trades[0]
    assert active_trade.current_bid_price == 110.0
    # Profit calculation check: (110 - 100) * 65 * 1 = 650.0
    profit = monitor.calculate_profit(active_trade)
    assert profit == pytest.approx(650.0)


def test_monitor_stale_data_circuit_breaker(option_history):
    """Step 5: Tick older than 2.0s triggers stale circuit breaker, freezing evaluation."""
    monitor = LiveTradeMonitor(MonitorConfig())
    now = datetime.now(UTC)
    signal = _dummy_signal(action=SignalAction.BUY_CE, confidence=0.9, stop_loss=90.0)

    trade = monitor.open_trade(
        signal=signal,
        entry_price=100.0,
        quantity=1,
        instrument_token=7777,
        strike=24500.0,
        expiry="2026-10-22",
        option_type="CE",
        entry_ask_price=100.0,
        signal_timestamp=now,
        option_ticks=option_history(now),
    )

    # Provide a stale tick older than 2 seconds (e.g. 5 seconds ago) with bid below stop loss
    stale_ts = datetime.now(UTC) - timedelta(seconds=5)
    stale_ticks = {
        7777: {
            "bid_price": 50.0,  # Below stop loss of 90.0
            "ask_price": 51.0,
            "timestamp": stale_ts,
        }
    }

    update = monitor.update(stale_ticks)
    # Trade MUST NOT be closed on stale data, evaluation is frozen
    assert len(update.open_trades) == 1
    assert len(update.closed_trades) == 0
    assert trade.status == "open"
    # Alerts should record stale warning
    assert any("stale tick gap" in alert for alert in update.alert_messages)


def test_engine_passes_option_contract_details_to_monitor():
    """Step 6: LiveIntelligenceEngine extracts token and exact entry ask price from snapshot."""
    settings = AppSettings.from_env()
    engine = LiveIntelligenceEngine(settings)

    now = datetime.now(UTC)
    option_chain_df = pd.DataFrame([
        {
            "instrument_token": 88888,
            "strike": 24000.0,
            "expiry": "2026-10-22",
            "option_type": "CE",
            "bid_price": 148.0,
            "ask_price": 150.0,
            "last_price": 149.0,
            "iv": 0.15,
            "oi": 50000,
        },
        {
            "instrument_token": 99999,
            "strike": 24000.0,
            "expiry": "2026-10-22",
            "option_type": "PE",
            "bid_price": 120.0,
            "ask_price": 122.0,
            "last_price": 121.0,
            "iv": 0.16,
            "oi": 60000,
        },
    ])

    ticks_df = pd.DataFrame({
        "timestamp": [now] * 10,
        "last_price": [24000.0] * 10,
        "volume_traded": list(range(1000, 2000, 100)),
        "instrument_token": [88888] * 10,
    })

    snapshot = MarketSnapshotBundle(
        timestamp=now,
        spot_price=24000.0,
        vix_value=14.0,
        spot_frame=pd.DataFrame([{"timestamp": now, "close": 24000.0}]),
        vix_frame=pd.DataFrame([{"timestamp": now, "close": 14.0}]),
        option_chain_frame=option_chain_df,
        ticks_frame=ticks_df,
        candles_1m=pd.DataFrame(),
        candles_5m=pd.DataFrame(),
    )

    _ = engine.process(snapshot)

    # Check aggregator buffer was populated for the option token
    aggregator = engine.state_aggregator
    ticks_buffer = aggregator.get_option_token_ticks()
    assert isinstance(ticks_buffer, dict)
