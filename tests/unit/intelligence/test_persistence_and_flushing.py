"""Unit tests for DuckDB persistence of signals and trade outcomes, and Parquet tick flushing.

Covers:
Step 1: DuckDBManager initialization creates 'signals' and 'trade_outcomes' tables.
Step 2: Prepared statement insertion methods 'insert_signal' and 'insert_trade_outcome'.
Step 3: MarketStateAggregator flush mechanism when option token buffer exceeds 5000 rows.
Step 4: LiveIntelligenceEngine persists evaluated setups (accepted and rejected) to 'signals'.
Step 5: LiveTradeMonitor persists trade outcomes upon closing to 'trade_outcomes', and lot size is 65.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from config.settings import AppSettings
from storage.duckdb_manager import DuckDBManager
from storage.parquet_writer import ParquetWriter
from trading_engine.intelligence.engine import LiveIntelligenceEngine
from trading_engine.intelligence.models import (
    DirectionalBias,
    LiquidityState,
    MarketRegime,
    MarketSnapshotBundle,
    MarketState,
    MarketTransition,
    MomentumState,
    OptionSide,
    ProbabilityResult,
    RegimeTransitionStage,
    SessionPhase,
    SignalAction,
    SignalResult,
    StrikeResult,
    TradeGrade,
    TrapResult,
    TrendState,
    VolatilityState,
)
from trading_engine.intelligence.monitor import LiveTradeMonitor, MonitorConfig
from trading_engine.intelligence.state import MarketStateAggregator, StateConfig


def _make_state(close: float = 24000.0) -> MarketState:
    now = datetime.now(UTC)
    transition = MarketTransition(
        stage=RegimeTransitionStage.TREND_EXPANSION,
        from_regime=MarketRegime.SIDEWAYS_COMPRESSION,
        to_regime=MarketRegime.TREND_BULLISH,
        confidence=0.85,
        persistence_probability=0.75,
        instability_probability=0.15,
        exhaustion_probability=0.10,
    )
    return MarketState(
        timestamp=now,
        trend_state=TrendState.TREND_BULLISH,
        volatility_state=VolatilityState.NORMAL,
        session_state=SessionPhase.TREND_WINDOW,
        liquidity_state=LiquidityState.NORMAL,
        momentum_state=MomentumState.BUILDING,
        option_chain_bias=0.25,
        breakout_pressure=0.002,
        trap_probability=0.15,
        mean_reversion_probability=0.20,
        trend_strength=0.004,
        directional_bias=DirectionalBias.BULLISH,
        regime=MarketRegime.TREND_BULLISH,
        regime_confidence=0.85,
        transition=transition,
        chop_probability=0.20,
        persistence_probability=0.75,
        instability_probability=0.15,
        exhaustion_probability=0.10,
        quality_score=0.80,
        trade_grade=TradeGrade.A,
        session_quality=0.85,
        signal_ttl_candles=3,
        is_tradeable=True,
        feature_snapshot={"close": close, "1m_trend_strength": 0.004},
        notes=["Test market state"],
    )


def test_step1_and_step2_duckdb_manager_tables_and_prepared_insertions(tmp_path: Path):
    """Step 1 & 2: DuckDBManager initializes relational tables and performs prepared writes."""
    db_file = tmp_path / "test_engine.duckdb"
    manager = DuckDBManager(db_file)

    now = datetime.now(UTC)

    # Test insert_signal
    manager.insert_signal(
        signal_id="SIG-001",
        timestamp=now,
        spot_price=24150.5,
        strike=24200.0,
        expiry="2026-10-22",
        option_type="CE",
        action="BUY_CE",
        confidence=0.88,
        trap_score=0.12,
        spread_percentage=0.015,
        iv_percentile=0.45,
        trade_grade="A",
        filter_passed=True,
        filter_rejection_reason=None,
    )

    # Test insert_trade_outcome
    manager.insert_trade_outcome(
        trade_id="TR-001",
        signal_id="SIG-001",
        instrument_token=998877,
        entry_timestamp=now,
        exit_timestamp=now + timedelta(minutes=8),
        entry_ask_price=120.0,
        exit_bid_price=135.0,
        quantity=2,
        holding_time_minutes=8.0,
        exit_reason="target_hit",
        realized_pnl=1950.0,  # 15 * 65 * 2
    )

    signals_df = manager.get_signals()
    assert len(signals_df) == 1
    row = signals_df.iloc[0]
    assert row["signal_id"] == "SIG-001"
    assert row["strike"] == 24200.0
    assert row["option_type"] == "CE"
    assert row["action"] == "BUY_CE"
    assert bool(row["filter_passed"]) is True
    assert row["filter_rejection_reason"] is None or pd.isna(row["filter_rejection_reason"])

    outcomes_df = manager.get_trade_outcomes()
    assert len(outcomes_df) == 1
    t_row = outcomes_df.iloc[0]
    assert t_row["trade_id"] == "TR-001"
    assert t_row["signal_id"] == "SIG-001"
    assert t_row["instrument_token"] == 998877
    assert t_row["entry_ask_price"] == 120.0
    assert t_row["exit_bid_price"] == 135.0
    assert t_row["quantity"] == 2
    assert t_row["exit_reason"] == "target_hit"
    assert t_row["realized_pnl"] == 1950.0

    manager.close()


def test_step3_state_aggregator_flushes_buffer_exceeding_5000_rows(tmp_path: Path):
    """Step 3: MarketStateAggregator flushes option token buffer to Parquet when rows > 5000."""
    settings = AppSettings.from_env()
    settings.storage.processed_dir = tmp_path / "processed"
    writer = ParquetWriter(settings.storage)
    aggregator = MarketStateAggregator(StateConfig(), parquet_writer=writer)

    token = 12345
    now = datetime.now(UTC)

    # Push 5000 rows -> buffer should be at 5000 rows without flushing yet
    for i in range(5000):
        aggregator.push_option_tick(
            instrument_token=token,
            bid=100.0 + (i % 10),
            ask=101.0 + (i % 10),
            timestamp=now + timedelta(seconds=i),
        )

    buf = aggregator.get_option_token_ticks(token)
    assert len(buf) == 5000

    # Push 5001st row -> should trigger flush and clear the buffer
    aggregator.push_option_tick(
        instrument_token=token,
        bid=105.0,
        ask=106.0,
        timestamp=now + timedelta(seconds=5001),
    )

    cleared_buf = aggregator.get_option_token_ticks(token)
    assert len(cleared_buf) == 0

    # Verify Parquet dataset was created on disk partitioned by date and instrument token
    dataset_dir = settings.storage.processed_dir / "option_ticks"
    assert dataset_dir.exists()
    parquet_files = list(dataset_dir.glob("**/*.parquet"))
    assert len(parquet_files) >= 1

    # Read back parquet and verify rows
    df_read = pd.read_parquet(dataset_dir)
    assert len(df_read) == 5001


def test_step4_live_intelligence_engine_persists_accepted_and_rejected_signals(tmp_path: Path, current_option_chain, option_history):
    """Step 4: LiveIntelligenceEngine writes evaluated setups to DuckDB signals table."""
    db_file = tmp_path / "test_signals.duckdb"
    duckdb_mgr = DuckDBManager(db_file)

    settings = AppSettings.from_env()
    engine = LiveIntelligenceEngine(settings=settings, duckdb_manager=duckdb_mgr)
    engine._momentum_engine.predict = lambda s, f: ProbabilityResult(
        bullish_probability=0.85,
        bearish_probability=0.08,
        sideways_probability=0.07,
        confidence=0.85,
        horizon_minutes=4,
    )

    now = datetime.now(UTC)

    # 1. Accepted setup: tight spread (0.01) and low IV (0.15)
    opt_chain_accepted = pd.DataFrame([
        {
            "instrument_token": 88001,
            "strike": 24000.0,
            "expiry": "2026-10-22",
            "option_type": "CE",
            "bid_price": 100.0,
            "ask_price": 101.0,  # 1% spread
            "iv": 0.15,
        }
    ])

    opt_chain_accepted = current_option_chain(opt_chain_accepted, now)
    opt_chain_accepted["timestamp"] = now
    key = (opt_chain_accepted.expiry.iloc[0], 24000., "CE")
    engine._signal_engine._historical_iv[key] = [.3] * 20
    underlying = option_history(now).rename(columns={"bid_price": "last_price"})
    underlying["last_price"] += 23900
    snapshot_accepted = MarketSnapshotBundle(
        timestamp=now,
        spot_price=24000.0,
        vix_value=13.0,
        spot_frame=underlying,
        vix_frame=pd.DataFrame([{"timestamp": now, "close": 13.0}]),
        option_chain_frame=opt_chain_accepted,
        ticks_frame=pd.DataFrame([{"timestamp": now, "instrument_token": 88001, "last_price": 100.5}]),
        candles_1m=pd.DataFrame([{"timestamp": now, "close": 24000.0}]),
        candles_5m=pd.DataFrame([{"timestamp": now, "close": 24000.0}]),
    )

    state = _make_state(close=24000.0)
    rep1 = engine.process(snapshot_accepted, state=state)
    assert rep1.signal.action == SignalAction.BUY_CE

    # Check signals table in DuckDB
    signals_df = duckdb_mgr.get_signals()
    assert len(signals_df) == 1
    s1 = signals_df.iloc[0]
    assert bool(s1["filter_passed"]) is True
    assert s1["action"].lower() == "buy_ce"
    assert s1["strike"] == 24000.0

    # 2. Rejected setup: wide spread (0.10 > 0.05)
    opt_chain_rejected = pd.DataFrame([
        {
            "instrument_token": 88001,
            "strike": 24000.0,
            "expiry": "2026-10-22",
            "option_type": "CE",
            "bid_price": 100.0,
            "ask_price": 110.0,  # 10% spread -> rejected
            "iv": 0.15,
        }
    ])

    opt_chain_rejected = current_option_chain(opt_chain_rejected, now)
    snapshot_rejected = MarketSnapshotBundle(
        timestamp=now + timedelta(seconds=10),
        spot_price=24000.0,
        vix_value=13.0,
        spot_frame=pd.DataFrame([{"timestamp": now + timedelta(seconds=10), "close": 24000.0}]),
        vix_frame=pd.DataFrame([{"timestamp": now + timedelta(seconds=10), "close": 13.0}]),
        option_chain_frame=opt_chain_rejected,
        ticks_frame=pd.DataFrame(),
        candles_1m=pd.DataFrame(),
        candles_5m=pd.DataFrame(),
    )

    rep2 = engine.process(snapshot_rejected, state=state)
    assert rep2.signal.action == SignalAction.NO_TRADE

    signals_df = duckdb_mgr.get_signals()
    assert len(signals_df) == 2
    s2 = signals_df.iloc[0]  # ordered by timestamp DESC
    assert bool(s2["filter_passed"]) is False
    assert "exceeds max allowed" in s2["filter_rejection_reason"] or "liquidity filter" in s2["filter_rejection_reason"]

    duckdb_mgr.close()


def test_step5_monitor_persists_trade_outcomes_and_uses_65_multiplier(tmp_path: Path, option_history):
    """Step 5: LiveTradeMonitor persists outcomes to DuckDB on close and calculates profit with 65 multiplier."""
    db_file = tmp_path / "test_monitor.duckdb"
    duckdb_mgr = DuckDBManager(db_file)

    assert LiveTradeMonitor.NIFTY_LOT_SIZE == 65

    monitor = LiveTradeMonitor(
        config=MonitorConfig(max_holding_minutes=5),
        duckdb_manager=duckdb_mgr,
    )

    now = datetime.now(UTC)
    sig = SignalResult(
        action=SignalAction.BUY_CE,
        confidence=0.85,
        regime=MarketRegime.TREND_BULLISH,
        probability=ProbabilityResult(0.85, 0.10, 0.05, 0.85, 4),
        trap=TrapResult(0.1, 0.1, 0.1, 0.1),
        entry_reference=100.0,
        stop_loss=90.0,
        target=130.0,
        strike=StrikeResult(OptionSide.CE, 24500.0, "2026-10-22", "ATM", delta=.7),
        quantity=2, underlying_atr=10., entry_iv=20., entry_pcr=1.6,
        signal_id="SIG-TEST-65",
    )

    # 1. Test profit calculation multiplier = 65
    trade = monitor.open_trade(
        signal=sig,
        entry_price=100.0,
        quantity=2,
        instrument_token=7777,
        strike=24500.0,
        expiry="2026-10-22",
        option_type="CE",
        entry_ask_price=100.0,
        signal_timestamp=now,
        option_ticks=option_history(now),
    )

    trade.current_bid_price = 110.0
    profit = monitor.calculate_profit(trade)
    # (110 - 100) * 65 * 2 = 10 * 65 * 2 = 1300.0
    assert profit == pytest.approx(1300.0)

    # 2. Trigger stop loss exit: bid drops to 88.0 <= 90.0
    fresh_ticks = {
        7777: {
            "bid_price": 88.0,
            "ask_price": 89.0,
            "timestamp": now + timedelta(seconds=1),
        }
    }
    update = monitor.update(fresh_ticks)

    # Trade must be removed from active open trades
    assert len(update.open_trades) == 0
    assert len(monitor.open_trades) == 0
    assert len(update.closed_trades) == 1

    # Trade outcome must be written to DuckDB table
    outcomes = duckdb_mgr.get_trade_outcomes()
    assert len(outcomes) == 1
    closed = outcomes.iloc[0]
    assert closed["trade_id"] == trade.trade_id
    assert closed["signal_id"] == "SIG-TEST-65"
    assert closed["instrument_token"] == 7777
    assert closed["entry_ask_price"] == 100.0
    assert closed["exit_bid_price"] == 88.0
    assert closed["quantity"] == 2
    assert closed["exit_reason"] == "EXIT_STOP_LOSS"
    # PnL: (88.0 - 100.0) * 65 * 2 = -1560.0
    assert closed["realized_pnl"] == pytest.approx(-1560.0)

    duckdb_mgr.close()
