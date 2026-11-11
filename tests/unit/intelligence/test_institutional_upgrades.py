"""Contract-specific OI/DTE, option ATR lifecycle, and schema migration tests."""
from datetime import UTC, datetime, timedelta

import duckdb
import pandas as pd
import pytest

from storage.duckdb_manager import DuckDBManager
from trading_engine.intelligence.engines import SignalGenerationEngine
from trading_engine.intelligence.models import (
    DirectionalBias,
    LiquidityState,
    MarketRegime,
    MarketState,
    MarketTransition,
    MomentumState,
    OptionSide,
    ProbabilityResult,
    RegimeResult,
    RegimeTransitionStage,
    SessionPhase,
    SignalAction,
    StrikeResult,
    TradeGrade,
    TrapResult,
    TrendState,
    VolatilityState,
)
from trading_engine.intelligence.monitor import LiveTradeMonitor, calculate_option_atr

pytestmark = pytest.mark.usefixtures("market_clock")

def tick_history(now, *, base=100.0):
    end = pd.Timestamp(now).floor("min")
    rows = []
    for minute in range(15, 0, -1):
        for seconds, mid in ((1, base), (30, base + 2)):
            rows.append({"timestamp": end - pd.Timedelta(minutes=minute) + pd.Timedelta(seconds=seconds),
                         "bid_price": mid - 0.1, "ask_price": mid + 0.1})
    return pd.DataFrame(rows)


def state(now):
    regime = MarketRegime.TREND_BULLISH
    transition = MarketTransition(RegimeTransitionStage.TREND_EXPANSION, regime, regime, 0.9)
    return MarketState(now, TrendState.TREND_BULLISH, VolatilityState.NORMAL,
        SessionPhase.TREND_WINDOW, LiquidityState.NORMAL, MomentumState.BUILDING,
        0.1, 0.1, 0.1, 0.1, 0.01, DirectionalBias.BULLISH, regime, 0.9,
        transition, 0.1, 0.8, 0.1, 0.1, 0.9, TradeGrade.A, 0.9, 2, True,
        feature_snapshot={"close": 24500.0})


def emit(*, ratio=1.0, action=SignalAction.BUY_CE, days=0, missing_oi=False):
    now = datetime.now(UTC)
    context = state(now)
    expiry = (pd.Timestamp(now).tz_convert("Asia/Kolkata").date() + timedelta(days=days)).isoformat()
    chain = pd.DataFrame([{"strike": 24500.0, "expiry": expiry, "option_type": side,
        "bid_price": 100.0, "ask_price": 101.0, "iv": 0.2,
        "open_interest": None if missing_oi else (100 if side == "CE" else 100 * ratio)} for side in ("CE", "PE")])
    probs = ProbabilityResult(0.9, 0.05, 0.05, 0.9, 4) if action == SignalAction.BUY_CE else ProbabilityResult(0.05, 0.9, 0.05, 0.9, 4)
    signal = SignalGenerationEngine().generate(state=context, probabilities=probs,
        trap=TrapResult(0.1, 0.1, 0.1, 0.1), regime=RegimeResult(context.regime, 0.9),
        selected_strike=StrikeResult(OptionSide.CE if action == SignalAction.BUY_CE else OptionSide.PE, 24500.0, expiry, "atm"),
        option_chain=chain, snapshot_timestamp=now)
    return signal


@pytest.mark.parametrize("days,limit", [(0, 15), (1, 45), (7, 45)])
def test_dte_limits_embedded_in_signal(days, limit):
    signal = emit(days=days)
    assert signal.action == SignalAction.BUY_CE
    assert signal.days_to_expiry == days
    assert signal.max_holding_minutes == limit


@pytest.mark.parametrize("action,ratio,accepted", [
    (SignalAction.BUY_CE, 0.79, False), (SignalAction.BUY_CE, 0.8, True),
    (SignalAction.BUY_PE, 1.2, True), (SignalAction.BUY_PE, 1.21, False),
])
def test_oi_directional_threshold_boundaries(action, ratio, accepted):
    signal = emit(action=action, ratio=ratio)
    assert (signal.action == action) is accepted
    assert signal.strike_oi_ratio == pytest.approx(ratio)
    if not accepted:
        assert signal.action == SignalAction.NO_TRADE
        assert "OI imbalance" in signal.filter_rejection_reason


def test_missing_oi_and_expired_contract_fail_closed():
    assert emit(missing_oi=True).action == SignalAction.NO_TRADE
    assert emit(days=-1).action == SignalAction.NO_TRADE


def test_atr_uses_completed_minutes_and_rejects_gaps():
    now = datetime.now(UTC)
    frame = tick_history(now)
    assert calculate_option_atr(frame, now=now) == pytest.approx(2)
    forming = pd.DataFrame([{"timestamp": now, "bid_price": 1, "ask_price": 99999}])
    assert calculate_option_atr(pd.concat([frame, forming]), now=now) == pytest.approx(2)
    assert calculate_option_atr(frame.iloc[-10:], now=now) is None
    assert calculate_option_atr(frame.drop(frame.index[-12:-10]), now=now) is None


def test_atr_captures_close_gaps():
    now = datetime.now(UTC)
    frame = tick_history(now)
    frame.loc[frame.index[-2:], ["bid_price", "ask_price"]] += 10
    # Last bar high=112, previous close=102 => TR=10, other 13 bars TR=2.
    assert calculate_option_atr(frame, now=now) == pytest.approx((13 * 2 + 10) / 14)


def test_entry_stop_trails_only_after_one_r_and_never_widens():
    now = datetime.now(UTC)
    signal = emit()
    monitor = LiveTradeMonitor()
    trade = monitor.open_trade(signal, 100, 1, instrument_token=7,
                               entry_ask_price=100, signal_timestamp=now, option_ticks=tick_history(now))
    assert trade.stop_loss_price == pytest.approx(97)
    assert trade.signal.stop_loss == pytest.approx(97)
    assert trade.initial_risk == pytest.approx(3)
    def mark(bid):
        return monitor.update({7: {"timestamp": datetime.now(UTC), "bid_price": bid, "ask_price": bid + 0.1}})
    mark(102.9)
    assert not trade.trailing_active
    mark(103)
    assert trade.trailing_active
    assert trade.stop_loss_price == pytest.approx(101)
    mark(102)
    assert trade.stop_loss_price == pytest.approx(101)
    closed = mark(100.5)
    assert closed.closed_trades[0].exit_reason == "trailing_stop"


@pytest.mark.parametrize("days,minutes", [(0, 15), (1, 45)])
def test_holding_limit_enforced_independently_of_admission_ttl(days, minutes):
    now = datetime.now(UTC)
    signal = emit(days=days)
    monitor = LiveTradeMonitor()
    trade = monitor.open_trade(signal, 100, 1, instrument_token=7, option_ticks=tick_history(now))
    trade.opened_at = datetime.now(UTC) - timedelta(minutes=minutes)
    update = monitor.update({7: {"timestamp": datetime.now(UTC), "bid_price": 100, "ask_price": 101}})
    assert update.closed_trades[0].exit_reason == "time_stop"


def test_missing_atr_suppresses_admission():
    monitor = LiveTradeMonitor()
    signal = emit()
    signal.quantity = 1
    assert monitor.open_trade_from_signal(signal, instrument_token=7, entry_ask_price=100) is None
    assert not monitor.open_trades


def test_old_duckdb_schema_is_migrated_and_filters_round_trip(tmp_path):
    path = tmp_path / "old.duckdb"
    original = DuckDBManager(path)
    original.close()
    with duckdb.connect(str(path)) as con:
        con.execute("DROP INDEX idx_signals_timestamp")
        for column in ("max_holding_minutes", "days_to_expiry", "strike_oi_ratio"):
            con.execute(f"ALTER TABLE signals DROP COLUMN {column}")
    store = DuckDBManager(path)
    store.insert_signal(signal_id="s1", timestamp=datetime.now(UTC), spot_price=24500,
        strike=24500, expiry="2026-11-11", option_type="CE", action="buy_ce", confidence=0.9,
        trap_score=0.1, spread_percentage=0.01, iv_percentile=0.5, trade_grade="A", filter_passed=True,
        max_holding_minutes=15, days_to_expiry=0, strike_oi_ratio=1.1)
    row = store.get_signals().iloc[0]
    assert row["max_holding_minutes"] == 15
    assert row["days_to_expiry"] == 0
    assert row["strike_oi_ratio"] == pytest.approx(1.1)
    store.close()


def test_snapshot_market_date_and_exact_expiry_oi_pair():
    # UTC Nov 10 is already Nov 11 in the market timezone. Later expiry OI
    # deliberately disagrees, so combining expiries would reject this call.
    snapshot_time = datetime(2026, 11, 10, 19, 0, tzinfo=UTC)
    context = state(snapshot_time)
    chain = pd.DataFrame([
        {"strike": 24500, "expiry": expiry, "option_type": side,
         "bid_price": 100, "ask_price": 101, "iv": 0.2, "open_interest": oi}
        for expiry, side, oi in [
            ("11-11-2026", "CE", 100), ("11-11-2026", "PE", 100),
            ("12-11-2026", "CE", 10000), ("12-11-2026", "PE", 1),
        ]
    ])
    signal = SignalGenerationEngine().generate(
        state=context, probabilities=ProbabilityResult(0.9, 0.05, 0.05, 0.9, 4),
        trap=TrapResult(0.1, 0.1, 0.1, 0.1), regime=RegimeResult(context.regime, 0.9),
        selected_strike=StrikeResult(OptionSide.CE, 24500, "2026-11-11", "atm"),
        option_chain=chain, snapshot_timestamp=snapshot_time,
    )
    assert signal.action == SignalAction.BUY_CE
    assert signal.days_to_expiry == 0
    assert signal.max_holding_minutes == 15
    assert signal.strike_oi_ratio == 1.0
