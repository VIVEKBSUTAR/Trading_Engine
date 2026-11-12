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


def emit(*, ratio=1.6, action=SignalAction.BUY_CE, days=0, missing_oi=False):
    now = datetime.now(UTC)
    context = state(now)
    expiry = (pd.Timestamp(now).tz_convert("Asia/Kolkata").date() + timedelta(days=days)).isoformat()
    chain = pd.DataFrame([{"strike": 24500.0, "expiry": expiry, "option_type": side,
        "bid_price": 100.0, "ask_price": 101.0, "iv": 20.0, "delta": .7 if side == "CE" else -.7,
        "open_interest": None if missing_oi else (100 if side == "CE" else 100 * ratio)} for side in ("CE", "PE")])
    probs = ProbabilityResult(0.9, 0.05, 0.05, 0.9, 4) if action == SignalAction.BUY_CE else ProbabilityResult(0.05, 0.9, 0.05, 0.9, 4)
    signal = SignalGenerationEngine().generate(state=context, probabilities=probs,
        trap=TrapResult(0.1, 0.1, 0.1, 0.1), regime=RegimeResult(context.regime, 0.9),
        selected_strike=StrikeResult(OptionSide.CE if action == SignalAction.BUY_CE else OptionSide.PE, 24500.0, expiry, "atm"),
        option_chain=chain, snapshot_timestamp=now, historical_iv_window=[25.0] * 20)
    signal.underlying_atr = 10
    return signal


@pytest.mark.parametrize("days,limit", [(0, 15), (1, 45), (7, 45)])
def test_dte_limits_embedded_in_signal(days, limit):
    signal = emit(days=days)
    assert signal.action == SignalAction.BUY_CE
    assert signal.days_to_expiry == days
    assert signal.max_holding_minutes == limit


@pytest.mark.parametrize("action,ratio,accepted", [
    (SignalAction.BUY_CE, 1.5, False), (SignalAction.BUY_CE, 1.51, True),
    (SignalAction.BUY_PE, 1.6, True), (SignalAction.BUY_PE, 1.49, False),
])
def test_oi_directional_threshold_boundaries(action, ratio, accepted):
    signal = emit(action=action, ratio=ratio)
    assert (signal.action == action) is accepted
    if accepted:
        assert signal.strike_oi_ratio == pytest.approx(ratio)
    if not accepted:
        assert signal.action == SignalAction.NO_TRADE
        assert "PCR" in signal.filter_rejection_reason


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


def test_delta_adjusted_stop_trails_immediately_and_never_widens():
    now = datetime.now(UTC)
    signal = emit()
    monitor = LiveTradeMonitor()
    trade = monitor.open_trade(signal, 100, 1, instrument_token=7, expiry=signal.strike.expiry,
                               strike=24500, option_type="CE")
    assert trade.stop_loss_price == pytest.approx(93)
    monitor.context = {"atr": 10, "spot": 24500}
    monitor.contract_context[(trade.expiry, 24500, "CE")] = {"timestamp": now, "delta": .7, "iv": 20}
    def mark(bid):
        return monitor.update({7: {"timestamp": datetime.now(UTC), "bid_price": bid, "ask_price": bid + .1}})
    mark(103)
    assert trade.stop_loss_price == pytest.approx(96)
    monitor.context["atr"] = 20
    mark(102)
    assert trade.stop_loss_price == pytest.approx(96)
    closed = mark(96)
    assert closed.closed_trades[0].exit_reason == "EXIT_STOP_LOSS"


@pytest.mark.parametrize("days,minutes", [(0, 15), (1, 45)])
def test_holding_limit_enforced_independently_of_admission_ttl(days, minutes):
    now = datetime.now(UTC)
    signal = emit(days=days)
    monitor = LiveTradeMonitor()
    trade = monitor.open_trade(signal, 100, 1, instrument_token=7, option_ticks=tick_history(now))
    trade.opened_at = datetime.now(UTC) - timedelta(minutes=minutes)
    update = monitor.update({7: {"timestamp": datetime.now(UTC), "bid_price": 100, "ask_price": 101}})
    assert update.closed_trades[0].exit_reason == "EXIT_TIME_LIMIT"


def test_missing_atr_suppresses_admission():
    monitor = LiveTradeMonitor()
    signal = emit()
    signal.underlying_atr = None
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
         "bid_price": 100, "ask_price": 101, "iv": 20, "delta": .7 if side == "CE" else -.7, "open_interest": oi}
        for expiry, side, oi in [
            ("11-11-2026", "CE", 100), ("11-11-2026", "PE", 160),
            ("12-11-2026", "CE", 10000), ("12-11-2026", "PE", 1),
        ]
    ])
    signal = SignalGenerationEngine().generate(
        state=context, probabilities=ProbabilityResult(0.9, 0.05, 0.05, 0.9, 4),
        trap=TrapResult(0.1, 0.1, 0.1, 0.1), regime=RegimeResult(context.regime, 0.9),
        selected_strike=StrikeResult(OptionSide.CE, 24500, "2026-11-11", "atm"),
        option_chain=chain, snapshot_timestamp=snapshot_time, historical_iv_window=[25] * 20,
    )
    assert signal.action == SignalAction.BUY_CE
    assert signal.days_to_expiry == 0
    assert signal.max_holding_minutes == 15
    assert signal.strike_oi_ratio == 1.6


@pytest.mark.parametrize("side,expected", [("CE", .5398278373), ("PE", -.4601721627)])
def test_black_scholes_delta_reference_and_invalid_inputs(side, expected):
    from trading_engine.intelligence.options_math import option_delta
    now = datetime(2025, 1, 1, 10, 0, tzinfo=UTC)  # exactly 365 days to 15:30 IST expiry
    assert option_delta(100, 100, "2026-01-01", 20, side, now) == pytest.approx(expected)
    for iv in (0, None, float("nan")):
        result = option_delta(100, 100, "2026-01-01", iv, side, now)
        assert result is None or not pd.notna(result)
    assert option_delta(100, 100, "2024-01-01", 20, side, now) is None


def test_nifty_atr_uses_index_prices_and_complete_bars():
    from trading_engine.intelligence.options_math import underlying_atr
    now = datetime.now(UTC)
    frame = tick_history(now).rename(columns={"bid_price": "last_price"})
    frame.last_price += 24400
    assert underlying_atr(frame, now=now) == pytest.approx(2)
    assert underlying_atr(frame.iloc[-20:], now=now) is None
    assert underlying_atr(pd.concat([frame, pd.DataFrame([{"timestamp": now, "last_price": 1e9}])]), now=now) == pytest.approx(2)


@pytest.mark.parametrize("reason,bid,iv,pcr,spot,age", [
    ("EXIT_STOP_LOSS", 60, 15, 1.0, 24520, 21),
    ("EXIT_VOLATILITY_CRUSH", 99, 17.9, 1.6, 24500, 0),
    ("EXIT_TIME_DECAY", 97, 20, 1.6, 24500, 21),
    ("EXIT_TAKE_PROFIT", 105, 20, 1.6, 24520, 0),
    ("EXIT_PCR_DROP", 100, 20, 1.28, 24500, 0),
])
def test_exit_routing_and_persistence_exactly_once(tmp_path, reason, bid, iv, pcr, spot, age):
    now = datetime.now(UTC)
    store = DuckDBManager(tmp_path / "lifecycle.duckdb")
    signal = emit(days=1)
    signal.underlying_atr = 50  # wide dynamic stop isolates non-stop exit cases
    monitor = LiveTradeMonitor(duckdb_manager=store)
    monitor.context = {"spot": 24500}
    trade = monitor.open_trade(signal, 100, 1, instrument_token=7, strike=24500,
                               expiry=signal.strike.expiry, option_type="CE")
    trade.opened_at = now - timedelta(minutes=age)
    trade.last_high_at = trade.opened_at
    monitor.context = {"spot": spot, "atr": 50}
    monitor.contract_context[(trade.expiry, 24500, "CE")] = {
        "timestamp": now, "delta": .7, "iv": iv, "pcr": pcr, "max_pain": 24510}
    ticks = {7: {"timestamp": now, "bid_price": bid, "ask_price": bid + 1}}
    update = monitor.update(ticks)
    assert update.closed_trades[0].exit_reason == reason
    assert update.lifecycle_alerts[-1]["alert_type"] == reason
    monitor.update(ticks)
    events = store.get_lifecycle_alerts(now=now)
    assert events.alert_type.tolist().count(reason) == 1
    assert set(events.alert_type) == {"ENTRY_BUY", reason}
    assert store.get_trade_outcomes().exit_reason.iloc[0] == reason
    store.close()


def test_exit_boundaries_stale_context_and_missing_greeks_do_not_crash():
    now = datetime.now(UTC)
    signal = emit(days=1)
    signal.underlying_atr = 50
    monitor = LiveTradeMonitor()
    trade = monitor.open_trade(signal, 100, 1, instrument_token=7, strike=24500,
                               expiry=signal.strike.expiry, option_type="CE")
    trade.last_high_at = now - timedelta(minutes=20)
    monitor.context = {"spot": 24500, "atr": None}
    key = (trade.expiry, 24500, "CE")
    monitor.contract_context[key] = {"timestamp": now, "iv": 18, "delta": float("nan"), "pcr": 1.6}
    ticks = {7: {"timestamp": now, "bid_price": 98, "ask_price": 99}}
    assert monitor.update(ticks).open_trades  # exactly 20 minutes, 2%, 10% are not > thresholds
    monitor.contract_context[key] = {"timestamp": now - timedelta(seconds=46), "iv": 1, "pcr": .1}
    assert monitor.update(ticks).open_trades
    assert trade.live_iv is None
    ticks[7]["bid_price"] = 60
    assert monitor.update(ticks).closed_trades[0].exit_reason == "EXIT_STOP_LOSS"


def test_iv_warmup_counts_distinct_snapshots_not_ticks():
    engine = SignalGenerationEngine()
    now = datetime.now(UTC)
    chain = pd.DataFrame([{"expiry": "2026-11-19", "strike": 24500, "option_type": "CE",
                           "bid_price": 100, "ask_price": 101, "iv": 20, "timestamp": now}])
    for _ in range(30):
        passed, _, percentile, _ = engine.check_volatility_filter(24500, "CE", chain)
        assert not passed and percentile is None
    assert len(next(iter(engine._historical_iv.values()))) == 1


def test_delta_selector_handles_put_signs_and_invalid_data():
    from trading_engine.intelligence.engines import StrikeSelectionEngine
    now = datetime.now(UTC)
    context = state(now)
    signal = emit(action=SignalAction.BUY_PE, days=1)
    chain = pd.DataFrame([{"expiry": signal.strike.expiry, "strike": strike, "option_type": "PE", "delta": delta}
                          for strike, delta in [(24500, -.5), (24550, -.69), (24600, -.75)]])
    assert StrikeSelectionEngine().select(context, signal, chain).strike == 24550
    chain.delta = float("nan")
    assert StrikeSelectionEngine().select(context, signal, chain).strike is None


@pytest.mark.parametrize("current,expected", [(18, True), (19, False), (20, False)])
def test_iv_percentile_strict_fiftieth_boundary(current, expected):
    chain = pd.DataFrame([{"expiry": "2026-11-19", "strike": 24500, "option_type": "CE",
                           "bid_price": 100, "ask_price": 101, "iv": current}])
    passed, _, _, _ = SignalGenerationEngine().check_volatility_filter(
        24500, "CE", chain, historical_iv_window=list(range(10, 30)))
    assert passed is expected


def test_trap_point_five_suppresses_candidate():
    now = datetime.now(UTC)
    context = state(now)
    signal = SignalGenerationEngine().generate(state=context,
        probabilities=ProbabilityResult(.9, .05, .05, .9, 4),
        trap=TrapResult(.1, .1, .1, .1, trap_score=.5),
        regime=RegimeResult(context.regime, .9), option_chain=pd.DataFrame())
    assert signal.action == SignalAction.NO_TRADE


def test_positioning_deduplicates_snapshots_and_rejects_mixed_expiries():
    from trading_engine.intelligence.options_math import positioning
    frame = pd.DataFrame([{"expiry": "2026-11-19", "strike": 24500, "option_type": side,
                           "open_interest": oi} for side, oi in [("CE", 100), ("PE", 160)]])
    pcr, pain = positioning(pd.concat([frame, frame]))
    assert pcr == pytest.approx(1.6) and pain == 24500
    other = frame.copy()
    other.expiry = "2026-11-26"
    assert positioning(pd.concat([frame, other])) == (None, None)


def test_fyers_nested_greeks_units_and_no_synthetic_quotes():
    from broker.fyers_client import FyersMarketClient
    client = object.__new__(FyersMarketClient)
    frame = client.normalize_option_chain({"data": {"optionsChain": [{
        "strike_price": 24500, "option_type": "PE", "symbol": "NSE:NIFTYTESTPE",
        "expiry": "19-11-2026", "ltp": 100, "oi": 160,
        "greeks": {"delta": -.7, "iv": 20}}]}})
    assert frame.delta.iloc[0] == -.7
    assert frame.implied_volatility.iloc[0] == 20  # provider percentage units
    assert frame.bid_price.iloc[0] == 0 and frame.ask_price.iloc[0] == 0
