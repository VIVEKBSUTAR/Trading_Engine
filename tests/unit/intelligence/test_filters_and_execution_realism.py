"""Unit tests for liquidity filter, volatility filter, config settings, and execution timing.

Verifies:
Step 1: Liquidity filter in SignalGenerationEngine (spread > 0.05 -> NO_TRADE).
Step 2: Volatility filter in SignalGenerationEngine (IV >= 90th percentile -> NO_TRADE).
Step 3: IntelligenceSettings in config/settings.py (max_spread_percentage=0.05, max_implied_volatility_percentile=0.90).
Step 4: DirectionalSignalStrategy fixes lookahead bias by executing at T+1 using the T+1 option chain ask price.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from config.settings import AppSettings, IntelligenceSettings
from trading_engine.backtest.engine import (
    BacktestConfig,
    run_directional_backtest,
)
from trading_engine.intelligence.engines import (
    RegimeResult,
    SignalEngineConfig,
    SignalGenerationEngine,
)
from trading_engine.intelligence.models import (
    DirectionalBias,
    LiquidityState,
    MarketRegime,
    MarketState,
    MarketTransition,
    MomentumState,
    OptionSide,
    ProbabilityResult,
    RegimeTransitionStage,
    SessionPhase,
    SignalAction,
    TradeGrade,
    TrapResult,
    TrendState,
    VolatilityState,
)


def _make_state(close: float = 24000.0) -> MarketState:
    now = datetime.now(UTC)
    transition = MarketTransition(
        stage=RegimeTransitionStage.UNKNOWN,
        from_regime=MarketRegime.TREND_BULLISH,
        to_regime=MarketRegime.TREND_BULLISH,
        confidence=0.85,
    )
    state = MarketState(
        timestamp=now,
        trend_state=TrendState.TREND_BULLISH,
        volatility_state=VolatilityState.NORMAL,
        session_state=SessionPhase.TREND_WINDOW,
        liquidity_state=LiquidityState.NORMAL,
        momentum_state=MomentumState.EXTENDING,
        option_chain_bias=0.1,
        breakout_pressure=0.02,
        trap_probability=0.10,
        mean_reversion_probability=0.10,
        trend_strength=0.01,
        directional_bias=DirectionalBias.BULLISH,
        regime=MarketRegime.TREND_BULLISH,
        regime_confidence=0.85,
        transition=transition,
        chop_probability=0.20,
        persistence_probability=0.80,
        instability_probability=0.20,
        exhaustion_probability=0.10,
        quality_score=0.85,
        trade_grade=TradeGrade.A_PLUS,
        session_quality=0.85,
        signal_ttl_candles=2,
        is_tradeable=True,
        feature_snapshot={
            "close": close,
            "1m_close": close,
            "session_quality": 0.85,
        },
        notes=[],
    )
    return state


def _make_bullish_probabilities() -> ProbabilityResult:
    return ProbabilityResult(
        bullish_probability=0.80,
        bearish_probability=0.10,
        sideways_probability=0.10,
        confidence=0.80,
        horizon_minutes=4,
    )


def _make_trap_result() -> TrapResult:
    return TrapResult(
        fake_breakout_probability=0.1,
        buyer_trap_probability=0.1,
        seller_trap_probability=0.1,
        stoploss_hunt_probability=0.1,
        trap_score=0.1,
    )


def _make_regime_result() -> RegimeResult:
    return RegimeResult(
        regime=MarketRegime.TREND_BULLISH,
        confidence=0.85,
    )


def test_step3_intelligence_settings_defaults():
    """Step 3: Validate max_spread_percentage and max_implied_volatility_percentile settings."""
    settings = IntelligenceSettings()
    assert hasattr(settings, "max_spread_percentage")
    assert settings.max_spread_percentage == pytest.approx(0.05)
    assert hasattr(settings, "max_implied_volatility_percentile")
    assert settings.max_implied_volatility_percentile == pytest.approx(0.90)

    app_settings = AppSettings.from_env()
    assert app_settings.intelligence.max_spread_percentage == pytest.approx(0.05)
    assert app_settings.intelligence.max_implied_volatility_percentile == pytest.approx(0.90)


def test_step1_liquidity_filter_rejects_wide_spread(current_option_chain):
    """Step 1: Spread percentage (ask - bid) / bid > 0.05 returns NO_TRADE signal."""
    engine = SignalGenerationEngine(SignalEngineConfig(max_spread_percentage=0.05))
    state = _make_state(24000.0)
    probs = _make_bullish_probabilities()
    trap = _make_trap_result()
    regime = _make_regime_result()

    # Create option chain with 6% spread: bid=100, ask=106 -> (106 - 100) / 100 = 0.06 > 0.05
    option_chain = pd.DataFrame([
        {
            "strike": 24000.0,
            "option_type": "CE",
            "bid_price": 100.0,
            "ask_price": 106.0,
            "iv": 0.15,
            "expiry": "2026-10-22",
        }
    ])
    option_chain = current_option_chain(option_chain, state.timestamp)

    # Check direct filter method
    passed, spread_pct, reason = engine.check_liquidity_filter(24000.0, OptionSide.CE, option_chain)
    assert passed is False
    assert spread_pct == pytest.approx(0.06)
    assert "spread" in reason.lower()

    # Check signal generation returns NO_TRADE
    signal = engine.generate(
        state=state,
        probabilities=probs,
        trap=trap,
        regime=regime,
        option_chain=option_chain,
    )
    assert signal.action == SignalAction.NO_TRADE
    assert any("liquidity filter" in r.lower() for r in signal.reasoning)


def test_step1_liquidity_filter_accepts_tight_spread(current_option_chain):
    """Step 1: Spread percentage (ask - bid) / bid <= 0.05 passes and emits trade."""
    engine = SignalGenerationEngine(SignalEngineConfig(max_spread_percentage=0.05))
    state = _make_state(24000.0)
    probs = _make_bullish_probabilities()
    trap = _make_trap_result()
    regime = _make_regime_result()

    # Create option chain with 3% spread: bid=100, ask=103 -> (103 - 100) / 100 = 0.03 <= 0.05
    option_chain = pd.DataFrame([
        {
            "strike": 24000.0,
            "option_type": "CE",
            "bid_price": 100.0,
            "ask_price": 103.0,
            "iv": 0.15,
            "expiry": "2026-10-22",
        }
    ])
    option_chain = current_option_chain(option_chain, state.timestamp)

    passed, spread_pct, reason = engine.check_liquidity_filter(24000.0, OptionSide.CE, option_chain)
    assert passed is True
    assert spread_pct == pytest.approx(0.03)

    signal = engine.generate(
        state=state,
        probabilities=probs,
        trap=trap,
        regime=regime,
        option_chain=option_chain,
    )
    assert signal.action == SignalAction.BUY_CE


def test_step2_volatility_filter_rejects_top_decile_iv(current_option_chain):
    """Step 2: Implied volatility in 90th percentile or higher returns NO_TRADE signal."""
    engine = SignalGenerationEngine(SignalEngineConfig(max_implied_volatility_percentile=0.90))
    state = _make_state(24000.0)
    probs = _make_bullish_probabilities()
    trap = _make_trap_result()
    regime = _make_regime_result()

    # Rolling window of IV from 0.10 to 0.20 (10 points)
    historical_iv = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
    # Current IV is 0.25 (well above 90th percentile)
    option_chain = pd.DataFrame([
        {
            "strike": 24000.0,
            "option_type": "CE",
            "bid_price": 100.0,
            "ask_price": 102.0,  # Tight spread (2%)
            "iv": 0.25,          # Top decile IV
            "expiry": "2026-10-22",
        }
    ])
    option_chain = current_option_chain(option_chain, state.timestamp)

    # Check direct filter method
    passed, iv_val, percentile, reason = engine.check_volatility_filter(
        24000.0, OptionSide.CE, option_chain, historical_iv_window=historical_iv
    )
    assert passed is False
    assert percentile >= 0.90
    assert "volatility filter" in reason.lower()

    # Check signal generation returns NO_TRADE
    signal = engine.generate(
        state=state,
        probabilities=probs,
        trap=trap,
        regime=regime,
        option_chain=option_chain,
        historical_iv_window=historical_iv,
    )
    assert signal.action == SignalAction.NO_TRADE
    assert any("volatility filter" in r.lower() for r in signal.reasoning)


def test_step2_volatility_filter_accepts_normal_iv(current_option_chain):
    """Step 2: Implied volatility below 90th percentile passes and allows trade."""
    engine = SignalGenerationEngine(SignalEngineConfig(max_implied_volatility_percentile=0.90))
    state = _make_state(24000.0)
    probs = _make_bullish_probabilities()
    trap = _make_trap_result()
    regime = _make_regime_result()

    historical_iv = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28]
    # Current IV is 0.15 (around median, well below 90th percentile)
    option_chain = pd.DataFrame([
        {
            "strike": 24000.0,
            "option_type": "CE",
            "bid_price": 100.0,
            "ask_price": 102.0,
            "iv": 0.15,
            "expiry": "2026-10-22",
        }
    ])
    option_chain = current_option_chain(option_chain, state.timestamp)

    passed, iv_val, percentile, reason = engine.check_volatility_filter(
        24000.0, OptionSide.CE, option_chain, historical_iv_window=historical_iv
    )
    assert passed is True
    assert percentile < 0.90

    signal = engine.generate(
        state=state,
        probabilities=probs,
        trap=trap,
        regime=regime,
        option_chain=option_chain,
        historical_iv_window=historical_iv,
    )
    assert signal.action == SignalAction.BUY_CE


def test_step4_directional_strategy_executes_at_t_plus_one_with_option_ask():
    """Step 4: Signal generated at T executes at T+1 using the option chain tick ask price."""
    # Construct a 5-bar series where signal fires at bar index 1 (T=1).
    # Bar 1 (T=1): close=100.0, signal_long=1, confidence=0.8, ask_price=102.0
    # Bar 2 (T=2, T+1 relative to signal): close=105.0, ask_price=107.5
    # The simulated execution MUST enter at bar 2 with entry_price=107.5 (T+1 ask), NOT 100.0 or 102.0 (T)!
    now = datetime(2026, 10, 16, 9, 15, tzinfo=UTC)
    timestamps = [now + timedelta(minutes=i) for i in range(5)]

    frame = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0, 101.0, 105.0, 106.0, 107.0],
        "high": [102.0, 103.0, 108.0, 109.0, 110.0],
        "low": [99.0, 100.0, 104.0, 105.0, 106.0],
        "close": [100.0, 101.0, 105.0, 106.0, 107.0],
        "volume": [1000, 1200, 1500, 1400, 1300],
        "ask_price": [101.5, 102.0, 107.5, 108.0, 109.0],  # Bar 1 ask=102.0, Bar 2 ask=107.5
        "signal_long": [0, 1, 0, 0, 0],                     # Signal fires at Bar 1 (T)
        "signal_short": [0, 0, 0, 0, 0],
        "confidence": [0.0, 0.85, 0.0, 0.0, 0.0],
    })

    config = BacktestConfig(
        initial_cash=1_000_000.0,
        confidence_threshold=0.60,
        stop_loss_pct=0.05,
        max_holding_bars=1,
    )

    result = run_directional_backtest(frame, config=config)

    # Check that a trade was executed
    assert len(result.trade_ledger) == 1
    trade = result.trade_ledger.iloc[0]

    # Bar 1 (length=2) was signal bar T.
    # Bar 2 (length=3) was execution bar T+1.
    assert trade["entry_bar"] == 3  # len(self) at bar 2 is 3
    # Entry price MUST be the ask price at T+1 (107.5), NOT 101.0 (close at T) or 102.0 (ask at T)
    assert trade["entry_price"] == pytest.approx(107.5)
