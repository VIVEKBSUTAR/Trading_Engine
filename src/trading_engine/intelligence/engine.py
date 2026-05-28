"""Orchestrator for live market state, probabilities, signals, strike choice, and risk."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from config.settings import AppSettings
from trading_engine.intelligence.engines import (
    MarketRegimeEngine,
    MomentumPredictionEngine,
    RiskEngineConfig,
    RiskManagementEngine,
    RegimeEngineConfig,
    SignalEngineConfig,
    SignalGenerationEngine,
    StrikeSelectionConfig,
    StrikeSelectionEngine,
    TrapDetectionEngine,
)
from trading_engine.intelligence.features import build_live_features
from trading_engine.intelligence.models import IntelligenceReport, MarketSnapshotBundle, MarketState, SignalAction, utcnow
from trading_engine.intelligence.monitor import LiveTradeMonitor
from trading_engine.intelligence.state import MarketStateAggregator, StateConfig


@dataclass(slots=True)
class IntelligenceEngineConfig:
    """Runtime configuration for the live intelligence orchestrator."""

    lookback_bars: int = 50


class LiveIntelligenceEngine:
    """Combine state, features, regime, probability, strike, and risk outputs."""

    def __init__(self, settings: AppSettings, monitor: LiveTradeMonitor | None = None) -> None:
        self._settings = settings
        self._monitor = monitor or LiveTradeMonitor()
        self._state_aggregator = MarketStateAggregator(
            StateConfig(
                max_spot_rows=settings.intelligence.state_buffer_rows,
                max_vix_rows=settings.intelligence.state_buffer_rows,
                max_option_rows=settings.intelligence.state_buffer_rows * 4,
                max_tick_rows=settings.intelligence.state_buffer_rows * 12,
                trend_strength_threshold=settings.intelligence.trend_strength_threshold,
                compression_threshold=settings.intelligence.compression_threshold,
                volatility_high_threshold=settings.intelligence.volatility_high_threshold,
                tradeable_quality_threshold=settings.intelligence.trade_quality_threshold,
                signal_ttl_min_candles=settings.intelligence.signal_ttl_min_candles,
                signal_ttl_max_candles=settings.intelligence.signal_ttl_max_candles,
            )
        )
        self._regime_engine = MarketRegimeEngine(
            RegimeEngineConfig(
                trend_strength_threshold=settings.intelligence.trend_strength_threshold,
                compression_threshold=settings.intelligence.compression_threshold,
                volatility_threshold=settings.intelligence.volatility_high_threshold,
            )
        )
        self._momentum_engine = MomentumPredictionEngine(horizon_minutes=4)
        self._trap_engine = TrapDetectionEngine()
        self._signal_engine = SignalGenerationEngine(
            SignalEngineConfig(
                bullish_threshold=settings.intelligence.bullish_probability_threshold,
                bearish_threshold=settings.intelligence.bearish_probability_threshold,
                confidence_floor=settings.intelligence.confidence_floor,
                trap_cutoff=settings.intelligence.trap_probability_threshold,
                min_trade_quality=settings.intelligence.trade_quality_threshold,
            )
        )
        self._strike_engine = StrikeSelectionEngine(StrikeSelectionConfig())
        self._risk_engine = RiskManagementEngine(
            RiskEngineConfig(
                capital=settings.risk.capital,
                risk_per_trade=settings.risk.risk_per_trade,
                max_capital_exposure=settings.risk.max_capital_exposure,
                contract_multiplier=settings.risk.contract_multiplier,
                min_units=settings.risk.min_units,
                max_daily_loss_pct=settings.intelligence.max_daily_loss_pct,
                max_trades_per_day=settings.intelligence.max_trades_per_day,
                partial_exit_fraction=settings.intelligence.partial_exit_fraction,
                time_stop_minutes=settings.intelligence.time_stop_minutes,
            )
        )

    @property
    def monitor(self) -> LiveTradeMonitor:
        """Expose the live trade monitor."""
        return self._monitor

    def process(
        self,
        snapshot: MarketSnapshotBundle,
        *,
        state: MarketState | None = None,
        trade_count_today: int = 0,
        realized_daily_pnl: float = 0.0,
    ) -> IntelligenceReport:
        """Generate a full intelligence report from a live market snapshot."""
        features = build_live_features(snapshot, lookback_bars=self._settings.intelligence.structure_lookback_bars)
        if features.frame.empty:
            logger.warning("Live feature frame empty; returning neutral report")

        if state is None:
            state = self._build_state(snapshot)
        state.feature_snapshot.update(features.feature_map)
        if snapshot.spot_price is not None:
            state.feature_snapshot.setdefault("close", snapshot.spot_price)
            state.feature_snapshot.setdefault("1m_close", snapshot.spot_price)

        trap = self._trap_engine.assess(state, features.frame)
        state.trap_probability = max(state.trap_probability, trap.trap_score)
        regime = self._regime_engine.classify(state)
        probabilities = self._momentum_engine.predict(state, features.frame)
        signal = self._signal_engine.generate(state=state, regime=regime, probabilities=probabilities, trap=trap)
        strike = self._strike_engine.select(state, signal, snapshot.option_chain_frame)
        signal.strike = strike
        risk = self._risk_engine.plan(
            state=state,
            signal=signal,
            regime=regime,
            trade_count_today=trade_count_today,
            realized_daily_pnl=realized_daily_pnl,
        )
        signal.quantity = risk.quantity if risk.allowed else 0
        signal.stop_loss = risk.stop_loss
        signal.target = risk.target
        signal.signal_ttl_candles = max(int(signal.signal_ttl_candles or 1), int(state.signal_ttl_candles or 1))

        if risk.allowed and signal.action != SignalAction.NO_TRADE:
            if not any(trade.signal.action == signal.action for trade in self._monitor.open_trades):
                self._monitor.open_trade_from_signal(signal, risk.quantity)

        trade_update = self._monitor.update(snapshot.spot_price, latest_state=state)
        report = IntelligenceReport(
            timestamp=utcnow(),
            snapshot=snapshot,
            state=state,
            features=features,
            regime=regime,
            transition=state.transition,
            probabilities=probabilities,
            trap=trap,
            signal=signal,
            strike=strike,
            risk=risk,
            trade_update=trade_update,
        )
        logger.info(
            "Generated intelligence report",
            regime=regime.regime.value,
            action=signal.action.value,
            confidence=signal.confidence,
            quantity=signal.quantity,
        )
        return report

    def _build_state(self, snapshot: MarketSnapshotBundle) -> MarketState:
        self._state_aggregator.update_from_snapshot(snapshot)
        return self._state_aggregator.build_market_state(snapshot)
