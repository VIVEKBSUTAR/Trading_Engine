"""Orchestrator for live market state, probabilities, signals, strike choice, and risk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

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
from trading_engine.intelligence.models import IntelligenceReport, MarketSnapshotBundle, TradeUpdate, utcnow
from trading_engine.intelligence.monitor import LiveTradeMonitor


@dataclass(slots=True)
class IntelligenceEngineConfig:
    """Runtime configuration for the live intelligence orchestrator."""

    lookback_bars: int = 50


class LiveIntelligenceEngine:
    """Combine state, features, regime, probability, strike, and risk outputs."""

    def __init__(self, settings: AppSettings, monitor: LiveTradeMonitor | None = None) -> None:
        self._settings = settings
        self._monitor = monitor or LiveTradeMonitor()
        self._regime_engine = MarketRegimeEngine(
            RegimeEngineConfig(
                trend_strength_threshold=0.0025,
                compression_threshold=0.55,
                volatility_threshold=0.012,
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
            )
        )

    @property
    def monitor(self) -> LiveTradeMonitor:
        """Expose the live trade monitor."""
        return self._monitor

    def process(self, snapshot: MarketSnapshotBundle, *, trade_count_today: int = 0, realized_daily_pnl: float = 0.0) -> IntelligenceReport:
        """Generate a full intelligence report from a live market snapshot."""
        features = build_live_features(snapshot, lookback_bars=self._settings.intelligence.structure_lookback_bars)
        if features.frame.empty:
            logger.warning("Live feature frame empty; returning neutral report")

        trap = self._trap_engine.assess(features.frame)
        regime = self._regime_engine.classify(features.frame, trap)
        probabilities = self._momentum_engine.predict(features.frame, regime)
        signal = self._signal_engine.generate(features=features.frame, regime=regime, probabilities=probabilities, trap=trap)
        strike = self._strike_engine.select(features.frame, signal, snapshot.option_chain_frame)
        signal.strike = strike
        risk = self._risk_engine.plan(
            features=features.frame,
            signal=signal,
            regime=regime,
            trade_count_today=trade_count_today,
            realized_daily_pnl=realized_daily_pnl,
        )
        signal.quantity = risk.quantity if risk.allowed else 0
        signal.stop_loss = risk.stop_loss
        signal.target = risk.target

        trade_update = self._monitor.update(snapshot.spot_price, latest_features=features.frame)
        report = IntelligenceReport(
            timestamp=utcnow(),
            snapshot=snapshot,
            features=features,
            regime=regime,
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
