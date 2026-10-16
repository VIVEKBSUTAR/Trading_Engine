"""Orchestrator for live market state, probabilities, signals, strike choice, and risk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from config.settings import AppSettings
from storage.duckdb_manager import DuckDBManager
from trading_engine.intelligence.engines import (
    MarketRegimeEngine,
    MomentumPredictionEngine,
    RegimeEngineConfig,
    RiskEngineConfig,
    RiskManagementEngine,
    SignalEngineConfig,
    SignalGenerationEngine,
    StrikeSelectionConfig,
    StrikeSelectionEngine,
    TrapDetectionEngine,
)
from trading_engine.intelligence.features import build_live_features
from trading_engine.intelligence.micro_momentum import MicroMomentumEngine
from trading_engine.intelligence.models import (
    IntelligenceReport,
    LatencyTrace,
    MarketSnapshotBundle,
    MarketState,
    SignalAction,
    SignalResult,
    SignalScore,
    TrapResult,
    utcnow,
)
from trading_engine.intelligence.monitor import LiveTradeMonitor
from trading_engine.intelligence.state import MarketStateAggregator, StateConfig


@dataclass(slots=True)
class IntelligenceEngineConfig:
    """Runtime configuration for the live intelligence orchestrator."""

    lookback_bars: int = 50


class LiveIntelligenceEngine:
    """Combine state, features, regime, probability, strike, and risk outputs."""

    def __init__(
        self,
        settings: AppSettings,
        monitor: LiveTradeMonitor | None = None,
        duckdb_manager: DuckDBManager | None = None,
    ) -> None:
        self._settings = settings
        if duckdb_manager is not None:
            self._duckdb_manager = duckdb_manager
        else:
            try:
                self._duckdb_manager = DuckDBManager(settings.storage.duckdb_path)
            except Exception:
                self._duckdb_manager = DuckDBManager()
        self._monitor = monitor or LiveTradeMonitor(duckdb_manager=self._duckdb_manager)
        if self._monitor.duckdb_manager is None:
            self._monitor.duckdb_manager = self._duckdb_manager
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
                max_spread_percentage=getattr(settings.intelligence, "max_spread_percentage", 0.05),
                max_implied_volatility_percentile=getattr(settings.intelligence, "max_implied_volatility_percentile", 0.90),
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
        self._micro_engine = MicroMomentumEngine()

    @property
    def monitor(self) -> LiveTradeMonitor:
        """Expose the live trade monitor."""
        return self._monitor

    @property
    def duckdb_manager(self) -> DuckDBManager:
        """Expose the DuckDB manager."""
        return self._duckdb_manager

    @property
    def state_aggregator(self) -> MarketStateAggregator:
        """Expose the market state aggregator."""
        return self._state_aggregator

    def process(
        self,
        snapshot: MarketSnapshotBundle,
        *,
        state: MarketState | None = None,
        trade_count_today: int = 0,
        realized_daily_pnl: float = 0.0,
        data_received_at: datetime | None = None,
    ) -> IntelligenceReport:
        """Generate a full intelligence report from a live market snapshot."""
        latency = LatencyTrace(data_received_at=data_received_at or datetime.now(UTC))

        features = build_live_features(snapshot, lookback_bars=self._settings.intelligence.structure_lookback_bars)
        latency.features_computed_at = datetime.now(UTC)
        if features.frame.empty:
            logger.warning("Live feature frame empty; returning neutral report")

        if state is None:
            state = self._build_state(snapshot)
        state.feature_snapshot.update(features.feature_map)
        if snapshot.spot_price is not None:
            state.feature_snapshot.setdefault("close", snapshot.spot_price)
            state.feature_snapshot.setdefault("1m_close", snapshot.spot_price)

        # -- Micro-momentum assessment (tick-level velocity/acceleration) --
        atm_ce_ticks, atm_pe_ticks = self._state_aggregator.get_option_tick_buffers()
        micro = self._micro_engine.assess(
            underlying_ticks=snapshot.ticks_frame,
            atm_ce_ticks=atm_ce_ticks,
            atm_pe_ticks=atm_pe_ticks,
        )

        # Propagate simultaneous spike trap into state trap probability
        if micro.simultaneous_spike and micro.trap_via_spike > 0.0:
            state.trap_probability = min(1.0, state.trap_probability + micro.trap_via_spike * 0.4)

        # Enrich feature_snapshot with micro features for downstream engines
        state.feature_snapshot["micro_ce_velocity"] = float(micro.ce_premium_velocity) if not _isnan(micro.ce_premium_velocity) else 0.0
        state.feature_snapshot["micro_pe_velocity"] = float(micro.pe_premium_velocity) if not _isnan(micro.pe_premium_velocity) else 0.0
        state.feature_snapshot["micro_ce_pe_relative"] = micro.ce_pe_relative_strength
        state.feature_snapshot["micro_buy_sell_imbalance"] = micro.buy_sell_imbalance
        state.feature_snapshot["micro_volume_acceleration"] = micro.volume_acceleration
        state.feature_snapshot["micro_simultaneous_spike"] = 1.0 if micro.simultaneous_spike else 0.0

        trap = self._trap_engine.assess(state, features.frame)
        state.trap_probability = max(state.trap_probability, trap.trap_score)
        regime = self._regime_engine.classify(state)
        probabilities = self._momentum_engine.predict(state, features.frame)
        signal = self._signal_engine.generate(
            state=state,
            regime=regime,
            probabilities=probabilities,
            trap=trap,
            option_chain=snapshot.option_chain_frame,
        )
        latency.signal_generated_at = datetime.now(UTC)

        # -- Build SignalScore (0-10 named breakdown) -----------------------
        signal.score = self._build_signal_score(state, micro, signal)
        signal.signal_validity_seconds = self._settings.intelligence.signal_ttl_max_candles * 60

        if signal.strike is None:
            strike = self._strike_engine.select(state, signal, snapshot.option_chain_frame)
            signal.strike = strike
        else:
            strike = signal.strike
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

        # Resolve specific option instrument token and exact entry ask price from the option chain snapshot
        token, ask_price, bid_price, chosen_strike, chosen_expiry, chosen_side = self._resolve_option_contract_details(
            signal, snapshot.option_chain_frame
        )

        # Step 4: Write complete record to signals table for both accepted and rejected setups
        if signal.filter_passed is not None or signal.action != SignalAction.NO_TRADE:
            self._persist_signal_record(
                signal=signal,
                snapshot=snapshot,
                state=state,
                trap=trap,
                chosen_strike=chosen_strike,
                chosen_expiry=chosen_expiry,
                chosen_side=chosen_side,
            )

        # Update state aggregator with fresh option bid/ask ticks for tracked instruments
        if signal.action != SignalAction.NO_TRADE:
            self._state_aggregator.push_option_tick(
                instrument_token=token,
                bid=bid_price,
                ask=ask_price,
                timestamp=snapshot.timestamp or utcnow(),
                last_price=ask_price,
            )

        # Refresh option ticks for existing open trades if available in current snapshot
        for trade in self._monitor.open_trades:
            if trade.instrument_token != token and not snapshot.option_chain_frame.empty:
                t_token, t_ask, t_bid, _, _, _ = self._resolve_option_contract_details(
                    trade.signal, snapshot.option_chain_frame
                )
                if t_token == trade.instrument_token:
                    self._state_aggregator.push_option_tick(
                        instrument_token=trade.instrument_token,
                        bid=t_bid,
                        ask=t_ask,
                        timestamp=snapshot.timestamp or utcnow(),
                        last_price=t_bid,
                    )

        if (
            risk.allowed
            and signal.action != SignalAction.NO_TRADE
            and not any(trade.signal.action == signal.action for trade in self._monitor.open_trades)
        ):
            self._monitor.open_trade_from_signal(
                    signal,
                    quantity=risk.quantity,
                    instrument_token=token,
                    entry_ask_price=ask_price,
                    strike=chosen_strike,
                    expiry=chosen_expiry,
                    option_type=chosen_side,
                )

        # Evaluate trade monitor using actual option tick data instead of underlying spot price
        option_ticks = self._state_aggregator.get_option_token_ticks()
        trade_update = self._monitor.update(option_ticks, latest_state=state)
        latency.report_published_at = datetime.now(UTC)

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
            latency=latency,
            ce_score=signal.score.ce_momentum_score_display if signal.score is not None else 0.0,
            pe_score=signal.score.pe_momentum_score_display if signal.score is not None else 0.0,
            ce_premium_velocity=micro.ce_premium_velocity,
            pe_premium_velocity=micro.pe_premium_velocity,
        )
        logger.info(
            "Generated intelligence report",
            regime=regime.regime.value,
            action=signal.action.value,
            confidence=signal.confidence,
            quantity=signal.quantity,
            latency_ms=f"{latency.total_latency_ms:.1f}",
        )
        return report

    def _build_signal_score(self, state: MarketState, micro, signal) -> SignalScore:
        """Assemble named SignalScore from state features and micro-momentum."""
        import numpy as np

        fm = state.feature_snapshot
        trend = float(fm.get("1m_trend_strength", 0.0) or 0.0)
        breakout = abs(float(fm.get("1m_breakout_pressure", 0.0) or 0.0))

        # underlying_direction: EMA alignment, clipped to [0, 1]
        underlying_direction = float(np.clip(abs(trend) / 0.005, 0.0, 1.0))

        # underlying_acceleration: second derivative of returns via micro engine
        underlying_acceleration = float(np.clip(abs(micro.underlying_acceleration) / 0.003, 0.0, 1.0))

        # premium velocity (raw fractional velocity, clipped at 1.5%)
        pv_ce = float(np.clip(abs(micro.ce_premium_velocity) / 0.015, 0.0, 1.0)) if not _isnan(micro.ce_premium_velocity) else 0.0
        pv_pe = float(np.clip(abs(micro.pe_premium_velocity) / 0.015, 0.0, 1.0)) if not _isnan(micro.pe_premium_velocity) else 0.0

        # premium acceleration: positive is building momentum
        pa_ce = float(np.clip(micro.ce_premium_acceleration / 0.010, 0.0, 1.0)) if not _isnan(micro.ce_premium_acceleration) else 0.0
        pa_pe = float(np.clip(micro.pe_premium_acceleration / 0.010, 0.0, 1.0)) if not _isnan(micro.pe_premium_acceleration) else 0.0
        premium_acceleration = max(pa_ce, pa_pe)

        # volume acceleration: already normalised [-1, 1] from engine
        volume_acceleration = float(np.clip(micro.volume_acceleration, 0.0, 1.0))

        # relative_ce_pe: directional (positive = CE dominant, negative = PE dominant)
        rel = micro.ce_pe_relative_strength  # [-1, 1]
        # For CE signal: positive rel is good; for PE: negative rel is good. Use abs for score.
        relative_ce_pe = float(np.clip(abs(rel), 0.0, 1.0))

        # breakout_pressure: clipped to [0, 1]
        breakout_score = float(np.clip(breakout / 0.003, 0.0, 1.0))

        # trap_inverse: reward for low trap probability
        trap_inverse = float(np.clip(1.0 - state.trap_probability, 0.0, 1.0))

        score = SignalScore(
            underlying_direction=underlying_direction,
            underlying_acceleration=underlying_acceleration,
            premium_velocity_ce=pv_ce,
            premium_velocity_pe=pv_pe,
            premium_acceleration=premium_acceleration,
            volume_acceleration=volume_acceleration,
            relative_ce_pe=relative_ce_pe,
            breakout_pressure=breakout_score,
            trap_inverse=trap_inverse,
        )
        # Attach display scores to the score object for easy report access
        score.ce_momentum_score_display = float(np.clip(micro.ce_momentum_score * 10.0, 0.0, 10.0))
        score.pe_momentum_score_display = float(np.clip(micro.pe_momentum_score * 10.0, 0.0, 10.0))
        return score

    def _build_state(self, snapshot: MarketSnapshotBundle) -> MarketState:
        self._state_aggregator.update_from_snapshot(snapshot)
        return self._state_aggregator.build_market_state(snapshot)

    def _resolve_option_contract_details(
        self,
        signal: SignalResult,
        option_chain: pd.DataFrame,
    ) -> tuple[int, float, float, float, str, str]:
        """Extract instrument_token, ask_price, bid_price, strike, expiry, and side from option chain."""
        target_strike = signal.strike.strike if signal.strike else None
        target_side = (
            signal.strike.option_side.value
            if signal.strike and hasattr(signal.strike, "option_side")
            else ("CE" if signal.action == SignalAction.BUY_CE else "PE")
        )
        target_expiry = signal.strike.expiry if signal.strike else None

        token: int | None = None
        ask_price: float | None = None
        bid_price: float | None = None
        strike_val: float = float(target_strike or 24000.0)
        expiry_val: str = str(target_expiry or "")
        side_val: str = target_side

        if option_chain is not None and not option_chain.empty:
            matches = option_chain.copy()
            if "option_type" in matches.columns:
                matches = matches.loc[matches["option_type"].astype(str).str.upper() == target_side.upper()]
            if "strike" in matches.columns and target_strike is not None:
                matches = matches.loc[pd.to_numeric(matches["strike"], errors="coerce") == float(target_strike)]
            if target_expiry is not None and "expiry" in matches.columns:
                exp_matches = matches.loc[matches["expiry"].astype(str) == str(target_expiry)]
                if not exp_matches.empty:
                    matches = exp_matches

            if not matches.empty:
                row = matches.iloc[0]
                strike_val = float(row.get("strike", strike_val))
                expiry_val = str(row.get("expiry", expiry_val))
                side_val = str(row.get("option_type", side_val))

                ask_price = _safe_float(row.get("ask_price") or row.get("ask") or row.get("last_price"))
                bid_price = _safe_float(row.get("bid_price") or row.get("bid"))

                tok = row.get("instrument_token")
                if tok is not None and not pd.isna(tok):
                    try:
                        token = int(tok)
                    except (ValueError, TypeError):
                        token = None

        # Fallbacks for pricing and token
        if ask_price is None or ask_price <= 0:
            ask_price = float(signal.entry_reference or 100.0)
        if bid_price is None or bid_price <= 0:
            bid_price = float(ask_price * 0.99)
        if token is None or token <= 0:
            side_mult = 1 if target_side.upper() == "CE" else 2
            token = int(strike_val * 10 + side_mult)

        return token, ask_price, bid_price, strike_val, expiry_val, side_val

    def _persist_signal_record(
        self,
        signal: SignalResult,
        snapshot: MarketSnapshotBundle,
        state: MarketState,
        trap: TrapResult,
        chosen_strike: float | None = None,
        chosen_expiry: str | None = None,
        chosen_side: str | None = None,
    ) -> None:
        """Write evaluated signal record to DuckDB signals table."""
        if self._duckdb_manager is None:
            return

        sig_id = signal.signal_id or f"SIG-{utcnow().strftime('%Y%m%d%H%M%S%f')}"
        signal.signal_id = sig_id

        strike_val: float | None = chosen_strike
        if strike_val is None and signal.strike and signal.strike.strike:
            strike_val = float(signal.strike.strike)

        expiry_val: str | None = chosen_expiry
        if expiry_val is None and signal.strike and signal.strike.expiry:
            expiry_val = str(signal.strike.expiry)

        side_val: str | None = chosen_side
        if side_val is None and signal.strike and hasattr(signal.strike, "option_side"):
            side_val = str(signal.strike.option_side.value)

        filter_passed = (
            bool(signal.filter_passed)
            if signal.filter_passed is not None
            else (signal.action != SignalAction.NO_TRADE)
        )
        rejection_reason = signal.filter_rejection_reason
        if rejection_reason is None and not filter_passed:
            rejection_reason = signal.reasoning[-1] if signal.reasoning else "Setup rejected"

        spot_px = snapshot.spot_price or state.feature_snapshot.get("close", 0.0)

        try:
            self._duckdb_manager.insert_signal(
                signal_id=sig_id,
                timestamp=snapshot.timestamp or state.timestamp or utcnow(),
                spot_price=float(spot_px) if spot_px is not None else None,
                strike=strike_val,
                expiry=expiry_val,
                option_type=side_val,
                action=signal.action.value,
                confidence=float(signal.confidence),
                trap_score=float(trap.trap_score),
                spread_percentage=float(signal.spread_percentage) if signal.spread_percentage is not None else None,
                iv_percentile=float(signal.iv_percentile) if signal.iv_percentile is not None else None,
                trade_grade=signal.trade_grade.value if hasattr(signal.trade_grade, "value") else str(signal.trade_grade),
                filter_passed=filter_passed,
                filter_rejection_reason=rejection_reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to persist signal to DuckDB", signal_id=sig_id, error=str(exc))


def _isnan(value: float) -> bool:
    """Safe NaN check that works for both float and numpy floating."""
    import math
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, float, str)):
            return float(value)
        return float(str(value))
    except (TypeError, ValueError):
        return None
