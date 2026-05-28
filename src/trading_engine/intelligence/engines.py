"""Probabilistic live signal engines for NIFTY options intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import exp
from typing import Iterable

import numpy as np
import pandas as pd

from trading_engine.intelligence.models import (
    DirectionalBias,
    MarketRegime,
    MarketState,
    MarketTransition,
    MomentumState,
    OptionSide,
    ProbabilityResult,
    RegimeResult,
    RiskResult,
    SignalAction,
    SignalResult,
    SessionPhase,
    StrikeResult,
    TradeGrade,
    RegimeTransitionStage,
    TrapResult,
)


@dataclass(slots=True)
class RegimeEngineConfig:
    """Thresholds for live regime classification."""

    trend_strength_threshold: float = 0.0025
    compression_threshold: float = 0.55
    volatility_threshold: float = 0.80
    fake_breakout_threshold: float = 0.68


@dataclass(slots=True)
class ProbabilityWeightConfig:
    """Configurable weights for the formal probability engine."""

    trend_strength: float = 0.20
    vwap_alignment: float = 0.15
    oi_confirmation: float = 0.25
    volume_expansion: float = 0.15
    session_quality: float = 0.10
    volatility_support: float = 0.15
    trap_penalty: float = 0.20
    sideways_compression: float = 0.20
    breakout_pressure: float = 0.15
    momentum_persistence: float = 0.10


@dataclass(slots=True)
class SignalEngineConfig:
    """Signal thresholds for the live alert engine."""

    bullish_threshold: float = 0.65
    bearish_threshold: float = 0.65
    confidence_floor: float = 0.60
    trap_cutoff: float = 0.60
    min_trade_quality: float = 0.55


@dataclass(slots=True)
class StrikeSelectionConfig:
    """Strike selection parameters for live NIFTY options."""

    default_step: float = 50.0
    aggressive_otm_offset: int = 1
    conservative_atm_offset: int = 0


@dataclass(slots=True)
class RiskEngineConfig:
    """Risk management controls for live signals."""

    capital: float = 10_000_000.0
    risk_per_trade: float = 0.005
    max_capital_exposure: float = 0.10
    contract_multiplier: float = 1.0
    min_units: int = 1
    max_daily_loss_pct: float = 0.02
    max_trades_per_day: int = 3
    partial_exit_fraction: float = 0.50
    time_stop_minutes: int = 12


class MarketRegimeEngine:
    """Classify the current market state before any signal is emitted."""

    def __init__(self, config: RegimeEngineConfig | None = None) -> None:
        self._config = config or RegimeEngineConfig()

    def classify(self, state: MarketState) -> RegimeResult:
        """Return the current regime and a confidence score."""
        reasons = list(state.notes)
        return RegimeResult(regime=state.regime, confidence=state.regime_confidence, reasons=reasons, transition=state.transition)


class MomentumPredictionEngine:
    """Estimate direction probabilities over a 1-5 minute horizon."""

    def __init__(self, horizon_minutes: int = 4, weights: ProbabilityWeightConfig | None = None) -> None:
        self._horizon_minutes = horizon_minutes
        self._weights = weights or ProbabilityWeightConfig()

    def predict(self, state: MarketState, features: pd.DataFrame | None = None) -> ProbabilityResult:
        feature_map = state.feature_snapshot
        evidence: list[str] = []

        trend_15 = float(feature_map.get("15m_trend_strength", 0.0) or 0.0)
        trend_5 = float(feature_map.get("5m_trend_strength", 0.0) or 0.0)
        trend_1 = float(feature_map.get("1m_trend_strength", 0.0) or 0.0)
        vwap_15 = float(feature_map.get("15m_vwap_alignment", 0.0) or 0.0)
        vwap_5 = float(feature_map.get("5m_vwap_alignment", 0.0) or 0.0)
        vwap_1 = float(feature_map.get("1m_vwap_alignment", 0.0) or 0.0)
        oi_bias = float(feature_map.get("option_chain_bias", state.option_chain_bias) or 0.0)
        volume_expansion = float(feature_map.get("1m_volume_expansion", 1.0) or 1.0)
        session_quality = float(feature_map.get("session_quality", state.session_quality) or 0.5)
        volatility_support = 1.0 - min(float(feature_map.get("1m_atr_compression", 0.0) or 0.0), 1.0)
        breakout_pressure = float(state.breakout_pressure)
        persistence = float(feature_map.get("1m_directional_persistence", 0.5) or 0.5)
        trap_penalty = float(np.clip(state.trap_probability, 0.0, 1.0))
        sideways_hint = float(feature_map.get("1m_narrow_range_compression", 0.0) or 0.0)

        bullish_score = (
            self._weights.trend_strength * max(0.0, (trend_15 + trend_5 + trend_1) / 3.0 * 100.0)
            + self._weights.vwap_alignment * max(0.0, (vwap_15 + vwap_5 + vwap_1) / 3.0 * 100.0)
            + self._weights.oi_confirmation * max(0.0, oi_bias)
            + self._weights.volume_expansion * max(0.0, volume_expansion - 1.0)
            + self._weights.session_quality * session_quality
            + self._weights.volatility_support * volatility_support
            + self._weights.breakout_pressure * max(0.0, breakout_pressure * 100.0)
            + self._weights.momentum_persistence * persistence
            - self._weights.trap_penalty * trap_penalty
        )
        bearish_score = (
            self._weights.trend_strength * max(0.0, -(trend_15 + trend_5 + trend_1) / 3.0 * 100.0)
            + self._weights.vwap_alignment * max(0.0, -(vwap_15 + vwap_5 + vwap_1) / 3.0 * 100.0)
            + self._weights.oi_confirmation * max(0.0, -oi_bias)
            + self._weights.volume_expansion * max(0.0, volume_expansion - 1.0)
            + self._weights.session_quality * session_quality
            + self._weights.volatility_support * volatility_support
            + self._weights.breakout_pressure * max(0.0, -breakout_pressure * 100.0)
            + self._weights.momentum_persistence * persistence
            - self._weights.trap_penalty * trap_penalty
        )
        sideways_score = (
            self._weights.sideways_compression * max(0.0, sideways_hint)
            + 0.20 * max(0.0, 1.0 - abs(trend_1) * 100.0)
            + 0.15 * max(0.0, 1.0 - abs(breakout_pressure) * 100.0)
            + 0.15 * max(0.0, 1.0 - persistence)
            + 0.10 * max(0.0, 1.0 - session_quality)
            + 0.10 * trap_penalty
        )

        if state.regime in {MarketRegime.SIDEWAYS_COMPRESSION, MarketRegime.HIGH_NOISE_ENVIRONMENT}:
            sideways_score += 0.35
            evidence.append("regime favors low participation")
        if state.regime in {MarketRegime.TREND_BULLISH, MarketRegime.TREND_BEARISH}:
            evidence.append("trend regime alignment")
            bullish_score += 0.10 if state.regime == MarketRegime.TREND_BULLISH else 0.0
            bearish_score += 0.10 if state.regime == MarketRegime.TREND_BEARISH else 0.0
        if state.transition.stage in {RegimeTransitionStage.COMPRESSION, RegimeTransitionStage.BREAKOUT_BUILDUP}:
            sideways_score += 0.10
            evidence.append("transition still building")
        if state.transition.stage == RegimeTransitionStage.REVERSAL:
            sideways_score += 0.15
            evidence.append("reversal stage increases caution")

        bullish_probability, bearish_probability, sideways_probability = _softmax([bullish_score, bearish_score, sideways_score])
        confidence = float(max(bullish_probability, bearish_probability, sideways_probability))
        calibration_band = _calibration_band(confidence)
        feature_contributions = {
            "trend_strength": float((trend_15 + trend_5 + trend_1) / 3.0),
            "vwap_alignment": float((vwap_15 + vwap_5 + vwap_1) / 3.0),
            "oi_confirmation": oi_bias,
            "volume_expansion": volume_expansion,
            "session_quality": session_quality,
            "volatility_support": volatility_support,
            "sideways_hint": sideways_hint,
            "trap_penalty": trap_penalty,
        }
        if bullish_probability == confidence:
            evidence.append("bullish score dominant")
        elif bearish_probability == confidence:
            evidence.append("bearish score dominant")
        else:
            evidence.append("sideways score dominant")

        return ProbabilityResult(
            bullish_probability=float(bullish_probability),
            bearish_probability=float(bearish_probability),
            sideways_probability=float(sideways_probability),
            confidence=confidence,
            horizon_minutes=self._horizon_minutes,
            bullish_score=float(bullish_score),
            bearish_score=float(bearish_score),
            sideways_score=float(sideways_score),
            calibration_band=calibration_band,
            feature_contributions=feature_contributions,
            evidence=evidence,
        )


class TrapDetectionEngine:
    """Detect stoploss hunts, fake breakouts, and exhaustion behavior."""

    def assess(self, state: MarketState, features: pd.DataFrame | None = None) -> TrapResult:
        feature_map = state.feature_snapshot
        evidence: list[str] = []

        breakout_pressure = float(state.breakout_pressure)
        trap_probability = float(np.clip(state.trap_probability, 0.0, 1.0))
        failed_breakout = float(feature_map.get("1m_failed_breakout_frequency", 0.0) or 0.0)
        wick_rejection = float(feature_map.get("1m_wick_rejection", 0.0) or 0.0)
        overlap = float(feature_map.get("1m_overlap_ratio", 0.0) or 0.0)
        persistence = float(feature_map.get("1m_directional_persistence", 0.5) or 0.5)
        vix_spread = float(feature_map.get("iv_realized_spread", 0.0) or 0.0)
        oi_bias = float(feature_map.get("option_chain_bias", 0.0) or 0.0)

        fake_breakout_probability = float(np.clip(0.45 * wick_rejection + 0.25 * failed_breakout + 0.15 * overlap + 0.10 * trap_probability + 0.05 * max(0.0, abs(vix_spread)), 0.0, 1.0))
        buyer_trap_probability = float(np.clip(0.45 * max(0.0, wick_rejection) + 0.25 * max(0.0, oi_bias) + 0.15 * max(0.0, 1.0 - persistence) + 0.15 * trap_probability, 0.0, 1.0))
        seller_trap_probability = float(np.clip(0.45 * max(0.0, wick_rejection) + 0.25 * max(0.0, -oi_bias) + 0.15 * max(0.0, 1.0 - persistence) + 0.15 * trap_probability, 0.0, 1.0))
        stoploss_hunt_probability = float(np.clip(0.40 * overlap + 0.30 * failed_breakout + 0.15 * wick_rejection + 0.15 * trap_probability, 0.0, 1.0))
        failed_breakout_probability = float(np.clip(0.60 * failed_breakout + 0.25 * overlap + 0.15 * trap_probability, 0.0, 1.0))
        liquidity_grab_probability = float(np.clip(0.40 * wick_rejection + 0.30 * failed_breakout + 0.30 * max(0.0, abs(breakout_pressure) * 100.0), 0.0, 1.0))
        trap_score = float(np.clip((fake_breakout_probability + stoploss_hunt_probability + liquidity_grab_probability) / 3.0, 0.0, 1.0))

        if fake_breakout_probability >= 0.5:
            evidence.append("wick rejection and weak follow-through")
        if failed_breakout_probability >= 0.5:
            evidence.append("failed breakout frequency elevated")
        if liquidity_grab_probability >= 0.5:
            evidence.append("liquidity grab pattern detected")
        if trap_probability >= 0.6:
            evidence.append("state trap probability elevated")

        return TrapResult(
            fake_breakout_probability=fake_breakout_probability,
            buyer_trap_probability=buyer_trap_probability,
            seller_trap_probability=seller_trap_probability,
            stoploss_hunt_probability=stoploss_hunt_probability,
            failed_breakout_probability=failed_breakout_probability,
            liquidity_grab_probability=liquidity_grab_probability,
            trap_score=trap_score,
            evidence=evidence,
        )


class SignalGenerationEngine:
    """Translate probabilities and trap diagnostics into a filtered trading signal."""

    def __init__(self, config: SignalEngineConfig | None = None) -> None:
        self._config = config or SignalEngineConfig()

    def generate(
        self,
        *,
        state: MarketState,
        probabilities: ProbabilityResult,
        trap: TrapResult,
        regime: RegimeResult,
    ) -> SignalResult:
        feature_map = state.feature_snapshot
        close = _safe_float(feature_map.get("close")) or _safe_float(feature_map.get("1m_close"))
        reasons: list[str] = []

        if probabilities.confidence < self._config.confidence_floor:
            reasons.append("confidence below floor")
            return self._neutral_signal(state, regime, probabilities, trap, close, reasons)

        if trap.trap_score >= self._config.trap_cutoff or state.trade_grade == TradeGrade.AVOID:
            reasons.append("trap or trade grade filter rejected entry")
            return self._neutral_signal(state, regime, probabilities, trap, close, reasons)

        if state.regime in {MarketRegime.SIDEWAYS_COMPRESSION, MarketRegime.HIGH_NOISE_ENVIRONMENT}:
            reasons.append("state indicates range/chop")
            return self._neutral_signal(state, regime, probabilities, trap, close, reasons)

        if state.quality_score < self._config.min_trade_quality:
            reasons.append("trade quality below minimum")
            return self._neutral_signal(state, regime, probabilities, trap, close, reasons)

        if probabilities.bullish_probability >= self._config.bullish_threshold and probabilities.bullish_probability > probabilities.bearish_probability:
            reasons.append("bullish probability dominates")
            return self._build_directional_signal(
                action=SignalAction.BUY_CE,
                confidence=probabilities.bullish_probability,
                state=state,
                regime=regime,
                probabilities=probabilities,
                trap=trap,
                close=close,
                reasons=reasons,
            )

        if probabilities.bearish_probability >= self._config.bearish_threshold and probabilities.bearish_probability > probabilities.bullish_probability:
            reasons.append("bearish probability dominates")
            return self._build_directional_signal(
                action=SignalAction.BUY_PE,
                confidence=probabilities.bearish_probability,
                state=state,
                regime=regime,
                probabilities=probabilities,
                trap=trap,
                close=close,
                reasons=reasons,
            )

        reasons.append("no directional edge")
        return self._neutral_signal(state, regime, probabilities, trap, close, reasons)

    def _neutral_signal(
        self,
        state: MarketState,
        regime: RegimeResult,
        probabilities: ProbabilityResult,
        trap: TrapResult,
        close: float | None,
        reasons: list[str],
    ) -> SignalResult:
        ttl = int(state.signal_ttl_candles or 1)
        expires_at = state.timestamp + timedelta(minutes=max(ttl, 1))
        return SignalResult(
            action=SignalAction.NO_TRADE,
            confidence=probabilities.confidence,
            regime=regime.regime,
            probability=probabilities,
            trap=trap,
            entry_reference=close,
            stop_loss=None,
            target=None,
            strike=None,
            quantity=0,
            expires_at=expires_at,
            signal_ttl_candles=ttl,
            trade_grade=TradeGrade.AVOID,
            quality_score=state.quality_score,
            state=state,
            transition=state.transition,
            reasoning=reasons,
        )

    def _build_directional_signal(
        self,
        *,
        action: SignalAction,
        confidence: float,
        state: MarketState,
        regime: RegimeResult,
        probabilities: ProbabilityResult,
        trap: TrapResult,
        close: float | None,
        reasons: list[str],
    ) -> SignalResult:
        ttl = state.signal_ttl_candles
        expires_at = state.timestamp + timedelta(minutes=max(ttl, 1))
        stop_loss = None
        target = None
        if close is not None:
            stop_loss = close * (0.995 if action == SignalAction.BUY_CE else 1.005)
            target = close * (1.012 if action == SignalAction.BUY_CE else 0.988)
        return SignalResult(
            action=action,
            confidence=confidence,
            regime=regime.regime,
            probability=probabilities,
            trap=trap,
            entry_reference=close,
            stop_loss=stop_loss,
            target=target,
            strike=None,
            quantity=0,
            expires_at=expires_at,
            signal_ttl_candles=ttl,
            trade_grade=state.trade_grade,
            quality_score=state.quality_score,
            state=state,
            transition=state.transition,
            reasoning=reasons + list(state.notes),
        )


class StrikeSelectionEngine:
    """Select ATM/ITM/OTM strikes using state, IV, and directional confidence."""

    def __init__(self, config: StrikeSelectionConfig | None = None) -> None:
        self._config = config or StrikeSelectionConfig()

    def select(self, state: MarketState, signal: SignalResult, option_chain: pd.DataFrame) -> StrikeResult:
        feature_map = state.feature_snapshot
        spot_price = _safe_float(feature_map.get("close")) or _safe_float(feature_map.get("1m_close"))
        confidence = signal.confidence
        side = OptionSide.CE if signal.action in {SignalAction.BUY_CE, SignalAction.SELL_PE} else OptionSide.PE
        strike_candidates = self._available_strikes(option_chain, side)
        step = self._infer_step(strike_candidates) if strike_candidates else self._config.default_step
        expected_move = float(max(1.0, state.quality_score) * step * max(confidence, 0.5))

        if not strike_candidates or spot_price is None:
            if spot_price is None:
                return StrikeResult(option_side=side, strike=None, expiry=None, moneyness="unknown", strike_distance=None, expected_move=expected_move, selection_grade=state.trade_grade, rationale=["no spot or strike data"])
            fallback = round(spot_price / step) * step
            if confidence >= 0.75 and state.volatility_state in {state.volatility_state.EXPANDING, state.volatility_state.HIGH}:
                fallback += step if side == OptionSide.CE else -step
                moneyness = "otm"
            elif confidence >= 0.70 and state.trade_grade in {TradeGrade.A_PLUS, TradeGrade.A}:
                moneyness = "itm"
            else:
                moneyness = "atm"
            strike_distance = float((fallback - spot_price) / max(spot_price, 1e-9))
            return StrikeResult(option_side=side, strike=float(max(fallback, 0.0)), expiry=None, moneyness=moneyness, strike_distance=strike_distance, expected_move=expected_move, selection_grade=state.trade_grade, rationale=["fallback strike from spot"])

        strikes = sorted(strike_candidates)
        if confidence >= 0.78 and state.trade_grade == TradeGrade.A_PLUS:
            offset = self._config.aggressive_otm_offset if side == OptionSide.CE else -self._config.aggressive_otm_offset
            target_strike = spot_price + offset * step if side == OptionSide.CE else spot_price - offset * step
            moneyness = "otm"
        elif confidence >= 0.70:
            target_strike = spot_price
            moneyness = "atm"
        else:
            target_strike = spot_price
            moneyness = "conservative_atm"
        chosen = min(strikes, key=lambda strike: abs(strike - target_strike))
        expiry = self._nearest_expiry(option_chain)
        strike_distance = float((chosen - spot_price) / max(spot_price, 1e-9))
        rationale = [f"selected {moneyness}", f"confidence={confidence:.2f}", f"step={step:.2f}", f"grade={state.trade_grade.value}"]
        return StrikeResult(option_side=side, strike=float(chosen), expiry=expiry, moneyness=moneyness, strike_distance=strike_distance, expected_move=expected_move, selection_grade=state.trade_grade, rationale=rationale)

    @staticmethod
    def _available_strikes(option_chain: pd.DataFrame, side: OptionSide) -> list[float]:
        if option_chain.empty:
            return []
        working = option_chain.copy()
        if "option_type" not in working.columns or "strike" not in working.columns:
            return []
        strikes = working.loc[working["option_type"].astype(str) == side.value, "strike"]
        return [float(value) for value in pd.to_numeric(strikes, errors="coerce").dropna().unique().tolist()]

    @staticmethod
    def _infer_step(strikes: list[float]) -> float:
        if len(strikes) < 2:
            return 50.0
        diffs = np.diff(sorted(strikes))
        step = float(np.median(diffs)) if len(diffs) else 50.0
        return step if step > 0 else 50.0

    @staticmethod
    def _nearest_expiry(option_chain: pd.DataFrame) -> str | None:
        if option_chain.empty or "expiry" not in option_chain.columns:
            return None
        expiries = option_chain["expiry"].dropna().astype(str).unique().tolist()
        return sorted(expiries)[0] if expiries else None


class RiskManagementEngine:
    """Convert signal quality into a bounded risk plan."""

    def __init__(self, config: RiskEngineConfig | None = None) -> None:
        self._config = config or RiskEngineConfig()

    def plan(
        self,
        *,
        state: MarketState,
        signal: SignalResult,
        regime: RegimeResult,
        trade_count_today: int,
        realized_daily_pnl: float,
    ) -> RiskResult:
        feature_map = state.feature_snapshot
        close = _safe_float(feature_map.get("close")) or _safe_float(feature_map.get("1m_close")) or 0.0
        if trade_count_today >= self._config.max_trades_per_day:
            return RiskResult(False, 0, None, None, False, "max trades per day reached", partial_exit_fraction=0.0, time_stop_minutes=self._config.time_stop_minutes, position_value=0.0)

        daily_loss_limit = -self._config.capital * self._config.max_daily_loss_pct
        if realized_daily_pnl <= daily_loss_limit:
            return RiskResult(False, 0, None, None, True, "max daily loss reached", partial_exit_fraction=0.0, time_stop_minutes=self._config.time_stop_minutes, position_value=0.0)

        if not state.is_tradeable or signal.action == SignalAction.NO_TRADE or signal.trade_grade == TradeGrade.AVOID:
            return RiskResult(False, 0, signal.stop_loss, signal.target, False, "state or signal filtered", partial_exit_fraction=0.0, time_stop_minutes=self._config.time_stop_minutes, position_value=0.0)

        stop_loss = signal.stop_loss
        if stop_loss is None and close > 0:
            stop_loss = close * (0.995 if signal.action == SignalAction.BUY_CE else 1.005)

        target = signal.target
        if target is None and stop_loss is not None:
            target = close + 2 * (close - stop_loss) if signal.action == SignalAction.BUY_CE else close - 2 * (stop_loss - close)

        risk_amount = self._config.capital * self._config.risk_per_trade * max(signal.confidence, 0.35)
        per_unit_risk = abs((close or 1.0) - (stop_loss or close)) * self._config.contract_multiplier
        max_exposure_units = int((self._config.capital * self._config.max_capital_exposure) / max((close or 1.0) * self._config.contract_multiplier, 1e-9))
        risk_units = int(risk_amount / max(per_unit_risk, 1e-9))
        quantity = int(max(min(risk_units, max_exposure_units), 0))

        if state.trade_grade == TradeGrade.A:
            quantity = int(max(1, quantity * 0.85))
        elif state.trade_grade == TradeGrade.B:
            quantity = int(max(1, quantity * 0.65))
        if state.volatility_state.name == "HIGH":
            quantity = max(1, int(quantity * 0.50))

        if quantity < self._config.min_units:
            return RiskResult(False, 0, stop_loss, target, False, "size below minimum", partial_exit_fraction=0.0, time_stop_minutes=self._config.time_stop_minutes, position_value=0.0)

        partial_exit_fraction = self._config.partial_exit_fraction if state.trade_grade in {TradeGrade.A_PLUS, TradeGrade.A} else 0.25
        position_value = float(quantity * close * self._config.contract_multiplier)
        return RiskResult(True, quantity, stop_loss, target, False, "risk checks passed", partial_exit_fraction=partial_exit_fraction, time_stop_minutes=self._config.time_stop_minutes, position_value=position_value)


def _latest_row(features: pd.DataFrame) -> pd.Series:
    if features.empty:
        return pd.Series(dtype=float)
    return features.iloc[-1]


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _softmax(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.array(list(values), dtype=float)
    arr = arr - np.max(arr)
    exp_arr = np.exp(arr)
    total = exp_arr.sum()
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    probs = exp_arr / total
    return float(probs[0]), float(probs[1]), float(probs[2])


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + exp(-value)))


def _calibration_band(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.65:
        return "moderate"
    return "low"
