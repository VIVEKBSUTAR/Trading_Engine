"""Probabilistic live signal engines for NIFTY options intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from trading_engine.intelligence.models import (
    MarketRegime,
    OptionSide,
    ProbabilityResult,
    RegimeResult,
    RiskResult,
    SignalAction,
    SignalResult,
    StrikeResult,
    TrapResult,
)


@dataclass(slots=True)
class RegimeEngineConfig:
    """Thresholds for live regime classification."""

    trend_strength_threshold: float = 0.0025
    compression_threshold: float = 0.55
    volatility_threshold: float = 0.012
    fake_breakout_threshold: float = 0.60


class MarketRegimeEngine:
    """Classify the current market state before any signal is emitted."""

    def __init__(self, config: RegimeEngineConfig | None = None) -> None:
        self._config = config or RegimeEngineConfig()

    def classify(self, features: pd.DataFrame, trap: TrapResult) -> RegimeResult:
        """Return the current regime and a confidence score."""
        latest = _latest_row(features)
        reasons: list[str] = []

        trend_strength = _safe_float(latest.get("trend_strength"))
        breakout_pressure = _safe_float(latest.get("breakout_pressure"))
        range_expansion = _safe_float(latest.get("range_expansion"))
        compression_zone = _safe_float(latest.get("compression_zone"))
        vix_level = _safe_float(latest.get("vix_level"))
        vix_change = _safe_float(latest.get("vix_ret_1"))

        if trap.fake_breakout_probability >= self._config.fake_breakout_threshold:
            reasons.append("trap score elevated")
            return RegimeResult(MarketRegime.FAKE_BREAKOUT_ENVIRONMENT, trap.fake_breakout_probability, reasons)

        if trend_strength is not None and trend_strength >= self._config.trend_strength_threshold:
            if breakout_pressure is not None and breakout_pressure > 0.0:
                reasons.append("trend and breakout pressure aligned")
                if vix_change is not None and vix_change > 0.01:
                    reasons.append("vix expanding")
                    return RegimeResult(MarketRegime.VOLATILE_EXPANSION, min(0.95, 0.5 + abs(vix_change)), reasons)
                return RegimeResult(MarketRegime.BULLISH_TRENDING if trend_strength > 0 else MarketRegime.BEARISH_TRENDING, min(0.95, abs(trend_strength) * 100), reasons)

        if compression_zone is not None and compression_zone >= self._config.compression_threshold:
            reasons.append("price compression detected")
            return RegimeResult(MarketRegime.SIDEWAYS_CHOPPY, float(compression_zone), reasons)

        if vix_level is not None and vix_level > 0 and vix_change is not None and vix_change > self._config.volatility_threshold:
            reasons.append("high volatility regime")
            return RegimeResult(MarketRegime.VOLATILE_EXPANSION, min(0.9, vix_change * 100), reasons)

        if breakout_pressure is not None and breakout_pressure < 0 and range_expansion is not None and range_expansion < 1.0:
            reasons.append("bearish pressure building")
            return RegimeResult(MarketRegime.BEARISH_TRENDING, min(0.8, abs(breakout_pressure) * 10), reasons)

        if breakout_pressure is not None and breakout_pressure > 0 and range_expansion is not None and range_expansion > 1.0:
            reasons.append("bullish pressure building")
            return RegimeResult(MarketRegime.BULLISH_TRENDING, min(0.8, breakout_pressure * 10), reasons)

        reasons.append("defaulting to sideways")
        return RegimeResult(MarketRegime.SIDEWAYS_CHOPPY, 0.5, reasons)


class MomentumPredictionEngine:
    """Estimate direction probabilities over a 1-5 minute horizon."""

    def __init__(self, horizon_minutes: int = 4) -> None:
        self._horizon_minutes = horizon_minutes

    def predict(self, features: pd.DataFrame, regime: RegimeResult) -> ProbabilityResult:
        latest = _latest_row(features)
        evidence: list[str] = []

        trend_strength = _safe_float(latest.get("trend_strength")) or 0.0
        ema20_slope = _safe_float(latest.get("ema20_slope")) or 0.0
        volume_expansion = _safe_float(latest.get("volume_expansion")) or 0.0
        breakout_pressure = _safe_float(latest.get("breakout_pressure")) or 0.0
        price_velocity = _safe_float(latest.get("price_velocity")) or 0.0
        pcr = _safe_float(latest.get("put_call_ratio")) or 1.0
        oi_imbalance = _safe_float(latest.get("oi_imbalance")) or 0.0
        vix_change = _safe_float(latest.get("vix_ret_1")) or 0.0
        compression_zone = _safe_float(latest.get("compression_zone")) or 0.0

        bullish_score = (
            1.4 * max(trend_strength, 0.0)
            + 1.2 * max(ema20_slope, 0.0)
            + 1.0 * max(volume_expansion - 1.0, 0.0)
            + 1.2 * max(breakout_pressure, 0.0)
            + 0.8 * max(price_velocity, 0.0)
            + 0.6 * max(1.2 - pcr, 0.0)
            + 0.6 * max(oi_imbalance, 0.0)
            - 0.7 * max(vix_change, 0.0)
        )
        bearish_score = (
            1.4 * max(-trend_strength, 0.0)
            + 1.2 * max(-ema20_slope, 0.0)
            + 1.0 * max(volume_expansion - 1.0, 0.0)
            + 1.2 * max(-breakout_pressure, 0.0)
            + 0.8 * max(-price_velocity, 0.0)
            + 0.6 * max(pcr - 1.0, 0.0)
            + 0.6 * max(-oi_imbalance, 0.0)
            + 0.7 * max(vix_change, 0.0)
        )
        sideways_score = (
            1.5 * max(compression_zone, 0.0)
            + 1.0 * max(1.0 - abs(trend_strength) * 100.0, 0.0)
            + 0.6 * max(1.0 - abs(breakout_pressure) * 10.0, 0.0)
        )

        if regime.regime == MarketRegime.FAKE_BREAKOUT_ENVIRONMENT:
            sideways_score += 0.6
            evidence.append("fake breakout regime bias")
        elif regime.regime == MarketRegime.VOLATILE_EXPANSION:
            bullish_score += 0.1
            bearish_score += 0.1
            evidence.append("volatility expansion increases directional odds")
        elif regime.regime == MarketRegime.BULLISH_TRENDING:
            bullish_score += 0.3
            evidence.append("bullish regime alignment")
        elif regime.regime == MarketRegime.BEARISH_TRENDING:
            bearish_score += 0.3
            evidence.append("bearish regime alignment")
        else:
            sideways_score += 0.2

        probs = _softmax([bullish_score, bearish_score, sideways_score])
        confidence = float(max(probs))
        if bullish_score == max(bullish_score, bearish_score, sideways_score):
            evidence.append("bullish score dominant")
        elif bearish_score == max(bullish_score, bearish_score, sideways_score):
            evidence.append("bearish score dominant")
        else:
            evidence.append("sideways score dominant")

        return ProbabilityResult(
            bullish_probability=float(probs[0]),
            bearish_probability=float(probs[1]),
            sideways_probability=float(probs[2]),
            confidence=confidence,
            horizon_minutes=self._horizon_minutes,
            evidence=evidence,
        )


class TrapDetectionEngine:
    """Detect stoploss hunts, fake breakouts, and exhaustion behavior."""

    def assess(self, features: pd.DataFrame) -> TrapResult:
        latest = _latest_row(features)
        evidence: list[str] = []

        close = _safe_float(latest.get("close")) or 0.0
        high = _safe_float(latest.get("high")) or close
        low = _safe_float(latest.get("low")) or close
        open_price = _safe_float(latest.get("open")) or close
        volume_expansion = _safe_float(latest.get("volume_expansion")) or 0.0
        breakout_pressure = _safe_float(latest.get("breakout_pressure")) or 0.0
        range_expansion = _safe_float(latest.get("range_expansion")) or 1.0
        pcr = _safe_float(latest.get("put_call_ratio")) or 1.0
        oi_imbalance = _safe_float(latest.get("oi_imbalance")) or 0.0

        candle_range = max(high - low, 1e-9)
        upper_wick = max(high - max(close, open_price), 0.0) / candle_range
        lower_wick = max(min(close, open_price) - low, 0.0) / candle_range
        close_location = (close - low) / candle_range

        fake_breakout_probability = _sigmoid(
            1.6 * max(upper_wick, lower_wick)
            + 0.8 * max(1.0 - close_location, 0.0)
            + 0.6 * max(volume_expansion - 1.0, 0.0)
            + 0.5 * max(1.0 - abs(breakout_pressure) * 10.0, 0.0)
        )
        buyer_trap_probability = _sigmoid(
            1.2 * max(upper_wick, 0.0)
            + 0.7 * max(pcr - 1.0, 0.0)
            + 0.5 * max(-oi_imbalance, 0.0)
        )
        seller_trap_probability = _sigmoid(
            1.2 * max(lower_wick, 0.0)
            + 0.7 * max(1.0 - pcr, 0.0)
            + 0.5 * max(oi_imbalance, 0.0)
        )
        stoploss_hunt_probability = _sigmoid(
            1.3 * max(upper_wick, lower_wick)
            + 0.8 * max(range_expansion - 1.0, 0.0)
            + 0.4 * max(volume_expansion - 1.0, 0.0)
        )

        if fake_breakout_probability > 0.5:
            evidence.append("wick rejection and weak follow-through")
        if buyer_trap_probability > seller_trap_probability:
            evidence.append("buyer trap bias")
        else:
            evidence.append("seller trap bias")

        return TrapResult(
            fake_breakout_probability=float(fake_breakout_probability),
            buyer_trap_probability=float(buyer_trap_probability),
            seller_trap_probability=float(seller_trap_probability),
            stoploss_hunt_probability=float(stoploss_hunt_probability),
            evidence=evidence,
        )


@dataclass(slots=True)
class SignalEngineConfig:
    """Signal thresholds for the live alert engine."""

    bullish_threshold: float = 0.65
    bearish_threshold: float = 0.65
    confidence_floor: float = 0.60
    trap_cutoff: float = 0.60


class SignalGenerationEngine:
    """Translate probabilities and trap diagnostics into a filtered trading signal."""

    def __init__(self, config: SignalEngineConfig | None = None) -> None:
        self._config = config or SignalEngineConfig()

    def generate(
        self,
        *,
        features: pd.DataFrame,
        regime: RegimeResult,
        probabilities: ProbabilityResult,
        trap: TrapResult,
    ) -> SignalResult:
        latest = _latest_row(features)
        close = _safe_float(latest.get("close"))
        reasons: list[str] = []

        if probabilities.confidence < self._config.confidence_floor:
            reasons.append("confidence below floor")
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
                reasoning=reasons,
            )

        if trap.fake_breakout_probability >= self._config.trap_cutoff:
            reasons.append("fake breakout probability too high")
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
                reasoning=reasons,
            )

        if probabilities.bullish_probability >= self._config.bullish_threshold and probabilities.bullish_probability > probabilities.bearish_probability:
            reasons.append("bullish probability dominates")
            return SignalResult(
                action=SignalAction.BUY_CE,
                confidence=probabilities.bullish_probability,
                regime=regime.regime,
                probability=probabilities,
                trap=trap,
                entry_reference=close,
                stop_loss=close * 0.995 if close else None,
                target=close * 1.01 if close else None,
                strike=None,
                quantity=0,
                reasoning=reasons,
            )

        if probabilities.bearish_probability >= self._config.bearish_threshold and probabilities.bearish_probability > probabilities.bullish_probability:
            reasons.append("bearish probability dominates")
            return SignalResult(
                action=SignalAction.BUY_PE,
                confidence=probabilities.bearish_probability,
                regime=regime.regime,
                probability=probabilities,
                trap=trap,
                entry_reference=close,
                stop_loss=close * 1.005 if close else None,
                target=close * 0.99 if close else None,
                strike=None,
                quantity=0,
                reasoning=reasons,
            )

        reasons.append("no directional edge")
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
            reasoning=reasons,
        )


@dataclass(slots=True)
class StrikeSelectionConfig:
    """Strike selection parameters for live NIFTY options."""

    default_step: float = 50.0


class StrikeSelectionEngine:
    """Select ATM/ITM/OTM strikes using spot, IV, and directional confidence."""

    def __init__(self, config: StrikeSelectionConfig | None = None) -> None:
        self._config = config or StrikeSelectionConfig()

    def select(self, features: pd.DataFrame, signal: SignalResult, option_chain: pd.DataFrame) -> StrikeResult:
        latest = _latest_row(features)
        spot_price = _safe_float(latest.get("close"))
        vix_level = _safe_float(latest.get("vix_level")) or 0.0
        confidence = signal.confidence

        side = OptionSide.CE if signal.action in {SignalAction.BUY_CE, SignalAction.SELL_PE} else OptionSide.PE
        strike_candidates = self._available_strikes(option_chain, side)
        if not strike_candidates:
            if spot_price is None:
                return StrikeResult(option_side=side, strike=None, expiry=None, moneyness="unknown", rationale=["no spot or strike data"])
            rounded = round(spot_price / self._config.default_step) * self._config.default_step
            moneyness = "atm"
            if confidence >= 0.75 and vix_level < 0.02:
                rounded += self._config.default_step if side == OptionSide.CE else -self._config.default_step
                moneyness = "otm"
            elif confidence >= 0.75 and vix_level >= 0.02:
                moneyness = "itm"
            return StrikeResult(option_side=side, strike=float(max(rounded, 0.0)), expiry=None, moneyness=moneyness, rationale=["fallback strike from spot"])

        strikes = sorted(strike_candidates)
        step = self._infer_step(strikes)
        if spot_price is None:
            return StrikeResult(option_side=side, strike=float(strikes[len(strikes) // 2]), expiry=None, moneyness="atm", rationale=["spot unavailable, used median strike"])

        offset_steps = 0
        moneyness = "atm"
        if confidence >= 0.75 and vix_level < 0.02:
            offset_steps = 1 if side == OptionSide.CE else -1
            moneyness = "otm"
        elif confidence >= 0.75 and vix_level >= 0.02:
            offset_steps = 0
            moneyness = "itm"
        elif confidence >= 0.65:
            offset_steps = 0
            moneyness = "atm"

        target_strike = spot_price + (offset_steps * step if side == OptionSide.CE else -offset_steps * step)
        chosen = min(strikes, key=lambda strike: abs(strike - target_strike))
        expiry = self._nearest_expiry(option_chain)
        rationale = [f"selected {moneyness}", f"confidence={confidence:.2f}", f"step={step:.2f}"]
        return StrikeResult(option_side=side, strike=float(chosen), expiry=expiry, moneyness=moneyness, rationale=rationale)

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


class RiskManagementEngine:
    """Convert signal quality into a bounded risk plan."""

    def __init__(self, config: RiskEngineConfig | None = None) -> None:
        self._config = config or RiskEngineConfig()

    def plan(
        self,
        *,
        features: pd.DataFrame,
        signal: SignalResult,
        regime: RegimeResult,
        trade_count_today: int,
        realized_daily_pnl: float,
    ) -> RiskResult:
        latest = _latest_row(features)
        close = _safe_float(latest.get("close")) or 0.0
        vix_level = _safe_float(latest.get("vix_level")) or 0.0

        if trade_count_today >= self._config.max_trades_per_day:
            return RiskResult(False, 0, None, None, False, "max trades per day reached")

        if realized_daily_pnl <= -self._config.capital * self._config.max_daily_loss_pct:
            return RiskResult(False, 0, None, None, True, "max daily loss reached")

        if regime.regime == MarketRegime.SIDEWAYS_CHOPPY:
            return RiskResult(False, 0, None, None, False, "sideways regime filtered")

        if signal.action == SignalAction.NO_TRADE:
            return RiskResult(False, 0, None, None, False, "no trade signal")

        stop_loss = signal.stop_loss
        if stop_loss is None and close > 0:
            stop_loss = close * (0.995 if signal.action == SignalAction.BUY_CE else 1.005)

        target = signal.target
        if target is None and stop_loss is not None:
            target = close + 2 * (close - stop_loss) if signal.action == SignalAction.BUY_CE else close - 2 * (stop_loss - close)

        risk_amount = self._config.capital * self._config.risk_per_trade * signal.confidence
        per_unit_risk = abs((close or 1.0) - (stop_loss or close)) * self._config.contract_multiplier
        quantity = int(max(min(risk_amount / max(per_unit_risk, 1e-9), (self._config.capital * self._config.max_capital_exposure) / max(close, 1e-9)), 0))
        if quantity < self._config.min_units:
            return RiskResult(False, 0, stop_loss, target, False, "size below minimum")

        if vix_level > 0.02:
            quantity = max(self._config.min_units, int(quantity * 0.5))

        return RiskResult(True, quantity, stop_loss, target, False, "risk checks passed")


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
