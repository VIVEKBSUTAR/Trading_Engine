"""Probabilistic live signal engines for NIFTY options intelligence."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp

import numpy as np
import pandas as pd

from trading_engine.intelligence.models import (
    MarketRegime,
    MarketState,
    OptionSide,
    ProbabilityResult,
    RegimeResult,
    RegimeTransitionStage,
    RiskResult,
    SignalAction,
    SignalResult,
    StrikeResult,
    TradeGrade,
    TrapResult,
    TrendState,
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
    chop_cutoff: float = 0.68
    instability_cutoff: float = 0.65
    max_spread_percentage: float = 0.05
    max_implied_volatility_percentile: float = 0.90


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

        trend_regime = state.regime in {MarketRegime.TREND_BULLISH, MarketRegime.TREND_BEARISH}
        compression_regime = state.regime == MarketRegime.SIDEWAYS_COMPRESSION or state.trend_state == TrendState.COMPRESSION
        volatile_regime = state.regime in {MarketRegime.VOLATILE_EXPANSION, MarketRegime.HIGH_NOISE_ENVIRONMENT}
        session_quality = float(feature_map.get("session_quality", state.session_quality) or 0.5)

        trend_weight = self._weights.trend_strength * (1.35 if trend_regime else 1.0)
        vwap_weight = self._weights.vwap_alignment * (1.20 if trend_regime else 1.0)
        vol_support_weight = self._weights.volatility_support * (1.25 if compression_regime else (0.85 if volatile_regime else 1.0))
        trap_weight = self._weights.trap_penalty * (1.25 if volatile_regime else 1.0)
        session_weight = self._weights.session_quality * (1.15 if session_quality >= 0.8 else 0.9)

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
            trend_weight * max(0.0, (trend_15 + trend_5 + trend_1) / 3.0 * 100.0)
            + vwap_weight * max(0.0, (vwap_15 + vwap_5 + vwap_1) / 3.0 * 100.0)
            + self._weights.oi_confirmation * max(0.0, oi_bias)
            + self._weights.volume_expansion * max(0.0, volume_expansion - 1.0)
            + session_weight * session_quality
            + vol_support_weight * volatility_support
            + self._weights.breakout_pressure * max(0.0, breakout_pressure * 100.0)
            + self._weights.momentum_persistence * persistence
            - trap_weight * trap_penalty
        )
        bearish_score = (
            trend_weight * max(0.0, -(trend_15 + trend_5 + trend_1) / 3.0 * 100.0)
            + vwap_weight * max(0.0, -(vwap_15 + vwap_5 + vwap_1) / 3.0 * 100.0)
            + self._weights.oi_confirmation * max(0.0, -oi_bias)
            + self._weights.volume_expansion * max(0.0, volume_expansion - 1.0)
            + session_weight * session_quality
            + vol_support_weight * volatility_support
            + self._weights.breakout_pressure * max(0.0, -breakout_pressure * 100.0)
            + self._weights.momentum_persistence * persistence
            - trap_weight * trap_penalty
        )
        sideways_score = (
            self._weights.sideways_compression * max(0.0, sideways_hint)
            + 0.20 * max(0.0, 1.0 - abs(trend_1) * 100.0)
            + 0.15 * max(0.0, 1.0 - abs(breakout_pressure) * 100.0)
            + 0.15 * max(0.0, 1.0 - persistence)
            + 0.10 * max(0.0, 1.0 - session_quality)
            + 0.10 * trap_penalty
        )

        if state.regime in {MarketRegime.SIDEWAYS_COMPRESSION, MarketRegime.HIGH_NOISE_ENVIRONMENT} or state.chop_probability >= 0.68:
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
        if state.transition.stage == RegimeTransitionStage.LIQUIDATION_PHASE:
            sideways_score += 0.20
            evidence.append("liquidation phase suppresses participation")

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
        self._historical_iv: dict[tuple[float, str], list[float]] = {}
        self._signal_counter = itertools.count(1)

    def extract_strike_quote(
        self,
        strike: float,
        side: OptionSide | str,
        option_chain: pd.DataFrame,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        """Return bid, ask, IV, call OI, put OI from one strike/expiry."""
        if option_chain is None or option_chain.empty:
            return None, None, None, None, None

        side_str = side.value.upper() if isinstance(side, OptionSide) else str(side).upper()
        working = option_chain.copy()

        if "strike" in working.columns:
            working = working.loc[pd.to_numeric(working["strike"], errors="coerce") == float(strike)]

        if working.empty:
            return None, None, None, None, None

        if "expiry" in working.columns:
            working = working.sort_values("expiry")
            working = working.loc[working["expiry"] == working.iloc[0]["expiry"]]

        def oi(option_side: str) -> float | None:
            rows = working.loc[working["option_type"].astype(str).str.upper() == option_side]
            if rows.empty:
                return None
            row = rows.iloc[0]
            return _safe_float(row.get("open_interest", row.get("oi")))

        call_oi, put_oi = oi("CE"), oi("PE")
        working = working.loc[working["option_type"].astype(str).str.upper() == side_str]
        if working.empty:
            return None, None, None, call_oi, put_oi

        row = working.iloc[0]
        bid = _safe_float(row.get("bid_price") or row.get("bid") or row.get("best_bid"))
        ask = _safe_float(row.get("ask_price") or row.get("ask") or row.get("best_ask") or row.get("last_price"))
        iv = _safe_float(row.get("iv") or row.get("implied_volatility") or row.get("implied_vol"))

        return bid, ask, iv, call_oi, put_oi

    def calculate_spread_percentage(self, bid: float, ask: float) -> float | None:
        """Calculate spread percentage: (ask - bid) / bid."""
        if bid <= 0.0 or ask <= 0.0:
            return None
        return float((ask - bid) / bid)

    def check_liquidity_filter(
        self,
        strike: float,
        side: OptionSide | str,
        option_chain: pd.DataFrame,
    ) -> tuple[bool, float | None, str | None]:
        """Verify the selected strike spread percentage does not exceed max_spread_percentage (default 0.05)."""
        bid, ask, _, _, _ = self.extract_strike_quote(strike, side, option_chain)
        if bid is None or ask is None:
            return True, None, None

        spread_pct = self.calculate_spread_percentage(bid, ask)
        if spread_pct is None:
            return False, None, "liquidity filter: non-positive bid/ask quote"

        if spread_pct > self._config.max_spread_percentage:
            return (
                False,
                spread_pct,
                f"liquidity filter: spread {spread_pct:.4f} ({spread_pct * 100:.2f}%) exceeds "
                f"max allowed {self._config.max_spread_percentage:.2f} ({self._config.max_spread_percentage * 100:.1f}%) penalty",
            )

        return True, spread_pct, None

    def check_volatility_filter(
        self,
        strike: float,
        side: OptionSide | str,
        option_chain: pd.DataFrame,
        historical_iv_window: pd.Series | list[float] | None = None,
    ) -> tuple[bool, float | None, float | None, str | None]:
        """Verify implied volatility is below max_implied_volatility_percentile (default 90th percentile)."""
        _, _, current_iv, _, _ = self.extract_strike_quote(strike, side, option_chain)
        if current_iv is None:
            return True, None, None, None

        side_str = side.value.upper() if isinstance(side, OptionSide) else str(side).upper()
        key = (float(strike), side_str)

        history: list[float] = []
        if historical_iv_window is not None:
            if isinstance(historical_iv_window, pd.Series):
                history = [float(v) for v in historical_iv_window.dropna().tolist()]
            else:
                history = [float(v) for v in historical_iv_window if v is not None and not np.isnan(v)]
        else:
            history = list(self._historical_iv.get(key, []))

        # Record current observation in internal rolling window (keep up to 100 observations)
        buf = self._historical_iv.setdefault(key, [])
        buf.append(current_iv)
        if len(buf) > 100:
            self._historical_iv[key] = buf[-100:]

        if len(history) < 2:
            return True, current_iv, None, None

        hist_arr = np.array(history, dtype=float)
        percentile = float(np.mean(hist_arr <= current_iv))

        if (
            percentile >= self._config.max_implied_volatility_percentile
            or current_iv >= float(np.percentile(hist_arr, self._config.max_implied_volatility_percentile * 100.0))
        ):
            return (
                False,
                current_iv,
                percentile,
                f"volatility filter: iv {current_iv:.4f} is at {percentile * 100:.1f}th percentile "
                f"(>= {self._config.max_implied_volatility_percentile * 100:.0f}th percentile) of rolling window - crush risk",
            )

        return True, current_iv, percentile, None

    def _resolve_candidate_strike(
        self,
        state: MarketState,
        action: SignalAction,
        confidence: float,
        option_chain: pd.DataFrame | None,
        selected_strike: StrikeResult | float | None,
    ) -> StrikeResult | None:
        if isinstance(selected_strike, StrikeResult):
            return selected_strike
        if isinstance(selected_strike, (int, float)):
            side = OptionSide.CE if action == SignalAction.BUY_CE else OptionSide.PE
            return StrikeResult(
                option_side=side,
                strike=float(selected_strike),
                expiry=None,
                moneyness="atm",
            )
        if option_chain is not None and not option_chain.empty:
            provisional = SignalResult(
                action=action,
                confidence=confidence,
                regime=state.regime,
                probability=ProbabilityResult(0.0, 0.0, 0.0, confidence, 4),
                trap=TrapResult(0.0, 0.0, 0.0, 0.0),
                entry_reference=None,
                stop_loss=None,
                target=None,
                strike=None,
                quantity=0,
            )
            return StrikeSelectionEngine().select(state, provisional, option_chain)
        return None

    def generate(
        self,
        *,
        state: MarketState,
        probabilities: ProbabilityResult,
        trap: TrapResult,
        regime: RegimeResult,
        selected_strike: StrikeResult | float | None = None,
        option_chain: pd.DataFrame | None = None,
        historical_iv_window: pd.Series | list[float] | None = None,
        snapshot_timestamp: datetime | None = None,
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

        if state.chop_probability >= self._config.chop_cutoff or state.instability_probability >= self._config.instability_cutoff:
            reasons.append("chop or instability filter rejected entry")
            return self._neutral_signal(state, regime, probabilities, trap, close, reasons)

        if state.regime in {MarketRegime.SIDEWAYS_COMPRESSION, MarketRegime.HIGH_NOISE_ENVIRONMENT}:
            reasons.append("state indicates range/chop")
            return self._neutral_signal(state, regime, probabilities, trap, close, reasons)

        if state.quality_score < self._config.min_trade_quality:
            reasons.append("trade quality below minimum")
            return self._neutral_signal(state, regime, probabilities, trap, close, reasons)

        is_bullish = probabilities.bullish_probability >= self._config.bullish_threshold and probabilities.bullish_probability > probabilities.bearish_probability
        is_bearish = probabilities.bearish_probability >= self._config.bearish_threshold and probabilities.bearish_probability > probabilities.bullish_probability

        if is_bullish or is_bearish:
            target_action = SignalAction.BUY_CE if is_bullish else SignalAction.BUY_PE
            target_side = OptionSide.CE if is_bullish else OptionSide.PE
            target_conf = probabilities.bullish_probability if is_bullish else probabilities.bearish_probability

            candidate_strike = self._resolve_candidate_strike(
                state=state,
                action=target_action,
                confidence=target_conf,
                option_chain=option_chain,
                selected_strike=selected_strike,
            )

            spread_pct: float | None = None
            iv_perc: float | None = None

            def reject(reason: str) -> SignalResult:
                sig = self._neutral_signal(state, regime, probabilities, trap, close, [reason])
                sig.strike = candidate_strike
                sig.filter_passed = False
                sig.filter_rejection_reason = reason
                return sig

            if candidate_strike is None or candidate_strike.strike is None or not candidate_strike.expiry:
                return reject("DTE filter: missing contract expiry")
            expiry_date = _contract_expiry(candidate_strike.expiry)
            if pd.isna(expiry_date):
                return reject("DTE filter: invalid contract expiry")
            snapshot_date = pd.Timestamp(snapshot_timestamp or state.timestamp)
            if snapshot_date.tzinfo is None:
                snapshot_date = snapshot_date.tz_localize("UTC")
            snapshot_date = snapshot_date.tz_convert("Asia/Kolkata").date()
            dte = (expiry_date.date() - snapshot_date).days
            if dte < 0:
                return reject("DTE filter: contract already expired")
            # All quote/IV/OI checks must refer to the exact emitted contract expiry.
            chain = option_chain if option_chain is not None else pd.DataFrame()
            if "expiry" in chain:
                dates = chain["expiry"].map(lambda value: _contract_expiry(value).date() if not pd.isna(_contract_expiry(value)) else None)
                chain = chain.loc[dates == expiry_date.date()]
            if chain.empty or "option_type" not in chain or "strike" not in chain:
                return reject("OI filter: missing contract chain")
            _, _, _, call_oi, put_oi = self.extract_strike_quote(candidate_strike.strike, target_side, chain)
            if (call_oi is None or put_oi is None or not np.isfinite(call_oi)
                    or not np.isfinite(put_oi) or call_oi <= 0 or put_oi < 0):
                return reject("OI filter: missing or invalid CE/PE open interest")
            oi_ratio = put_oi / call_oi
            if (is_bullish and oi_ratio < 0.8) or (is_bearish and oi_ratio > 1.2):
                rejected = reject(f"OI imbalance filter: put/call ratio {oi_ratio:.4f} rejects {target_action.value}")
                rejected.strike_oi_ratio = oi_ratio
                rejected.days_to_expiry = dte
                rejected.max_holding_minutes = 15 if dte == 0 else 45
                return rejected

            if (
                candidate_strike is not None
                and candidate_strike.strike is not None
                and option_chain is not None
                and not option_chain.empty
            ):
                # Step 1: Liquidity Filter (spread percentage <= 0.05)
                liq_ok, spread_pct, liq_reason = self.check_liquidity_filter(
                    strike=candidate_strike.strike,
                    side=target_side,
                    option_chain=chain,
                )
                if not liq_ok:
                    reasons.append(liq_reason or "liquidity filter rejected entry")
                    sig = self._neutral_signal(state, regime, probabilities, trap, close, reasons)
                    sig.strike = candidate_strike
                    sig.spread_percentage = spread_pct
                    sig.iv_percentile = None
                    sig.filter_passed = False
                    sig.filter_rejection_reason = liq_reason or "liquidity filter rejected entry"
                    return sig

                # Step 2: Volatility Filter (IV < 90th percentile of rolling historical window)
                vol_ok, cur_iv, iv_perc, vol_reason = self.check_volatility_filter(
                    strike=candidate_strike.strike,
                    side=target_side,
                    option_chain=chain,
                    historical_iv_window=historical_iv_window,
                )
                if not vol_ok:
                    reasons.append(vol_reason or "volatility filter rejected entry")
                    sig = self._neutral_signal(state, regime, probabilities, trap, close, reasons)
                    sig.strike = candidate_strike
                    sig.spread_percentage = spread_pct
                    sig.iv_percentile = iv_perc
                    sig.filter_passed = False
                    sig.filter_rejection_reason = vol_reason or "volatility filter rejected entry"
                    return sig

            reasons.append("bullish probability dominates" if is_bullish else "bearish probability dominates")
            sig = self._build_directional_signal(
                action=target_action,
                confidence=target_conf,
                state=state,
                regime=regime,
                probabilities=probabilities,
                trap=trap,
                close=close,
                reasons=reasons,
            )
            if candidate_strike is not None:
                sig.strike = candidate_strike
            sig.spread_percentage = spread_pct
            sig.iv_percentile = iv_perc
            sig.filter_passed = True
            sig.filter_rejection_reason = None
            sig.days_to_expiry = dte
            sig.max_holding_minutes = 15 if dte == 0 else 45
            sig.strike_oi_ratio = oi_ratio
            return sig

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
            signal_id=f"SIG-{state.timestamp.strftime('%Y%m%d%H%M%S')}-{next(self._signal_counter):05d}",
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
            signal_id=f"SIG-{state.timestamp.strftime('%Y%m%d%H%M%S')}-{next(self._signal_counter):05d}",
            filter_passed=True,
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
        expiry = self._nearest_expiry(option_chain, state.timestamp)
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
    def _nearest_expiry(option_chain: pd.DataFrame, timestamp: datetime | None = None) -> str | None:
        if option_chain.empty or "expiry" not in option_chain.columns:
            return None
        now = pd.Timestamp(timestamp or datetime.now().astimezone())
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        day = now.tz_convert("Asia/Kolkata").date()
        expiries = [_contract_expiry(value) for value in option_chain["expiry"].dropna().unique()]
        valid = [value for value in expiries if not pd.isna(value) and value.date() >= day]
        return min(valid).date().isoformat() if valid else None


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


def _contract_expiry(value: object) -> pd.Timestamp:
    """Parse ISO dates, Fyers DD-MM-YYYY dates, and epoch expiry identifiers."""
    text = str(value)
    if text.isdigit() and len(text) in {10, 13}:
        return pd.to_datetime(int(text), unit="ms" if len(text) == 13 else "s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
    if len(text) == 10 and text[2] == "-" and text[5] == "-":
        return pd.to_datetime(text, format="%d-%m-%Y", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


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
