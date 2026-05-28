"""Live market state aggregation for probabilistic options intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from loguru import logger

from broker.kite_client import LiveMarketSnapshot, MarketQuote
from broker.normalizer import NormalizedTickBatch
from trading_engine.features.options_analytics import (
    compute_dealer_positioning_proxy,
    compute_max_pain,
    compute_oi_imbalance,
    compute_put_call_ratio,
    compute_strike_concentration_zones,
)
from trading_engine.intelligence.features import build_live_features
from trading_engine.intelligence.models import (
    DirectionalBias,
    LiquidityState,
    MarketRegime,
    MarketSnapshotBundle,
    MarketState,
    MarketTransition,
    MomentumState,
    RegimeTransitionStage,
    SessionPhase,
    TradeGrade,
    TrendState,
    VolatilityState,
    utcnow,
)


@dataclass(slots=True)
class StateConfig:
    """Limits for in-memory live history buffers."""

    max_spot_rows: int = 2000
    max_vix_rows: int = 2000
    max_option_rows: int = 20000
    max_tick_rows: int = 50000
    candle_frequency: str = "1min"
    sideway_threshold: float = 0.58
    tradeable_quality_threshold: float = 0.55
    trend_strength_threshold: float = 0.0025
    compression_threshold: float = 0.60
    volatility_high_threshold: float = 18.0
    signal_ttl_min_candles: int = 2
    signal_ttl_max_candles: int = 4


class MarketStateAggregator:
    """Maintain rolling live state from broker and NSE market feeds."""

    def __init__(self, config: StateConfig | None = None) -> None:
        self._config = config or StateConfig()
        self._spot_history = pd.DataFrame()
        self._vix_history = pd.DataFrame()
        self._option_chain = pd.DataFrame()
        self._ticks = pd.DataFrame()
        self._last_snapshot: MarketSnapshotBundle | None = None
        self._last_state: MarketState | None = None

    def update_spot(self, quote: MarketQuote) -> None:
        row = pd.DataFrame(
            [
                {
                    "timestamp": utcnow(),
                    "instrument": quote.instrument,
                    "last_price": quote.last_price,
                    "change": quote.change,
                    "open": quote.ohlc.get("open"),
                    "high": quote.ohlc.get("high"),
                    "low": quote.ohlc.get("low"),
                    "close": quote.ohlc.get("close"),
                }
            ]
        )
        self._spot_history = self._append_and_trim(self._spot_history, row, self._config.max_spot_rows)
        logger.debug("Updated spot history", rows=len(self._spot_history))

    def update_vix(self, quote: MarketQuote) -> None:
        row = pd.DataFrame(
            [
                {
                    "timestamp": utcnow(),
                    "instrument": quote.instrument,
                    "last_price": quote.last_price,
                    "change": quote.change,
                    "close": quote.last_price,
                }
            ]
        )
        self._vix_history = self._append_and_trim(self._vix_history, row, self._config.max_vix_rows)
        logger.debug("Updated VIX history", rows=len(self._vix_history))

    def update_option_chain(self, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            logger.warning("Ignoring empty option-chain update")
            return

        working = frame.copy()
        if "timestamp" in working.columns:
            working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        self._option_chain = self._append_and_trim(self._option_chain, working, self._config.max_option_rows)
        logger.debug("Updated option chain history", rows=len(self._option_chain))

    def update_ticks(self, batch: NormalizedTickBatch) -> None:
        if batch.frame.empty:
            return

        working = batch.frame.copy()
        self._ticks = self._append_and_trim(self._ticks, working, self._config.max_tick_rows)
        logger.debug("Updated tick history", rows=len(self._ticks))

    def update_from_snapshot(self, snapshot: LiveMarketSnapshot | MarketSnapshotBundle, option_chain: pd.DataFrame | None = None) -> None:
        if hasattr(snapshot, "nifty_spot") and hasattr(snapshot, "india_vix"):
            self.update_spot(snapshot.nifty_spot)
            self.update_vix(snapshot.india_vix)
            if option_chain is not None:
                self.update_option_chain(option_chain)
            return

        if isinstance(snapshot, MarketSnapshotBundle):
            if not snapshot.spot_frame.empty:
                self._spot_history = self._append_and_trim(self._spot_history, snapshot.spot_frame.copy(), self._config.max_spot_rows)
            if not snapshot.vix_frame.empty:
                self._vix_history = self._append_and_trim(self._vix_history, snapshot.vix_frame.copy(), self._config.max_vix_rows)
            if not snapshot.option_chain_frame.empty:
                self._option_chain = self._append_and_trim(self._option_chain, snapshot.option_chain_frame.copy(), self._config.max_option_rows)
            if not snapshot.ticks_frame.empty:
                self._ticks = self._append_and_trim(self._ticks, snapshot.ticks_frame.copy(), self._config.max_tick_rows)
            return

        raise TypeError(f"Unsupported snapshot type: {type(snapshot)!r}")

    def build_snapshot(self) -> MarketSnapshotBundle:
        spot_frame = self._spot_history.copy()
        vix_frame = self._vix_history.copy()
        option_frame = self._option_chain.copy()
        tick_frame = self._ticks.copy()
        candles_1m = self._build_candles(tick_frame, "1min")
        candles_5m = self._build_candles(tick_frame, "5min")

        spot_price = self._latest_numeric(spot_frame, "last_price")
        vix_value = self._latest_numeric(vix_frame, "last_price")

        bundle = MarketSnapshotBundle(
            timestamp=utcnow(),
            spot_price=spot_price,
            vix_value=vix_value,
            spot_frame=spot_frame,
            vix_frame=vix_frame,
            option_chain_frame=option_frame,
            ticks_frame=tick_frame,
            candles_1m=candles_1m,
            candles_5m=candles_5m,
        )
        self._last_snapshot = bundle
        return bundle

    def build_market_state(self, snapshot: MarketSnapshotBundle | None = None) -> MarketState:
        current_snapshot = snapshot or self._last_snapshot or self.build_snapshot()
        features = build_live_features(current_snapshot, lookback_bars=50)
        feature_map = features.feature_map
        trend_strength = float(feature_map.get("1m_trend_strength", feature_map.get("trend_strength", 0.0)) or 0.0)
        breakout_pressure = float(feature_map.get("1m_breakout_pressure", feature_map.get("breakout_pressure", 0.0)) or 0.0)
        mean_reversion_probability = float(feature_map.get("1m_mean_reversion_probability", feature_map.get("mean_reversion_probability", 0.0)) or 0.0)
        trap_probability = self._derive_trap_probability(feature_map)
        option_bias = float(feature_map.get("option_chain_bias", 0.0) or 0.0)
        regime, regime_confidence, notes = self._classify_regime(feature_map, current_snapshot.spot_price)
        transition = self._classify_transition(regime, regime_confidence, feature_map, notes)
        trade_grade = self._grade_trade(feature_map, regime, trap_probability)

        state = MarketState(
            timestamp=current_snapshot.timestamp,
            trend_state=self._classify_trend_state(feature_map),
            volatility_state=self._classify_volatility_state(feature_map, current_snapshot.vix_value),
            session_state=self._classify_session_state(current_snapshot.timestamp),
            liquidity_state=self._classify_liquidity_state(feature_map),
            momentum_state=self._classify_momentum_state(feature_map),
            option_chain_bias=option_bias,
            breakout_pressure=breakout_pressure,
            trap_probability=trap_probability,
            mean_reversion_probability=mean_reversion_probability,
            trend_strength=trend_strength,
            directional_bias=self._classify_directional_bias(feature_map, regime),
            regime=regime,
            regime_confidence=regime_confidence,
            transition=transition,
            quality_score=self._quality_score(feature_map, regime, trap_probability, current_snapshot.vix_value),
            trade_grade=trade_grade,
            session_quality=float(feature_map.get("session_quality", 0.5) or 0.5),
            signal_ttl_candles=self._signal_ttl(feature_map, regime),
            is_tradeable=trade_grade in {TradeGrade.A_PLUS, TradeGrade.A, TradeGrade.B}
            and float(feature_map.get("session_quality", 0.5) or 0.5) >= self._config.tradeable_quality_threshold,
            feature_snapshot={key: float(value) for key, value in feature_map.items() if isinstance(value, (int, float, np.floating)) and pd.notna(value)},
            notes=notes,
        )
        self._last_state = state
        return state

    @property
    def last_snapshot(self) -> MarketSnapshotBundle | None:
        return self._last_snapshot

    @property
    def last_state(self) -> MarketState | None:
        return self._last_state

    @staticmethod
    def _append_and_trim(existing: pd.DataFrame, new_rows: pd.DataFrame, limit: int) -> pd.DataFrame:
        combined = pd.concat([existing, new_rows], ignore_index=True)
        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp")
        if limit > 0 and len(combined) > limit:
            combined = combined.tail(limit)
        return combined.reset_index(drop=True)

    @staticmethod
    def _build_candles(ticks: pd.DataFrame, frequency: str) -> pd.DataFrame:
        if ticks.empty or "timestamp" not in ticks.columns or "last_price" not in ticks.columns:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        working = ticks.copy().dropna(subset=["timestamp", "last_price"])
        if working.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        working = working.dropna(subset=["timestamp"])
        working = working.set_index("timestamp").sort_index()

        price_col = "last_price"
        volume_col = "volume_traded" if "volume_traded" in working.columns else None

        agg: dict[str, object] = {price_col: ["first", "max", "min", "last"]}
        if volume_col:
            agg[volume_col] = "last"

        resampled = working.resample(frequency).agg(agg)
        resampled.columns = ["open", "high", "low", "close"] + (["volume"] if volume_col else [])
        if "volume" not in resampled.columns:
            resampled["volume"] = 0.0

        return resampled.reset_index().rename(columns={"timestamp": "timestamp"})

    @staticmethod
    def _latest_numeric(frame: pd.DataFrame, column: str) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])

    def _classify_trend_state(self, feature_map: dict[str, float]) -> TrendState:
        trend = float(feature_map.get("15m_trend_strength", feature_map.get("5m_trend_strength", feature_map.get("1m_trend_strength", 0.0))) or 0.0)
        compression = float(feature_map.get("1m_narrow_range_compression", 0.0) or 0.0)
        if compression >= self._config.compression_threshold and abs(trend) < self._config.trend_strength_threshold:
            return TrendState.COMPRESSION
        if trend >= self._config.trend_strength_threshold:
            return TrendState.TREND_BULLISH
        if trend <= -self._config.trend_strength_threshold:
            return TrendState.TREND_BEARISH
        if float(feature_map.get("1m_reversal_probability", 0.0) or 0.0) >= 0.55:
            return TrendState.EXHAUSTION
        return TrendState.RANGE_BOUND

    def _classify_volatility_state(self, feature_map: dict[str, float], vix_value: float | None) -> VolatilityState:
        atr_compression = float(feature_map.get("1m_atr_compression", 0.0) or 0.0)
        if vix_value is not None and vix_value >= self._config.volatility_high_threshold:
            return VolatilityState.HIGH
        if atr_compression >= self._config.compression_threshold:
            return VolatilityState.CONTRACTING
        if float(feature_map.get("1m_breakout_pressure", 0.0) or 0.0) >= 0.003:
            return VolatilityState.EXPANDING
        if vix_value is not None and vix_value <= 11.5:
            return VolatilityState.LOW
        return VolatilityState.NORMAL

    def _classify_session_state(self, timestamp: datetime) -> SessionPhase:
        from trading_engine.intelligence.features import _session_phase

        return _session_phase(timestamp)

    def _classify_liquidity_state(self, feature_map: dict[str, float]) -> LiquidityState:
        volume_expansion = float(feature_map.get("1m_volume_expansion", 1.0) or 1.0)
        overlap = float(feature_map.get("1m_overlap_ratio", 0.0) or 0.0)
        if volume_expansion < 0.75 or overlap > 0.8:
            return LiquidityState.STRESSED
        if volume_expansion > 1.25 and overlap < 0.45:
            return LiquidityState.DEPTH
        if volume_expansion > 1.05:
            return LiquidityState.NORMAL
        return LiquidityState.THIN

    def _classify_momentum_state(self, feature_map: dict[str, float]) -> MomentumState:
        persistence = float(feature_map.get("1m_directional_persistence", 0.5) or 0.5)
        breakout = float(feature_map.get("1m_breakout_pressure", 0.0) or 0.0)
        if abs(breakout) < 0.0005 and persistence < 0.45:
            return MomentumState.FLAT
        if persistence >= 0.65 and abs(breakout) > 0.0015:
            return MomentumState.EXTENDING
        if persistence >= 0.55:
            return MomentumState.BUILDING
        if float(feature_map.get("1m_reversal_probability", 0.0) or 0.0) >= 0.55:
            return MomentumState.REVERSING
        return MomentumState.WEAKENING

    def _classify_directional_bias(self, feature_map: dict[str, float], regime: MarketRegime) -> DirectionalBias:
        bullish_alignment = sum(
            float(feature_map.get(key, 0.0) or 0.0)
            for key in ("15m_trend_strength", "5m_trend_strength", "1m_trend_strength")
        )
        bearish_alignment = -bullish_alignment
        if regime in {MarketRegime.TREND_BULLISH, MarketRegime.BULLISH_TRENDING} and bullish_alignment > 0:
            return DirectionalBias.BULLISH
        if regime in {MarketRegime.TREND_BEARISH, MarketRegime.BEARISH_TRENDING} and bearish_alignment > 0:
            return DirectionalBias.BEARISH
        if abs(bullish_alignment) < 0.0025:
            return DirectionalBias.NEUTRAL
        return DirectionalBias.TWO_WAY

    def _classify_regime(self, feature_map: dict[str, float], spot_price: float | None) -> tuple[MarketRegime, float, list[str]]:
        reasons: list[str] = []
        trend_strength = float(feature_map.get("15m_trend_strength", feature_map.get("1m_trend_strength", 0.0)) or 0.0)
        compression = float(feature_map.get("1m_narrow_range_compression", 0.0) or 0.0)
        trap = self._derive_trap_probability(feature_map)
        mean_reversion = float(feature_map.get("1m_mean_reversion_probability", 0.0) or 0.0)
        breakout = float(feature_map.get("1m_breakout_pressure", 0.0) or 0.0)
        iv_percentile = float(feature_map.get("iv_percentile", 0.5) or 0.5)
        vwap_alignment = float(feature_map.get("1m_vwap_alignment", 0.0) or 0.0)

        if trap >= 0.68:
            reasons.append("trap probability elevated")
            return MarketRegime.HIGH_NOISE_ENVIRONMENT, trap, reasons
        if compression >= self._config.compression_threshold and abs(trend_strength) < self._config.trend_strength_threshold:
            reasons.append("compression dominant")
            return MarketRegime.SIDEWAYS_COMPRESSION, max(compression, 0.55), reasons
        if trend_strength >= self._config.trend_strength_threshold and breakout >= 0.001 and vwap_alignment >= -0.001:
            reasons.append("bullish trend alignment")
            confidence = min(0.95, 0.55 + abs(trend_strength) * 120 + max(breakout, 0.0) * 40)
            return MarketRegime.TREND_BULLISH, confidence, reasons
        if trend_strength <= -self._config.trend_strength_threshold and breakout <= -0.001 and vwap_alignment <= 0.001:
            reasons.append("bearish trend alignment")
            confidence = min(0.95, 0.55 + abs(trend_strength) * 120 + abs(min(breakout, 0.0)) * 40)
            return MarketRegime.TREND_BEARISH, confidence, reasons
        if mean_reversion >= 0.62 and compression >= 0.45:
            reasons.append("exhaustion and mean reversion building")
            return MarketRegime.EXHAUSTION_REVERSAL, mean_reversion, reasons
        if iv_percentile >= 0.80:
            reasons.append("high volatility regime")
            return MarketRegime.VOLATILE_EXPANSION, iv_percentile, reasons
        reasons.append("defaulting to range-bound state")
        return MarketRegime.SIDEWAYS_COMPRESSION, max(0.45, 0.55 - abs(trend_strength) * 50), reasons

    def _classify_transition(
        self,
        current_regime: MarketRegime,
        confidence: float,
        feature_map: dict[str, float],
        reasons: list[str],
    ) -> MarketTransition:
        previous = self._last_state.regime if self._last_state is not None else current_regime
        path = [previous.value, current_regime.value]
        stage = RegimeTransitionStage.UNKNOWN
        breakout = float(feature_map.get("1m_breakout_pressure", 0.0) or 0.0)
        compression = float(feature_map.get("1m_narrow_range_compression", 0.0) or 0.0)
        trend_strength = float(feature_map.get("1m_trend_strength", 0.0) or 0.0)
        reversal = float(feature_map.get("1m_reversal_probability", 0.0) or 0.0)

        if compression >= 0.6 and current_regime == MarketRegime.SIDEWAYS_COMPRESSION:
            stage = RegimeTransitionStage.COMPRESSION
        elif previous == MarketRegime.SIDEWAYS_COMPRESSION and abs(breakout) >= 0.001:
            stage = RegimeTransitionStage.BREAKOUT_BUILDUP
        elif current_regime in {MarketRegime.TREND_BULLISH, MarketRegime.TREND_BEARISH} and abs(trend_strength) >= 0.0025:
            stage = RegimeTransitionStage.TREND_EXPANSION
        elif reversal >= 0.58:
            stage = RegimeTransitionStage.REVERSAL
        elif self._last_state is not None and self._last_state.regime in {MarketRegime.TREND_BULLISH, MarketRegime.TREND_BEARISH} and current_regime in {MarketRegime.EXHAUSTION_REVERSAL, MarketRegime.HIGH_NOISE_ENVIRONMENT}:
            stage = RegimeTransitionStage.EXHAUSTION
        else:
            stage = RegimeTransitionStage.SIDEWAYS if current_regime == MarketRegime.SIDEWAYS_COMPRESSION else RegimeTransitionStage.UNKNOWN

        path_notes = [f"transition:{previous.value}->{current_regime.value}", f"breakout:{breakout:.4f}", f"reversal:{reversal:.2f}"]
        return MarketTransition(stage=stage, from_regime=previous, to_regime=current_regime, confidence=float(np.clip(confidence, 0.0, 0.99)), path=path_notes, reasons=list(reasons))

    def _derive_trap_probability(self, feature_map: dict[str, float]) -> float:
        wick = float(feature_map.get("1m_wick_rejection", 0.0) or 0.0)
        failed = float(feature_map.get("1m_failed_breakout_frequency", 0.0) or 0.0)
        persistence = float(feature_map.get("1m_directional_persistence", 0.5) or 0.5)
        overlap = float(feature_map.get("1m_overlap_ratio", 0.0) or 0.0)
        spread = float(feature_map.get("iv_realized_spread", 0.0) or 0.0)
        return float(np.clip(0.35 * wick + 0.3 * failed + 0.2 * max(0.0, 1.0 - persistence) + 0.1 * overlap + 0.05 * max(0.0, spread), 0.0, 1.0))

    def _quality_score(self, feature_map: dict[str, float], regime: MarketRegime, trap_probability: float, vix_value: float | None) -> float:
        session_quality = float(feature_map.get("session_quality", 0.5) or 0.5)
        trend = abs(float(feature_map.get("15m_trend_strength", feature_map.get("1m_trend_strength", 0.0)) or 0.0))
        compression = float(feature_map.get("1m_narrow_range_compression", 0.0) or 0.0)
        liquidity = float(feature_map.get("1m_volume_expansion", 1.0) or 1.0)
        regime_bonus = 0.15 if regime in {MarketRegime.TREND_BULLISH, MarketRegime.TREND_BEARISH, MarketRegime.VOLATILE_EXPANSION} else -0.05
        vix_penalty = 0.08 if vix_value is not None and vix_value >= self._config.volatility_high_threshold else 0.0
        score = 0.35 * session_quality + 0.25 * min(trend * 120.0, 1.0) + 0.15 * compression + 0.15 * min(liquidity / 1.25, 1.0) + regime_bonus - 0.4 * trap_probability - vix_penalty
        return float(np.clip(score, 0.0, 1.0))

    def _grade_trade(self, feature_map: dict[str, float], regime: MarketRegime, trap_probability: float) -> TradeGrade:
        quality = self._quality_score(feature_map, regime, trap_probability, None)
        session_quality = float(feature_map.get("session_quality", 0.5) or 0.5)
        if quality >= 0.78 and session_quality >= 0.8 and trap_probability < 0.25:
            return TradeGrade.A_PLUS
        if quality >= 0.65 and trap_probability < 0.35:
            return TradeGrade.A
        if quality >= 0.50 and trap_probability < 0.55:
            return TradeGrade.B
        return TradeGrade.AVOID

    def _signal_ttl(self, feature_map: dict[str, float], regime: MarketRegime) -> int:
        base = 3 if regime in {MarketRegime.TREND_BULLISH, MarketRegime.TREND_BEARISH} else 2
        if float(feature_map.get("1m_failed_breakout_frequency", 0.0) or 0.0) >= 0.4:
            base = 1
        if float(feature_map.get("session_quality", 0.5) or 0.5) >= 0.85:
            base += 1
        return int(max(self._config.signal_ttl_min_candles, min(base, self._config.signal_ttl_max_candles)))
