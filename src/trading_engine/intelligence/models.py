"""Shared data models for live market intelligence outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class MarketRegime(StrEnum):
    """Supported live market regime labels."""

    TREND_BULLISH = "trend_bullish"
    BULLISH_TRENDING = "trend_bullish"
    TREND_BEARISH = "trend_bearish"
    BEARISH_TRENDING = "trend_bearish"
    SIDEWAYS_COMPRESSION = "sideways_compression"
    SIDEWAYS_CHOPPY = "sideways_compression"
    VOLATILE_EXPANSION = "volatile_expansion"
    EXHAUSTION_REVERSAL = "exhaustion_reversal"
    FAKE_BREAKOUT_ENVIRONMENT = "high_noise_environment"
    HIGH_NOISE_ENVIRONMENT = "high_noise_environment"
    UNKNOWN = "unknown"


class TrendState(StrEnum):
    """Macro trend polarity and strength labels."""

    TREND_BULLISH = "trend_bullish"
    TREND_BEARISH = "trend_bearish"
    RANGE_BOUND = "range_bound"
    COMPRESSION = "compression"
    EXHAUSTION = "exhaustion"
    UNKNOWN = "unknown"


class VolatilityState(StrEnum):
    """Volatility regime labels."""

    LOW = "low"
    NORMAL = "normal"
    EXPANDING = "expanding"
    HIGH = "high"
    CONTRACTING = "contracting"


class SessionPhase(StrEnum):
    """Intraday session phases for NIFTY markets."""

    PREOPEN = "preopen"
    OPENING_EXPANSION = "opening_expansion"
    TREND_WINDOW = "trend_window"
    MIDDAY_COMPRESSION = "midday_compression"
    TRANSITIONAL_NOISE = "transitional_noise"
    AFTERNOON_EXPANSION = "afternoon_expansion"
    CLOSING_VOLATILITY = "closing_volatility"
    UNKNOWN = "unknown"


class LiquidityState(StrEnum):
    """Liquidity quality labels."""

    THIN = "thin"
    NORMAL = "normal"
    DEPTH = "depth"
    STRESSED = "stressed"


class MomentumState(StrEnum):
    """Momentum profile labels."""

    BUILDING = "building"
    EXTENDING = "extending"
    WEAKENING = "weakening"
    REVERSING = "reversing"
    FLAT = "flat"


class DirectionalBias(StrEnum):
    """Directional bias derived from multi-timeframe alignment."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    TWO_WAY = "two_way"


class RegimeTransitionStage(StrEnum):
    """Early transition stages used to anticipate breakouts or reversals."""

    SIDEWAYS = "sideways"
    COMPRESSION = "compression"
    BREAKOUT_BUILDUP = "breakout_buildup"
    TREND_EXPANSION = "trend_expansion"
    EXHAUSTION = "exhaustion"
    REVERSAL = "reversal"
    LIQUIDATION_PHASE = "liquidation_phase"
    UNKNOWN = "unknown"


class TradeGrade(StrEnum):
    """Trade quality grading for participation filters."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    AVOID = "Avoid"


class SignalAction(StrEnum):
    """Supported directional option signal outputs."""

    BUY_CE = "buy_ce"
    BUY_PE = "buy_pe"
    SELL_CE = "sell_ce"
    SELL_PE = "sell_pe"
    NO_TRADE = "no_trade"


class OptionSide(StrEnum):
    """Call/put leg selectors."""

    CE = "CE"
    PE = "PE"


@dataclass(slots=True)
class MarketSnapshotBundle:
    """Latest aligned live market state across inputs."""

    timestamp: datetime
    spot_price: float | None
    vix_value: float | None
    spot_frame: pd.DataFrame
    vix_frame: pd.DataFrame
    option_chain_frame: pd.DataFrame
    ticks_frame: pd.DataFrame
    candles_1m: pd.DataFrame
    candles_5m: pd.DataFrame


@dataclass(slots=True)
class MarketTransition:
    """Interpretable regime transition signal."""

    stage: RegimeTransitionStage
    from_regime: MarketRegime
    to_regime: MarketRegime
    confidence: float
    persistence_probability: float = 0.0
    instability_probability: float = 0.0
    exhaustion_probability: float = 0.0
    path: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MarketState:
    """Centralized market state that all signals should be derived from."""

    timestamp: datetime
    trend_state: TrendState
    volatility_state: VolatilityState
    session_state: SessionPhase
    liquidity_state: LiquidityState
    momentum_state: MomentumState
    option_chain_bias: float
    breakout_pressure: float
    trap_probability: float
    mean_reversion_probability: float
    trend_strength: float
    directional_bias: DirectionalBias
    regime: MarketRegime
    regime_confidence: float
    transition: MarketTransition
    chop_probability: float
    persistence_probability: float
    instability_probability: float
    exhaustion_probability: float
    quality_score: float
    trade_grade: TradeGrade
    session_quality: float
    signal_ttl_candles: int
    is_tradeable: bool
    feature_snapshot: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LiveFeatures:
    """Live feature frame and convenience payload for downstream engines."""

    frame: pd.DataFrame
    timestamp: datetime
    feature_map: dict[str, float] = field(default_factory=dict)
    multi_timeframe_map: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(slots=True)
class RegimeResult:
    """Market regime decision with confidence and rationale."""

    regime: MarketRegime
    confidence: float
    reasons: list[str] = field(default_factory=list)
    transition: MarketTransition | None = None


@dataclass(slots=True)
class ProbabilityResult:
    """Directional expansion probability estimates."""

    bullish_probability: float
    bearish_probability: float
    sideways_probability: float
    confidence: float
    horizon_minutes: int
    bullish_score: float = 0.0
    bearish_score: float = 0.0
    sideways_score: float = 0.0
    calibration_band: str = "uncalibrated"
    feature_contributions: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrapResult:
    """Trap and fake-breakout diagnostics."""

    fake_breakout_probability: float
    buyer_trap_probability: float
    seller_trap_probability: float
    stoploss_hunt_probability: float
    failed_breakout_probability: float = 0.0
    liquidity_grab_probability: float = 0.0
    trap_score: float = 0.0
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StrikeResult:
    """Automatic strike recommendation."""

    option_side: OptionSide
    strike: float | None
    expiry: str | None
    moneyness: str
    strike_distance: float | None = None
    expected_move: float | None = None
    delta: float | None = None
    selection_grade: TradeGrade = TradeGrade.B
    rationale: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RiskResult:
    """Sizing and risk-control output."""

    allowed: bool
    quantity: int
    stop_loss: float | None
    target: float | None
    max_daily_loss_hit: bool
    reason: str
    partial_exit_fraction: float = 0.0
    time_stop_minutes: int = 0
    position_value: float = 0.0


@dataclass(slots=True)
class LatencyTrace:
    """Processing timestamps for a single engine cycle."""

    data_received_at: datetime
    features_computed_at: datetime | None = None
    signal_generated_at: datetime | None = None
    report_published_at: datetime | None = None

    @property
    def feature_latency_ms(self) -> float:
        """Milliseconds between data receipt and feature computation."""
        if self.features_computed_at is None:
            return float("nan")
        return (self.features_computed_at - self.data_received_at).total_seconds() * 1000.0

    @property
    def signal_latency_ms(self) -> float:
        """Milliseconds between data receipt and signal generation."""
        if self.signal_generated_at is None:
            return float("nan")
        return (self.signal_generated_at - self.data_received_at).total_seconds() * 1000.0

    @property
    def total_latency_ms(self) -> float:
        """End-to-end milliseconds from data receipt to report publication."""
        end = self.report_published_at or self.signal_generated_at or self.features_computed_at
        if end is None:
            return float("nan")
        return (end - self.data_received_at).total_seconds() * 1000.0


@dataclass(slots=True)
class SignalScore:
    """Named component scores (0–1 each) and aggregate 0–10 display score."""

    underlying_direction: float = 0.0      # alignment of EMAs / trend strength
    underlying_acceleration: float = 0.0  # rate of change of underlying returns
    premium_velocity_ce: float = 0.0      # CE ATM premium pct change per tick
    premium_velocity_pe: float = 0.0      # PE ATM premium pct change per tick
    premium_acceleration: float = 0.0    # rate of change of premium velocity
    volume_acceleration: float = 0.0     # second derivative of traded volume
    relative_ce_pe: float = 0.0          # CE vs PE relative strength (directional)
    breakout_pressure: float = 0.0       # distance from rolling high/low
    trap_inverse: float = 0.0            # 1 - trap_probability (reward component)
    ce_momentum_score_display: float = 0.0  # CE score on 0-10 scale (filled by engine)
    pe_momentum_score_display: float = 0.0  # PE score on 0-10 scale (filled by engine)

    @property
    def total(self) -> float:
        """Weighted composite score on a 0–10 scale."""
        weights = {
            "underlying_direction": 1.5,
            "underlying_acceleration": 1.0,
            "premium_velocity_ce": 1.5,
            "premium_velocity_pe": 1.5,
            "premium_acceleration": 1.0,
            "volume_acceleration": 0.8,
            "relative_ce_pe": 1.2,
            "breakout_pressure": 0.8,
            "trap_inverse": 0.7,
        }
        raw = sum(
            float(np.clip(getattr(self, key), 0.0, 1.0)) * weight
            for key, weight in weights.items()
        )
        # max possible raw = sum of weights = 10.0
        return float(np.clip(raw, 0.0, 10.0))

    def label(self) -> str:
        """Human-readable strength label based on total score."""
        t = self.total
        if t >= 8.0:
            return "VERY STRONG"
        if t >= 6.5:
            return "STRONG"
        if t >= 5.0:
            return "MODERATE"
        if t >= 3.5:
            return "WEAK"
        return "NO SIGNAL"


@dataclass(slots=True)
class SignalResult:
    """Final filtered signal generated by the intelligence engine."""

    action: SignalAction
    confidence: float
    regime: MarketRegime
    probability: ProbabilityResult
    trap: TrapResult
    entry_reference: float | None
    stop_loss: float | None
    target: float | None
    strike: StrikeResult | None
    quantity: int
    expires_at: datetime | None = None
    signal_ttl_candles: int = 0
    signal_validity_seconds: int = 60  # real-time validity window in seconds
    trade_grade: TradeGrade = TradeGrade.AVOID
    quality_score: float = 0.0
    score: SignalScore | None = None    # named 0–10 component breakdown
    state: MarketState | None = None
    transition: MarketTransition | None = None
    reasoning: list[str] = field(default_factory=list)
    signal_id: str | None = None
    spread_percentage: float | None = None
    iv_percentile: float | None = None
    filter_passed: bool | None = None
    filter_rejection_reason: str | None = None
    days_to_expiry: int | None = None
    max_holding_minutes: int | None = None
    strike_oi_ratio: float | None = None
    entry_iv: float | None = None
    entry_pcr: float | None = None
    underlying_atr: float | None = None

    def signal_age_seconds(self) -> float:
        """Seconds elapsed since signal was generated (uses expires_at as anchor)."""
        if self.expires_at is None:
            return float("nan")
        remaining = (self.expires_at - datetime.now(UTC)).total_seconds()
        return float(self.signal_validity_seconds - remaining)


@dataclass(slots=True)
class OpenTrade:
    """Live trade state tracked by the monitor."""

    trade_id: str
    signal: SignalResult
    entry_price: float
    quantity: int
    opened_at: datetime
    expires_at: datetime | None = None
    status: str = "open"
    last_price: float | None = None
    peak_pnl: float = 0.0
    realized_pnl: float = 0.0
    grade: TradeGrade = TradeGrade.B
    exit_reason: str | None = None
    partial_exit_done: bool = False
    instrument_token: int = 0
    strike: float = 0.0
    expiry: str = ""
    option_type: str = ""
    entry_ask_price: float = 0.0
    current_bid_price: float = 0.0
    signal_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_frozen: bool = False
    option_atr: float | None = None
    initial_risk: float | None = None
    stop_loss_price: float | None = None
    trailing_active: bool = False
    delta: float | None = None
    entry_iv: float | None = None
    live_iv: float | None = None
    entry_pcr: float | None = None
    last_high_at: datetime | None = None
    high_premium: float = 0.0
    previous_spot: float | None = None


@dataclass(slots=True)
class TradeUpdate:
    """Status update emitted by the live trade monitor."""

    open_trades: list[OpenTrade]
    closed_trades: list[OpenTrade]
    expired_trades: list[OpenTrade]
    total_pnl: float
    alert_messages: list[str] = field(default_factory=list)
    lifecycle_alerts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class IntelligenceReport:
    """Complete market intelligence output for a single refresh cycle."""

    timestamp: datetime
    snapshot: MarketSnapshotBundle
    state: MarketState
    features: LiveFeatures
    regime: RegimeResult
    transition: MarketTransition
    probabilities: ProbabilityResult
    trap: TrapResult
    signal: SignalResult
    strike: StrikeResult
    risk: RiskResult
    trade_update: TradeUpdate
    latency: LatencyTrace | None = None
    ce_score: float = 0.0             # component CE signal score 0–10
    pe_score: float = 0.0             # component PE signal score 0–10
    ce_premium_velocity: float = float("nan")  # CE ATM pct velocity
    pe_premium_velocity: float = float("nan")  # PE ATM pct velocity

    def as_dict(self) -> dict[str, Any]:
        """Serialize a report to a plain dictionary for dashboard rendering."""
        latency_dict: dict[str, Any] = {}
        if self.latency is not None:
            latency_dict = {
                "data_received_at": self.latency.data_received_at.isoformat(),
                "feature_latency_ms": self.latency.feature_latency_ms,
                "signal_latency_ms": self.latency.signal_latency_ms,
                "total_latency_ms": self.latency.total_latency_ms,
            }
        score_dict: dict[str, Any] = {}
        if self.signal.score is not None:
            score_dict = {
                "total": self.signal.score.total,
                "label": self.signal.score.label(),
                "components": {
                    "underlying_direction": self.signal.score.underlying_direction,
                    "underlying_acceleration": self.signal.score.underlying_acceleration,
                    "premium_velocity_ce": self.signal.score.premium_velocity_ce,
                    "premium_velocity_pe": self.signal.score.premium_velocity_pe,
                    "premium_acceleration": self.signal.score.premium_acceleration,
                    "volume_acceleration": self.signal.score.volume_acceleration,
                    "relative_ce_pe": self.signal.score.relative_ce_pe,
                    "breakout_pressure": self.signal.score.breakout_pressure,
                    "trap_inverse": self.signal.score.trap_inverse,
                },
            }
        return {
            "timestamp": self.timestamp.isoformat(),
            "snapshot": {
                "timestamp": self.snapshot.timestamp.isoformat(),
                "spot_price": self.snapshot.spot_price,
                "vix_value": self.snapshot.vix_value,
            },
            "state": _serialize_dataclass(self.state),
            "regime": _serialize_dataclass(self.regime),
            "transition": _serialize_dataclass(self.transition),
            "probabilities": _serialize_dataclass(self.probabilities),
            "trap": _serialize_dataclass(self.trap),
            "signal": {
                **_serialize_dataclass(self.signal),
                "action": self.signal.action.value,
                "regime": self.signal.regime.value,
                "trade_grade": self.signal.trade_grade.value,
                "strike": None if self.signal.strike is None else _serialize_dataclass(self.signal.strike),
                "score": score_dict,
            },
            "strike": _serialize_dataclass(self.strike),
            "risk": _serialize_dataclass(self.risk),
            "trade_update": {
                "open_trades": [_serialize_dataclass(item) for item in self.trade_update.open_trades],
                "closed_trades": [_serialize_dataclass(item) for item in self.trade_update.closed_trades],
                "expired_trades": [_serialize_dataclass(item) for item in self.trade_update.expired_trades],
                "total_pnl": self.trade_update.total_pnl,
                "alert_messages": list(self.trade_update.alert_messages),
            },
            "latency": latency_dict,
            "ce_score": self.ce_score,
            "pe_score": self.pe_score,
            "ce_premium_velocity": self.ce_premium_velocity,
            "pe_premium_velocity": self.pe_premium_velocity,
        }


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def _serialize_dataclass(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize_dataclass(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_dataclass(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        data = asdict(value)
        return {key: _serialize_dataclass(item) for key, item in data.items()}
    return value
