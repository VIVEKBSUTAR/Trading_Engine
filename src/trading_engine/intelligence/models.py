"""Shared data models for live market intelligence outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

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
    trade_grade: TradeGrade = TradeGrade.AVOID
    quality_score: float = 0.0
    state: MarketState | None = None
    transition: MarketTransition | None = None
    reasoning: list[str] = field(default_factory=list)


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


@dataclass(slots=True)
class TradeUpdate:
    """Status update emitted by the live trade monitor."""

    open_trades: list[OpenTrade]
    closed_trades: list[OpenTrade]
    expired_trades: list[OpenTrade]
    total_pnl: float
    alert_messages: list[str] = field(default_factory=list)


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

    def as_dict(self) -> dict[str, Any]:
        """Serialize a report to a plain dictionary for dashboard rendering."""
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
        }


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def _serialize_dataclass(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize_dataclass(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_dataclass(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        data = asdict(value)
        return {key: _serialize_dataclass(item) for key, item in data.items()}
    return value
