"""Micro-momentum engine: tick-level velocity and acceleration for NIFTY options signals.

This module operates on the raw tick buffers (last_price, volume_traded,
total_buy_quantity, total_sell_quantity) to compute sub-minute features that
are invisible to candle-based analysis. All computation is incremental — only
the last N ticks are used, so per-tick overhead is bounded and predictable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MicroMomentumResult:
    """Tick-level signal features for a single evaluation cycle."""

    # Underlying (NIFTY index) features
    underlying_velocity: float = 0.0        # pct return over last N ticks
    underlying_acceleration: float = 0.0   # change in velocity vs prior window
    buy_sell_imbalance: float = 0.0         # (buy_qty - sell_qty) / total  [-1, +1]
    volume_acceleration: float = 0.0        # second derivative of traded volume

    # Option premium features (ATM CE)
    ce_premium_velocity: float = float("nan")      # pct change of CE LTP per tick
    ce_premium_acceleration: float = float("nan")  # rate of change of CE velocity

    # Option premium features (ATM PE)
    pe_premium_velocity: float = float("nan")      # pct change of PE LTP per tick
    pe_premium_acceleration: float = float("nan")  # rate of change of PE velocity

    # Cross-side relative features
    ce_pe_relative_strength: float = 0.0   # (ce_vel - pe_vel) / denom, [-1, +1]
    simultaneous_spike: bool = False        # both CE and PE rising rapidly -> trap
    trap_via_spike: float = 0.0             # probability contribution from spike

    # Summary scores (0-1 each)
    ce_momentum_score: float = 0.0  # how strong the CE directional signal is
    pe_momentum_score: float = 0.0  # how strong the PE directional signal is
    confidence: float = 0.0         # overall micro-momentum data quality

    evidence: list[str] = field(default_factory=list)


class MicroMomentumEngine:
    """Compute micro-momentum features from per-tick market data.

    Design decisions:
    - velocity_window: number of ticks used for one velocity estimate.
      At ~2 ticks/second for FULL mode, window=5 ~= 2.5 seconds.
    - acceleration_windows: (short, long) tick counts for computing the
      rate of velocity change. Acceleration = short_velocity - long_velocity.
    - spike_threshold: minimum velocity (fraction of last_price) for both
      CE and PE that triggers the simultaneous-spike trap flag.
    """

    def __init__(
        self,
        velocity_window: int = 5,
        acceleration_windows: tuple[int, int] = (5, 15),
        spike_threshold: float = 0.003,  # 0.3% per-window premium move
        min_ticks_required: int = 6,
    ) -> None:
        self._vel_window = velocity_window
        self._short_window, self._long_window = acceleration_windows
        self._spike_threshold = spike_threshold
        self._min_ticks = min_ticks_required

    def assess(
        self,
        underlying_ticks: pd.DataFrame,
        atm_ce_ticks: pd.DataFrame,
        atm_pe_ticks: pd.DataFrame,
    ) -> MicroMomentumResult:
        """Compute micro-momentum from raw tick DataFrames.

        Args:
            underlying_ticks: Normalised ticks for the NIFTY index instrument.
            atm_ce_ticks: Normalised ticks for the ATM CE option.
            atm_pe_ticks: Normalised ticks for the ATM PE option.

        Returns:
            MicroMomentumResult with all computed features.
        """
        result = MicroMomentumResult()

        # -- Underlying features -------------------------------------------
        if len(underlying_ticks) >= self._min_ticks:
            result.underlying_velocity = self._velocity(
                underlying_ticks, col="last_price", window=self._vel_window
            )
            result.underlying_acceleration = self._acceleration(
                underlying_ticks,
                col="last_price",
                short=self._short_window,
                long=self._long_window,
            )
            result.buy_sell_imbalance = self._imbalance(underlying_ticks)
            result.volume_acceleration = self._volume_acceleration(underlying_ticks)
            result.confidence = 0.3  # partial: only underlying available
            result.evidence.append(
                f"underlying_vel={result.underlying_velocity:+.4f} "
                f"accel={result.underlying_acceleration:+.4f} "
                f"imbalance={result.buy_sell_imbalance:+.3f}"
            )

        # -- CE premium features -------------------------------------------
        if len(atm_ce_ticks) >= self._min_ticks:
            result.ce_premium_velocity = self._velocity(
                atm_ce_ticks, col="last_price", window=self._vel_window
            )
            result.ce_premium_acceleration = self._acceleration(
                atm_ce_ticks,
                col="last_price",
                short=self._short_window,
                long=self._long_window,
            )
            result.evidence.append(
                f"ce_vel={result.ce_premium_velocity:+.4f} "
                f"ce_accel={result.ce_premium_acceleration:+.4f}"
            )

        # -- PE premium features -------------------------------------------
        if len(atm_pe_ticks) >= self._min_ticks:
            result.pe_premium_velocity = self._velocity(
                atm_pe_ticks, col="last_price", window=self._vel_window
            )
            result.pe_premium_acceleration = self._acceleration(
                atm_pe_ticks,
                col="last_price",
                short=self._short_window,
                long=self._long_window,
            )
            result.evidence.append(
                f"pe_vel={result.pe_premium_velocity:+.4f} "
                f"pe_accel={result.pe_premium_acceleration:+.4f}"
            )

        # -- Cross-side relative analysis ----------------------------------
        ce_valid = not (isinstance(result.ce_premium_velocity, float) and np.isnan(result.ce_premium_velocity))
        pe_valid = not (isinstance(result.pe_premium_velocity, float) and np.isnan(result.pe_premium_velocity))

        if ce_valid and pe_valid:
            result.confidence = 1.0
            ce_vel = result.ce_premium_velocity
            pe_vel = result.pe_premium_velocity
            denom = abs(ce_vel) + abs(pe_vel) + 1e-9
            result.ce_pe_relative_strength = float(np.clip((ce_vel - pe_vel) / denom, -1.0, 1.0))

            # Simultaneous spike: both rising fast -> not a clean directional -> trap
            both_rising = ce_vel > self._spike_threshold and pe_vel > self._spike_threshold
            result.simultaneous_spike = bool(both_rising)
            if both_rising:
                avg_spike = (ce_vel + pe_vel) / 2.0
                result.trap_via_spike = float(
                    np.clip(avg_spike / (self._spike_threshold * 3.0), 0.0, 1.0)
                )
                result.evidence.append(
                    f"SIMULTANEOUS_SPIKE ce={ce_vel:.4f} pe={pe_vel:.4f} "
                    f"trap_contrib={result.trap_via_spike:.2f}"
                )
            else:
                result.trap_via_spike = 0.0

            result.evidence.append(
                f"ce_pe_rel={result.ce_pe_relative_strength:+.3f}"
            )
        elif ce_valid or pe_valid:
            result.confidence = 0.6

        # -- Momentum scores (0-1) -----------------------------------------
        result.ce_momentum_score = self._momentum_score(
            velocity=result.ce_premium_velocity if ce_valid else 0.0,
            acceleration=result.ce_premium_acceleration if ce_valid else 0.0,
            underlying_velocity=result.underlying_velocity,
            direction=1,  # CE profits from upward move
        )
        result.pe_momentum_score = self._momentum_score(
            velocity=result.pe_premium_velocity if pe_valid else 0.0,
            acceleration=result.pe_premium_acceleration if pe_valid else 0.0,
            underlying_velocity=result.underlying_velocity,
            direction=-1,  # PE profits from downward move
        )

        return result

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _velocity(self, ticks: pd.DataFrame, col: str, window: int) -> float:
        """Fractional price change over the last window ticks."""
        prices = pd.to_numeric(ticks[col], errors="coerce").dropna()
        if len(prices) < 2:
            return 0.0
        tail = prices.iloc[-min(window, len(prices)):]
        first = float(tail.iloc[0])
        last = float(tail.iloc[-1])
        if first <= 0.0:
            return 0.0
        return float((last - first) / first)

    def _acceleration(
        self, ticks: pd.DataFrame, col: str, short: int, long: int
    ) -> float:
        """Rate of change of velocity: short_window_velocity - long_window_velocity."""
        prices = pd.to_numeric(ticks[col], errors="coerce").dropna()
        if len(prices) < long + 1:
            return 0.0
        short_vel = self._velocity_from_series(prices, short)
        long_vel = self._velocity_from_series(prices, long)
        return float(short_vel - long_vel)

    @staticmethod
    def _velocity_from_series(prices: pd.Series, window: int) -> float:
        tail = prices.iloc[-min(window, len(prices)):]
        first = float(tail.iloc[0])
        last = float(tail.iloc[-1])
        if first <= 0.0:
            return 0.0
        return float((last - first) / first)

    def _imbalance(self, ticks: pd.DataFrame) -> float:
        """Normalised buy/sell quantity imbalance from KiteTicker FULL mode fields."""
        required = {"total_buy_quantity", "total_sell_quantity"}
        if not required.issubset(ticks.columns):
            return 0.0
        tail = ticks.tail(self._vel_window)
        buy = pd.to_numeric(tail["total_buy_quantity"], errors="coerce").fillna(0.0).sum()
        sell = pd.to_numeric(tail["total_sell_quantity"], errors="coerce").fillna(0.0).sum()
        total = buy + sell
        if total <= 0.0:
            return 0.0
        return float(np.clip((buy - sell) / total, -1.0, 1.0))

    def _volume_acceleration(self, ticks: pd.DataFrame) -> float:
        """Second derivative of cumulative volume (acceleration of traded volume)."""
        if "volume_traded" not in ticks.columns:
            return 0.0
        vol = pd.to_numeric(ticks["volume_traded"], errors="coerce").dropna()
        if len(vol) < self._long_window + 2:
            return 0.0
        diff = vol.diff().dropna()
        if len(diff) < self._long_window:
            return 0.0
        short_rate = float(diff.iloc[-self._short_window:].mean())
        long_rate = float(diff.iloc[-self._long_window:].mean())
        if abs(long_rate) < 1e-9:
            return 0.0
        return float(np.clip((short_rate - long_rate) / abs(long_rate), -1.0, 1.0))

    def _momentum_score(
        self,
        velocity: float,
        acceleration: float,
        underlying_velocity: float,
        direction: int,  # +1 for CE (bullish), -1 for PE (bearish)
    ) -> float:
        """Composite 0-1 momentum score for one option side.

        Combines:
        - Premium velocity in the correct direction
        - Premium acceleration (building momentum)
        - Underlying confirmation (velocity in the correct direction)
        """
        # 1.5% premium move in window -> score 1.0
        vel_component = float(np.clip(velocity / 0.015, 0.0, 1.0)) if velocity * direction > 0 else 0.0
        # positive acceleration even with moderate velocity is a good sign
        accel_component = float(np.clip(acceleration / 0.010, 0.0, 1.0)) if acceleration > 0 else 0.0
        # underlying must confirm direction; 0.3% move -> score 1.0
        underlying_component = float(np.clip(underlying_velocity * direction / 0.003, 0.0, 1.0))

        score = 0.45 * vel_component + 0.30 * accel_component + 0.25 * underlying_component
        return float(np.clip(score, 0.0, 1.0))
