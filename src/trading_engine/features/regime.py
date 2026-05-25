"""Regime detection for directional-first strategy activation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RegimeConfig:
    """Thresholds used for vectorized regime labeling."""

    trend_window: int = 20
    vol_window: int = 20
    trend_threshold: float = 0.012
    expansion_quantile: float = 0.80
    compression_quantile: float = 0.20
    mean_reversion_abs_return_threshold: float = 0.003


class RegimeDetector:
    """Classifies market state before strategy activation."""

    def __init__(self, config: RegimeConfig) -> None:
        self._config = config

    def classify(
        self,
        frame: pd.DataFrame,
        *,
        close_col: str,
        realized_vol_col: str,
        iv_realized_spread_col: str | None = None,
        expiry_day_col: str | None = None,
    ) -> pd.Series:
        """Return a timestamp-indexed regime label series."""
        close = frame[close_col].astype(float)
        returns = close.pct_change()

        trend = returns.rolling(self._config.trend_window, min_periods=self._config.trend_window).mean()
        trend_strength = trend.abs()

        realized_vol = frame[realized_vol_col].astype(float)
        vol_expansion_threshold = realized_vol.rolling(
            self._config.vol_window,
            min_periods=self._config.vol_window,
        ).quantile(self._config.expansion_quantile)
        vol_compression_threshold = realized_vol.rolling(
            self._config.vol_window,
            min_periods=self._config.vol_window,
        ).quantile(self._config.compression_quantile)

        regime = np.full(shape=len(frame), fill_value="neutral", dtype=object)

        trend_up = trend > self._config.trend_threshold
        trend_down = trend < -self._config.trend_threshold
        mean_reverting = (
            trend_strength < self._config.mean_reversion_abs_return_threshold
        ) & (realized_vol <= vol_expansion_threshold)
        panic_expansion = realized_vol >= vol_expansion_threshold
        low_vol_compression = realized_vol <= vol_compression_threshold

        regime = np.where(trend_up, "trending_up", regime)
        regime = np.where(trend_down, "trending_down", regime)
        regime = np.where(mean_reverting, "mean_reverting", regime)
        regime = np.where(low_vol_compression, "vol_compression", regime)
        regime = np.where(panic_expansion, "vol_expansion", regime)

        if iv_realized_spread_col and iv_realized_spread_col in frame.columns:
            iv_spread = frame[iv_realized_spread_col].astype(float)
            skewed_risk_off = iv_spread > iv_spread.rolling(self._config.vol_window, min_periods=5).median()
            regime = np.where(skewed_risk_off & panic_expansion, "risk_off_expansion", regime)

        if expiry_day_col and expiry_day_col in frame.columns:
            expiry_distortion = frame[expiry_day_col].astype(bool)
            regime = np.where(expiry_distortion, "expiry_distortion", regime)

        return pd.Series(regime, index=frame.index, name="regime")
