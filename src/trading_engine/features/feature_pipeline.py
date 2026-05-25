"""Feature engineering pipeline for directional-first Nifty modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_engine.features.options_analytics import (
    compute_iv_percentile,
    compute_oi_imbalance,
    compute_put_call_ratio,
    compute_realized_volatility,
    compute_rolling_correlation,
    compute_volatility_spread,
)
from trading_engine.features.regime import RegimeConfig, RegimeDetector


@dataclass(slots=True)
class FeaturePipelineConfig:
    """Configuration holder for feature windows and NaN policy."""

    feature_version: str = "v1.0.0"
    realized_vol_window: int = 20
    rolling_corr_window: int = 20
    iv_percentile_window: int = 252
    nan_fill_method: str = "ffill"
    drop_remaining_nan: bool = True
    prevent_lookahead: bool = True
    include_target: bool = True
    confidence_floor: float = 0.60
    target_horizon_bars: int = 1


class FeaturePipeline:
    """Transforms aligned raw data into ML-ready feature frames."""

    def __init__(self, config: FeaturePipelineConfig) -> None:
        self._config = config
        self._regime_detector = RegimeDetector(RegimeConfig())

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        option_chain: pd.DataFrame | None = None,
        timestamp_col: str = "timestamp",
        nifty_open_col: str = "nifty50_spot__open",
        nifty_high_col: str = "nifty50_spot__high",
        nifty_low_col: str = "nifty50_spot__low",
        nifty_close_col: str = "nifty50_spot__close",
        gift_close_col: str = "gift_nifty_fut__close",
        vix_close_col: str = "india_vix__close",
    ) -> pd.DataFrame:
        """Create ML-ready features and optional target without lookahead leakage."""
        required = {
            timestamp_col,
            nifty_open_col,
            nifty_high_col,
            nifty_low_col,
            nifty_close_col,
            gift_close_col,
            vix_close_col,
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns in aligned frame: {sorted(missing)}")

        working = frame.copy().sort_values(timestamp_col).reset_index(drop=True)
        close = working[nifty_close_col].astype(float)
        gift_close = working[gift_close_col].astype(float)
        vix_close = working[vix_close_col].astype(float)

        features = pd.DataFrame({timestamp_col: working[timestamp_col]})
        features["feature_version"] = self._config.feature_version

        features["ret_1"] = close.pct_change(1)
        features["ret_3"] = close.pct_change(3)
        features["ret_5"] = close.pct_change(5)

        features["overnight_gap"] = working[nifty_open_col].astype(float) - close.shift(1)
        features["gift_lead"] = gift_close - close.shift(1)
        features["intraday_range_frac"] = (
            (working[nifty_high_col].astype(float) - working[nifty_low_col].astype(float))
            / close.replace(0.0, np.nan)
        )

        realized_vol = compute_realized_volatility(
            close,
            window=self._config.realized_vol_window,
        )
        features["realized_vol"] = realized_vol

        nifty_returns = close.pct_change()
        vix_returns = vix_close.pct_change()
        features["nifty_vix_corr"] = compute_rolling_correlation(
            nifty_returns,
            vix_returns,
            window=self._config.rolling_corr_window,
        )

        features["vix_ret_1"] = vix_returns
        features["vix_level"] = vix_close

        if option_chain is not None and not option_chain.empty:
            option_features = self._aggregate_option_chain(option_chain, timestamp_col=timestamp_col)
            features = features.merge(option_features, on=timestamp_col, how="left")

            if "atm_iv" in features.columns:
                features["iv_percentile"] = compute_iv_percentile(
                    features["atm_iv"],
                    window=self._config.iv_percentile_window,
                )
                features["iv_realized_spread"] = compute_volatility_spread(
                    features["atm_iv"],
                    features["realized_vol"],
                )

        regime_input = features.copy()
        regime_input[nifty_close_col] = close
        regime_input["realized_vol"] = features["realized_vol"]
        features["regime"] = self._regime_detector.classify(
            regime_input,
            close_col=nifty_close_col,
            realized_vol_col="realized_vol",
            iv_realized_spread_col="iv_realized_spread" if "iv_realized_spread" in features.columns else None,
        )

        features = self._handle_missing_values(features)
        features = self._encode_regime(features)

        if self._config.prevent_lookahead:
            features = self._shift_features(features, exclude_columns={timestamp_col, "feature_version"})

        if self._config.include_target:
            future_close = close.shift(-self._config.target_horizon_bars)
            target = (future_close > close).astype(float)
            features["target_direction_up"] = target

        if self._config.drop_remaining_nan:
            features = features.dropna().reset_index(drop=True)

        return features

    def _aggregate_option_chain(
        self,
        option_chain: pd.DataFrame,
        *,
        timestamp_col: str,
    ) -> pd.DataFrame:
        output = pd.DataFrame({timestamp_col: pd.to_datetime(option_chain[timestamp_col]).sort_values().unique()})

        pcr = compute_put_call_ratio(option_chain, timestamp_col=timestamp_col)
        oi_imbalance = compute_oi_imbalance(option_chain, timestamp_col=timestamp_col)

        output = output.merge(pcr, on=timestamp_col, how="left")
        output = output.merge(oi_imbalance, on=timestamp_col, how="left")

        atm_iv = (
            option_chain.groupby(timestamp_col, observed=True)["iv"]
            .median()
            .rename("atm_iv")
            .reset_index()
        )
        output = output.merge(atm_iv, on=timestamp_col, how="left")

        return output

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._config.nan_fill_method == "ffill":
            return features.ffill()
        if self._config.nan_fill_method == "bfill":
            return features.bfill()
        if self._config.nan_fill_method == "zero":
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            output = features.copy()
            output[numeric_columns] = output[numeric_columns].fillna(0.0)
            return output
        return features

    @staticmethod
    def _shift_features(features: pd.DataFrame, *, exclude_columns: set[str]) -> pd.DataFrame:
        output = features.copy()
        for column in output.columns:
            if column in exclude_columns:
                continue
            output[column] = output[column].shift(1)
        return output

    @staticmethod
    def _encode_regime(features: pd.DataFrame) -> pd.DataFrame:
        output = features.copy()
        if "regime" not in output.columns:
            return output

        dummies = pd.get_dummies(output["regime"], prefix="regime", dtype=float)
        return pd.concat([output, dummies], axis=1)
