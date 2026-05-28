"""Live feature generation for short-horizon NIFTY options intelligence."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from trading_engine.features.options_analytics import (
    compute_oi_imbalance,
    compute_put_call_ratio,
    compute_realized_volatility,
    compute_rolling_correlation,
)
from trading_engine.intelligence.models import LiveFeatures, MarketSnapshotBundle


def build_live_features(snapshot: MarketSnapshotBundle, *, lookback_bars: int = 50) -> LiveFeatures:
    """Build a live feature frame from the current rolling market snapshot."""
    candles = snapshot.candles_1m.copy()
    if candles.empty:
        return LiveFeatures(frame=pd.DataFrame(), timestamp=snapshot.timestamp)

    candles = candles.sort_values("timestamp").reset_index(drop=True)
    if len(candles) > lookback_bars:
        candles = candles.tail(lookback_bars).reset_index(drop=True)

    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    volume = candles["volume"].fillna(0.0).astype(float)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema20_slope = ema20.diff()
    ema50_slope = ema50.diff()

    ret_1 = close.pct_change(1)
    ret_3 = close.pct_change(3)
    price_velocity = close.diff(3) / close.shift(3)
    candle_range = (high - low) / close.replace(0.0, np.nan)
    range_expansion = candle_range / candle_range.rolling(10, min_periods=3).mean()
    volume_expansion = volume / volume.rolling(10, min_periods=3).mean()
    breakout_pressure = (close - high.rolling(20, min_periods=5).max().shift(1)) / close
    support_lookback = low.rolling(20, min_periods=5).min()
    resistance_lookback = high.rolling(20, min_periods=5).max()
    compression_zone = 1.0 - (candle_range / candle_range.rolling(20, min_periods=5).max())

    features = pd.DataFrame(
        {
            "timestamp": candles["timestamp"],
            "close": close,
            "open": candles["open"].astype(float),
            "high": high,
            "low": low,
            "volume": volume,
            "ema20": ema20,
            "ema50": ema50,
            "ema20_slope": ema20_slope,
            "ema50_slope": ema50_slope,
            "trend_strength": (ema20 - ema50) / close.replace(0.0, np.nan),
            "candle_acceleration": ret_1.diff(),
            "volume_expansion": volume_expansion,
            "breakout_pressure": breakout_pressure,
            "price_velocity": price_velocity,
            "range_expansion": range_expansion,
            "compression_zone": compression_zone,
            "support_level": support_lookback,
            "resistance_level": resistance_lookback,
            "intraday_range_frac": candle_range,
            "hour": pd.to_datetime(candles["timestamp"], utc=True).dt.hour,
            "minute": pd.to_datetime(candles["timestamp"], utc=True).dt.minute,
            "is_open_window": pd.to_datetime(candles["timestamp"], utc=True).dt.hour.between(9, 10).astype(int),
            "is_midday": pd.to_datetime(candles["timestamp"], utc=True).dt.hour.between(12, 14).astype(int),
            "is_close_window": pd.to_datetime(candles["timestamp"], utc=True).dt.hour.between(15, 15).astype(int),
        }
    )

    if snapshot.vix_frame.empty:
        features["vix_level"] = np.nan
        features["vix_ret_1"] = np.nan
        features["vix_corr"] = np.nan
        features["iv_percentile"] = np.nan
        features["iv_realized_spread"] = np.nan
    else:
        vix_frame = snapshot.vix_frame.copy().sort_values("timestamp").reset_index(drop=True)
        vix_close = vix_frame["last_price"].astype(float)
        vix_ret_1 = vix_close.pct_change(1)
        aligned_vix = vix_ret_1.reindex(features.index, method="ffill")
        features["vix_level"] = vix_close.reindex(features.index, method="ffill")
        features["vix_ret_1"] = aligned_vix
        features["vix_corr"] = compute_rolling_correlation(ret_1.fillna(0.0), aligned_vix.fillna(0.0), window=10)
        realized_vol = compute_realized_volatility(close, window=10)
        features["iv_percentile"] = np.nan
        features["iv_realized_spread"] = np.nan

    option_chain = snapshot.option_chain_frame.copy()
    if not option_chain.empty:
        option_chain = _normalize_option_chain(option_chain)
        pcr = compute_put_call_ratio(option_chain, timestamp_col="timestamp", option_type_col="option_type", oi_col="oi")
        oi_imbalance = compute_oi_imbalance(option_chain, timestamp_col="timestamp", option_type_col="option_type", oi_col="oi")
        latest_option = option_chain.sort_values("timestamp").groupby("timestamp", as_index=False).tail(1)
        latest_timestamp = latest_option["timestamp"].iloc[-1]
        latest_pcr = pcr.loc[pcr["timestamp"] == latest_timestamp, "put_call_ratio"].tail(1)
        latest_imb = oi_imbalance.loc[oi_imbalance["timestamp"] == latest_timestamp, "oi_imbalance"].tail(1)

        features["put_call_ratio"] = float(latest_pcr.iloc[-1]) if not latest_pcr.empty else np.nan
        features["oi_imbalance"] = float(latest_imb.iloc[-1]) if not latest_imb.empty else np.nan
        features["call_oi"] = float(option_chain.loc[option_chain["option_type"] == "CE", "oi"].tail(1).mean())
        features["put_oi"] = float(option_chain.loc[option_chain["option_type"] == "PE", "oi"].tail(1).mean())
        features["atm_iv"] = float(option_chain.loc[_atm_mask(option_chain, snapshot.spot_price), "iv"].tail(1).mean())
        atm_iv_series = option_chain.groupby("timestamp")["iv"].median().dropna().sort_index()
        features["iv_percentile"] = _last_percentile(atm_iv_series)
        realized_vol = compute_realized_volatility(close, window=10)
        features["iv_realized_spread"] = float(features["atm_iv"].iloc[-1] - realized_vol.iloc[-1]) if not realized_vol.empty and pd.notna(features["atm_iv"].iloc[-1]) and pd.notna(realized_vol.iloc[-1]) else np.nan
        total_oi_by_ts = option_chain.groupby("timestamp")["oi"].sum().dropna().sort_index()
        if len(total_oi_by_ts) >= 2:
            oi_delta = float(total_oi_by_ts.iloc[-1] - total_oi_by_ts.iloc[-2])
            features["oi_buildup"] = max(oi_delta, 0.0)
            features["oi_unwinding"] = max(-oi_delta, 0.0)
        else:
            features["oi_buildup"] = np.nan
            features["oi_unwinding"] = np.nan
        features["strike_concentration"] = _strike_concentration(option_chain, snapshot.spot_price)
    else:
        for column in [
            "put_call_ratio",
            "oi_imbalance",
            "call_oi",
            "put_oi",
            "atm_iv",
            "oi_buildup",
            "oi_unwinding",
            "strike_concentration",
        ]:
            features[column] = np.nan

    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    features = features.dropna(subset=["close"]).reset_index(drop=True)

    feature_map = features.iloc[-1].drop(labels=["timestamp"]).to_dict() if not features.empty else {}
    feature_map = {key: float(value) for key, value in feature_map.items() if pd.notna(value) and isinstance(value, (int, float, np.floating))}
    return LiveFeatures(frame=features, timestamp=snapshot.timestamp, feature_map=feature_map)


def _normalize_option_chain(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    rename_map = {
        "open_interest": "oi",
        "implied_volatility": "iv",
        "last_price": "ltp",
        "bid_qty": "bid",
        "ask_qty": "ask",
    }
    working = working.rename(columns=rename_map)
    required = ["timestamp", "expiry", "strike", "option_type", "oi", "iv", "ltp"]
    for column in required:
        if column not in working.columns:
            working[column] = np.nan
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working["expiry"] = working["expiry"].astype(str)
    working["strike"] = pd.to_numeric(working["strike"], errors="coerce")
    working["oi"] = pd.to_numeric(working["oi"], errors="coerce")
    working["iv"] = pd.to_numeric(working["iv"], errors="coerce")
    return working.dropna(subset=["timestamp", "strike"]).reset_index(drop=True)


def _atm_mask(option_chain: pd.DataFrame, spot_price: float | None) -> pd.Series:
    if spot_price is None or option_chain.empty:
        return pd.Series(False, index=option_chain.index)
    return (option_chain["strike"] - float(spot_price)).abs() <= option_chain["strike"].diff().abs().median()


def _strike_concentration(option_chain: pd.DataFrame, spot_price: float | None) -> float:
    if spot_price is None or option_chain.empty:
        return float("nan")
    window = option_chain.loc[(option_chain["strike"] - spot_price).abs() <= max(option_chain["strike"].diff().abs().median(), 50.0)]
    if window.empty:
        return float("nan")
    return float(window.groupby("strike")["oi"].sum().max() / max(window["oi"].sum(), 1.0))


def _last_percentile(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    last_value = series.iloc[-1]
    return float((series <= last_value).mean())
