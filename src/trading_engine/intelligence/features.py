"""Live feature generation for short-horizon NIFTY options intelligence."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from trading_engine.features.options_analytics import (
    compute_dealer_positioning_proxy,
    compute_iv_percentile,
    compute_iv_skew,
    compute_max_pain,
    compute_oi_imbalance,
    compute_put_call_ratio,
    compute_realized_volatility,
    compute_rolling_correlation,
    compute_strike_concentration_zones,
)
from trading_engine.intelligence.models import LiveFeatures, MarketSnapshotBundle, SessionPhase


def build_live_features(snapshot: MarketSnapshotBundle, *, lookback_bars: int = 50) -> LiveFeatures:
    """Build a live feature frame from the current rolling market snapshot."""
    candles_1m = _prepare_candles(snapshot.candles_1m, lookback_bars)
    if candles_1m.empty:
        return LiveFeatures(frame=pd.DataFrame(), timestamp=snapshot.timestamp)

    candles_5m = _prepare_candles(snapshot.candles_5m, max(lookback_bars // 2, 10))
    candles_15m = _resample_candles(candles_1m, "15min")

    frame_1m, summary_1m = _build_timeframe_summary(candles_1m, "1m", snapshot.timestamp)
    frame_5m, summary_5m = _build_timeframe_summary(candles_5m if not candles_5m.empty else candles_1m, "5m", snapshot.timestamp)
    frame_15m, summary_15m = _build_timeframe_summary(candles_15m if not candles_15m.empty else candles_1m, "15m", snapshot.timestamp)

    option_chain = _normalize_option_chain(snapshot.option_chain_frame)
    option_summary = _build_option_chain_summary(option_chain, snapshot.spot_price, snapshot.timestamp)
    session_summary = _build_session_summary(snapshot.timestamp)

    combined_frame = frame_1m.copy()
    for key, value in {**summary_5m, **summary_15m, **option_summary, **session_summary}.items():
        combined_frame[key] = value

    combined_frame["timestamp"] = pd.to_datetime(combined_frame["timestamp"], utc=True)
    combined_frame = combined_frame.dropna(subset=["close"]).reset_index(drop=True)

    feature_map = _latest_numeric_map(combined_frame)
    feature_map.update({key: value for key, value in session_summary.items() if isinstance(value, (int, float, np.floating))})

    multi_timeframe_map = {
        "1m": summary_1m,
        "5m": summary_5m,
        "15m": summary_15m,
    }
    feature_map.update(option_summary)
    feature_map.update(session_summary)

    return LiveFeatures(
        frame=combined_frame,
        timestamp=snapshot.timestamp,
        feature_map=feature_map,
        multi_timeframe_map=multi_timeframe_map,
    )


def _prepare_candles(frame: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    if lookback_bars > 0 and len(working) > lookback_bars:
        working = working.tail(lookback_bars)
    required = ["open", "high", "low", "close"]
    for column in required:
        if column not in working.columns:
            working[column] = np.nan
    if "volume" not in working.columns:
        working["volume"] = 0.0
    return working.reset_index(drop=True)


def _resample_candles(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy().set_index("timestamp").sort_index()
    resampled = working.resample(frequency).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["close"]).reset_index()
    return resampled


def _build_timeframe_summary(frame: pd.DataFrame, label: str, timestamp: datetime) -> tuple[pd.DataFrame, dict[str, float]]:
    working = frame.copy().reset_index(drop=True)
    if working.empty:
        return working, {}

    close = working["close"].astype(float)
    open_price = working["open"].astype(float)
    high = working["high"].astype(float)
    low = working["low"].astype(float)
    volume = working["volume"].fillna(0.0).astype(float)

    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    ema_flat = close.ewm(span=34, adjust=False).mean()
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=min(14, len(working)), min_periods=max(3, min(14, len(working)) // 2)).mean()
    atr_baseline = atr.rolling(window=min(20, len(working)), min_periods=max(3, min(20, len(working)) // 2)).mean()
    vwap = (close * volume).cumsum() / volume.replace(0.0, np.nan).cumsum()
    returns = close.pct_change().fillna(0.0)

    overlap_ratio = _candle_overlap_ratio(working.tail(min(len(working), 12)))
    failed_breakout_frequency = _failed_breakout_frequency(working)
    directional_persistence = _directional_persistence(returns)
    wick_rejection = _wick_rejection_score(working)
    narrow_range_compression = _narrow_range_compression(working)
    volume_expansion = _volume_expansion(volume)
    atr_compression = _atr_compression(atr, atr_baseline)
    breakout_pressure = _breakout_pressure(close, high, low)
    vwap_alignment = _latest_value(close - vwap) / max(_latest_value(close), 1e-9)
    trend_strength = _latest_value((ema_fast - ema_slow) / close.replace(0.0, np.nan))
    trend_slope = _latest_value(ema_fast.diff()) / max(_latest_value(close), 1e-9)
    reversal_probability = _reversal_probability(wick_rejection, failed_breakout_frequency, directional_persistence)
    mean_reversion_probability = _mean_reversion_probability(overlap_ratio, atr_compression, narrow_range_compression)
    session_quality = _session_quality(timestamp)
    price_range = _latest_value((high - low) / close.replace(0.0, np.nan))

    summary = {
        f"{label}_trend_strength": trend_strength,
        f"{label}_trend_slope": trend_slope,
        f"{label}_vwap_alignment": vwap_alignment,
        f"{label}_atr_compression": atr_compression,
        f"{label}_overlap_ratio": overlap_ratio,
        f"{label}_failed_breakout_frequency": failed_breakout_frequency,
        f"{label}_directional_persistence": directional_persistence,
        f"{label}_wick_rejection": wick_rejection,
        f"{label}_narrow_range_compression": narrow_range_compression,
        f"{label}_volume_expansion": volume_expansion,
        f"{label}_breakout_pressure": breakout_pressure,
        f"{label}_reversal_probability": reversal_probability,
        f"{label}_mean_reversion_probability": mean_reversion_probability,
        f"{label}_session_quality": session_quality,
        f"{label}_price_range": price_range,
    }

    for key, value in summary.items():
        working[key] = value

    return working, summary


def _build_option_chain_summary(option_chain: pd.DataFrame, spot_price: float | None, timestamp: datetime) -> dict[str, float]:
    if option_chain.empty:
        return {
            "put_call_ratio": np.nan,
            "oi_imbalance": np.nan,
            "atm_iv": np.nan,
            "iv_percentile": np.nan,
            "iv_realized_spread": np.nan,
            "oi_buildup": np.nan,
            "oi_unwinding": np.nan,
            "strike_concentration": np.nan,
            "dealer_positioning_proxy": np.nan,
            "max_pain_distance": np.nan,
            "iv_skew": np.nan,
            "option_chain_bias": 0.0,
            "call_wall": np.nan,
            "put_wall": np.nan,
        }

    pcr = compute_put_call_ratio(option_chain, timestamp_col="timestamp", option_type_col="option_type", oi_col="oi")
    oi_imbalance = compute_oi_imbalance(option_chain, timestamp_col="timestamp", option_type_col="option_type", oi_col="oi")
    latest_timestamp = option_chain["timestamp"].max()

    latest_pcr = _latest_row_value(pcr, latest_timestamp, "put_call_ratio")
    latest_imbalance = _latest_row_value(oi_imbalance, latest_timestamp, "oi_imbalance")
    atm_iv = _atm_iv(option_chain, spot_price, latest_timestamp)
    iv_percentile = _rolling_latest_percentile(option_chain, latest_timestamp)
    iv_skew = _latest_iv_skew(option_chain, spot_price, latest_timestamp)
    realized_proxy = _realized_vol_proxy(option_chain, spot_price)
    iv_realized_spread = float(atm_iv - realized_proxy) if pd.notna(atm_iv) and pd.notna(realized_proxy) else np.nan
    total_oi_by_ts = option_chain.groupby("timestamp")["oi"].sum().dropna().sort_index()
    if len(total_oi_by_ts) >= 2:
        oi_delta = float(total_oi_by_ts.iloc[-1] - total_oi_by_ts.iloc[-2])
        oi_buildup = max(oi_delta, 0.0)
        oi_unwinding = max(-oi_delta, 0.0)
    else:
        oi_buildup = np.nan
        oi_unwinding = np.nan

    dealer_proxy = _dealer_positioning(option_chain, latest_timestamp)
    strike_concentration = _strike_concentration(option_chain, spot_price)
    max_pain_distance = _max_pain_distance(option_chain, spot_price)
    concentration_zones = compute_strike_concentration_zones(option_chain)
    call_wall = _latest_zone_value(concentration_zones.frame, latest_timestamp, "call_wall")
    put_wall = _latest_zone_value(concentration_zones.frame, latest_timestamp, "put_wall")

    option_bias = float(np.nan_to_num(latest_imbalance, nan=0.0) + np.nan_to_num(latest_pcr, nan=1.0) - 1.0)

    return {
        "put_call_ratio": latest_pcr,
        "oi_imbalance": latest_imbalance,
        "atm_iv": atm_iv,
        "iv_percentile": iv_percentile,
        "iv_realized_spread": iv_realized_spread,
        "oi_buildup": oi_buildup,
        "oi_unwinding": oi_unwinding,
        "strike_concentration": strike_concentration,
        "dealer_positioning_proxy": dealer_proxy,
        "max_pain_distance": max_pain_distance,
        "iv_skew": iv_skew,
        "option_chain_bias": option_bias,
        "call_wall": call_wall,
        "put_wall": put_wall,
    }


def _build_session_summary(timestamp: datetime) -> dict[str, float]:
    phase = _session_phase(timestamp)
    phase_quality = _session_quality(timestamp)
    return {
        "session_quality": phase_quality,
        "session_phase_score": float(_session_phase_score(phase)),
        "time_to_close_minutes": float(_minutes_to_close(timestamp)),
        "time_from_open_minutes": float(_minutes_from_open(timestamp)),
    }


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


def _latest_numeric_map(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    latest = frame.iloc[-1].to_dict()
    output: dict[str, float] = {}
    for key, value in latest.items():
        if key == "timestamp":
            continue
        if isinstance(value, (int, float, np.floating)) and pd.notna(value):
            output[key] = float(value)
    return output


def _latest_row_value(frame: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    filtered = frame.loc[frame["timestamp"] == timestamp, column]
    if filtered.empty:
        return float(frame[column].dropna().iloc[-1]) if frame[column].dropna().any() else float("nan")
    value = pd.to_numeric(filtered, errors="coerce").dropna()
    return float(value.iloc[-1]) if not value.empty else float("nan")


def _latest_zone_value(frame: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    filtered = frame.loc[frame["timestamp"] == timestamp, column]
    numeric = pd.to_numeric(filtered, errors="coerce").dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else float("nan")


def _latest_value(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else float("nan")


def _session_phase(timestamp: datetime) -> SessionPhase:
    hour_minute = pd.Timestamp(timestamp)
    if hour_minute.tzinfo is None:
        hour_minute = hour_minute.tz_localize("UTC")
    hour_minute = hour_minute.tz_convert("Asia/Kolkata")
    minutes = hour_minute.hour * 60 + hour_minute.minute
    if minutes < 9 * 60 + 15:
        return SessionPhase.PREOPEN
    if minutes < 10 * 60:
        return SessionPhase.OPENING_EXPANSION
    if minutes < 11 * 60 + 15:
        return SessionPhase.TREND_WINDOW
    if minutes < 13 * 60:
        return SessionPhase.MIDDAY_COMPRESSION
    if minutes < 14 * 60 + 15:
        return SessionPhase.TRANSITIONAL_NOISE
    if minutes < 15 * 60:
        return SessionPhase.AFTERNOON_EXPANSION
    return SessionPhase.CLOSING_VOLATILITY


def _session_quality(timestamp: datetime) -> float:
    phase = _session_phase(timestamp)
    return float(_session_phase_score(phase))


def _session_phase_score(phase: SessionPhase) -> float:
    scores = {
        SessionPhase.OPENING_EXPANSION: 0.92,
        SessionPhase.TREND_WINDOW: 0.88,
        SessionPhase.AFTERNOON_EXPANSION: 0.84,
        SessionPhase.CLOSING_VOLATILITY: 0.70,
        SessionPhase.MIDDAY_COMPRESSION: 0.42,
        SessionPhase.TRANSITIONAL_NOISE: 0.48,
        SessionPhase.PREOPEN: 0.30,
        SessionPhase.UNKNOWN: 0.50,
    }
    return scores.get(phase, 0.50)


def _minutes_from_open(timestamp: datetime) -> int:
    converted = pd.Timestamp(timestamp)
    if converted.tzinfo is None:
        converted = converted.tz_localize("UTC")
    converted = converted.tz_convert("Asia/Kolkata")
    return int(converted.hour * 60 + converted.minute - (9 * 60 + 15))


def _minutes_to_close(timestamp: datetime) -> int:
    converted = pd.Timestamp(timestamp)
    if converted.tzinfo is None:
        converted = converted.tz_localize("UTC")
    converted = converted.tz_convert("Asia/Kolkata")
    return int((15 * 60 + 30) - (converted.hour * 60 + converted.minute))


def _candle_overlap_ratio(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    body_high = frame[["open", "close"]].max(axis=1)
    body_low = frame[["open", "close"]].min(axis=1)
    overlap = 0
    total = 0
    prev_high = None
    prev_low = None
    for high, low in zip(body_high, body_low):
        if prev_high is not None and prev_low is not None:
            total += 1
            if high >= prev_low and low <= prev_high:
                overlap += 1
        prev_high = high
        prev_low = low
    return float(overlap / total) if total else 0.0


def _failed_breakout_frequency(frame: pd.DataFrame) -> float:
    if len(frame) < 5:
        return float("nan")
    closes = frame["close"].astype(float)
    highs = frame["high"].astype(float)
    lows = frame["low"].astype(float)
    rolling_high = highs.rolling(window=min(10, len(frame)), min_periods=3).max().shift(1)
    rolling_low = lows.rolling(window=min(10, len(frame)), min_periods=3).min().shift(1)
    breakout_up = closes > rolling_high
    breakout_down = closes < rolling_low
    failures = 0
    attempts = 0
    for idx in range(1, len(frame)):
        if bool(breakout_up.iloc[idx]):
            attempts += 1
            if idx + 1 < len(frame) and closes.iloc[idx + 1] < closes.iloc[idx]:
                failures += 1
        if bool(breakout_down.iloc[idx]):
            attempts += 1
            if idx + 1 < len(frame) and closes.iloc[idx + 1] > closes.iloc[idx]:
                failures += 1
    return float(failures / attempts) if attempts else 0.0


def _directional_persistence(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    signs = np.sign(returns.to_numpy(dtype=float))
    if len(signs) < 2:
        return float(0.5)
    persistence = np.mean(signs[1:] == signs[:-1])
    return float(persistence)


def _wick_rejection_score(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    body_high = frame[["open", "close"]].max(axis=1)
    body_low = frame[["open", "close"]].min(axis=1)
    candle_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    upper_wick = (frame["high"] - body_high) / candle_range
    lower_wick = (body_low - frame["low"]) / candle_range
    score = np.nanmean(np.maximum(upper_wick, lower_wick))
    return float(np.nan_to_num(score, nan=0.0))


def _narrow_range_compression(frame: pd.DataFrame) -> float:
    if len(frame) < 5:
        return 0.0
    candle_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    latest = float(candle_range.iloc[-1])
    median = float(candle_range.tail(min(10, len(frame))).median())
    if not np.isfinite(median) or median == 0.0:
        return 0.0
    return float(max(0.0, 1.0 - latest / median))


def _volume_expansion(volume: pd.Series) -> float:
    if volume.empty:
        return 0.0
    rolling_mean = volume.rolling(window=min(10, len(volume)), min_periods=max(3, min(10, len(volume)) // 2)).mean()
    baseline = float(rolling_mean.iloc[-1]) if not rolling_mean.empty else float("nan")
    latest = float(volume.iloc[-1])
    if not np.isfinite(baseline) or baseline <= 0:
        return 0.0
    return float(latest / baseline)


def _atr_compression(atr: pd.Series, baseline: pd.Series) -> float:
    if atr.empty or baseline.empty:
        return 0.0
    latest_atr = pd.to_numeric(atr, errors="coerce").dropna()
    latest_base = pd.to_numeric(baseline, errors="coerce").dropna()
    if latest_atr.empty or latest_base.empty:
        return 0.0
    base = float(latest_base.iloc[-1])
    value = float(latest_atr.iloc[-1])
    if base <= 0 or not np.isfinite(base):
        return 0.0
    return float(max(0.0, 1.0 - value / base))


def _breakout_pressure(close: pd.Series, high: pd.Series, low: pd.Series) -> float:
    if close.empty:
        return 0.0
    rolling_high = high.rolling(window=min(20, len(close)), min_periods=max(3, min(20, len(close)) // 2)).max().shift(1)
    rolling_low = low.rolling(window=min(20, len(close)), min_periods=max(3, min(20, len(close)) // 2)).min().shift(1)
    latest = float(close.iloc[-1])
    top = float(rolling_high.iloc[-1]) if not pd.isna(rolling_high.iloc[-1]) else latest
    bottom = float(rolling_low.iloc[-1]) if not pd.isna(rolling_low.iloc[-1]) else latest
    if latest >= top:
        return float((latest - top) / max(latest, 1e-9))
    if latest <= bottom:
        return float(-((bottom - latest) / max(latest, 1e-9)))
    midpoint = (top + bottom) / 2.0 if np.isfinite(top) and np.isfinite(bottom) else latest
    return float((latest - midpoint) / max(latest, 1e-9))


def _reversal_probability(wick_rejection: float, failed_breakouts: float, persistence: float) -> float:
    score = 0.45 * wick_rejection + 0.35 * failed_breakouts + 0.2 * max(0.0, 1.0 - persistence)
    return float(np.clip(score, 0.0, 1.0))


def _mean_reversion_probability(overlap_ratio: float, atr_compression: float, narrow_range: float) -> float:
    score = 0.4 * overlap_ratio + 0.35 * atr_compression + 0.25 * narrow_range
    return float(np.clip(score, 0.0, 1.0))


def _atm_iv(option_chain: pd.DataFrame, spot_price: float | None, timestamp: pd.Timestamp) -> float:
    if spot_price is None or option_chain.empty:
        return float("nan")
    latest = option_chain.loc[option_chain["timestamp"] == timestamp]
    if latest.empty:
        latest = option_chain
    latest = latest.copy()
    latest["distance"] = (latest["strike"] - float(spot_price)).abs()
    atm = latest.sort_values("distance").head(4)
    return float(pd.to_numeric(atm["iv"], errors="coerce").dropna().mean()) if not atm.empty else float("nan")


def _rolling_latest_percentile(option_chain: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    iv_series = option_chain.groupby("timestamp")["iv"].median().dropna().sort_index()
    if iv_series.empty:
        return float("nan")
    last_value = iv_series.iloc[-1]
    return float((iv_series <= last_value).mean())


def _latest_iv_skew(option_chain: pd.DataFrame, spot_price: float | None, timestamp: pd.Timestamp) -> float:
    if spot_price is None or option_chain.empty:
        return float("nan")
    skew = compute_iv_skew(option_chain, timestamp_col="timestamp", expiry_col="expiry", strike_col="strike", option_type_col="option_type", iv_col="iv")
    latest = skew.loc[skew["timestamp"] == timestamp]
    if latest.empty:
        latest = skew.tail(1)
    if latest.empty:
        return float("nan")
    return float(latest["iv_skew"].dropna().iloc[-1]) if not latest["iv_skew"].dropna().empty else float("nan")


def _realized_vol_proxy(option_chain: pd.DataFrame, spot_price: float | None) -> float:
    if spot_price is None or option_chain.empty:
        return float("nan")
    strikes = pd.to_numeric(option_chain["strike"], errors="coerce").dropna()
    if strikes.empty:
        return float("nan")
    return float(strikes.pct_change().abs().rolling(window=min(10, len(strikes)), min_periods=3).mean().iloc[-1])


def _dealer_positioning(option_chain: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    if option_chain.empty:
        return float("nan")
    underlying = pd.Series(index=option_chain["timestamp"], data=option_chain.groupby("timestamp")["strike"].median())
    proxy = compute_dealer_positioning_proxy(option_chain, underlying_price=underlying)
    latest = proxy.loc[proxy["timestamp"] == timestamp]
    if latest.empty:
        latest = proxy.tail(1)
    if latest.empty:
        return float("nan")
    return float(latest["dealer_positioning_proxy"].dropna().iloc[-1]) if not latest["dealer_positioning_proxy"].dropna().empty else float("nan")


def _strike_concentration(option_chain: pd.DataFrame, spot_price: float | None) -> float:
    if spot_price is None or option_chain.empty:
        return float("nan")
    window = option_chain.loc[(option_chain["strike"] - spot_price).abs() <= max(option_chain["strike"].diff().abs().median(), 50.0)]
    if window.empty:
        return float("nan")
    return float(window.groupby("strike")["oi"].sum().max() / max(window["oi"].sum(), 1.0))


def _max_pain_distance(option_chain: pd.DataFrame, spot_price: float | None) -> float:
    if spot_price is None or option_chain.empty:
        return float("nan")
    max_pain = compute_max_pain(option_chain)
    if max_pain.empty:
        return float("nan")
    latest = max_pain.iloc[-1]
    return float((float(latest["max_pain"]) - float(spot_price)) / max(float(spot_price), 1e-9))


def _build_option_chain_summary(option_chain: pd.DataFrame, spot_price: float | None, timestamp: datetime) -> dict[str, float]:
    if option_chain.empty:
        return {
            "put_call_ratio": np.nan,
            "oi_imbalance": np.nan,
            "atm_iv": np.nan,
            "iv_percentile": np.nan,
            "iv_realized_spread": np.nan,
            "oi_buildup": np.nan,
            "oi_unwinding": np.nan,
            "strike_concentration": np.nan,
            "dealer_positioning_proxy": np.nan,
            "max_pain_distance": np.nan,
            "iv_skew": np.nan,
            "option_chain_bias": 0.0,
            "call_wall": np.nan,
            "put_wall": np.nan,
        }

    pcr = compute_put_call_ratio(option_chain, timestamp_col="timestamp", option_type_col="option_type", oi_col="oi")
    oi_imbalance = compute_oi_imbalance(option_chain, timestamp_col="timestamp", option_type_col="option_type", oi_col="oi")
    latest_timestamp = option_chain["timestamp"].max()

    latest_pcr = _latest_row_value(pcr, latest_timestamp, "put_call_ratio")
    latest_imbalance = _latest_row_value(oi_imbalance, latest_timestamp, "oi_imbalance")
    atm_iv = _atm_iv(option_chain, spot_price, latest_timestamp)
    iv_percentile = _rolling_latest_percentile(option_chain, latest_timestamp)
    iv_skew = _latest_iv_skew(option_chain, spot_price, latest_timestamp)
    realized_proxy = _realized_vol_proxy(option_chain, spot_price)
    iv_realized_spread = float(atm_iv - realized_proxy) if pd.notna(atm_iv) and pd.notna(realized_proxy) else np.nan
    total_oi_by_ts = option_chain.groupby("timestamp")["oi"].sum().dropna().sort_index()
    if len(total_oi_by_ts) >= 2:
        oi_delta = float(total_oi_by_ts.iloc[-1] - total_oi_by_ts.iloc[-2])
        oi_buildup = max(oi_delta, 0.0)
        oi_unwinding = max(-oi_delta, 0.0)
    else:
        oi_buildup = np.nan
        oi_unwinding = np.nan

    dealer_proxy = _dealer_positioning(option_chain, latest_timestamp)
    strike_concentration = _strike_concentration(option_chain, spot_price)
    max_pain_distance = _max_pain_distance(option_chain, spot_price)
    concentration_zones = compute_strike_concentration_zones(option_chain)
    call_wall = _latest_zone_value(concentration_zones.frame, latest_timestamp, "call_wall")
    put_wall = _latest_zone_value(concentration_zones.frame, latest_timestamp, "put_wall")
    option_bias = float(np.nan_to_num(latest_imbalance, nan=0.0) + np.nan_to_num(latest_pcr, nan=1.0) - 1.0)

    return {
        "put_call_ratio": latest_pcr,
        "oi_imbalance": latest_imbalance,
        "atm_iv": atm_iv,
        "iv_percentile": iv_percentile,
        "iv_realized_spread": iv_realized_spread,
        "oi_buildup": oi_buildup,
        "oi_unwinding": oi_unwinding,
        "strike_concentration": strike_concentration,
        "dealer_positioning_proxy": dealer_proxy,
        "max_pain_distance": max_pain_distance,
        "iv_skew": iv_skew,
        "option_chain_bias": option_bias,
        "call_wall": call_wall,
        "put_wall": put_wall,
    }


def _latest_row_value(frame: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    filtered = frame.loc[frame["timestamp"] == timestamp, column]
    numeric = pd.to_numeric(filtered, errors="coerce").dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else float("nan")


def _latest_zone_value(frame: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    filtered = frame.loc[frame["timestamp"] == timestamp, column]
    numeric = pd.to_numeric(filtered, errors="coerce").dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else float("nan")
