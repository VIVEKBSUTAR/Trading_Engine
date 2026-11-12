"""Validated option context; no network access or broker order operations."""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd


def number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def option_delta(spot: float, strike: float, expiry, iv: float, side: str, now: datetime) -> float | None:
    """European Black-Scholes Delta, zero carry assumption; IV in percent units.

    Used only when provider Delta is absent. Missing/invalid IV never becomes zero.
    """
    try:
        end = pd.Timestamp(expiry).normalize().tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)
        years = (end - pd.Timestamp(now)).total_seconds() / (365 * 86400)
        sigma = float(iv) / 100
        if not all(math.isfinite(x) and x > 0 for x in (spot, strike, years, sigma)) or side not in {"CE", "PE"}:
            return None
        d1 = (math.log(spot / strike) + sigma * sigma * years / 2) / (sigma * math.sqrt(years))
        delta = .5 * (1 + math.erf(d1 / math.sqrt(2)))
        return delta if side == "CE" else delta - 1
    except (ValueError, TypeError, OverflowError):
        return None


def latest_chain(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or not {"expiry", "strike", "option_type"}.issubset(frame):
        return pd.DataFrame()
    frame = frame.copy()
    if "timestamp" in frame:
        frame = frame.sort_values("timestamp")
    return frame.drop_duplicates(["expiry", "strike", "option_type"], keep="last")


def positioning(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    """Aggregate same-expiry PCR and OI-weighted minimum settlement payout."""
    frame = latest_chain(frame)
    if frame.empty or frame.expiry.nunique() != 1:
        return None, None
    oi_col = "open_interest" if "open_interest" in frame else "oi"
    if oi_col not in frame:
        return None, None
    frame["strike"] = pd.to_numeric(frame.strike, errors="coerce")
    if not frame.strike.map(math.isfinite).all() or (frame.strike <= 0).any():
        return None, None
    frame["_oi"] = pd.to_numeric(frame[oi_col], errors="coerce")
    if not frame["_oi"].map(math.isfinite).all() or (frame["_oi"] < 0).any():
        return None, None
    calls = frame.loc[frame.option_type == "CE"]
    puts = frame.loc[frame.option_type == "PE"]
    call_oi, put_oi = calls._oi.sum(), puts._oi.sum()
    if calls.empty or puts.empty or call_oi <= 0:
        return None, None
    strikes = sorted(pd.to_numeric(frame.strike, errors="coerce").dropna().unique())
    pain = min(strikes, key=lambda x: ((x - calls.strike).clip(lower=0) * calls._oi).sum()
               + ((puts.strike - x).clip(lower=0) * puts._oi).sum())
    return float(put_oi / call_oi), float(pain)


def underlying_atr(frame: pd.DataFrame, *, now: datetime, period: int = 14) -> float | None:
    """14 completed contiguous one-minute NIFTY LTP bars, with preceding close.

    This deliberately never consumes mixed option tick candles or daily OHLC.
    """
    if frame is None or frame.empty or not {"timestamp", "last_price"}.issubset(frame):
        return None
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="coerce", utc=True)
    frame["last_price"] = pd.to_numeric(frame.last_price, errors="coerce")
    stamp = pd.Timestamp(now)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    frame = frame.dropna(subset=["timestamp", "last_price"])
    frame = frame.loc[(frame.timestamp < stamp.floor("min")) & (frame.last_price > 0)
                      & (frame.timestamp.dt.tz_convert("Asia/Kolkata").dt.date == stamp.tz_convert("Asia/Kolkata").date())]
    bars = frame.sort_values("timestamp").set_index("timestamp").last_price.resample("1min").ohlc().tail(period + 1)
    if len(bars) != period + 1 or bars.isna().any().any() or bars.index[-1] != stamp.floor("min") - pd.Timedelta(minutes=1):
        return None
    previous = bars.close.shift(1)
    ranges = pd.concat([bars.high - bars.low, (bars.high - previous).abs(), (bars.low - previous).abs()], axis=1).max(axis=1).iloc[1:]
    value = float(ranges.mean())
    return value if math.isfinite(value) and value > 0 else None


def enrich_chain(frame: pd.DataFrame, *, spot: float | None, now: datetime) -> pd.DataFrame:
    frame = latest_chain(frame)
    if frame.empty:
        return frame
    frame["delta"] = [
        number(row.get("delta")) if number(row.get("delta")) is not None else
        option_delta(spot or 0, number(row.get("strike")) or 0, row.get("expiry"),
                     number(row.get("implied_volatility", row.get("iv"))) or 0,
                     str(row.get("option_type")), now)
        for _, row in frame.iterrows()
    ]
    frame["delta"] = pd.to_numeric(frame["delta"], errors="coerce")
    return frame
