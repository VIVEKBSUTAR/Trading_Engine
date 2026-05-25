"""Vectorized options analytics primitives for Nifty options research and production."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def compute_put_call_ratio(
    option_chain: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    option_type_col: str = "option_type",
    oi_col: str = "oi",
) -> pd.DataFrame:
    """Compute put-call ratio by timestamp using total open interest."""
    grouped = (
        option_chain.groupby([timestamp_col, option_type_col], observed=True)[oi_col]
        .sum()
        .unstack(fill_value=0.0)
        .rename(columns={"CE": "call_oi", "PE": "put_oi"})
    )

    call_oi = grouped.get("call_oi", pd.Series(0.0, index=grouped.index))
    put_oi = grouped.get("put_oi", pd.Series(0.0, index=grouped.index))
    pcr = put_oi / call_oi.replace(0.0, np.nan)

    return pcr.rename("put_call_ratio").reset_index()


def compute_oi_imbalance(
    option_chain: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    option_type_col: str = "option_type",
    oi_col: str = "oi",
) -> pd.DataFrame:
    """Compute OI imbalance by timestamp as normalized net call-vs-put OI."""
    grouped = (
        option_chain.groupby([timestamp_col, option_type_col], observed=True)[oi_col]
        .sum()
        .unstack(fill_value=0.0)
        .rename(columns={"CE": "call_oi", "PE": "put_oi"})
    )

    call_oi = grouped.get("call_oi", pd.Series(0.0, index=grouped.index))
    put_oi = grouped.get("put_oi", pd.Series(0.0, index=grouped.index))

    denominator = (call_oi + put_oi).replace(0.0, np.nan)
    imbalance = (call_oi - put_oi) / denominator

    return imbalance.rename("oi_imbalance").reset_index()


def compute_max_pain(
    option_chain: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    expiry_col: str = "expiry",
    strike_col: str = "strike",
    option_type_col: str = "option_type",
    oi_col: str = "oi",
) -> pd.DataFrame:
    """Compute max pain strike for each timestamp and expiry bucket.

    A small group-level loop is used because the strike-payout matrix dimension varies
    per snapshot and cannot be fully vectorized across all snapshots.
    """
    required = {timestamp_col, expiry_col, strike_col, option_type_col, oi_col}
    missing = required - set(option_chain.columns)
    if missing:
        raise ValueError(f"Missing required columns for max pain: {sorted(missing)}")

    output_rows: list[dict[str, object]] = []
    grouped = option_chain.groupby([timestamp_col, expiry_col], observed=True)

    for (timestamp, expiry), group in grouped:
        pivot = (
            group.pivot_table(
                index=strike_col,
                columns=option_type_col,
                values=oi_col,
                aggfunc="sum",
                fill_value=0.0,
            )
            .rename(columns={"CE": "call_oi", "PE": "put_oi"})
            .sort_index()
        )

        strikes = pivot.index.to_numpy(dtype=float)
        call_oi = pivot.get("call_oi", pd.Series(0.0, index=pivot.index)).to_numpy(dtype=float)
        put_oi = pivot.get("put_oi", pd.Series(0.0, index=pivot.index)).to_numpy(dtype=float)

        settlement = strikes[:, None]
        strike_grid = strikes[None, :]

        call_payout = np.maximum(0.0, settlement - strike_grid) * call_oi[None, :]
        put_payout = np.maximum(0.0, strike_grid - settlement) * put_oi[None, :]
        total_payout = call_payout.sum(axis=1) + put_payout.sum(axis=1)

        max_pain_idx = int(np.argmin(total_payout))
        output_rows.append(
            {
                timestamp_col: timestamp,
                expiry_col: expiry,
                "max_pain": float(strikes[max_pain_idx]),
                "max_pain_total_payout": float(total_payout[max_pain_idx]),
            }
        )

    return pd.DataFrame(output_rows)


def compute_realized_volatility(
    close: pd.Series,
    *,
    window: int = 20,
    annualization_factor: int = 252,
) -> pd.Series:
    """Compute annualized realized volatility from close-to-close returns."""
    returns = close.pct_change()
    return returns.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(annualization_factor)


def compute_iv_percentile(iv: pd.Series, *, window: int = 252) -> pd.Series:
    """Compute rolling IV percentile rank for each observation."""

    def _rank_last(values: np.ndarray) -> float:
        last_value = values[-1]
        return float((values <= last_value).mean())

    return iv.rolling(window=window, min_periods=window).apply(_rank_last, raw=True)


def compute_volatility_spread(implied_vol: pd.Series, realized_vol: pd.Series) -> pd.Series:
    """Compute implied-realized volatility spread."""
    return implied_vol - realized_vol


def compute_rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    window: int = 20,
) -> pd.Series:
    """Compute rolling correlation between two return series."""
    return series_a.rolling(window=window, min_periods=window).corr(series_b)


def compute_iv_skew(
    option_chain: pd.DataFrame,
    *,
    underlying_price: pd.Series | None = None,
    timestamp_col: str = "timestamp",
    expiry_col: str = "expiry",
    strike_col: str = "strike",
    option_type_col: str = "option_type",
    iv_col: str = "iv",
    otm_put_distance: float = 0.02,
) -> pd.DataFrame:
    """Compute OTM put IV minus ATM IV for each timestamp/expiry."""
    required = {timestamp_col, expiry_col, strike_col, option_type_col, iv_col}
    missing = required - set(option_chain.columns)
    if missing:
        raise ValueError(f"Missing required columns for IV skew: {sorted(missing)}")

    rows: list[dict[str, object]] = []

    grouped = option_chain.groupby([timestamp_col, expiry_col], observed=True)
    for (timestamp, expiry), group in grouped:
        puts = group[group[option_type_col] == "PE"]
        all_options = group
        if puts.empty or all_options.empty:
            continue

        if underlying_price is not None and timestamp in underlying_price.index:
            ref_price = float(underlying_price.loc[timestamp])
        else:
            ref_price = float(all_options[strike_col].median())

        strikes = all_options[strike_col].to_numpy(dtype=float)
        nearest_idx = int(np.argmin(np.abs(strikes - ref_price)))
        atm_strike = float(strikes[nearest_idx])

        atm_iv_series = all_options.loc[all_options[strike_col] == atm_strike, iv_col]
        if atm_iv_series.empty:
            continue
        atm_iv = float(atm_iv_series.mean())

        otm_target = ref_price * (1.0 - otm_put_distance)
        candidate_puts = puts.loc[puts[strike_col] <= otm_target]
        if candidate_puts.empty:
            candidate_puts = puts

        put_strikes = candidate_puts[strike_col].to_numpy(dtype=float)
        put_idx = int(np.argmin(np.abs(put_strikes - otm_target)))
        otm_put_strike = float(put_strikes[put_idx])
        otm_put_iv = float(candidate_puts.loc[candidate_puts[strike_col] == otm_put_strike, iv_col].mean())

        rows.append(
            {
                timestamp_col: timestamp,
                expiry_col: expiry,
                "atm_iv": atm_iv,
                "otm_put_iv": otm_put_iv,
                "iv_skew": otm_put_iv - atm_iv,
            }
        )

    return pd.DataFrame(rows)


def compute_dealer_positioning_proxy(
    option_chain: pd.DataFrame,
    *,
    underlying_price: pd.Series,
    timestamp_col: str = "timestamp",
    option_type_col: str = "option_type",
    strike_col: str = "strike",
    oi_col: str = "oi",
    bandwidth: float = 0.02,
) -> pd.DataFrame:
    """Approximate dealer positioning from OI weighted gamma proxy.

    This is a directional proxy, not an exact dealer gamma model.
    """
    required = {timestamp_col, option_type_col, strike_col, oi_col}
    missing = required - set(option_chain.columns)
    if missing:
        raise ValueError(f"Missing required columns for dealer positioning: {sorted(missing)}")

    frame = option_chain.copy()
    aligned_underlying = underlying_price.reindex(frame[timestamp_col]).to_numpy(dtype=float)
    frame = frame.assign(underlying=aligned_underlying)
    frame = frame.dropna(subset=["underlying"])

    moneyness = (frame["underlying"] - frame[strike_col]) / (bandwidth * frame["underlying"].clip(lower=1e-6))
    call_delta = 1.0 / (1.0 + np.exp(-moneyness.to_numpy(dtype=float)))
    put_delta = call_delta - 1.0

    is_call = frame[option_type_col].eq("CE").to_numpy(dtype=float)
    delta = is_call * call_delta + (1.0 - is_call) * put_delta

    gamma = np.abs(delta * (1.0 - np.clip(np.abs(delta), 0.0, 1.0)))
    exposure = -1.0 * frame[oi_col].to_numpy(dtype=float) * gamma * np.sign(delta)

    output = (
        pd.DataFrame({timestamp_col: frame[timestamp_col], "dealer_positioning_proxy": exposure})
        .groupby(timestamp_col, observed=True)["dealer_positioning_proxy"]
        .sum()
        .reset_index()
    )
    return output


@dataclass(slots=True)
class StrikeConcentrationResult:
    """Container for call/put concentration and gamma-flip proxy zones."""

    frame: pd.DataFrame


def compute_strike_concentration_zones(
    option_chain: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    expiry_col: str = "expiry",
    strike_col: str = "strike",
    option_type_col: str = "option_type",
    oi_col: str = "oi",
) -> StrikeConcentrationResult:
    """Compute call wall, put wall, and midpoint gamma-flip proxy by snapshot."""
    grouped = (
        option_chain.groupby([timestamp_col, expiry_col, strike_col, option_type_col], observed=True)[oi_col]
        .sum()
        .unstack(fill_value=0.0)
        .rename(columns={"CE": "call_oi", "PE": "put_oi"})
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for (timestamp, expiry), group in grouped.groupby([timestamp_col, expiry_col], observed=True):
        call_idx = int(group["call_oi"].to_numpy(dtype=float).argmax()) if len(group) else 0
        put_idx = int(group["put_oi"].to_numpy(dtype=float).argmax()) if len(group) else 0

        call_wall = float(group.iloc[call_idx][strike_col])
        put_wall = float(group.iloc[put_idx][strike_col])

        rows.append(
            {
                timestamp_col: timestamp,
                expiry_col: expiry,
                "call_wall": call_wall,
                "put_wall": put_wall,
                "gamma_flip_zone": float((call_wall + put_wall) / 2.0),
            }
        )

    return StrikeConcentrationResult(frame=pd.DataFrame(rows))
