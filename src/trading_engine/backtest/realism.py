"""Execution realism helpers for backtesting option trades."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ExecutionRealismConfig:
    """Slippage and fill assumptions for realistic backtests."""

    slippage_bps: float = 5.0
    spread_bps: float = 12.0
    liquidity_penalty_bps: float = 6.0
    delay_bars: int = 1
    partial_fill_rate: float = 0.85
    iv_expansion_bps: float = 4.0


@dataclass(slots=True)
class FillEstimate:
    """Estimated fill outcome for a simulated execution."""

    fill_price: float
    fill_quantity: int
    slippage_bps: float
    spread_bps: float
    delay_bars: int
    partial_fill_fraction: float


def estimate_option_fill_price(
    *,
    mid_price: float,
    side: str,
    config: ExecutionRealismConfig | None = None,
    liquidity_multiplier: float = 1.0,
    iv_expansion_multiplier: float = 1.0,
) -> float:
    """Estimate an option fill using spread, slippage, and liquidity penalties."""
    cfg = config or ExecutionRealismConfig()
    direction = 1.0 if side.lower() in {"buy", "cover"} else -1.0
    spread = mid_price * cfg.spread_bps / 10_000.0
    slippage = mid_price * cfg.slippage_bps / 10_000.0
    liquidity_penalty = mid_price * cfg.liquidity_penalty_bps / 10_000.0 / max(liquidity_multiplier, 0.25)
    iv_penalty = mid_price * cfg.iv_expansion_bps / 10_000.0 * max(iv_expansion_multiplier - 1.0, 0.0)
    return float(mid_price + direction * (spread / 2.0 + slippage + liquidity_penalty + iv_penalty))


def estimate_fill_quantity(requested_quantity: int, *, config: ExecutionRealismConfig | None = None, liquidity_score: float = 1.0) -> int:
    """Estimate partial fills under weaker liquidity."""
    cfg = config or ExecutionRealismConfig()
    effective_fraction = max(0.05, min(cfg.partial_fill_rate * max(liquidity_score, 0.25), 1.0))
    return int(max(0, round(requested_quantity * effective_fraction)))


def simulate_execution_fill(
    *,
    mid_price: float,
    requested_quantity: int,
    side: str,
    config: ExecutionRealismConfig | None = None,
    liquidity_score: float = 1.0,
    iv_expansion_multiplier: float = 1.0,
) -> FillEstimate:
    """Simulate a realistic fill with delayed entry and partial execution."""
    cfg = config or ExecutionRealismConfig()
    fill_price = estimate_option_fill_price(
        mid_price=mid_price,
        side=side,
        config=cfg,
        liquidity_multiplier=liquidity_score,
        iv_expansion_multiplier=iv_expansion_multiplier,
    )
    fill_quantity = estimate_fill_quantity(requested_quantity, config=cfg, liquidity_score=liquidity_score)
    return FillEstimate(
        fill_price=float(fill_price),
        fill_quantity=int(fill_quantity),
        slippage_bps=cfg.slippage_bps,
        spread_bps=cfg.spread_bps,
        delay_bars=cfg.delay_bars,
        partial_fill_fraction=float(fill_quantity / max(requested_quantity, 1)),
    )


def apply_execution_realism(trades: pd.DataFrame, *, price_col: str = "entry_price", quantity_col: str = "quantity", side_col: str = "side", config: ExecutionRealismConfig | None = None) -> pd.DataFrame:
    """Apply execution realism to a trades DataFrame."""
    if trades.empty:
        return trades.copy()

    cfg = config or ExecutionRealismConfig()
    frame = trades.copy()
    frame["realistic_fill_price"] = [
        estimate_option_fill_price(mid_price=float(price), side=str(side), config=cfg)
        for price, side in zip(frame[price_col], frame[side_col], strict=False)
    ]
    frame["realistic_fill_quantity"] = [
        estimate_fill_quantity(int(quantity), config=cfg, liquidity_score=1.0)
        for quantity in frame[quantity_col]
    ]
    return frame
