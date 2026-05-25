"""Commission, slippage, and transaction cost models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CostModelConfig:
	"""Cost parameters for liquidity-aware execution simulation."""

	fixed_commission_bps: float = 2.0
	min_commission: float = 20.0
	spread_multiplier: float = 1.0
	impact_coefficient: float = 0.10


def estimate_slippage_fraction(*, bid: float, ask: float, mid: float, participation: float) -> float:
	"""Estimate fractional slippage from spread and participation pressure."""
	if mid <= 0.0:
		return 0.0
	spread = max(ask - bid, 0.0)
	spread_component = spread / mid
	impact_component = np.clip(participation, 0.0, 1.0) ** 2
	return float(spread_component + impact_component)


def compute_transaction_cost(
	*,
	notional: float,
	quantity: int,
	bid: float,
	ask: float,
	mid: float,
	participation: float,
	config: CostModelConfig,
) -> float:
	"""Compute total estimated transaction cost in currency units."""
	if quantity <= 0 or notional <= 0.0:
		return 0.0

	commission = max(notional * (config.fixed_commission_bps / 10_000.0), config.min_commission)
	slippage_fraction = estimate_slippage_fraction(
		bid=bid,
		ask=ask,
		mid=mid,
		participation=participation * config.impact_coefficient,
	)
	slippage_cost = notional * slippage_fraction * config.spread_multiplier
	return float(commission + slippage_cost)
