"""Execution and risk abstractions for directional-first trade orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SignalFilterConfig:
	"""Controls signal participation and regime eligibility."""

	probability_long_threshold: float = 0.60
	probability_short_threshold: float = 0.40
	allowed_regimes: tuple[str, ...] = (
		"trending_up",
		"trending_down",
		"vol_expansion",
		"risk_off_expansion",
	)


@dataclass(slots=True)
class RiskConfig:
	"""Risk-based position sizing settings."""

	capital: float
	risk_per_trade: float = 0.005
	max_capital_exposure: float = 0.10
	contract_multiplier: float = 1.0
	min_units: int = 1


@dataclass(slots=True)
class ExecutionInstruction:
	"""Represents a filtered, risk-sized execution decision."""

	timestamp: pd.Timestamp
	strategy: str
	side: str
	quantity: int
	confidence: float
	regime: str
	reason: str


def calculate_position_size(
	*,
	entry_price: float,
	stop_price: float,
	confidence: float,
	risk: RiskConfig,
) -> int:
	"""Size position from stop distance and capped capital allocation."""
	risk_amount = risk.capital * risk.risk_per_trade * np.clip(confidence, 0.0, 1.0)
	per_unit_risk = abs(entry_price - stop_price) * risk.contract_multiplier
	if per_unit_risk <= 0.0:
		return 0

	max_exposure_units = int(
		(risk.capital * risk.max_capital_exposure) / max(entry_price * risk.contract_multiplier, 1e-9)
	)
	risk_units = int(risk_amount / per_unit_risk)
	units = min(max_exposure_units, risk_units)
	return int(max(units, risk.min_units)) if units > 0 else 0


def filter_directional_signals(
	predictions: pd.DataFrame,
	*,
	timestamp_col: str = "timestamp",
	confidence_col: str = "proba_up",
	regime_col: str = "regime",
	config: SignalFilterConfig,
) -> pd.DataFrame:
	"""Apply confidence and regime constraints to model predictions."""
	required = {timestamp_col, confidence_col, regime_col}
	missing = required - set(predictions.columns)
	if missing:
		raise ValueError(f"Missing prediction columns: {sorted(missing)}")

	frame = predictions.copy()
	frame["long_ok"] = frame[confidence_col] >= config.probability_long_threshold
	frame["short_ok"] = frame[confidence_col] <= config.probability_short_threshold
	frame["regime_ok"] = frame[regime_col].astype(str).isin(config.allowed_regimes)

	frame["signal_long"] = (frame["long_ok"] & frame["regime_ok"]).astype(int)
	frame["signal_short"] = (frame["short_ok"] & frame["regime_ok"]).astype(int)
	return frame


def build_execution_instructions(
	filtered_signals: pd.DataFrame,
	*,
	price_col: str,
	timestamp_col: str = "timestamp",
	confidence_col: str = "proba_up",
	regime_col: str = "regime",
	stop_loss_pct: float = 0.005,
	risk: RiskConfig,
) -> list[ExecutionInstruction]:
	"""Create directional-first instructions (CE/PE buying) from filtered signals."""
	required = {timestamp_col, confidence_col, regime_col, price_col, "signal_long", "signal_short"}
	missing = required - set(filtered_signals.columns)
	if missing:
		raise ValueError(f"Missing filtered signal columns: {sorted(missing)}")

	instructions: list[ExecutionInstruction] = []
	for row in filtered_signals.itertuples(index=False):
		timestamp = getattr(row, timestamp_col)
		close_price = float(getattr(row, price_col))
		confidence = float(getattr(row, confidence_col))
		regime = str(getattr(row, regime_col))
		signal_long = int(getattr(row, "signal_long"))
		signal_short = int(getattr(row, "signal_short"))

		if signal_long == 0 and signal_short == 0:
			continue

		if signal_long == 1:
			side = "buy"
			strategy = "ce_buy"
			stop_price = close_price * (1.0 - stop_loss_pct)
			qty = calculate_position_size(
				entry_price=close_price,
				stop_price=stop_price,
				confidence=confidence,
				risk=risk,
			)
			reason = "Directional long with confidence/regime pass"
		else:
			side = "buy"
			strategy = "pe_buy"
			stop_price = close_price * (1.0 + stop_loss_pct)
			qty = calculate_position_size(
				entry_price=close_price,
				stop_price=stop_price,
				confidence=1.0 - confidence,
				risk=risk,
			)
			reason = "Directional short proxy via PE buy"

		if qty <= 0:
			continue

		instructions.append(
			ExecutionInstruction(
				timestamp=pd.Timestamp(timestamp),
				strategy=strategy,
				side=side,
				quantity=qty,
				confidence=confidence,
				regime=regime,
				reason=reason,
			)
		)

	return instructions
