"""Backtest metrics and risk reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BacktestMetrics:
	"""Core portfolio-level and trade-level performance diagnostics."""

	total_return: float
	annualized_return: float
	annualized_volatility: float
	sharpe_ratio: float
	max_drawdown: float
	win_rate: float
	profit_factor: float


def compute_backtest_metrics(
	equity_curve: pd.Series,
	trade_returns: pd.Series,
	*,
	annualization_factor: int = 252,
) -> BacktestMetrics:
	"""Compute institutional baseline diagnostics from equity and trade returns."""
	curve = equity_curve.dropna().astype(float)
	if curve.empty:
		raise ValueError("Equity curve is empty")

	returns = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
	total_return = float((curve.iloc[-1] / curve.iloc[0]) - 1.0)

	periods = max(len(returns), 1)
	annualized_return = float((1.0 + total_return) ** (annualization_factor / periods) - 1.0)
	annualized_volatility = float(returns.std(ddof=0) * np.sqrt(annualization_factor)) if not returns.empty else 0.0
	sharpe_ratio = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(annualization_factor)) if returns.std(ddof=0) > 0 else 0.0

	running_max = curve.cummax()
	drawdown = (curve / running_max) - 1.0
	max_drawdown = float(drawdown.min())

	trades = trade_returns.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
	wins = trades[trades > 0.0]
	losses = trades[trades < 0.0]
	win_rate = float((wins.shape[0] / trades.shape[0]) if trades.shape[0] else 0.0)
	gross_profit = float(wins.sum())
	gross_loss = float(-losses.sum())
	profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

	return BacktestMetrics(
		total_return=total_return,
		annualized_return=annualized_return,
		annualized_volatility=annualized_volatility,
		sharpe_ratio=sharpe_ratio,
		max_drawdown=max_drawdown,
		win_rate=win_rate,
		profit_factor=profit_factor,
	)
