"""Validation metrics and walk-forward slicing for time-series models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


@dataclass(slots=True)
class WalkForwardWindow:
	"""Defines a single chronological train-test window."""

	train_start: int
	train_end: int
	test_start: int
	test_end: int


@dataclass(slots=True)
class ValidationMetrics:
	"""Core model and strategy metrics."""

	precision: float
	recall: float
	win_rate: float
	profit_factor: float
	sharpe_ratio: float
	max_drawdown: float


def build_walk_forward_windows(
	n_rows: int,
	*,
	train_size: int,
	test_size: int,
	step_size: int | None = None,
) -> list[WalkForwardWindow]:
	"""Create chronological rolling windows for walk-forward validation."""
	if train_size <= 0 or test_size <= 0:
		raise ValueError("train_size and test_size must be > 0")
	if n_rows < train_size + test_size:
		return []

	step = step_size or test_size
	windows: list[WalkForwardWindow] = []

	start = 0
	while True:
		train_start = start
		train_end = train_start + train_size
		test_start = train_end
		test_end = test_start + test_size

		if test_end > n_rows:
			break

		windows.append(
			WalkForwardWindow(
				train_start=train_start,
				train_end=train_end,
				test_start=test_start,
				test_end=test_end,
			)
		)
		start += step

	return windows


def compute_strategy_metrics(
	y_true: pd.Series,
	y_pred: pd.Series,
	strategy_returns: pd.Series,
	*,
	annualization_factor: int = 252,
) -> ValidationMetrics:
	"""Compute precision/recall plus portfolio-level diagnostics."""
	y_true_int = y_true.astype(int)
	y_pred_int = y_pred.astype(int)

	precision = float(precision_score(y_true_int, y_pred_int, zero_division=0))
	recall = float(recall_score(y_true_int, y_pred_int, zero_division=0))

	traded = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna()
	wins = traded[traded > 0.0]
	losses = traded[traded < 0.0]

	win_rate = float((wins.shape[0] / traded.shape[0]) if traded.shape[0] else 0.0)

	gross_profit = float(wins.sum())
	gross_loss = float(-losses.sum())
	profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf")

	mean_ret = float(traded.mean()) if traded.shape[0] else 0.0
	std_ret = float(traded.std(ddof=0)) if traded.shape[0] else 0.0
	sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(annualization_factor)) if std_ret > 0 else 0.0

	equity_curve = (1.0 + traded.fillna(0.0)).cumprod()
	running_max = equity_curve.cummax()
	drawdown = (equity_curve / running_max) - 1.0
	max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

	return ValidationMetrics(
		precision=precision,
		recall=recall,
		win_rate=win_rate,
		profit_factor=profit_factor,
		sharpe_ratio=sharpe_ratio,
		max_drawdown=max_drawdown,
	)
