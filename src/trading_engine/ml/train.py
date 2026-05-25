"""Walk-forward RandomForest training for directional next-bar bias."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from trading_engine.ml.models import TrainedModelArtifact
from trading_engine.ml.validation import (
	ValidationMetrics,
	build_walk_forward_windows,
	compute_strategy_metrics,
)


@dataclass(slots=True)
class RandomForestTrainingConfig:
	"""Configuration for chronology-safe walk-forward training."""

	target_column: str = "target_direction_up"
	timestamp_column: str = "timestamp"
	train_size: int = 504
	test_size: int = 21
	step_size: int = 21
	confidence_threshold: float = 0.60
	n_estimators: int = 500
	max_depth: int | None = 8
	min_samples_leaf: int = 20
	random_state: int = 42


@dataclass(slots=True)
class WalkForwardTrainingResult:
	"""Container for fold predictions, metrics, and feature importances."""

	oos_predictions: pd.DataFrame
	metrics: ValidationMetrics
	feature_importances: pd.DataFrame
	artifact: TrainedModelArtifact


class RandomForestWalkForwardTrainer:
	"""Train and evaluate directional classifier with strict temporal ordering."""

	def __init__(self, config: RandomForestTrainingConfig) -> None:
		self._config = config

	def train(self, frame: pd.DataFrame) -> WalkForwardTrainingResult:
		"""Run walk-forward training and return out-of-sample diagnostics."""
		if self._config.target_column not in frame.columns:
			raise ValueError(f"Target column missing: {self._config.target_column}")
		if self._config.timestamp_column not in frame.columns:
			raise ValueError(f"Timestamp column missing: {self._config.timestamp_column}")

		data = frame.sort_values(self._config.timestamp_column).reset_index(drop=True)
		feature_columns = self._resolve_feature_columns(data)

		windows = build_walk_forward_windows(
			n_rows=len(data),
			train_size=self._config.train_size,
			test_size=self._config.test_size,
			step_size=self._config.step_size,
		)
		if not windows:
			raise ValueError("Insufficient rows for configured walk-forward windows")

		fold_predictions: list[pd.DataFrame] = []
		fold_importances: list[pd.DataFrame] = []

		for fold_idx, window in enumerate(windows):
			train_df = data.iloc[window.train_start : window.train_end]
			test_df = data.iloc[window.test_start : window.test_end]

			model = self._build_model()
			model.fit(train_df[feature_columns], train_df[self._config.target_column].astype(int))

			proba = model.predict_proba(test_df[feature_columns])[:, 1]
			pred_raw = (proba >= 0.5).astype(int)
			pred_filtered = np.where(proba >= self._config.confidence_threshold, pred_raw, 0)

			fold_predictions.append(
				pd.DataFrame(
					{
						self._config.timestamp_column: test_df[self._config.timestamp_column].to_numpy(),
						"y_true": test_df[self._config.target_column].astype(int).to_numpy(),
						"y_pred": pred_filtered,
						"proba_up": proba,
						"fold": fold_idx,
					}
				)
			)

			fold_importances.append(
				pd.DataFrame(
					{
						"feature": feature_columns,
						"importance": model.feature_importances_,
						"fold": fold_idx,
					}
				)
			)

		oos = pd.concat(fold_predictions, ignore_index=True)

		# Strategy return assumes long-only participation when y_pred == 1.
		future_ret = data[self._config.target_column].astype(float)
		aligned_future_ret = (
			data[[self._config.timestamp_column]]
			.assign(target_as_return=future_ret)
			.merge(oos[[self._config.timestamp_column, "y_pred"]], on=self._config.timestamp_column, how="right")
		)
		strategy_returns = aligned_future_ret["target_as_return"] * aligned_future_ret["y_pred"]

		metrics = compute_strategy_metrics(
			y_true=oos["y_true"],
			y_pred=oos["y_pred"],
			strategy_returns=strategy_returns,
		)

		feature_importances = (
			pd.concat(fold_importances, ignore_index=True)
			.groupby("feature", observed=True)["importance"]
			.mean()
			.sort_values(ascending=False)
			.reset_index()
		)

		final_model = self._build_model()
		final_model.fit(data[feature_columns], data[self._config.target_column].astype(int))

		artifact = TrainedModelArtifact(
			model=final_model,
			feature_columns=feature_columns,
			target_column=self._config.target_column,
			feature_version=str(data.get("feature_version", pd.Series(["unknown"])).iloc[-1]),
			confidence_threshold=self._config.confidence_threshold,
		)

		return WalkForwardTrainingResult(
			oos_predictions=oos,
			metrics=metrics,
			feature_importances=feature_importances,
			artifact=artifact,
		)

	def _build_model(self) -> RandomForestClassifier:
		return RandomForestClassifier(
			n_estimators=self._config.n_estimators,
			max_depth=self._config.max_depth,
			min_samples_leaf=self._config.min_samples_leaf,
			random_state=self._config.random_state,
			n_jobs=-1,
			class_weight="balanced_subsample",
		)

	def _resolve_feature_columns(self, frame: pd.DataFrame) -> list[str]:
		excluded = {
			self._config.timestamp_column,
			self._config.target_column,
		}
		feature_columns = [
			col
			for col in frame.columns
			if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
		]
		if not feature_columns:
			raise ValueError("No numeric feature columns available for training")
		return feature_columns
