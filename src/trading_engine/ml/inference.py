"""Inference utilities for directional probability scoring and filtering."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from trading_engine.ml.models import TrainedModelArtifact


def save_artifact(artifact: TrainedModelArtifact, path: Path) -> None:
	"""Persist trained model artifact to disk."""
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("wb") as handle:
		pickle.dump(artifact, handle)


def load_artifact(path: Path) -> TrainedModelArtifact:
	"""Load model artifact from disk."""
	with path.open("rb") as handle:
		artifact = pickle.load(handle)

	if not isinstance(artifact, TrainedModelArtifact):
		raise TypeError("Loaded object is not a TrainedModelArtifact")
	return artifact


def predict_with_confidence(
	artifact: TrainedModelArtifact,
	frame: pd.DataFrame,
	*,
	timestamp_col: str = "timestamp",
	confidence_threshold: float | None = None,
) -> pd.DataFrame:
	"""Generate directional predictions with confidence threshold filter."""
	threshold = confidence_threshold if confidence_threshold is not None else artifact.confidence_threshold

	missing = set(artifact.feature_columns) - set(frame.columns)
	if missing:
		raise ValueError(f"Missing inference feature columns: {sorted(missing)}")
	if timestamp_col not in frame.columns:
		raise ValueError(f"Missing timestamp column: {timestamp_col}")

	features = frame[artifact.feature_columns]
	proba_up = artifact.model.predict_proba(features)[:, 1]
	pred_raw = (proba_up >= 0.5).astype(int)
	pred_filtered = (proba_up >= threshold).astype(int) * pred_raw

	return pd.DataFrame(
		{
			timestamp_col: frame[timestamp_col].to_numpy(),
			"proba_up": proba_up,
			"signal_directional_long": pred_filtered,
			"signal_confident": (proba_up >= threshold).astype(int),
		}
	)
