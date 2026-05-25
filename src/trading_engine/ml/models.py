"""Shared model artifacts and payload types for ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier


@dataclass(slots=True)
class TrainedModelArtifact:
    """Serializable trained model with metadata needed for inference."""

    model: RandomForestClassifier
    feature_columns: list[str]
    target_column: str
    feature_version: str
    confidence_threshold: float
