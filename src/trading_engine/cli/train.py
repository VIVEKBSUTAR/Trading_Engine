"""CLI entrypoint to train a walk-forward RandomForest and persist artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading_engine.config.settings import get_settings
from trading_engine.data.storage import ParquetDuckDBStore
from trading_engine.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from trading_engine.ml.train import RandomForestWalkForwardTrainer, RandomForestTrainingConfig
from trading_engine.ml.inference import save_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="te-train")
    parser.add_argument("--dataset", default="features", help="Dataset name registered in Parquet store")
    parser.add_argument("--artifact", default="models/rf_walkforward.pkl", help="Output artifact path")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args(argv)

    settings = get_settings()
    store = ParquetDuckDBStore(settings)

    frame = store.load_dataset(dataset=args.dataset)
    if frame.empty:
        print("No data loaded from dataset; aborting.")
        return 2

    fp = FeaturePipeline(FeaturePipelineConfig())
    features = fp.transform(frame)

    train_cfg = RandomForestTrainingConfig(n_estimators=args.n_estimators, max_depth=args.max_depth)
    trainer = RandomForestWalkForwardTrainer(train_cfg)

    result = trainer.train(features)

    artifact_path = settings.project_root / Path(args.artifact)
    save_artifact(result.artifact, artifact_path)

    print(f"Training complete. Artifact saved to: {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
