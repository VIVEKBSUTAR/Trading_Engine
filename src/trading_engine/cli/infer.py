"""CLI entrypoint to run inference given a trained artifact and dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading_engine.config.settings import get_settings
from trading_engine.data.storage import ParquetDuckDBStore
from trading_engine.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from trading_engine.ml.inference import load_artifact, predict_with_confidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="te-infer")
    parser.add_argument("--dataset", default="features", help="Dataset name to run inference on")
    parser.add_argument("--artifact", default="models/rf_walkforward.pkl", help="Path to trained artifact")
    parser.add_argument("--out-csv", default=None, help="Optional output CSV path for signals")
    args = parser.parse_args(argv)

    settings = get_settings()
    store = ParquetDuckDBStore(settings)

    frame = store.load_dataset(dataset=args.dataset)
    if frame.empty:
        print("No data available for inference; aborting.")
        return 2

    fp_config = FeaturePipelineConfig(prevent_lookahead=False, include_target=False)
    fp = FeaturePipeline(fp_config)
    features = fp.transform(frame)

    artifact_path = settings.project_root / Path(args.artifact)
    artifact = load_artifact(artifact_path)

    signals = predict_with_confidence(artifact, features)

    if args.out_csv:
        out_path = settings.project_root / Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(out_path, index=False)
        print(f"Signals written to {out_path}")
    else:
        print(signals.head().to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
