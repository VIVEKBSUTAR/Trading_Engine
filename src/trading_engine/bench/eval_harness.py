"""Reproducible walk-forward evaluation harness for local CSV datasets.

Loads spot and VIX CSVs, constructs minimal aligned frame expected by
`FeaturePipeline`, runs walk-forward training, persists artifacts and
benchmarks, and runs a directional backtest using `run_directional_backtest`.

Assumptions (explicit):
- spot CSV contains columns: `date,open,high,low,close,volume` with ISO timestamps
- vix CSV contains the same timestamp column and `close` column for VIX level
- no option chain is used in this benchmark (option features are skipped)

This module does not invent data. It will fail loudly if inputs are missing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd

from trading_engine.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from trading_engine.ml.train import RandomForestWalkForwardTrainer, RandomForestTrainingConfig
from trading_engine.ml.inference import save_artifact, predict_with_confidence
from trading_engine.config.settings import get_settings

logger = logging.getLogger("te.eval")


def _load_csv_market(path: Path, *, time_col: str = "date") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Market CSV not found: {path}")
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"Expected time column '{time_col}' in {path}")
    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _build_aligned_frame(spot: pd.DataFrame, vix: pd.DataFrame | None) -> pd.DataFrame:
    """Return DataFrame with columns expected by FeaturePipeline.

    Maps `open,high,low,close,volume` -> `nifty50_spot__*` and `india_vix__close`.
    If VIX is missing, `india_vix__close` will be NaN.
    """
    out = spot.rename(columns={
        "open": "nifty50_spot__open",
        "high": "nifty50_spot__high",
        "low": "nifty50_spot__low",
        "close": "nifty50_spot__close",
        "volume": "volume",
    })[
        ["timestamp", "nifty50_spot__open", "nifty50_spot__high", "nifty50_spot__low", "nifty50_spot__close", "volume"]
    ].copy()

    if vix is not None:
        vix = vix.rename(columns={"close": "india_vix__close"})[["timestamp", "india_vix__close"]]
        out = out.merge(vix, on="timestamp", how="left")
    else:
        out["india_vix__close"] = pd.NA

    # For futures lead (Gift Nifty) we do not have futures CSV; use spot close as a proxy
    # NOTE: Using spot as a proxy for futures is an explicit assumption for this
    # local benchmark and may not reflect true futures lead/lag dynamics.
    out["gift_nifty_fut__close"] = out["nifty50_spot__close"].astype(float)

    return out


def run_evaluation(
    spot_csv: Path,
    vix_csv: Path | None,
    out_dir: Path,
    *,
    train_n_estimators: int = 200,
) -> dict[str, object]:
    settings = get_settings()

    spot = _load_csv_market(spot_csv)
    vix = _load_csv_market(vix_csv) if vix_csv is not None else None

    aligned = _build_aligned_frame(spot, vix)

    fp = FeaturePipeline(FeaturePipelineConfig())
    features = fp.transform(aligned)

    # Train
    train_cfg = RandomForestTrainingConfig(n_estimators=train_n_estimators)
    trainer = RandomForestWalkForwardTrainer(train_cfg)
    result = trainer.train(features)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest = out_dir / timestamp
    dest.mkdir(parents=True, exist_ok=True)

    # Persist artifact
    artifact_path = dest / "artifact.pkl"
    save_artifact(result.artifact, artifact_path)

    # Persist OOS predictions and feature importances
    result.oos_predictions.to_csv(dest / "oos_predictions.csv", index=False)
    result.feature_importances.to_csv(dest / "feature_importances.csv", index=False)

    # Persist validation metrics
    metrics = {
        "precision": result.metrics.precision,
        "recall": result.metrics.recall,
        "win_rate": result.metrics.win_rate,
        "profit_factor": result.metrics.profit_factor,
        "sharpe_ratio": result.metrics.sharpe_ratio,
        "max_drawdown": result.metrics.max_drawdown,
    }
    with (dest / "validation_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Generate signals for backtest
    signals = predict_with_confidence(result.artifact, features)

    # Merge signals back onto market-level candles for backtest
    backtest_df = aligned.merge(signals, on="timestamp", how="right").sort_values("timestamp")
    backtest_df = backtest_df.rename(columns={
        "nifty50_spot__open": "open",
        "nifty50_spot__high": "high",
        "nifty50_spot__low": "low",
        "nifty50_spot__close": "close",
    })

    # Add backtest-specific columns
    backtest_df["signal_long"] = backtest_df["signal_directional_long"].fillna(0).astype(int)
    backtest_df["signal_short"] = 0
    backtest_df["confidence"] = backtest_df["proba_up"].fillna(0.0).astype(float)

    # Try to run the backtest if backtrader is available. If not, skip gracefully.
    backtest_output: dict[str, object] | None = None
    try:
        from trading_engine.backtest.engine import run_directional_backtest, BacktestConfig

        bt_config = BacktestConfig()
        bt_result = run_directional_backtest(backtest_df, config=bt_config, datetime_col="timestamp")

        # Persist backtest results
        if getattr(bt_result, "equity_curve", None) is not None:
            bt_result.equity_curve.to_csv(dest / "equity_curve.csv", index=True)

        if getattr(bt_result, "metrics", None) is not None:
            with (dest / "backtest_metrics.json").open("w", encoding="utf-8") as fh:
                json.dump({
                    "total_return": bt_result.metrics.total_return,
                    "annualized_return": bt_result.metrics.annualized_return,
                    "sharpe_ratio": bt_result.metrics.sharpe_ratio,
                    "max_drawdown": bt_result.metrics.max_drawdown,
                    "win_rate": bt_result.metrics.win_rate,
                    "profit_factor": bt_result.metrics.profit_factor,
                }, fh, indent=2)

        # Persist detailed trade ledger if available
        if getattr(bt_result, "trade_ledger", None) is not None and not bt_result.trade_ledger.empty:
            bt_result.trade_ledger.to_csv(dest / "trade_ledger.csv", index=False)

        backtest_output = {
            "total_return": bt_result.metrics.total_return,
            "annualized_return": bt_result.metrics.annualized_return,
            "sharpe_ratio": bt_result.metrics.sharpe_ratio,
            "max_drawdown": bt_result.metrics.max_drawdown,
            "win_rate": bt_result.metrics.win_rate,
            "profit_factor": bt_result.metrics.profit_factor,
        }
    except Exception:  # noqa: BLE001 - allow any import/runtime error to skip backtest
        logger.warning("Backtest skipped: backtrader not available or backtest failed")

    logger.info("Evaluation complete", dest=str(dest))

    return {
        "dest": str(dest),
        "metrics": metrics,
        "backtest_metrics": backtest_output,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="te-eval")
    parser.add_argument("--spot-csv", required=True, help="Spot CSV (NIFTY 50) path")
    parser.add_argument("--vix-csv", required=False, help="VIX CSV path")
    parser.add_argument("--out-dir", default="outputs/benchmarks", help="Output directory")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of RF estimators for speed")
    args = parser.parse_args(argv)

    spot = Path(args.spot_csv)
    vix = Path(args.vix_csv) if args.vix_csv else None
    out_dir = Path(args.out_dir)

    try:
        result = run_evaluation(spot, vix, out_dir, train_n_estimators=args.n_estimators)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation failed")
        print(f"Evaluation failed: {exc}")
        return 2

    print("Evaluation outputs written to:", result["dest"])
    print("Validation metrics:", json.dumps(result["metrics"], indent=2))
    print("Backtest metrics:", json.dumps(result["backtest_metrics"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
