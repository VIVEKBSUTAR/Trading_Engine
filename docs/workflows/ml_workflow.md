# ML Workflow (Planned)

1. Read aligned feature data from parquet via DuckDB.
2. Generate leakage-safe feature matrix and forward target.
3. Walk-forward train/validation loops with rolling windows.
4. Evaluate Sharpe, drawdown, precision/recall, win rate, profit factor.
5. Persist model artifacts and feature importances.
