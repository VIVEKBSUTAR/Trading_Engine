from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_engine.config.models import AppConfig
from trading_engine.config.settings import RuntimeSettings
from trading_engine.data.storage import ParquetDuckDBStore


def test_storage_writes_dataset_and_indexes_metadata(tmp_path: Path) -> None:
    app = AppConfig.model_validate(
        {
            "paths": {
                "data_raw_dir": str(tmp_path / "raw"),
                "data_processed_dir": str(tmp_path / "processed"),
                "data_cache_dir": str(tmp_path / "cache"),
                "duckdb_path": str(tmp_path / "cache" / "metadata.duckdb"),
                "log_dir": str(tmp_path / "logs"),
            }
        }
    )
    runtime = RuntimeSettings(project_root=tmp_path, app=app)

    for directory in [
        runtime.raw_dir,
        runtime.processed_dir,
        runtime.cache_dir,
        runtime.app.paths.log_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:15:00+05:30", "2026-01-01 09:16:00+05:30"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
        }
    )

    store = ParquetDuckDBStore(runtime)
    try:
        store.write_dataset(
            frame,
            dataset="market_data",
            feed="nifty50_spot",
            timestamp_col="timestamp",
            mode="append",
        )
        metadata = store.query_metadata(dataset="market_data", feed="nifty50_spot")
    finally:
        store.close()

    assert len(metadata) >= 1
