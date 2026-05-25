from __future__ import annotations

import pandas as pd

from trading_engine.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig


def test_feature_pipeline_outputs_target_and_regime() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, freq="D"),
            "nifty50_spot__open": [100 + i for i in range(40)],
            "nifty50_spot__high": [101 + i for i in range(40)],
            "nifty50_spot__low": [99 + i for i in range(40)],
            "nifty50_spot__close": [100 + i for i in range(40)],
            "gift_nifty_fut__close": [100 + i for i in range(40)],
            "india_vix__close": [12 + (i % 5) for i in range(40)],
        }
    )

    pipeline = FeaturePipeline(
        FeaturePipelineConfig(
            realized_vol_window=5,
            rolling_corr_window=5,
            prevent_lookahead=True,
            include_target=True,
        )
    )
    output = pipeline.transform(frame)

    assert "target_direction_up" in output.columns
    assert "regime" in output.columns
    assert "ret_1" in output.columns
    assert len(output) > 0
