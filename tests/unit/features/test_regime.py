from __future__ import annotations

import pandas as pd

from trading_engine.features.regime import RegimeConfig, RegimeDetector


def test_regime_detector_outputs_labels() -> None:
    frame = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100],
            "realized_vol": [0.10, 0.11, 0.12, 0.15, 0.18, 0.20, 0.16, 0.14, 0.12, 0.10, 0.09],
            "iv_realized_spread": [0.01] * 11,
        }
    )

    detector = RegimeDetector(
        RegimeConfig(
            trend_window=3,
            vol_window=3,
            trend_threshold=0.001,
            expansion_quantile=0.8,
            compression_quantile=0.2,
        )
    )
    regime = detector.classify(
        frame,
        close_col="close",
        realized_vol_col="realized_vol",
        iv_realized_spread_col="iv_realized_spread",
    )

    assert len(regime) == len(frame)
    assert regime.name == "regime"
