"""Unit tests for SignalGenerationEngine and composite SignalScore outputs."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from config.settings import AppSettings
from trading_engine.intelligence.engine import LiveIntelligenceEngine
from trading_engine.intelligence.models import MarketSnapshotBundle


def test_signal_score_attached_to_signal():
    """Verify that SignalResult has a computed composite SignalScore."""
    settings = AppSettings.from_env()
    engine = LiveIntelligenceEngine(settings)

    empty_df = pd.DataFrame()
    ticks_df = pd.DataFrame({
        "timestamp": [datetime.now(UTC)] * 10,
        "last_price": [24000.0] * 10,
        "volume_traded": list(range(1000, 2000, 100))
    })

    snapshot = MarketSnapshotBundle(
        timestamp=datetime.now(UTC),
        spot_price=24000.0,
        vix_value=15.0,
        spot_frame=empty_df,
        vix_frame=empty_df,
        option_chain_frame=empty_df,
        ticks_frame=ticks_df,
        candles_1m=empty_df,
        candles_5m=empty_df
    )

    report = engine.process(snapshot)
    assert report.signal.score is not None
    assert 0.0 <= report.signal.score.total <= 10.0
    assert report.signal.score.label() in {"VERY STRONG", "STRONG", "MODERATE", "WEAK", "NO SIGNAL"}
