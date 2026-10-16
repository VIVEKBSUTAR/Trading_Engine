"""Unit and integration tests for signal pipeline latency tracing."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from config.settings import AppSettings
from trading_engine.intelligence.engine import LiveIntelligenceEngine
from trading_engine.intelligence.models import MarketSnapshotBundle


def test_orchestrator_populates_latency_trace():
    """Verify that LiveIntelligenceEngine.process records LatencyTrace timestamps."""
    settings = AppSettings.from_env()
    engine = LiveIntelligenceEngine(settings)

    # Build dummy snapshot bundles
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

    t_recv = datetime.now(UTC) - timedelta(milliseconds=10)
    report = engine.process(snapshot, data_received_at=t_recv)

    assert report.latency is not None
    assert report.latency.data_received_at == t_recv
    assert report.latency.features_computed_at is not None
    assert report.latency.signal_generated_at is not None
    assert report.latency.report_published_at is not None
    assert report.latency.total_latency_ms >= 0.0
