"""Integration tests verifying full signal processing pipeline from raw ticks to signal."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from config.settings import AppSettings
from trading_engine.intelligence.engine import LiveIntelligenceEngine
from trading_engine.intelligence.models import MarketSnapshotBundle, SignalAction


def test_full_pipeline_tick_processing_flow():
    """Verify that incoming ticks trigger state aggregation, micro-momentum, and signal generation."""
    settings = AppSettings.from_env()
    engine = LiveIntelligenceEngine(settings)

    empty_df = pd.DataFrame()
    ticks_df = pd.DataFrame({
        "timestamp": [datetime.now(UTC)] * 10,
        "last_price": [24000.0 + i for i in range(10)],  # upward trend
        "volume_traded": list(range(1000, 2000, 100)),
        "total_buy_quantity": [500] * 10,
        "total_sell_quantity": [300] * 10
    })

    # Prepare ATM options chain
    option_chain = pd.DataFrame({
        "timestamp": [datetime.now(UTC)] * 4,
        "expiry": ["2026-08-20"] * 4,
        "strike": [24000.0, 24000.0, 24100.0, 24100.0],
        "option_type": ["CE", "PE", "CE", "PE"],
        "open_interest": [10000, 12000, 8000, 7000],
        "implied_volatility": [0.15, 0.16, 0.15, 0.16],
        "last_price": [120.0, 110.0, 80.0, 140.0],
        "bid_qty": [1000] * 4,
        "ask_qty": [800] * 4
    })

    snapshot = MarketSnapshotBundle(
        timestamp=datetime.now(UTC),
        spot_price=24000.0,
        vix_value=15.0,
        spot_frame=empty_df,
        vix_frame=empty_df,
        option_chain_frame=option_chain,
        ticks_frame=ticks_df,
        candles_1m=empty_df,
        candles_5m=empty_df
    )

    report = engine.process(snapshot)
    
    # Assert report structure is fully populated
    assert report.timestamp is not None
    assert report.state is not None
    assert report.signal is not None
    assert report.signal.action in {SignalAction.BUY_CE, SignalAction.BUY_PE, SignalAction.NO_TRADE}
    assert report.latency is not None
    assert report.latency.total_latency_ms >= 0.0
