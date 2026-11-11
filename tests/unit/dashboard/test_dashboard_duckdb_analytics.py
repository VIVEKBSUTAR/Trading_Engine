"""Unit and integration tests for dashboard DuckDB analytics, equity curve, and active positions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config.settings import AppSettings
from storage.duckdb_manager import DuckDBManager
from trading_engine.dashboard.shared import (
    compute_equity_curve,
    compute_performance_metrics,
    compute_unrealized_pnl,
    get_dashboard_duckdb_manager,
    query_recent_signals,
)
from trading_engine.intelligence.models import (
    MarketRegime,
    OpenTrade,
    OptionSide,
    ProbabilityResult,
    SignalAction,
    SignalResult,
    StrikeResult,
    TradeGrade,
    TrapResult,
)
from trading_engine.intelligence.monitor import NIFTY_LOT_SIZE, LiveTradeMonitor, MonitorConfig


def test_step1_compute_performance_metrics_empty():
    """Step 1: Metric computation handles empty or missing trade outcomes safely."""
    empty_df = pd.DataFrame()
    metrics = compute_performance_metrics(empty_df)
    assert metrics["total_realized_profit"] == 0.0
    assert metrics["total_closed_trades"] == 0
    assert metrics["win_rate_pct"] == 0.0
    assert metrics["profit_factor"] == 0.0


def test_step1_compute_performance_metrics_with_trades():
    """Step 1: Metric computation accurately calculates total P&L, win rate, and profit factor."""
    trades = pd.DataFrame([
        {"realized_pnl": 5000.0},
        {"realized_pnl": -2000.0},
        {"realized_pnl": 3000.0},
        {"realized_pnl": -1000.0},
    ])
    metrics = compute_performance_metrics(trades)
    assert metrics["total_realized_profit"] == 5000.0
    assert metrics["total_closed_trades"] == 4
    assert metrics["win_rate_pct"] == 50.0  # 2 wins out of 4
    assert metrics["profit_factor"] == pytest.approx(8000.0 / 3000.0)  # 2.6667


def test_step1_compute_performance_metrics_all_wins():
    """Step 1: Profit factor is infinity when there are no losing trades."""
    trades = pd.DataFrame([
        {"realized_pnl": 1500.0},
        {"realized_pnl": 2500.0},
    ])
    metrics = compute_performance_metrics(trades)
    assert metrics["total_realized_profit"] == 4000.0
    assert metrics["total_closed_trades"] == 2
    assert metrics["win_rate_pct"] == 100.0
    assert metrics["profit_factor"] == float("inf")


def test_step2_compute_equity_curve_empty():
    """Step 2: Equity curve returns empty DataFrame when trade outcomes is empty."""
    empty_df = pd.DataFrame()
    curve = compute_equity_curve(empty_df)
    assert curve.empty
    assert list(curve.columns) == ["exit_timestamp", "cumulative_profit"]


def test_step2_compute_equity_curve_chronological_cum_pnl():
    """Step 2: Equity curve is sorted chronologically and calculates cumulative profit."""
    t0 = datetime(2026, 10, 16, 9, 30, tzinfo=UTC)
    t1 = datetime(2026, 10, 16, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 10, 16, 10, 30, tzinfo=UTC)

    # Insert out-of-order to verify sorting
    trades = pd.DataFrame([
        {"exit_timestamp": t2, "realized_pnl": 2000.0},
        {"exit_timestamp": t0, "realized_pnl": 1000.0},
        {"exit_timestamp": t1, "realized_pnl": -500.0},
    ])
    curve = compute_equity_curve(trades)
    assert len(curve) == 3
    assert list(curve["exit_timestamp"]) == [t0, t1, t2]
    assert list(curve["cumulative_profit"]) == [1000.0, 500.0, 2500.0]


def _make_dummy_signal(stop_loss: float = 120.0) -> SignalResult:
    return SignalResult(
        action=SignalAction.BUY_CE,
        confidence=0.85,
        regime=MarketRegime.TREND_BULLISH,
        probability=ProbabilityResult(0.85, 0.10, 0.05, 0.85, 4),
        trap=TrapResult(0.1, 0.1, 0.1, 0.1),
        entry_reference=150.0,
        stop_loss=stop_loss,
        target=180.0,
        strike=StrikeResult(OptionSide.CE, 24500.0, "2026-10-22", "ATM"),
        quantity=2,
        trade_grade=TradeGrade.A_PLUS,
    )


def test_step3_compute_unrealized_pnl_with_65_multiplier():
    """Step 3: Unrealized profit = (current_bid - entry_ask) * 65 * quantity."""
    assert NIFTY_LOT_SIZE == 65

    dummy_signal = _make_dummy_signal(stop_loss=120.0)

    trade = OpenTrade(
        trade_id="LT-00001",
        signal=dummy_signal,
        entry_price=150.0,
        quantity=2,  # 2 lots
        opened_at=datetime.now(UTC),
        strike=24500.0,
        option_type="CE",
        entry_ask_price=150.0,
        current_bid_price=170.0,  # +20 points gain
    )

    pnl = compute_unrealized_pnl(trade)
    # (170 - 150) * 65 * 2 = 20 * 65 * 2 = 2600.0
    assert pnl == 2600.0


def test_step3_frozen_trade_detection(option_history):
    """Step 3: Stale ticks mark trade as is_frozen."""
    dummy_signal = _make_dummy_signal(stop_loss=100.0)
    monitor = LiveTradeMonitor(config=MonitorConfig())
    trade = monitor.open_trade(
        dummy_signal,
        entry_price=120.0,
        quantity=1,
        instrument_token=9999,
        strike=24500.0,
        option_type="CE",
        entry_ask_price=120.0,
        option_ticks=option_history(datetime.now(UTC)),
    )
    assert bool(getattr(trade, "is_frozen", False)) is False

    # Stale tick (> 2.0s gap)
    now = datetime.now(UTC)
    stale_ts = now - timedelta(seconds=10)
    monitor.update({9999: {"bid": 125.0, "ask": 127.0, "timestamp": stale_ts}})

    assert trade.is_frozen is True


def test_step4_query_recent_signals_from_duckdb(tmp_path: Path):
    """Step 4: Query recent fifty records from signals table."""
    db_file = tmp_path / "test_signals.duckdb"
    duckdb_mgr = DuckDBManager(db_file)

    now = datetime.now(UTC)
    # Insert 60 signals (both accepted and rejected)
    for i in range(60):
        passed = (i % 2 == 0)
        rejection = None if passed else "liquidity filter: spread exceeds max 0.05"
        duckdb_mgr.insert_signal(
            signal_id=f"SIG-{i:04d}",
            timestamp=now + timedelta(seconds=i),
            spot_price=24000.0 + i,
            strike=24000.0,
            expiry="2026-10-22",
            option_type="CE",
            action="BUY_CE" if passed else "NO_TRADE",
            confidence=0.75,
            trap_score=0.20,
            spread_percentage=0.02 if passed else 0.08,
            iv_percentile=0.50,
            trade_grade="A" if passed else "AVOID",
            filter_passed=passed,
            filter_rejection_reason=rejection,
        )

    signals_df = query_recent_signals(duckdb_mgr, limit=50)
    assert len(signals_df) == 50
    assert "timestamp" in signals_df.columns
    assert "filter_passed" in signals_df.columns
    assert "rejection_reason" in signals_df.columns

    duckdb_mgr.close()


def test_step5_duckdb_read_only_manager(tmp_path: Path):
    """Step 5: DuckDBManager supports read_only=True without taking write lock."""
    db_file = tmp_path / "test_read_only.duckdb"

    # Initialize write manager
    writer = DuckDBManager(db_file)
    writer.insert_signal(
        signal_id="SIG-0001",
        timestamp=datetime.now(UTC),
        spot_price=24500.0,
        strike=24500.0,
        expiry="2026-10-22",
        option_type="CE",
        action="BUY_CE",
        confidence=0.85,
        trap_score=0.15,
        spread_percentage=0.015,
        iv_percentile=0.45,
        trade_grade="A+",
        filter_passed=True,
    )

    # Initialize read-only manager
    reader = DuckDBManager(db_file, read_only=True)
    df = reader.get_signals()
    assert len(df) == 1
    assert df.iloc[0]["signal_id"] == "SIG-0001"

    reader.close()
    writer.close()


def test_step5_get_dashboard_duckdb_manager_caches_in_session_state(tmp_path: Path):
    """Step 5: DuckDBManager is reused across Streamlit reruns via session state."""
    db_file = tmp_path / "session_test.duckdb"
    settings = AppSettings.from_env()
    settings.storage.duckdb_path = db_file

    with patch.dict("trading_engine.dashboard.shared.st.session_state", {}, clear=True):
        mgr1 = get_dashboard_duckdb_manager(settings)
        mgr2 = get_dashboard_duckdb_manager(settings)
        assert mgr1 is mgr2
        mgr1.close()


def test_render_performance_banner_and_equity_curve():
    """Verify performance banner and equity curve render without exception."""
    from trading_engine.dashboard.shared import (
        _render_equity_curve_section,
        _render_performance_banner,
    )

    metrics = {
        "total_realized_profit": 15000.0,
        "total_closed_trades": 8,
        "win_rate_pct": 75.0,
        "profit_factor": 3.5,
    }
    col_mocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    with patch("streamlit.columns", return_value=col_mocks), patch("streamlit.markdown"), patch("streamlit.plotly_chart") as mock_chart:
        _render_performance_banner(metrics)
        assert col_mocks[0].metric.called
        assert col_mocks[1].metric.called
        assert col_mocks[2].metric.called
        assert col_mocks[3].metric.called

        trades = pd.DataFrame([
            {"exit_timestamp": datetime.now(UTC), "realized_pnl": 5000.0},
        ])
        _render_equity_curve_section(trades)
        assert mock_chart.called


def test_render_active_positions_surfaces_frozen_alert():
    """Step 3: Frozen positions display alert badge and error message."""
    from trading_engine.dashboard.shared import _render_active_positions

    dummy_signal = _make_dummy_signal()
    trade = OpenTrade(
        trade_id="LT-FROZEN",
        signal=dummy_signal,
        entry_price=100.0,
        quantity=1,
        opened_at=datetime.now(UTC),
        strike=24000.0,
        option_type="CE",
        entry_ask_price=100.0,
        current_bid_price=110.0,
        is_frozen=True,
    )
    monitor = MagicMock()
    monitor.open_trades = {"LT-FROZEN": trade}

    with patch("streamlit.error") as mock_error, patch("streamlit.dataframe") as mock_df, patch("streamlit.metric"):
        _render_active_positions(monitor, None)
        assert mock_error.called
        assert "UNRESOLVABLE DATA ALERT" in mock_error.call_args[0][0]
        assert mock_df.called


def test_render_signal_log_section_displays_table_and_rejection_expander(tmp_path: Path):
    """Step 4: Signal log renders table with rejection styling and review expander."""
    from trading_engine.dashboard.shared import _render_signal_log_section

    db_file = tmp_path / "test_sig_render.duckdb"
    duckdb_mgr = DuckDBManager(db_file)
    duckdb_mgr.insert_signal(
        signal_id="SIG-REJ-01",
        timestamp=datetime.now(UTC),
        spot_price=24200.0,
        strike=24200.0,
        expiry="2026-10-22",
        option_type="PE",
        action="NO_TRADE",
        confidence=0.60,
        trap_score=0.45,
        spread_percentage=0.08,
        iv_percentile=0.92,
        trade_grade="AVOID",
        filter_passed=False,
        filter_rejection_reason="liquidity filter: spread 8.0% exceeds max 5.0%",
    )

    with patch("streamlit.dataframe") as mock_df, patch("streamlit.expander") as mock_expander:
        _render_signal_log_section(duckdb_mgr)
        assert mock_df.called
        assert mock_expander.called

    duckdb_mgr.close()
