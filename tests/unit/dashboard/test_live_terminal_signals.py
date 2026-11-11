"""Real database alert window, one-shot notifications, and position hierarchy."""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pandas as pd

from storage.duckdb_manager import DuckDBManager
from trading_engine.dashboard import shared


def insert(manager, signal_id, now, *, seconds=0, action="buy_ce", passed=True, entry=101):
    manager.insert_signal(signal_id=signal_id, timestamp=now - timedelta(seconds=seconds),
        spot_price=24500, strike=24500, expiry="2026-11-17", option_type="CE",
        action=action, confidence=0.9, trap_score=0.1, spread_percentage=0.01,
        iv_percentile=0.5, trade_grade="A", filter_passed=passed, entry_price=entry)


def test_live_query_is_independent_of_fifty_row_audit_and_excludes_stale_future_rejected(tmp_path):
    now = datetime.now(UTC)
    manager = DuckDBManager(tmp_path / "terminal.duckdb")
    insert(manager, "call", now, seconds=30)
    insert(manager, "put-boundary", now, seconds=60, action="BUY_PE")
    insert(manager, "old", now, seconds=61)
    insert(manager, "future", now, seconds=-1)
    insert(manager, "neutral", now, action="no_trade")
    for index in range(55):
        insert(manager, f"rejected-{index}", now, passed=False)
    live = shared.query_live_signals(manager, now=now)
    assert set(live["signal_id"]) == {"call", "put-boundary"}
    assert live["entry_price"].tolist() == [101, 101]
    assert "call" not in shared.query_recent_signals(manager)["signal_id"].tolist()
    manager.close()


def test_toast_is_emitted_once_per_new_batch_and_never_for_expired_signal():
    now = datetime.now(UTC)
    row = {"signal_id": "s1", "timestamp": now.isoformat(), "action": "BUY_CE",
           "strike": 24500, "entry_price": 101.25, "filter_passed": True}
    payload = {"source": "snapshot", "timestamp": now.isoformat(), "live_signals": [row]}
    with patch.object(shared.st, "session_state", {}), patch.object(shared.st, "toast") as toast, \
         patch.object(shared.st, "success") as success, patch.object(shared.st, "caption"):
        shared.render_live_alert(payload)
        shared.render_live_alert(payload)
        assert toast.call_count == 1
        assert "₹101.25" in toast.call_args.args[0]
        assert success.call_count == 2
        row["signal_id"] = "s2"
        shared.render_live_alert(payload)
        assert toast.call_count == 2
        row["timestamp"] = (now - timedelta(seconds=61)).isoformat()
        shared.render_live_alert(payload)
        assert toast.call_count == 2
        assert success.call_count == 3


def test_position_profit_loss_colors_and_bold_stop_are_cell_specific():
    frame = shared.positions_frame([
        {"trade_id": "profit", "strike": 24500, "option_type": "CE", "quantity": 1,
         "entry_ask": 100, "current_bid": 110, "stop_loss": 106},
        {"trade_id": "loss", "strike": 24500, "option_type": "PE", "quantity": 1,
         "entry_ask": 100, "current_bid": 95, "stop_loss": 94},
    ])
    assert frame["Unrealized PnL"].tolist() == [650, -325]
    styles = shared.style_positions(frame)._compute().ctx
    pnl_column = frame.columns.get_loc("Unrealized PnL")
    stop_column = frame.columns.get_loc("Trailing Stop")
    assert ("color", "#00FF00") in styles[(0, pnl_column)]
    assert ("color", "#FF5252") in styles[(1, pnl_column)]
    assert ("font-weight", "700") in styles[(0, stop_column)]
    assert ("font-weight", "700") in styles[(1, stop_column)]


def test_equity_chart_is_chronological_area_with_unified_hover():
    now = datetime.now(UTC)
    frame = pd.DataFrame([{"exit_timestamp": now, "realized_pnl": -100},
                          {"exit_timestamp": now - timedelta(minutes=1), "realized_pnl": 200}])
    figure = shared.equity_figure(frame)
    assert figure.data[0].fill == "tozeroy"
    assert list(figure.data[0].y) == [200, 100]
    assert figure.layout.hovermode == "x unified"
    assert figure.layout.xaxis.showgrid is False


def test_historical_reader_refreshes_inherited_temporary_copy(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    old = Mock(_tmp_copy=tmp_path / "copy.duckdb")
    fresh = Mock(_tmp_copy=None)
    settings = SimpleNamespace(storage=SimpleNamespace(duckdb_path=tmp_path / "source.duckdb"))
    with patch.object(shared.st, "session_state", {"duckdb_mgr": old}), \
         patch.object(shared, "DuckDBManager", return_value=fresh):
        assert shared.get_dashboard_duckdb_manager(settings) is fresh
        old.close.assert_called_once()
