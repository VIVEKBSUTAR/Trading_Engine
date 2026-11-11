"""Two-page terminal hierarchy and live-only alerts without broker connections."""
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[3]
DESK = ROOT / "src/trading_engine/dashboard/1_Execution_Desk.py"


def publication():
    now = datetime.now(UTC)
    return {"timestamp": now.isoformat(), "spot_price": 24500, "vix_value": 14,
        "websocket": {"connected": True, "last_tick_at": now.isoformat(), "safe_mode": False},
        "live_signals": [{"signal_id": "live-ce", "timestamp": now.isoformat(), "action": "buy_ce",
            "strike": 24500, "entry_price": 101, "filter_passed": True}],
        "signals": [{"timestamp": now.isoformat(), "action": "no_trade", "strike": 24500,
            "filter_passed": False, "filter_rejection_reason": "OI imbalance"}],
        "outcomes": [{"exit_timestamp": now.isoformat(), "realized_pnl": 650}],
        "positions": [{"trade_id": "p1", "strike": 24500, "option_type": "CE", "quantity": 1,
            "entry_ask": 100, "current_bid": 110, "stop_loss": 106, "is_frozen": False}]}


def test_execution_desk_has_only_health_alerts_and_active_positions(monkeypatch, tmp_path):
    monkeypatch.setenv("TE_DASHBOARD_SNAPSHOT_DIR", str(tmp_path))
    app = AppTest.from_file(str(DESK)).run(timeout=10)
    assert not app.exception
    assert "Waiting for the engine" in app.info[0].value
    (tmp_path / "latest.json").write_text(json.dumps(publication()))
    app.run(timeout=10)
    assert not app.exception
    assert [metric.label for metric in app.metric] == ["Fyers WebSocket", "NIFTY Spot", "India VIX"]
    assert app.metric[0].value == "Connected"
    assert len(app.success) == 1
    assert "BUY_CE" in app.success[0].value and "₹101.00" in app.success[0].value
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value["Unrealized PnL"].iloc[0] == 650
    assert not app.get("plotly_chart")
    assert not app.expander
    assert [heading.value for heading in app.subheader] == ["Active Positions"]


def test_analytics_navigation_moves_performance_chart_and_rejections_off_desk(monkeypatch, tmp_path):
    monkeypatch.setenv("TE_DASHBOARD_SNAPSHOT_DIR", str(tmp_path))
    (tmp_path / "latest.json").write_text(json.dumps(publication()))
    app = AppTest.from_file(str(DESK)).run(timeout=10)
    app.switch_page("pages/2_Analytics_And_Logs.py").run(timeout=10)
    assert not app.exception
    assert [metric.label for metric in app.metric] == ["Total Profit", "Closed Trades", "Win Rate", "Profit Factor"]
    assert app.metric[0].value == "₹650.00"
    assert len(app.get("plotly_chart")) == 1
    assert not app.success
    assert len(app.dataframe) == 1
    assert "Rejection Reason" in app.expander[0].label
    assert "OI imbalance" in [item.value for item in app.caption]


def test_stale_publication_disables_alerts_and_marks_cached_positions(monkeypatch, tmp_path):
    monkeypatch.setenv("TE_DASHBOARD_SNAPSHOT_DIR", str(tmp_path))
    payload = publication()
    payload["timestamp"] = (datetime.now(UTC) - timedelta(seconds=40)).isoformat()
    (tmp_path / "latest.json").write_text(json.dumps(payload))
    app = AppTest.from_file(str(DESK)).run(timeout=10)
    assert not app.exception
    assert app.metric[0].value == "Stale / paused"
    assert not app.success
    assert app.dataframe[0].value["Data"].iloc[0] == "STALE"


def test_execution_local_database_fallback_shows_no_invented_health(monkeypatch, tmp_path):
    from trading_engine.dashboard import shared
    monkeypatch.delenv("TE_DASHBOARD_SNAPSHOT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shared, "AppSettings", type("Settings", (), {
        "from_env": staticmethod(lambda: type("App", (), {"storage": type("Storage", (), {"duckdb_path": tmp_path / "missing.duckdb"})()})())}))
    result = shared.load_terminal_data()
    assert result["waiting"]
    assert not result.get("websocket")
