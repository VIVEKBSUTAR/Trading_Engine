"""The dashboard consumes one published document without opening engine DuckDB."""
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from trading_engine.cli.intelligence import publish_dashboard_snapshot
from trading_engine.intelligence.models import SignalAction


def test_dashboard_publication_replaces_complete_document_and_removes_temporary_files(tmp_path):
    report = SimpleNamespace(timestamp=datetime.now(UTC), regime=SimpleNamespace(regime=SignalAction.NO_TRADE),
        signal=SimpleNamespace(action=SignalAction.BUY_CE, confidence=0.9, reasoning=["validated"]),
        snapshot=SimpleNamespace(spot_price=24500, vix_value=14))
    store = Mock()
    store.get_signals.return_value = pd.DataFrame([{"signal_id": "s1", "max_holding_minutes": 15}])
    store.get_trade_outcomes.return_value = pd.DataFrame()
    store.get_live_signals.return_value = pd.DataFrame([{"signal_id": "s1", "entry_price": 101.0}])
    runtime = SimpleNamespace(engine=SimpleNamespace(duckdb_manager=store), monitor=SimpleNamespace(open_trades=[]),
        stream=SimpleNamespace(state=SimpleNamespace(connected=True, last_tick_at=datetime.now(UTC), safe_mode=False)))
    publish_dashboard_snapshot(tmp_path, report, runtime)
    first = json.loads((tmp_path / "latest.json").read_text())
    assert first["signals"][0]["max_holding_minutes"] == 15
    assert first["action"] == "buy_ce"
    assert first["spot_price"] == 24500
    assert first["websocket"]["connected"] is True
    assert first["live_signals"][0]["entry_price"] == 101.0
    report.signal.action = SignalAction.NO_TRADE
    publish_dashboard_snapshot(tmp_path, report, runtime)
    second = json.loads((tmp_path / "latest.json").read_text())
    assert second["action"] == "no_trade"
    assert first["action"] == "buy_ce"
    assert first["spot_price"] == 24500
    assert first["websocket"]["connected"] is True
    assert first["live_signals"][0]["entry_price"] == 101.0
    assert list(tmp_path.iterdir()) == [tmp_path / "latest.json"]
