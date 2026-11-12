"""IST presentation preserves UTC instants and leaves live freshness math intact."""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from trading_engine.dashboard import shared


@pytest.mark.parametrize("value,expected", [
    ("2026-11-12T06:00:07Z", "12 Nov 2026 11:30:07 AM IST"),
    ("2026-11-12 18:30:09", "13 Nov 2026 12:00:09 AM IST"),
    ("2026-11-12T06:30:01+00:00", "12 Nov 2026 12:00:01 PM IST"),
    ("2026-11-12T11:30:07+05:30", "12 Nov 2026 11:30:07 AM IST"),
    (datetime(2026, 11, 12, 6, 0, 7), "12 Nov 2026 11:30:07 AM IST"),
    (None, "—"), ("bad timestamp", "—"),
])
def test_ist_formats_naive_utc_and_aware_timestamps(value, expected):
    assert shared.format_ist_timestamp(value) == expected


def test_mixed_dataframe_timestamps_do_not_mutate_source():
    source = pd.Series(["2026-11-12T06:00:07Z", "2026-11-12 18:30:09", None])
    original = source.copy()
    assert shared.format_ist_timestamps(source).tolist() == [
        "12 Nov 2026 11:30:07 AM IST", "13 Nov 2026 12:00:09 AM IST", "—"]
    pd.testing.assert_series_equal(source, original)


def test_positions_and_telemetry_share_ist_format():
    stamp = "2026-11-12T06:00:07Z"
    frame = shared.positions_frame([{"opened_at": stamp, "signal_timestamp": stamp}])
    assert frame["Entry Time"].iloc[0] == "12 Nov 2026 11:30:07 AM IST"
    assert frame["Signal Time"].iloc[0] == frame["Entry Time"].iloc[0]
    with patch.object(shared.st, "caption") as caption:
        shared.render_telemetry({"timestamp": stamp, "websocket": {"last_tick_at": stamp}})
        assert caption.call_args.args[0].count("12 Nov 2026 11:30:07 AM IST") == 2
        assert "+00:00" not in caption.call_args.args[0]


def test_audit_table_and_rejection_details_format_ist():
    signals = pd.DataFrame([{"timestamp": "2026-11-12T06:00:07Z", "action": "no_trade",
                             "filter_passed": False, "filter_rejection_reason": "spread"}])
    with patch.object(shared.st, "dataframe") as table, patch.object(shared.st, "write") as write, \
         patch.object(shared.st, "subheader"), patch.object(shared.st, "caption"), \
         patch.object(shared.st, "expander"):
        shared.render_audit_log(signals)
        assert table.call_args.args[0].timestamp.iloc[0] == "12 Nov 2026 11:30:07 AM IST"
        assert "12 Nov 2026 11:30:07 AM IST" in write.call_args.args[0]
    assert signals.timestamp.iloc[0] == "2026-11-12T06:00:07Z"


def test_alert_formatting_preserves_utc_freshness_window():
    now = datetime.now(UTC)
    events = [{"event_id": str(age), "timestamp": (now - timedelta(seconds=age)).isoformat(),
               "alert_type": "ENTRY_BUY", "action": "buy_ce", "strike": 24500, "premium": 100}
              for age in (1, 61)]
    with patch.object(shared.st, "session_state", {}), patch.object(shared.st, "success") as success, \
         patch.object(shared.st, "toast"), patch.object(shared.st, "caption"):
        shared.render_live_alert({"source": "snapshot", "timestamp": now.isoformat(), "lifecycle_alerts": events})
        assert success.call_count == 1
        assert shared.format_ist_timestamp(events[0]["timestamp"]) in success.call_args.args[0]
    assert shared.publication_age({"timestamp": now.isoformat()}, now=now) == 0


def test_equity_hover_uses_ist_wall_clock_without_offsets():
    figure = shared.equity_figure(pd.DataFrame([{"exit_timestamp": "2026-11-12T06:00:07Z", "realized_pnl": 10}]))
    assert figure.data[0].customdata[0] == "12 Nov 2026 11:30:07 AM IST"
    assert pd.Timestamp(figure.data[0].x[0]) == pd.Timestamp("2026-11-12 11:30:07")
    assert "IST" in figure.layout.xaxis.tickformat
