"""A rejected packet must never close the socket or poison the next valid tick."""
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from loguru import logger

from broker.api_security import APISecurityError, APISecurityGuard
from broker.fyers_stream import FyersStream
from broker.normalizer import FyersTickNormalizer
from config.settings import FyersSettings, SecuritySettings

SYMBOL = "NSE:NIFTY26NOV25000CE"


def tick(**changes):
    return {"type": "sf", "symbol": SYMBOL, "ltp": 100.0,
            "bid_price": 99.9, "ask_price": 100.1, "vol_traded_today": 0,
            "exch_feed_time": datetime.now(UTC).timestamp(), **changes}


@pytest.fixture
def stream(monkeypatch):
    socket = Mock()
    monkeypatch.setattr("broker.fyers_stream.data_ws.FyersDataSocket", Mock(return_value=socket))
    instance = FyersStream(FyersSettings(client_id="APP-100", access_token="test-token"),
        auth_manager=SimpleNamespace(resolve_access_token=lambda: "test-token"),
        security_guard=APISecurityGuard(SecuritySettings(environment="paper")), aggregator=Mock())
    instance.register_symbol_token(SYMBOL, 12345)
    instance.state.connected = True
    instance.stop = Mock()
    return instance


@pytest.fixture
def warnings():
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="WARNING")
    yield records
    logger.remove(sink)


def assert_alive(instance):
    instance.stop.assert_not_called()
    instance._socket.close_connection.assert_not_called()
    instance._socket.close.assert_not_called()
    assert instance.state.connected
    assert not instance._stop_event.is_set()


@pytest.mark.parametrize("bad", [None, 42, "{broken", [], [None], {"type": "sf"},
    tick(bid_price=None), tick(ask_price=None), tick(vol_traded_today=None),
    tick(bid_price="bad"), tick(ask_price=float("nan")), tick(ltp=float("inf")),
    tick(bid_price=101, ask_price=100), tick(vol_traded_today=-1),
    tick(exch_feed_time="not-a-timestamp"), {"type": "sub", "s": "error"},
])
def test_bad_packet_warns_then_next_valid_tick_is_delivered(stream, warnings, bad):
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(bad)
    consumer.assert_not_called()
    stream.aggregator.push_option_tick.assert_not_called()
    assert stream.state.last_tick_at is None
    assert stream._security_guard.health_snapshot().last_stream_at is None
    assert stream.state.dropped_packets == 1
    assert warnings and all(record["level"].name == "WARNING" for record in warnings)
    assert not stream._security_guard.safe_mode
    assert_alive(stream)
    stream._on_message(tick())
    consumer.assert_called_once()
    stream.aggregator.push_option_tick.assert_called_once()
    assert consumer.call_args.args[0].iloc[0]["volume_traded"] == 0
    assert stream.state.last_tick_at is not None
    assert_alive(stream)


@pytest.mark.parametrize("message", [{"type": "cn"}, {"type": "sub"}, {"type": "unsub"},
                                     {"type": "hb"}, "pong"])
def test_control_packets_are_heartbeats_not_market_ticks(stream, warnings, message):
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(message)
    assert stream.state.last_heartbeat_at is not None
    assert stream.state.last_tick_at is None
    consumer.assert_not_called()
    assert not warnings
    assert_alive(stream)


def test_index_without_bid_ask_volume_is_preserved_without_invented_quotes(stream):
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message({"type": "if", "symbol": "NSE:NIFTY50-INDEX", "ltp": 25000})
    consumer.assert_called_once()
    frame = consumer.call_args.args[0]
    assert frame.iloc[0]["last_price"] == 25000
    assert pd.isna(frame.iloc[0]["bid_price"])
    assert pd.isna(frame.iloc[0]["ask_price"])
    stream.aggregator.push_option_tick.assert_not_called()
    assert_alive(stream)


def test_json_bytes_aliases_and_zero_volume(stream):
    consumer = Mock()
    stream.register_tick_handler(consumer)
    message = {"symbol": SYMBOL, "lp": 100, "bid": 99, "ask": 101, "vol": 0,
               "exch_feed_time": str(int(datetime.now(UTC).timestamp()))}
    stream._on_message(json.dumps(message).encode())
    frame = consumer.call_args.args[0]
    assert frame.iloc[0]["bid_price"] == 99
    assert frame.iloc[0]["volume_traded"] == 0
    assert_alive(stream)


def test_stale_then_fresh_packet_does_not_latch_safe_mode(stream, warnings):
    stream._on_message(tick())
    stream.aggregator.reset_mock()
    last_stream_at = stream._security_guard.health_snapshot().last_stream_at
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(tick(exch_feed_time=(datetime.now(UTC) - timedelta(minutes=2)).timestamp()))
    assert not stream._security_guard.safe_mode
    assert stream._security_guard.health_snapshot().last_stream_at == last_stream_at
    consumer.assert_not_called()
    stream.aggregator.push_option_tick.assert_not_called()
    stream._on_message(tick())
    consumer.assert_called_once()
    assert_alive(stream)
    assert any("Stale websocket" in record["message"] for record in warnings)


def test_existing_global_safe_mode_is_not_cleared_by_parser(stream):
    stream._security_guard.mark_safe_mode("Independent authentication policy")
    stream._on_message(tick())
    assert stream._security_guard.safe_mode
    assert stream.state.safe_mode
    assert_alive(stream)


def test_mixed_invalid_batch_is_rejected_before_any_state_write(stream):
    stream._on_message([tick(), tick(ask_price=None)])
    stream.aggregator.push_option_tick.assert_not_called()
    assert stream.state.last_tick_at is None
    stream._on_message(tick())
    stream.aggregator.push_option_tick.assert_called_once()
    assert_alive(stream)


@pytest.mark.parametrize("stage", ["normalizer", "integrity", "aggregator"])
def test_unexpected_exception_is_contained_and_warning_does_not_leak_payload(stream, warnings, stage, monkeypatch):
    if stage == "normalizer":
        original = stream.normalizer.normalize_ticks
        monkeypatch.setattr(stream.normalizer, "normalize_ticks", Mock(side_effect=ValueError("sensitive-value")))
    elif stage == "integrity":
        original = stream._security_guard.validate_stream_frame
        monkeypatch.setattr(stream._security_guard, "validate_stream_frame", Mock(side_effect=APISecurityError("Rejected test packet")))
    else:
        stream.aggregator.push_option_tick.side_effect = RuntimeError("sensitive-value")
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(tick())
    consumer.assert_not_called()
    assert warnings
    assert all("sensitive-value" not in record["message"] for record in warnings)
    assert_alive(stream)
    if stage == "normalizer":
        monkeypatch.setattr(stream.normalizer, "normalize_ticks", original)
    elif stage == "integrity":
        monkeypatch.setattr(stream._security_guard, "validate_stream_frame", original)
    else:
        stream.aggregator.push_option_tick.side_effect = None
    stream._on_message(tick())
    consumer.assert_called_once()
    assert_alive(stream)


def test_consumer_failure_does_not_block_other_consumers_or_future_packets(stream, warnings):
    good = Mock()
    stream.register_raw_tick_handler(Mock(side_effect=RuntimeError("sensitive-value")))
    stream.register_tick_handler(Mock(side_effect=RuntimeError("sensitive-value")))
    stream.register_tick_handler(good)
    stream._on_message(tick())
    stream._on_message(tick())
    assert good.call_count == 2
    assert all(record["level"].name == "WARNING" for record in warnings)
    assert all("sensitive-value" not in record["message"] for record in warnings)
    assert_alive(stream)


def test_normalizer_does_not_synthesize_quotes_and_handles_epoch_strings():
    normalizer = FyersTickNormalizer()
    result = normalizer.normalize_tick({"symbol": SYMBOL, "ltp": 100, "vol": 0,
                                      "exch_feed_time": "1794456000"})
    assert result["bid_price"] is None and result["ask_price"] is None
    assert result["volume_traded"] == 0
    assert result["timestamp"] == pd.to_datetime(1794456000, unit="s", utc=True)


def test_old_last_trade_time_does_not_replace_fresh_quote_arrival_time(stream):
    message = tick(last_traded_time=1700000000)
    del message["exch_feed_time"]
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(message)
    consumer.assert_called_once()
    assert_alive(stream)


@pytest.mark.parametrize("case", ["future", "duplicate", "out_of_order"])
def test_integrity_rejection_is_packet_local_and_fresh_data_recovers(stream, case):
    if case == "future":
        stream._on_message(tick())
        stream.aggregator.reset_mock()
    now = datetime.now(UTC)
    if case == "future":
        message = tick(exch_feed_time=(now + timedelta(hours=1)).timestamp())
    elif case == "duplicate":
        row = tick()
        message = [row.copy() for _ in range(4)]
    else:
        message = [tick(exch_feed_time=(now - timedelta(seconds=index)).timestamp()) for index in range(4)]
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(message)
    consumer.assert_not_called()
    stream.aggregator.push_option_tick.assert_not_called()
    assert not stream._security_guard.safe_mode
    assert stream.state.dropped_packets == 1
    assert_alive(stream)
    stream._on_message(tick())
    consumer.assert_called_once()
    assert_alive(stream)


@pytest.fixture
def integrity_clock(monkeypatch):
    now = datetime(2026, 11, 12, 6, 0, tzinfo=UTC)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    monkeypatch.setattr("broker.api_security.datetime", FixedDateTime)
    monkeypatch.setattr("broker.fyers_stream.datetime", FixedDateTime)
    monkeypatch.setattr("broker.normalizer.datetime", FixedDateTime)
    return now


@pytest.mark.parametrize("age,accepted", [(0, True), (1.999, True), (2, True), (2.001, False), (5, False)])
def test_normalized_two_second_boundary(stream, warnings, integrity_clock, age, accepted):
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=20)).timestamp()))
    offset = stream._clock_offset
    assert offset == timedelta(seconds=20)
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=20 + age)).timestamp()))
    assert bool(consumer.call_count) is accepted
    assert stream._clock_offset == offset  # stale packets must not recalibrate
    assert stream.state.dropped_packets == (0 if accepted else 1)
    if accepted:
        assert warnings == []
    else:
        assert any("Stale websocket" in record["message"] for record in warnings)
    assert_alive(stream)


def test_calibration_rejects_stale_rows_in_mixed_packet(stream, integrity_clock):
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message([
        tick(exch_feed_time=(integrity_clock - timedelta(seconds=23)).timestamp()),
        tick(exch_feed_time=(integrity_clock - timedelta(seconds=20)).timestamp()),
    ])
    consumer.assert_not_called()
    assert stream._clock_offset is None
    assert stream.state.dropped_packets == 1
    assert_alive(stream)


def test_generic_integrity_policy_keeps_configured_tolerance(stream, integrity_clock):
    frame = FyersTickNormalizer().normalize_ticks([
        tick(exch_feed_time=(integrity_clock - timedelta(seconds=21)).timestamp())], source="fyers").frame
    with pytest.raises(APISecurityError, match="Stale websocket"):
        stream._security_guard.validate_stream_frame(frame, packet_only=True)


@pytest.mark.parametrize("drift", [20, -20, 0, 120])
def test_first_valid_packet_calibrates_signed_clock_offset(stream, warnings, integrity_clock, drift):
    assert stream._clock_offset is None
    stamp = integrity_clock - timedelta(seconds=drift)
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(tick(exch_feed_time=stamp.timestamp()))
    assert stream._clock_offset == timedelta(seconds=drift)
    assert consumer.call_args.args[0].timestamp.iloc[0] == pd.Timestamp(stamp)
    assert not warnings
    assert_alive(stream)


def test_bad_packets_and_heartbeats_cannot_calibrate(stream, integrity_clock):
    stream._on_message({"type": "hb"})
    stream._on_message(tick(ask_price=None, exch_feed_time=integrity_clock.timestamp()))
    stream._on_message([tick(exch_feed_time=integrity_clock.timestamp())] * 4)
    assert stream._clock_offset is None
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=20)).timestamp()))
    assert stream._clock_offset == timedelta(seconds=20)


def test_receipt_time_does_not_calibrate_or_poison_exchange_offset(stream, integrity_clock):
    packet = tick()
    del packet["exch_feed_time"]
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(packet)
    assert stream._clock_offset is None
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=40)).timestamp()))
    assert stream._clock_offset == timedelta(seconds=40)
    stream._on_message(packet)
    assert consumer.call_count == 3
    assert_alive(stream)


def test_reconnect_resets_calibration(stream, integrity_clock):
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=20)).timestamp()))
    stream._on_connect()
    assert stream._clock_offset is None
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=30)).timestamp()))
    assert stream._clock_offset == timedelta(seconds=30)


def test_stale_filter_runs_even_without_security_guard(stream, integrity_clock):
    stream._security_guard = None
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=20)).timestamp()))
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=23)).timestamp()))
    consumer.assert_not_called()
    assert stream.state.dropped_packets == 1
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=20)).timestamp()))
    consumer.assert_called_once()
    assert_alive(stream)


def test_elapsed_local_time_uses_fixed_offset_for_later_packets(stream, integrity_clock, monkeypatch):
    stream._on_message(tick(exch_feed_time=(integrity_clock - timedelta(seconds=20)).timestamp()))
    later = integrity_clock + timedelta(minutes=1)
    clock = Mock()
    clock.now.return_value = later
    monkeypatch.setattr("broker.fyers_stream.datetime", clock)
    consumer = Mock()
    stream.register_tick_handler(consumer)
    stream._on_message(tick(exch_feed_time=(later - timedelta(seconds=20)).timestamp()))
    consumer.assert_called_once()
    assert stream._clock_offset == timedelta(seconds=20)
    assert_alive(stream)
