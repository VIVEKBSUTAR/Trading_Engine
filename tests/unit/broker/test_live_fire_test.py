"""Offline contract tests; these do not validate the real Fyers network."""

from __future__ import annotations

import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from threading import current_thread
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

spec = importlib.util.spec_from_file_location(
    "live_fire_probe", Path(__file__).resolve().parents[3] / "live_fire_test.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

SYMBOLS = ["NSE:NIFTY_TEST_CE", "NSE:NIFTY_TEST_PE"]


class ProbeTests(unittest.TestCase):
    def test_credentials_bare_prefixed_and_mismatch(self):
        for value in ("token", "APP-100:token"):
            with patch.dict("os.environ", {"FYERS_CLIENT_ID": "APP-100", "FYERS_ACCESS_TOKEN": value}, clear=True):
                self.assertEqual(probe.load_credentials(), ("APP-100", "token"))
        with patch.dict("os.environ", {"FYERS_CLIENT_ID": "APP-100", "FYERS_ACCESS_TOKEN": "OTHER:token"}, clear=True):
            with self.assertRaises(ValueError):
                probe.load_credentials()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                probe.load_credentials()

    def test_expiry_discovery_selects_nearest_atm_and_requests_expiry(self):
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        expiries = [
            {"date": (today + timedelta(days=days)).strftime("%d-%m-%Y"), "expiry": epoch}
            for days, epoch in [(7, "later"), (-1, "expired"), (0, "current")]
        ]
        rows = [{"symbol": probe.UNDERLYING, "ltp": 24526}]
        for strike in [24500, 24550]:
            for side in ["CE", "PE"]:
                rows.append({"symbol": f"NSE:NIFTY{strike}{side}", "strike_price": strike, "option_type": side})
        client = Mock()
        client.optionchain.side_effect = [
            {"s": "ok", "data": {"expiryData": expiries}},
            {"s": "ok", "data": {"optionsChain": rows}},
        ]
        symbols, expiry, strike = probe.resolve_atm_options(client)
        self.assertEqual(strike, 24550)
        self.assertEqual(expiry, today.isoformat())
        self.assertEqual(symbols, ["NSE:NIFTY24550CE", "NSE:NIFTY24550PE"])
        self.assertEqual(client.optionchain.call_args.kwargs["data"]["timestamp"], "current")

    def socket_factory(self, scenario):
        test = self

        class Socket:
            def __init__(self, **kwargs):
                self.callbacks = kwargs
                self.closed = False
                test.socket = self

            def connect(self):
                self.callbacks["on_connect"]()
                scenario(self.callbacks)

            def subscribe(self, **kwargs):
                test.assertEqual(kwargs, {"symbols": SYMBOLS, "data_type": "SymbolUpdate"})

            def close_connection(self):
                test.assertIs(current_thread(), test.owner)
                self.closed = True
                self.callbacks["on_close"]({"message": "closed"})

        self.owner = current_thread()
        return Socket

    def test_fifty_data_ticks_excludes_control_and_late_messages_then_closes(self):
        def scenario(callbacks):
            callback = callbacks["on_message"]
            callback({"type": "sub", "symbol": SYMBOLS[0], "ltp": 1})
            callback({"type": "if", "symbol": probe.UNDERLYING, "ltp": 24500})
            callback({"type": "sf", "symbol": "NSE:OTHER", "ltp": 1})
            callback(None)
            for index in range(70):
                callback({"type": "sf", "symbol": SYMBOLS[index % 2], "ltp": 100 + index})

        output = io.StringIO()
        with redirect_stdout(output):
            count = probe.stream_ticks(self.socket_factory(scenario), "APP:token", SYMBOLS, 1)
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(count, 50)
        self.assertEqual([line["tick"] for line in lines], list(range(1, 51)))
        self.assertEqual({line["data"]["symbol"] for line in lines}, set(SYMBOLS))
        self.assertTrue(self.socket.closed)
        self.assertFalse(self.socket.callbacks["reconnect"])
        self.assertFalse(self.socket.callbacks["litemode"])

    def test_timeout_error_early_close_and_interrupt_always_cleanup(self):
        def interrupt(_callbacks):
            raise KeyboardInterrupt

        scenarios = [
            (lambda callbacks: None, RuntimeError),
            (lambda callbacks: callbacks["on_error"]({"s": "error"}), RuntimeError),
            (lambda callbacks: callbacks["on_close"]({}), RuntimeError),
            (lambda callbacks: callbacks["on_message"]({"s": "error"}), RuntimeError),
            (interrupt, KeyboardInterrupt),
        ]
        for scenario, error in scenarios:
            with self.subTest(scenario=scenario):
                with self.assertRaises(error):
                    probe.stream_ticks(self.socket_factory(scenario), "APP:token", SYMBOLS, 0.001)
                self.assertTrue(self.socket.closed)

    def test_api_error_fails_without_exposing_response(self):
        client = Mock()
        client.optionchain.return_value = {"s": "error", "message": "sensitive"}
        with self.assertRaisesRegex(RuntimeError, "option-chain request failed") as result:
            probe.resolve_atm_options(client)
        self.assertNotIn("sensitive", str(result.exception))


if __name__ == "__main__":
    unittest.main()
