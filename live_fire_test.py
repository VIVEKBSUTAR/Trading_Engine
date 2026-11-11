"""Isolated FYERS network probe: print 50 option updates, then disconnect.

Requires FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in the process environment.
No engine imports, token persistence, signal generation, or order operations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from threading import Event, Lock
from time import monotonic
from typing import Any
from zoneinfo import ZoneInfo

UNDERLYING = "NSE:NIFTY50-INDEX"
TICK_LIMIT = 50


def load_credentials() -> tuple[str, str]:
    """Return client ID and bare REST token; accept prefixed socket tokens too."""
    client_id = os.environ.get("FYERS_CLIENT_ID", "").strip()
    token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
    if not client_id or not token:
        raise ValueError("Set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in the environment")
    if ":" in token:
        prefix, token = token.split(":", 1)
        if prefix != client_id or not token:
            raise ValueError("FYERS_ACCESS_TOKEN prefix must match FYERS_CLIENT_ID")
    return client_id, token


def chain_data(client: Any, expiry: str = "") -> dict[str, Any]:
    response = client.optionchain(
        data={"symbol": UNDERLYING, "strikecount": 2, "timestamp": expiry}
    )
    if not isinstance(response, dict) or response.get("s") != "ok":
        # Avoid printing broker payloads that may contain sensitive information.
        raise RuntimeError("Fyers option-chain request failed; check token and data access")
    data = response.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Fyers option-chain response has no data object")
    return data


def resolve_atm_options(client: Any) -> tuple[list[str], str, float]:
    """Discover nearest unexpired expiry and nearest listed strike to index LTP."""
    discovery = chain_data(client)
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    expiries = []
    for item in discovery.get("expiryData", []):
        expiry_day = datetime.strptime(item["date"], "%d-%m-%Y").date()
        if expiry_day >= today:
            expiries.append((expiry_day, str(item["expiry"])))
    if not expiries:
        raise RuntimeError("Fyers returned no current or upcoming NIFTY expiry")
    expiry_day, expiry = min(expiries)
    data = chain_data(client, expiry)
    rows = data.get("optionsChain", [])
    spot = next((float(row["ltp"]) for row in rows if row.get("symbol") == UNDERLYING), 0)
    if not math.isfinite(spot) or spot <= 0:
        raise RuntimeError("Fyers returned no valid NIFTY index price for ATM selection")
    contracts: dict[float, dict[str, str]] = {}
    for row in rows:
        side = row.get("option_type")
        if side not in {"CE", "PE"}:
            continue
        strike = float(row["strike_price"])
        symbol = row.get("symbol", "")
        if math.isfinite(strike) and strike > 0 and symbol.startswith("NSE:NIFTY"):
            contracts.setdefault(strike, {})[side] = symbol
    if not contracts:
        raise RuntimeError("Fyers returned no NIFTY option contracts")
    strike = min(contracts, key=lambda value: (abs(value - spot), value))
    pair = contracts[strike]
    if set(pair) != {"CE", "PE"}:
        raise RuntimeError("Nearest ATM strike does not have both CE and PE contracts")
    return [pair["CE"], pair["PE"]], expiry_day.isoformat(), strike


def stream_ticks(socket_factory: Any, access_token: str, symbols: list[str], timeout: float) -> int:
    """Count only subscribed option data; close on the controlling thread."""
    done = Event()
    lock = Lock()
    count = 0
    failure: str | None = None

    def fail(message: str) -> None:
        nonlocal failure
        with lock:
            if not done.is_set():
                failure = message
                done.set()

    def on_message(message: Any) -> None:
        nonlocal count
        if not isinstance(message, dict):
            return
        if message.get("s") == "error":
            fail("Fyers rejected the websocket request")
            return
        # cn/sub/heartbeat frames are not ticks. Full option updates use sf.
        if message.get("type") != "sf" or message.get("symbol") not in symbols:
            return
        with lock:
            if done.is_set():
                return
            count += 1
            print(json.dumps({"tick": count, "data": message}), flush=True)
            if count == TICK_LIMIT:
                done.set()

    def on_connect() -> None:
        try:
            socket.subscribe(symbols=symbols, data_type="SymbolUpdate")
        except Exception:
            fail("Fyers websocket subscription failed")

    def on_error(*_args: Any) -> None:
        fail("Fyers websocket error; check credentials, connectivity, and subscription access")

    def on_close(*_args: Any) -> None:
        fail("Fyers websocket closed before fifty ticks arrived")

    socket = socket_factory(
        access_token=access_token,
        log_path="",
        litemode=False,
        write_to_file=False,
        reconnect=False,
        on_connect=on_connect,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    started = monotonic()
    try:
        # The SDK creates its own worker threads. Do not call keep_running(),
        # which creates an additional loop; our Event keeps this process alive.
        socket.connect()
        if not done.wait(max(0, timeout - (monotonic() - started))):
            fail(f"Timed out after {timeout:g}s with {count}/{TICK_LIMIT} ticks")
    finally:
        # Gate late callbacks and use the SDK's actual shutdown method. Calling
        # this from a callback would try to join the websocket's own thread.
        with lock:
            done.set()
        socket.close_connection()
    if failure:
        raise RuntimeError(failure)
    if count != TICK_LIMIT:
        raise RuntimeError(f"Incomplete stream: {count}/{TICK_LIMIT} ticks")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=120, help="Websocket timeout in seconds (default: 120)")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    try:
        client_id, token = load_credentials()
        from fyers_apiv3 import fyersModel
        from fyers_apiv3.FyersWebsocket import data_ws

        client = fyersModel.FyersModel(client_id=client_id, token=token, is_async=False, log_path="")
        symbols, expiry, strike = resolve_atm_options(client)
        print(f"NIFTY expiry={expiry} ATM={strike:g} symbols={','.join(symbols)}", file=sys.stderr)
        stream_ticks(data_ws.FyersDataSocket, f"{client_id}:{token}", symbols, args.timeout)
    except KeyboardInterrupt:
        print("Interrupted; websocket cleanup requested.", file=sys.stderr)
        return 130
    except Exception as exc:
        # SDK exceptions can include auth material; print only our known errors.
        message = str(exc) if type(exc) in {ValueError, RuntimeError} else type(exc).__name__
        print(f"Live fire test failed: {message}", file=sys.stderr)
        return 1
    print("Received exactly 50 option ticks; websocket closed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
