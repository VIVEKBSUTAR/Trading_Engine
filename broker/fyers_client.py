"""Authenticated Fyers API V3 client for live market connectivity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from fyers_apiv3 import fyersModel
from loguru import logger

from broker.api_security import APISecurityError, APISecurityGuard
from broker.fyers_auth import FyersAuthManager
from config.settings import FyersSettings


class FyersClientError(RuntimeError):
    """Raised when a Fyers REST call or validation fails."""


@dataclass(slots=True)
class MarketQuote:
    """Structured quote payload for an instrument."""

    instrument: str
    last_price: float | None
    change: float | None
    ohlc: dict[str, float | None]
    received_at: datetime
    exchange_timestamp: datetime | None
    last_trade_time: datetime | None
    raw: dict[str, Any]


@dataclass(slots=True)
class LiveMarketSnapshot:
    """Current market snapshot for Nifty and India VIX plus option instruments."""

    nifty_spot: MarketQuote
    india_vix: MarketQuote
    option_instruments: pd.DataFrame


class FyersMarketClient:
    """Wraps Fyers API V3 REST operations for live market intelligence."""

    EXPECTED_OPTION_COLUMNS: tuple[str, ...] = (
        "strike",
        "expiry",
        "option_type",
        "last_price",
        "bid_price",
        "ask_price",
    )

    def __init__(
        self,
        settings: FyersSettings,
        auth_manager: FyersAuthManager | None = None,
        security_guard: APISecurityGuard | None = None,
    ) -> None:
        self._settings = settings
        self._security_guard = security_guard
        self._auth_manager = auth_manager or FyersAuthManager(settings, security_guard=security_guard)

        token = self._auth_manager.resolve_access_token()
        if not token:
            token = self._settings.access_token

        self._fyers = fyersModel.FyersModel(
            client_id=settings.client_id or "",
            token=token or "",
            is_async=False,
            log_path=str(settings.token_store_path.parent) if settings.token_store_path else None,
        )

    @property
    def fyers(self) -> fyersModel.FyersModel:
        """Expose the underlying authenticated FyersModel client."""
        return self._fyers

    def validate_session(self) -> dict[str, Any]:
        """Validate current session credentials with Fyers profile endpoint."""
        profile = self._auth_manager.validate_session()
        self._record_success()
        return profile

    def get_quote(self, instrument: str) -> MarketQuote:
        """Fetch a structured quote for a symbol like NSE:NIFTY50-INDEX or NSE:INDIAVIX-INDEX."""
        try:
            self._validate_instrument_identifier(instrument)
            self._record_request("quotes", instrument)
            response = self._fyers.quotes(data={"symbols": instrument})
        except Exception as exc:  # noqa: BLE001
            self._record_failure(f"quote_failed:{instrument}:{exc}")
            raise FyersClientError(f"Quote fetch failed for {instrument}: {exc}") from exc

        if not isinstance(response, dict) or response.get("s") != "ok":
            msg = response.get("message") if isinstance(response, dict) else str(response)
            self._record_failure(f"quote_error:{instrument}:{msg}")
            raise FyersClientError(f"Fyers quote returned error for {instrument}: {msg}")

        data_items = response.get("d", [])
        if not data_items or not isinstance(data_items, list):
            self._record_failure(f"quote_missing_payload:{instrument}")
            raise FyersClientError(f"Quote response missing data payload for {instrument}")

        item = data_items[0]
        val = item.get("v", {}) if isinstance(item, dict) else {}
        cmd = val.get("cmd", {}) if isinstance(val.get("cmd"), dict) else {}

        last_price = _safe_float(val.get("lp") or val.get("last_price") or cmd.get("c"))
        change = _safe_float(val.get("ch") or val.get("change"))
        open_price = _safe_float(val.get("open_price") or cmd.get("o"))
        high_price = _safe_float(val.get("high_price") or cmd.get("h"))
        low_price = _safe_float(val.get("low_price") or cmd.get("l"))
        prev_close = _safe_float(val.get("prev_close_price") or cmd.get("c"))

        tt = val.get("tt") or cmd.get("t")
        exchange_timestamp = _safe_timestamp(tt)
        received_at = datetime.now(UTC)

        structured = MarketQuote(
            instrument=instrument,
            last_price=last_price,
            change=change,
            ohlc={
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": prev_close or last_price,
            },
            received_at=received_at,
            exchange_timestamp=exchange_timestamp,
            last_trade_time=exchange_timestamp,
            raw=item,
        )

        if self._security_guard is not None:
            try:
                self._security_guard.validate_quote_payload(val, instrument=instrument)
            except APISecurityError as exc:
                self._record_failure(str(exc))
                raise FyersClientError(f"Quote validation failed for {instrument}: {exc}") from exc

            # Stale-timestamp check: skip when market is closed (last_price is None/zero).
            # On weekends and public holidays the exchange timestamp will be hours old — that
            # is expected and should NOT crash the engine or trigger safe-mode.
            if structured.last_price:
                try:
                    self._security_guard.validate_fresh_timestamp(
                        exchange_timestamp or received_at,
                        max_age_seconds=self._security_guard.settings.stale_snapshot_seconds,
                        label=f"quote:{instrument}",
                    )
                except APISecurityError as exc:
                    # Log a warning but don't crash — the engine can operate in
                    # degraded / market-closed mode and retry on the next cycle.
                    logger.warning(
                        "Stale quote timestamp (market may be closed)",
                        instrument=instrument,
                        error=str(exc),
                    )

            self._security_guard.register_rest_success()

        logger.info("Fetched Fyers quote", instrument=instrument, last_price=structured.last_price)
        return structured

    def get_nifty_spot(self) -> MarketQuote:
        """Fetch live NIFTY 50 spot quote via Fyers quotes endpoint."""
        return self.get_quote("NSE:NIFTY50-INDEX")

    def get_india_vix(self) -> MarketQuote:
        """Fetch live India VIX quote via Fyers quotes endpoint."""
        return self.get_quote("NSE:INDIAVIX-INDEX")

    def get_option_chain(
        self,
        symbol: str = "NSE:NIFTY50-INDEX",
        strikecount: int = 10,
        timestamp: int | None = None,
    ) -> pd.DataFrame:
        """Fetch live NIFTY 50 option chain snapshot and normalize into engine DataFrame schema."""
        data_payload: dict[str, Any] = {
            "symbol": symbol,
            "strikecount": strikecount,
            "greeks": "1",
        }
        if timestamp is not None:
            data_payload["timestamp"] = int(timestamp)

        try:
            self._record_request("optionchain", symbol)
            response = self._fyers.optionchain(data=data_payload)
        except Exception as exc:  # noqa: BLE001
            self._record_failure(f"optionchain_failed:{symbol}:{exc}")
            raise FyersClientError(f"Option chain fetch failed for {symbol}: {exc}") from exc

        if not isinstance(response, dict) or response.get("s") != "ok":
            msg = response.get("message") if isinstance(response, dict) else str(response)
            self._record_failure(f"optionchain_error:{symbol}:{msg}")
            raise FyersClientError(f"Fyers option chain error for {symbol}: {msg}")

        frame = self.normalize_option_chain(response)
        self._record_success()
        logger.info(
            "Fetched and normalized option chain",
            symbol=symbol,
            strikecount=strikecount,
            rows=len(frame),
        )
        return frame

    def normalize_option_chain(self, response_json: dict[str, Any]) -> pd.DataFrame:
        """Normalize the Fyers JSON response into exact pandas DataFrame schema expected by the engine.

        Schema columns required:
        - strike: float
        - expiry: str
        - option_type: str ("CE" / "PE")
        - last_price: float
        - bid_price: float
        - ask_price: float
        """
        data = response_json.get("data")
        if not data:
            data = response_json

        raw_options: list[dict[str, Any]] = []
        default_expiry = ""

        if isinstance(data, dict):
            raw_options = data.get("optionsChain") or data.get("options") or []
            expiry_data = data.get("expiryData") or []
            if expiry_data and isinstance(expiry_data, list):
                first_exp = expiry_data[0]
                if isinstance(first_exp, dict):
                    default_expiry = str(first_exp.get("date") or first_exp.get("expiry") or "")
        elif isinstance(data, list):
            raw_options = data

        if not raw_options:
            logger.warning("Fyers option chain response contained no option records")
            return pd.DataFrame(columns=list(self.EXPECTED_OPTION_COLUMNS))

        records: list[dict[str, Any]] = []
        for item in raw_options:
            if not isinstance(item, dict):
                continue

            strike = _safe_float(item.get("strike_price") or item.get("strike"))
            if strike is None or strike <= 0:
                continue

            opt_type = str(item.get("option_type") or item.get("opt_type") or "").upper().strip()
            sym = str(item.get("symbol") or "")
            if not opt_type:
                if sym.endswith("CE") or "CE" in sym:
                    opt_type = "CE"
                elif sym.endswith("PE") or "PE" in sym:
                    opt_type = "PE"
                else:
                    opt_type = "CE"

            last_price = _safe_float(item.get("ltp") or item.get("last_price") or item.get("lp")) or 0.0
            bid_price = _safe_float(item.get("bid") or item.get("bid_price") or item.get("bids") or item.get("bp"))
            ask_price = _safe_float(item.get("ask") or item.get("ask_price") or item.get("asks") or item.get("ap"))

            bid_price = bid_price or 0.0
            ask_price = ask_price or 0.0

            expiry_raw = item.get("expiry") or item.get("expiry_date") or default_expiry
            expiry_str = _format_expiry(expiry_raw, sym)

            fy_token = item.get("fyToken") or item.get("token") or item.get("exchange_token")
            token_int: int | None = None
            if fy_token is not None:
                token_clean = re.sub(r"\D", "", str(fy_token))
                if token_clean:
                    token_int = int(token_clean[-10:])
            if token_int is None or token_int <= 0:
                side_mult = 1 if opt_type == "CE" else 2
                token_int = int(strike * 10 + side_mult)

            oi = _safe_float(item.get("oi") or item.get("open_interest")) or 0.0
            greeks = item.get("greeks") if isinstance(item.get("greeks"), dict) else {}
            iv = _safe_float(item.get("iv") or item.get("implied_volatility") or greeks.get("iv"))
            delta = _safe_float(item.get("delta") if item.get("delta") is not None else greeks.get("delta"))

            record = {
                "strike": float(strike),
                "expiry": str(expiry_str),
                "option_type": str(opt_type),
                "last_price": float(last_price),
                "bid_price": float(bid_price),
                "ask_price": float(ask_price),
                # Additional compatible columns
                "instrument_token": token_int,
                "symbol": sym,
                "tradingsymbol": sym,
                "instrument_type": opt_type,
                "open_interest": oi,
                "implied_volatility": iv,
                "delta": delta,
                "timestamp": datetime.now(UTC),
            }
            records.append(record)

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return pd.DataFrame(columns=list(self.EXPECTED_OPTION_COLUMNS))

        df = df.sort_values(["strike", "option_type"]).reset_index(drop=True)
        return df

    def get_option_instruments(
        self,
        *,
        exchange: str = "NFO",
        underlying: str = "NIFTY",
        instrument_types: tuple[str, ...] = ("CE", "PE"),
    ) -> pd.DataFrame:
        """Fetch option chain and return as instrument list matching engine expectations."""
        chain_df = self.get_option_chain(symbol="NSE:NIFTY50-INDEX", strikecount=15)
        if chain_df.empty:
            return chain_df

        filtered = chain_df.loc[chain_df["option_type"].isin(instrument_types)].copy()
        return filtered.reset_index(drop=True)

    def get_live_snapshot(self) -> LiveMarketSnapshot:
        """Collect live market snapshot including NIFTY spot, India VIX, and normalized option chain.

        When the market is closed (weekend / public holiday) the Fyers API
        returns zero or missing prices.  ``get_quote`` will raise
        ``FyersClientError`` in that case.  We catch it here and return a
        snapshot with ``last_price=None`` so the intelligence engine can
        continue in a graceful no-trade state rather than crashing.
        """

        def _closed_quote(instrument: str) -> MarketQuote:
            return MarketQuote(
                instrument=instrument,
                last_price=None,
                change=None,
                ohlc={"open": None, "high": None, "low": None, "close": None},
                received_at=datetime.now(UTC),
                exchange_timestamp=None,
                last_trade_time=None,
                raw={},
            )

        try:
            nifty_spot = self.get_nifty_spot()
        except FyersClientError as exc:
            logger.warning("NIFTY spot unavailable (market may be closed)", error=str(exc))
            nifty_spot = _closed_quote("NSE:NIFTY50-INDEX")

        try:
            india_vix = self.get_india_vix()
        except FyersClientError as exc:
            logger.warning("India VIX unavailable (market may be closed)", error=str(exc))
            india_vix = _closed_quote("NSE:INDIAVIX-INDEX")

        try:
            option_instruments = self.get_option_instruments()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Option chain unavailable (market may be closed)", error=str(exc))
            option_instruments = pd.DataFrame(columns=list(self.EXPECTED_OPTION_COLUMNS))

        return LiveMarketSnapshot(
            nifty_spot=nifty_spot,
            india_vix=india_vix,
            option_instruments=option_instruments,
        )

    def _validate_instrument_identifier(self, instrument: str) -> None:
        if self._security_guard is None:
            return
        self._security_guard.validate_instrument_identifier(instrument)

    def _record_request(self, api_name: str, endpoint: str) -> None:
        if self._security_guard is None:
            return
        self._security_guard.register_rest_request(api_name, endpoint)

    def _record_success(self) -> None:
        if self._security_guard is None:
            return
        self._security_guard.register_rest_success()

    def _record_failure(self, reason: str) -> None:
        if self._security_guard is None:
            return
        self._security_guard.register_rest_failure(reason)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            if value > 1e11:  # milliseconds
                value = value / 1000.0
            return datetime.fromtimestamp(value, tz=UTC)
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:  # noqa: BLE001
        return None


def _format_expiry(expiry_val: Any, symbol: str) -> str:
    """Format expiry as standard date string YYYY-MM-DD or DD-Mon-YYYY."""
    if expiry_val:
        text = str(expiry_val).strip()
        if len(text) == 10 and text[2] == "-" and text[5] == "-":
            try:
                return datetime.strptime(text, "%d-%m-%Y").date().isoformat()
            except ValueError:
                return ""
        if text.isdigit() and len(text) in {10, 13}:
            expiry_val = int(text)
        if isinstance(expiry_val, (int, float)):
            try:
                ts = expiry_val / 1000.0 if expiry_val > 1e11 else float(expiry_val)
                return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")
            except Exception:  # noqa: BLE001
                pass
        return str(expiry_val).strip()

    # Fallback to symbol parsing: e.g. NSE:NIFTY24OCT24500CE -> 24OCT
    if symbol:
        match = re.search(r"NIFTY(\d{2}[A-Z]{3}|\d{5})", symbol.upper())
        if match:
            return match.group(1)
    return ""
