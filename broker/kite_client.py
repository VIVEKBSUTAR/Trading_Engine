"""Authenticated Kite Connect client helpers for live market connectivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from broker.kite_auth import KiteAuthError, KiteAuthManager
from config.settings import KiteSettings


class KiteClientError(RuntimeError):
    """Raised when a Kite REST call or validation fails."""


@dataclass(slots=True)
class MarketQuote:
    """Structured quote payload for an instrument."""

    instrument: str
    last_price: float | None
    change: float | None
    ohlc: dict[str, float | None]
    raw: dict[str, Any]


@dataclass(slots=True)
class LiveMarketSnapshot:
    """Current market snapshot for Nifty and India VIX plus option instruments."""

    nifty_spot: MarketQuote
    india_vix: MarketQuote
    option_instruments: pd.DataFrame


class KiteMarketClient:
    """Wraps KiteConnect REST calls needed for live market intelligence."""

    def __init__(self, settings: KiteSettings, auth_manager: KiteAuthManager | None = None) -> None:
        self._settings = settings
        self._auth_manager = auth_manager or KiteAuthManager(settings)
        self._kite = self._auth_manager.kite

    @property
    def kite(self):
        """Expose the underlying authenticated KiteConnect client."""
        return self._kite

    def validate_session(self) -> dict[str, Any]:
        """Validate session state using the official profile endpoint."""
        return self._auth_manager.validate_session()

    def get_quote(self, instrument: str) -> MarketQuote:
        """Fetch a structured quote for an instrument string like NSE:INFY."""
        try:
            response = self._kite.quote(instrument)
        except Exception as exc:  # noqa: BLE001
            raise KiteClientError(f"Quote fetch failed for {instrument}: {exc}") from exc

        payload = response.get(instrument)
        if not isinstance(payload, dict):
            raise KiteClientError(f"Quote response missing payload for {instrument}")

        ohlc = payload.get("ohlc") if isinstance(payload.get("ohlc"), dict) else {}
        structured = MarketQuote(
            instrument=instrument,
            last_price=_safe_float(payload.get("last_price")),
            change=_safe_float(payload.get("change")),
            ohlc={
                "open": _safe_float(ohlc.get("open")),
                "high": _safe_float(ohlc.get("high")),
                "low": _safe_float(ohlc.get("low")),
                "close": _safe_float(ohlc.get("close")),
            },
            raw=payload,
        )
        logger.info("Fetched quote", instrument=instrument, last_price=structured.last_price)
        return structured

    def get_nifty_spot(self) -> MarketQuote:
        """Fetch live Nifty spot quote via the official quote API."""
        return self.get_quote("NSE:NIFTY 50")

    def get_india_vix(self) -> MarketQuote:
        """Fetch live India VIX quote via the official quote API."""
        return self.get_quote("NSE:INDIA VIX")

    def get_option_instruments(
        self,
        *,
        exchange: str = "NFO",
        underlying: str = "NIFTY",
        instrument_types: tuple[str, ...] = ("CE", "PE"),
    ) -> pd.DataFrame:
        """Fetch and filter live option instruments for the requested underlying."""
        try:
            records = self._kite.instruments(exchange)
        except Exception as exc:  # noqa: BLE001
            raise KiteClientError(f"Instrument fetch failed for exchange={exchange}: {exc}") from exc

        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            logger.warning("Instrument frame is empty", exchange=exchange)
            return frame

        filtered = frame.copy()
        if "name" in filtered.columns:
            filtered = filtered.loc[filtered["name"].astype(str).str.upper() == underlying.upper()]
        if "instrument_type" in filtered.columns:
            filtered = filtered.loc[filtered["instrument_type"].astype(str).isin(instrument_types)]
        if "segment" in filtered.columns:
            filtered = filtered.loc[filtered["segment"].astype(str).str.contains("OPT", na=False)]

        sort_columns = [column for column in ("expiry", "strike", "tradingsymbol") if column in filtered.columns]
        if sort_columns:
            filtered = filtered.sort_values(sort_columns)

        logger.info(
            "Fetched option instruments",
            exchange=exchange,
            underlying=underlying,
            rows=len(filtered),
        )
        return filtered.reset_index(drop=True)

    def get_live_snapshot(self) -> LiveMarketSnapshot:
        """Collect the live market snapshot used by downstream analytics."""
        nifty_spot = self.get_nifty_spot()
        india_vix = self.get_india_vix()
        option_instruments = self.get_option_instruments()
        return LiveMarketSnapshot(
            nifty_spot=nifty_spot,
            india_vix=india_vix,
            option_instruments=option_instruments,
        )


def _safe_float(value: Any) -> float | None:
    """Best-effort float coercion without raising on missing values."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
