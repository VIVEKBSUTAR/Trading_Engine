"""Canonical schemas for market data feeds."""

from __future__ import annotations

from enum import StrEnum


class FeedName(StrEnum):
    """Supported feed names for ingestion and storage."""

    NIFTY50_SPOT = "nifty50_spot"
    GIFT_NIFTY_FUT = "gift_nifty_fut"
    INDIA_VIX = "india_vix"
    NSE_OPTION_CHAIN = "nse_option_chain"


TIMESTAMP_COLUMN = "timestamp"

DEFAULT_REQUIRED_COLUMNS: dict[FeedName, set[str]] = {
    FeedName.NIFTY50_SPOT: {"timestamp", "open", "high", "low", "close"},
    FeedName.GIFT_NIFTY_FUT: {"timestamp", "open", "high", "low", "close"},
    FeedName.INDIA_VIX: {"timestamp", "close"},
    FeedName.NSE_OPTION_CHAIN: {
        "timestamp",
        "expiry",
        "strike",
        "option_type",
        "ltp",
        "iv",
        "oi",
    },
}

NUMERIC_COLUMNS_BY_FEED: dict[FeedName, set[str]] = {
    FeedName.NIFTY50_SPOT: {"open", "high", "low", "close", "volume"},
    FeedName.GIFT_NIFTY_FUT: {"open", "high", "low", "close", "volume", "open_interest"},
    FeedName.INDIA_VIX: {"close"},
    FeedName.NSE_OPTION_CHAIN: {"strike", "ltp", "iv", "oi", "bid", "ask"},
}
