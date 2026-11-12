"""Public NSE endpoints cannot be reintroduced through generic API ingestion."""
from unittest.mock import patch
from urllib.request import Request

import pytest

from trading_engine.common.exceptions import DataIngestionError
from trading_engine.data.loaders import AsyncFeedLoader, FeedRequest, _NoPublicNSERedirectHandler


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
    "https://nseindia.com/",
    "https://WWW.NSEINDIA.COM./option-chain",
    "https://nsearchives.nseindia.com/data.csv",
])
async def test_public_exchange_requests_are_rejected_before_network(url):
    with patch("trading_engine.data.loaders.build_opener") as network:
        with pytest.raises(DataIngestionError, match="Direct NSE website ingestion is disabled"):
            await AsyncFeedLoader().load_one(FeedRequest("nifty50_spot", "api", url))
        network.assert_not_called()


def test_redirect_to_public_exchange_is_rejected_before_following():
    with pytest.raises(DataIngestionError, match="Direct NSE website ingestion is disabled"):
        _NoPublicNSERedirectHandler().redirect_request(
            Request("https://api-t1.fyers.in/example"), None, 302, "Found", {},
            "https://www.nseindia.com/api/option-chain-indices",
        )
