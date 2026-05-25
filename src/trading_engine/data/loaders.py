"""Asynchronous data feed loaders for CSV and API sources."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen

import pandas as pd

from trading_engine.common.exceptions import DataIngestionError
from trading_engine.common.logging import get_logger

SourceType = Literal["csv", "api"]


@dataclass(slots=True)
class FeedRequest:
    """Request parameters for loading a single feed."""

    feed: str
    source_type: SourceType
    source: str
    read_kwargs: dict[str, Any] | None = None


class AsyncFeedLoader:
    """Load one or many feed payloads concurrently."""

    def __init__(self, timeout_seconds: int = 30) -> None:
        self._timeout_seconds = timeout_seconds
        self._logger = get_logger("ingestion")

    async def load_many(self, requests: list[FeedRequest]) -> dict[str, pd.DataFrame]:
        """Load all requests concurrently and return feed->DataFrame mapping."""
        tasks = [self.load_one(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, pd.DataFrame] = {}
        failures: list[str] = []

        for request, result in zip(requests, results, strict=True):
            if isinstance(result, Exception):
                failures.append(f"{request.feed}: {result}")
                continue
            output[request.feed] = result

        if failures:
            raise DataIngestionError("; ".join(failures))

        return output

    async def load_one(self, request: FeedRequest) -> pd.DataFrame:
        """Load a single feed request based on source type."""
        if request.source_type == "csv":
            return await self._load_csv(Path(request.source), request.read_kwargs or {})
        if request.source_type == "api":
            return await self._load_api(request.source)

        raise DataIngestionError(f"Unsupported source type: {request.source_type}")

    async def _load_csv(self, path: Path, read_kwargs: dict[str, Any]) -> pd.DataFrame:
        if not path.exists():
            raise DataIngestionError(f"CSV path does not exist: {path}")

        self._logger.info("Loading CSV feed", path=str(path))
        frame = await asyncio.wait_for(
            asyncio.to_thread(pd.read_csv, path, **read_kwargs),
            timeout=self._timeout_seconds,
        )
        return frame

    async def _load_api(self, url: str) -> pd.DataFrame:
        self._logger.info("Loading API feed", url=url)

        def _fetch() -> dict[str, Any] | list[dict[str, Any]]:
            request = Request(url, headers={"User-Agent": "trading-engine/0.1"})
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                body = response.read().decode("utf-8")
            return json.loads(body)

        payload = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=self._timeout_seconds)

        if isinstance(payload, dict):
            records = payload.get("data")
            if not isinstance(records, list):
                raise DataIngestionError("API payload missing list under 'data'")
            return pd.DataFrame(records)

        if isinstance(payload, list):
            return pd.DataFrame(payload)

        raise DataIngestionError("Unsupported API payload shape")
