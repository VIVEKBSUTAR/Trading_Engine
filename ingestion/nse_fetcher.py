"""NSE option chain HTTP fetcher with session bootstrap and retry handling."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import requests
from loguru import logger

from config.settings import APISettings


class NSEFetchError(RuntimeError):
    """Raised when NSE option-chain fetch fails after all retries."""


@dataclass(slots=True)
class FetchResult:
    """Container for raw and parsed response payload from NSE."""

    fetched_at_utc: datetime
    raw_text: str
    payload: dict


class NSEOptionChainFetcher:
    """Fetch NIFTY option-chain snapshots from official NSE API endpoint."""

    def __init__(self, settings: APISettings) -> None:
        self._settings = settings
        self._session = requests.Session()
        self._session.headers.update(self._settings.headers)
        self._bootstrapped = False

    def _bootstrap_session(self) -> None:
        """Initialize NSE cookies by hitting the NSE homepage."""
        if self._bootstrapped:
            return

        logger.info("Bootstrapping NSE session")
        urls = self._settings.bootstrap_urls if self._settings.bootstrap_urls else (self._settings.bootstrap_url,)

        for url in urls:
            try:
                response = self._session.get(
                    url,
                    timeout=self._settings.timeout_seconds,
                )
                # 2xx/3xx considered successful warm-up.
                if 200 <= response.status_code < 400:
                    self._bootstrapped = True
                    logger.info("NSE session bootstrap completed", url=url, status_code=response.status_code)
                    return

                logger.warning("Bootstrap warm-up returned non-success", url=url, status_code=response.status_code)
            except requests.RequestException as exc:
                logger.warning("Bootstrap warm-up request failed", url=url, error=str(exc))

        # Keep running with retries at fetch-call level; in some network setups
        # bootstrap can fail while API may still become reachable in subsequent attempts.
        self._bootstrapped = True
        logger.warning("NSE bootstrap did not return success status on warm-up URLs")

    def fetch_option_chain(self) -> FetchResult:
        """Fetch one option-chain snapshot with retries and safe JSON parsing."""
        self._bootstrap_session()

        last_error: Exception | None = None

        for attempt in range(1, self._settings.max_retries + 1):
            try:
                started = time.perf_counter()
                response = self._session.get(
                    self._settings.option_chain_url,
                    timeout=self._settings.timeout_seconds,
                )

                if response.status_code == 401 or response.status_code == 403:
                    logger.warning(
                        "NSE denied request, re-bootstrap session",
                        status_code=response.status_code,
                        attempt=attempt,
                    )
                    self._bootstrapped = False
                    self._bootstrap_session()
                    raise requests.HTTPError(f"NSE denied request: {response.status_code}")

                response.raise_for_status()
                raw_text = response.text
                payload = self._parse_json(raw_text)
                latency_ms = (time.perf_counter() - started) * 1000.0

                logger.info(
                    "Fetched NSE option chain",
                    attempt=attempt,
                    latency_ms=round(latency_ms, 2),
                    bytes=len(raw_text),
                )
                return FetchResult(
                    fetched_at_utc=datetime.now(UTC),
                    raw_text=raw_text,
                    payload=payload,
                )
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "NSE fetch attempt failed",
                    attempt=attempt,
                    max_retries=self._settings.max_retries,
                    error=str(exc),
                )
                if attempt < self._settings.max_retries:
                    sleep_seconds = self._settings.backoff_seconds * attempt
                    time.sleep(sleep_seconds)

        raise NSEFetchError(f"NSE fetch failed after retries: {last_error}")

    @staticmethod
    def _parse_json(raw_text: str) -> dict:
        """Parse JSON payload with explicit type guard."""
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise ValueError("NSE API payload is not a JSON object")
        return payload
