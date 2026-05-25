"""CLI command for feed ingestion into partitioned Parquet + DuckDB catalog."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from trading_engine.common.logging import configure_logging, get_logger
from trading_engine.config.settings import get_settings
from trading_engine.data.loaders import AsyncFeedLoader, FeedRequest
from trading_engine.data.pipeline import IngestionPipeline
from trading_engine.data.schemas import DEFAULT_REQUIRED_COLUMNS, NUMERIC_COLUMNS_BY_FEED, FeedName


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest market feed data")
    parser.add_argument("--feed", required=True, choices=[item.value for item in FeedName])
    parser.add_argument("--source", required=True, choices=["csv", "api"])
    parser.add_argument("--path", required=True, help="CSV file path or API URL")
    parser.add_argument("--dataset", default="market_data")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--frequency", default="1min")
    return parser


async def _run_ingestion(args: argparse.Namespace) -> None:
    settings = get_settings()
    configure_logging(
        log_dir=settings.app.paths.log_dir,
        level=settings.app.logging.level,
        serialize=settings.app.logging.json,
    )
    logger = get_logger("ingestion")

    loader = AsyncFeedLoader()
    pipeline = IngestionPipeline(settings)

    request = FeedRequest(feed=args.feed, source_type=args.source, source=args.path)
    loaded = await loader.load_one(request)

    feed_name = FeedName(args.feed)
    result = pipeline.run(
        loaded,
        dataset=args.dataset,
        feed=args.feed,
        timestamp_col=args.timestamp_col,
        required_columns=DEFAULT_REQUIRED_COLUMNS[feed_name],
        numeric_columns=NUMERIC_COLUMNS_BY_FEED[feed_name],
        frequency=args.frequency,
    )

    logger.info(
        "Ingestion completed",
        feed=args.feed,
        rows=len(result.cleaned_frame),
        dataset_path=result.dataset_path,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    asyncio.run(_run_ingestion(args))


if __name__ == "__main__":
    main()
