"""Validation utilities for normalized NSE option-chain DataFrames."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass(slots=True)
class ValidationResult:
    """Validated frame with diagnostics about dropped/corrupted records."""

    frame: pd.DataFrame
    input_rows: int
    output_rows: int
    dropped_rows: int


class OptionChainValidator:
    """Validate schema, coerce numeric columns, and drop invalid rows."""

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "expiry",
        "strike",
        "option_type",
        "open_interest",
        "change_in_oi",
        "implied_volatility",
        "traded_volume",
        "last_price",
        "bid_qty",
        "ask_qty",
        "underlying_price",
    )

    NUMERIC_COLUMNS: tuple[str, ...] = (
        "strike",
        "open_interest",
        "change_in_oi",
        "implied_volatility",
        "traded_volume",
        "last_price",
        "bid_qty",
        "ask_qty",
        "underlying_price",
    )

    def validate(self, frame: pd.DataFrame) -> ValidationResult:
        """Run schema and quality checks and return cleaned frame."""
        if frame is None:
            raise ValueError("Input frame is None")

        input_rows = len(frame)
        if input_rows == 0:
            logger.warning("Validation skipped for empty frame")
            return ValidationResult(frame=frame, input_rows=0, output_rows=0, dropped_rows=0)

        missing = [col for col in self.REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        cleaned = frame.copy()

        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce", utc=True)

        for column in self.NUMERIC_COLUMNS:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

        valid_option_type = cleaned["option_type"].isin(["CE", "PE"])

        non_negative_columns = [
            "strike",
            "open_interest",
            "traded_volume",
            "last_price",
            "bid_qty",
            "ask_qty",
            "underlying_price",
        ]
        non_negative_mask = pd.Series(True, index=cleaned.index)
        for column in non_negative_columns:
            non_negative_mask &= cleaned[column].fillna(-1) >= 0

        required_not_null = (
            cleaned["timestamp"].notna()
            & cleaned["expiry"].notna()
            & cleaned["strike"].notna()
            & cleaned["option_type"].notna()
        )

        valid_mask = required_not_null & valid_option_type & non_negative_mask
        invalid_count = int((~valid_mask).sum())
        if invalid_count > 0:
            logger.warning("Dropping invalid option-chain rows", invalid_rows=invalid_count)

        cleaned = cleaned.loc[valid_mask].drop_duplicates(
            subset=["timestamp", "expiry", "strike", "option_type"],
            keep="last",
        )

        output_rows = len(cleaned)
        dropped_rows = input_rows - output_rows

        logger.info(
            "Validation complete",
            input_rows=input_rows,
            output_rows=output_rows,
            dropped_rows=dropped_rows,
        )
        return ValidationResult(
            frame=cleaned.reset_index(drop=True),
            input_rows=input_rows,
            output_rows=output_rows,
            dropped_rows=dropped_rows,
        )
