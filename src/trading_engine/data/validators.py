"""Data quality validators for ingestion pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    """Single validation finding emitted by DataValidator."""

    code: str
    severity: str
    message: str
    affected_rows: int = 0


class ValidationReport(BaseModel):
    """Aggregate validation report for a DataFrame."""

    is_valid: bool
    rows_total: int
    rows_corrupted: int
    corruption_ratio: float
    issues: list[ValidationIssue] = Field(default_factory=list)


class DataValidator:
    """Detect structural and statistical anomalies in feed data."""

    def validate(
        self,
        frame: pd.DataFrame,
        *,
        required_columns: Iterable[str],
        timestamp_col: str,
        numeric_columns: Iterable[str] | None = None,
    ) -> ValidationReport:
        required = set(required_columns)
        numeric = set(numeric_columns or [])

        issues: list[ValidationIssue] = []
        rows_total = len(frame)

        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            issues.append(
                ValidationIssue(
                    code="missing_columns",
                    severity="error",
                    message=f"Missing required columns: {missing_columns}",
                    affected_rows=rows_total,
                )
            )
            return ValidationReport(
                is_valid=False,
                rows_total=rows_total,
                rows_corrupted=rows_total,
                corruption_ratio=1.0 if rows_total else 0.0,
                issues=issues,
            )

        working = frame.copy()
        corrupted_mask = pd.Series(False, index=working.index)

        timestamp_parsed = pd.to_datetime(working[timestamp_col], errors="coerce", utc=False)
        ts_invalid_mask = timestamp_parsed.isna()
        if ts_invalid_mask.any():
            count = int(ts_invalid_mask.sum())
            issues.append(
                ValidationIssue(
                    code="invalid_timestamp",
                    severity="error",
                    message=f"Found {count} rows with invalid timestamps",
                    affected_rows=count,
                )
            )
            corrupted_mask |= ts_invalid_mask

        duplicate_count = int(working.duplicated().sum())
        if duplicate_count > 0:
            issues.append(
                ValidationIssue(
                    code="duplicate_rows",
                    severity="warning",
                    message=f"Found {duplicate_count} exact duplicate rows",
                    affected_rows=duplicate_count,
                )
            )

        for column in numeric:
            if column not in working.columns:
                continue
            converted = pd.to_numeric(working[column], errors="coerce")
            invalid_mask = converted.isna() & working[column].notna()
            non_finite_mask = ~np.isfinite(converted.fillna(0.0))
            combined = invalid_mask | non_finite_mask
            if combined.any():
                count = int(combined.sum())
                issues.append(
                    ValidationIssue(
                        code="non_numeric",
                        severity="error",
                        message=f"Column '{column}' has {count} non-numeric or non-finite values",
                        affected_rows=count,
                    )
                )
                corrupted_mask |= combined

        null_required_mask = working[list(required)].isna().any(axis=1)
        if null_required_mask.any():
            count = int(null_required_mask.sum())
            issues.append(
                ValidationIssue(
                    code="null_required",
                    severity="error",
                    message=f"Found {count} rows with nulls in required columns",
                    affected_rows=count,
                )
            )
            corrupted_mask |= null_required_mask

        rows_corrupted = int(corrupted_mask.sum())
        corruption_ratio = float(rows_corrupted / rows_total) if rows_total else 0.0
        is_valid = rows_corrupted == 0

        return ValidationReport(
            is_valid=is_valid,
            rows_total=rows_total,
            rows_corrupted=rows_corrupted,
            corruption_ratio=corruption_ratio,
            issues=issues,
        )

    @staticmethod
    def drop_corrupted_rows(frame: pd.DataFrame, report: ValidationReport, *, timestamp_col: str) -> pd.DataFrame:
        """Best-effort cleanup for rows likely to be corrupted."""
        if report.rows_corrupted == 0:
            return frame

        cleaned = frame.copy()
        cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
        cleaned = cleaned.dropna(subset=[timestamp_col])
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.dropna(how="all")
        return cleaned.reset_index(drop=True)
