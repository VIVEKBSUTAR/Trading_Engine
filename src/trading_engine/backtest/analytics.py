"""Performance analytics utilities for stratified backtest reporting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class PerformanceSummary:
    """Container for summary tables derived from a trade ledger."""

    by_regime: pd.DataFrame
    by_session: pd.DataFrame
    by_volatility: pd.DataFrame
    by_weekday: pd.DataFrame
    by_expiry_proximity: pd.DataFrame
    by_confidence_band: pd.DataFrame
    by_trade_grade: pd.DataFrame


def build_performance_summary(trade_ledger: pd.DataFrame) -> PerformanceSummary:
    """Create stratified performance tables from a trade ledger."""
    frame = trade_ledger.copy()
    if frame.empty:
        empty = pd.DataFrame()
        return PerformanceSummary(empty, empty, empty, empty, empty, empty, empty)

    frame = frame.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        frame["weekday"] = frame["timestamp"].dt.day_name()
    if "expiry_days" not in frame.columns and "expiry" in frame.columns and "timestamp" in frame.columns:
        expiry = pd.to_datetime(frame["expiry"], errors="coerce", utc=True)
        frame["expiry_days"] = (expiry - frame["timestamp"]).dt.total_seconds() / 86_400.0

    aggregations = {
        "trade_count": ("pnl", "count"),
        "win_rate": ("pnl", lambda s: float((s > 0).mean())),
        "avg_pnl": ("pnl", "mean"),
        "expectancy": ("pnl", "mean"),
        "avg_confidence": ("confidence", "mean"),
        "max_drawdown": ("equity", _max_drawdown_series),
    }

    by_regime = _group_summary(frame, "regime", aggregations)
    by_session = _group_summary(frame, "session_state", aggregations)
    by_volatility = _group_summary(frame, "volatility_state", aggregations)
    by_weekday = _group_summary(frame, "weekday", aggregations)
    by_expiry_proximity = _group_summary(frame, "expiry_band", aggregations)
    by_confidence_band = _group_summary(frame, "confidence_band", aggregations)
    by_trade_grade = _group_summary(frame, "trade_grade", aggregations)

    return PerformanceSummary(
        by_regime=by_regime,
        by_session=by_session,
        by_volatility=by_volatility,
        by_weekday=by_weekday,
        by_expiry_proximity=by_expiry_proximity,
        by_confidence_band=by_confidence_band,
        by_trade_grade=by_trade_grade,
    )


def _group_summary(frame: pd.DataFrame, column: str, aggregations: dict[str, tuple[str, object]]) -> pd.DataFrame:
    if column not in frame.columns:
        return pd.DataFrame()
    named_aggs = {output: pd.NamedAgg(column=source, aggfunc=agg) for output, (source, agg) in aggregations.items() if source in frame.columns}
    if not named_aggs:
        return pd.DataFrame()
    grouped = frame.groupby(column, dropna=False).agg(**named_aggs).reset_index()
    return grouped


def _max_drawdown_series(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    running_max = values.cummax()
    drawdown = (values / running_max) - 1.0
    return float(drawdown.min())
