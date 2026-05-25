"""Feed alignment and merge utilities."""

from __future__ import annotations

from functools import reduce

import pandas as pd


class FeedMerger:
    """Align feed dataframes on shared timestamp axis."""

    def merge(
        self,
        feeds: dict[str, pd.DataFrame],
        *,
        timestamp_col: str,
        how: str = "outer",
        prefix_columns: bool = True,
    ) -> pd.DataFrame:
        """Merge multiple feeds into one aligned dataframe.

        Each feed's non-timestamp columns can be prefixed to avoid collisions.
        """
        if not feeds:
            return pd.DataFrame(columns=[timestamp_col])

        prepared: list[pd.DataFrame] = []
        for feed_name, frame in feeds.items():
            if timestamp_col not in frame.columns:
                raise ValueError(f"Timestamp column '{timestamp_col}' missing in feed '{feed_name}'")

            working = frame.copy().sort_values(timestamp_col)
            if prefix_columns:
                rename_map = {
                    column: f"{feed_name}__{column}"
                    for column in working.columns
                    if column != timestamp_col
                }
                working = working.rename(columns=rename_map)

            prepared.append(working)

        merged = reduce(
            lambda left, right: pd.merge(left, right, on=timestamp_col, how=how, sort=True),
            prepared,
        )

        return merged.sort_values(timestamp_col).reset_index(drop=True)
