from __future__ import annotations

from trading_engine.ml.validation import build_walk_forward_windows


def test_walk_forward_windows_are_chronological() -> None:
    windows = build_walk_forward_windows(
        n_rows=100,
        train_size=40,
        test_size=10,
        step_size=10,
    )

    assert len(windows) > 0
    assert windows[0].train_end == windows[0].test_start
    assert windows[0].train_start == 0
