"""Unit tests for MicroMomentumEngine — tick-level velocity/acceleration features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_engine.intelligence.micro_momentum import MicroMomentumEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_price_ticks(prices: list[float], include_volume: bool = True) -> pd.DataFrame:
    """Build a minimal tick DataFrame from a price list."""
    data: dict = {"last_price": prices}
    if include_volume:
        # Cumulative volume increasing by 200 each tick
        data["volume_traded"] = list(range(1000, 1000 + len(prices) * 200, 200))
    return pd.DataFrame(data)


def make_ticks_with_qty(prices: list[float], buy_qty: int, sell_qty: int) -> pd.DataFrame:
    df = make_price_ticks(prices)
    df["total_buy_quantity"] = buy_qty
    df["total_sell_quantity"] = sell_qty
    return df


ENGINE = MicroMomentumEngine(velocity_window=5, acceleration_windows=(5, 10), min_ticks_required=6)


# ---------------------------------------------------------------------------
# Tests: velocity
# ---------------------------------------------------------------------------

class TestVelocity:
    def test_flat_price_has_zero_velocity(self):
        ticks = make_price_ticks([100.0] * 15)
        result = ENGINE.assess(ticks, pd.DataFrame(), pd.DataFrame())
        assert result.underlying_velocity == pytest.approx(0.0, abs=1e-9)

    def test_rising_price_has_positive_velocity(self):
        prices = [100.0 + i * 0.5 for i in range(15)]
        ticks = make_price_ticks(prices)
        result = ENGINE.assess(ticks, pd.DataFrame(), pd.DataFrame())
        assert result.underlying_velocity > 0.0

    def test_falling_price_has_negative_velocity(self):
        prices = [100.0 - i * 0.5 for i in range(15)]
        ticks = make_price_ticks(prices)
        result = ENGINE.assess(ticks, pd.DataFrame(), pd.DataFrame())
        assert result.underlying_velocity < 0.0

    def test_velocity_magnitude_proportional_to_move(self):
        slow = [100.0 + i * 0.1 for i in range(15)]
        fast = [100.0 + i * 1.0 for i in range(15)]
        slow_ticks = make_price_ticks(slow)
        fast_ticks = make_price_ticks(fast)
        slow_res = ENGINE.assess(slow_ticks, pd.DataFrame(), pd.DataFrame())
        fast_res = ENGINE.assess(fast_ticks, pd.DataFrame(), pd.DataFrame())
        assert abs(fast_res.underlying_velocity) > abs(slow_res.underlying_velocity)


# ---------------------------------------------------------------------------
# Tests: insufficient data
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_empty_underlying_returns_zero_velocity(self):
        result = ENGINE.assess(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert result.underlying_velocity == 0.0
        assert result.confidence == 0.0

    def test_too_few_ticks_returns_nan_for_options(self):
        ticks = make_price_ticks([100.0] * 15)  # enough for underlying
        result = ENGINE.assess(ticks, make_price_ticks([90.0] * 3), pd.DataFrame())
        assert np.isnan(result.ce_premium_velocity)
        assert np.isnan(result.pe_premium_velocity)

    def test_confidence_increases_with_more_data(self):
        ticks = make_price_ticks([100.0] * 15)
        ce_ticks = make_price_ticks([90.0] * 15)
        pe_ticks = make_price_ticks([85.0] * 15)
        only_underlying = ENGINE.assess(ticks, pd.DataFrame(), pd.DataFrame())
        with_options = ENGINE.assess(ticks, ce_ticks, pe_ticks)
        assert with_options.confidence > only_underlying.confidence


# ---------------------------------------------------------------------------
# Tests: simultaneous spike (trap detection)
# ---------------------------------------------------------------------------

class TestSimultaneousSpike:
    def test_both_options_rising_triggers_spike(self):
        # CE rising 2% in window, PE rising 2% in window -> both above 0.3% threshold
        underlying = make_price_ticks([24000.0] * 15)
        ce = make_price_ticks([100.0, 100.5, 101.0, 101.5, 102.0,
                                102.5, 103.0, 103.5, 104.0, 104.5,
                                105.0, 105.5, 106.0, 106.5, 107.0])
        pe = make_price_ticks([100.0, 100.4, 100.8, 101.2, 101.6,
                                102.0, 102.4, 102.8, 103.2, 103.6,
                                104.0, 104.4, 104.8, 105.2, 105.6])
        result = ENGINE.assess(underlying, ce, pe)
        assert result.simultaneous_spike is True
        assert result.trap_via_spike > 0.0

    def test_only_ce_rising_does_not_trigger_spike(self):
        underlying = make_price_ticks([24000.0 + i * 0.5 for i in range(15)])
        # CE rising, PE flat
        ce = make_price_ticks([100.0 + i for i in range(15)])
        pe = make_price_ticks([90.0] * 15)
        result = ENGINE.assess(underlying, ce, pe)
        assert result.simultaneous_spike is False
        assert result.trap_via_spike == 0.0


# ---------------------------------------------------------------------------
# Tests: buy/sell imbalance
# ---------------------------------------------------------------------------

class TestBuySellImbalance:
    def test_all_buy_gives_positive_imbalance(self):
        ticks = make_ticks_with_qty([100.0] * 15, buy_qty=1000, sell_qty=0)
        result = ENGINE.assess(ticks, pd.DataFrame(), pd.DataFrame())
        assert result.buy_sell_imbalance == pytest.approx(1.0, abs=1e-6)

    def test_all_sell_gives_negative_imbalance(self):
        ticks = make_ticks_with_qty([100.0] * 15, buy_qty=0, sell_qty=1000)
        result = ENGINE.assess(ticks, pd.DataFrame(), pd.DataFrame())
        assert result.buy_sell_imbalance == pytest.approx(-1.0, abs=1e-6)

    def test_equal_qty_gives_zero_imbalance(self):
        ticks = make_ticks_with_qty([100.0] * 15, buy_qty=500, sell_qty=500)
        result = ENGINE.assess(ticks, pd.DataFrame(), pd.DataFrame())
        assert result.buy_sell_imbalance == pytest.approx(0.0, abs=1e-6)

    def test_missing_qty_columns_gives_zero(self):
        ticks = make_price_ticks([100.0] * 15, include_volume=False)
        result = ENGINE.assess(ticks, pd.DataFrame(), pd.DataFrame())
        assert result.buy_sell_imbalance == 0.0


# ---------------------------------------------------------------------------
# Tests: momentum scores
# ---------------------------------------------------------------------------

class TestMomentumScores:
    def test_strong_ce_move_with_underlying_up_gives_high_ce_score(self):
        underlying = make_price_ticks([24000.0 + i * 5 for i in range(15)])  # rising index
        ce = make_price_ticks([100.0 + i * 2 for i in range(15)])            # CE rising strongly
        pe = make_price_ticks([90.0] * 15)                                   # PE flat
        result = ENGINE.assess(underlying, ce, pe)
        assert result.ce_momentum_score > 0.3
        assert result.pe_momentum_score < result.ce_momentum_score

    def test_scores_clipped_to_unit_interval(self):
        # Extreme prices should not produce scores outside [0, 1]
        underlying = make_price_ticks([24000.0 + i * 100 for i in range(15)])
        ce = make_price_ticks([100.0 + i * 50 for i in range(15)])
        pe = make_price_ticks([100.0 - i * 50 for i in range(15)])
        result = ENGINE.assess(underlying, ce, pe)
        assert 0.0 <= result.ce_momentum_score <= 1.0
        assert 0.0 <= result.pe_momentum_score <= 1.0


# ---------------------------------------------------------------------------
# Tests: SignalScore dataclass
# ---------------------------------------------------------------------------

class TestSignalScore:
    def test_all_zeros_gives_zero_total(self):
        from trading_engine.intelligence.models import SignalScore
        score = SignalScore()
        assert score.total == pytest.approx(0.0, abs=1e-9)

    def test_all_ones_gives_max_total(self):
        from trading_engine.intelligence.models import SignalScore
        score = SignalScore(
            underlying_direction=1.0,
            underlying_acceleration=1.0,
            premium_velocity_ce=1.0,
            premium_velocity_pe=1.0,
            premium_acceleration=1.0,
            volume_acceleration=1.0,
            relative_ce_pe=1.0,
            breakout_pressure=1.0,
            trap_inverse=1.0,
        )
        assert score.total == pytest.approx(10.0, abs=1e-6)

    def test_label_progression(self):
        from trading_engine.intelligence.models import SignalScore
        # Very strong
        strong = SignalScore(
            underlying_direction=1.0, underlying_acceleration=1.0,
            premium_velocity_ce=1.0, premium_velocity_pe=1.0,
            premium_acceleration=1.0, volume_acceleration=1.0,
            relative_ce_pe=1.0, breakout_pressure=1.0, trap_inverse=1.0,
        )
        assert strong.label() == "VERY STRONG"
        # No signal (all zeros)
        empty = SignalScore()
        assert empty.label() == "NO SIGNAL"


# ---------------------------------------------------------------------------
# Tests: LatencyTrace
# ---------------------------------------------------------------------------

class TestLatencyTrace:
    def test_latency_ms_computed_correctly(self):
        from datetime import UTC, datetime, timedelta

        from trading_engine.intelligence.models import LatencyTrace
        t0 = datetime.now(UTC)
        lt = LatencyTrace(
            data_received_at=t0,
            features_computed_at=t0 + timedelta(milliseconds=10),
            signal_generated_at=t0 + timedelta(milliseconds=35),
            report_published_at=t0 + timedelta(milliseconds=50),
        )
        assert lt.feature_latency_ms == pytest.approx(10.0, abs=1.0)
        assert lt.signal_latency_ms == pytest.approx(35.0, abs=1.0)
        assert lt.total_latency_ms == pytest.approx(50.0, abs=1.0)

    def test_none_fields_return_nan(self):
        from datetime import UTC, datetime

        from trading_engine.intelligence.models import LatencyTrace
        lt = LatencyTrace(data_received_at=datetime.now(UTC))
        assert np.isnan(lt.feature_latency_ms)
        assert np.isnan(lt.signal_latency_ms)
        assert np.isnan(lt.total_latency_ms)
