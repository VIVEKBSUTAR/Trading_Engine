"""Warmed option quote history shared by monitor and dashboard tests."""
import pandas as pd
import pytest


@pytest.fixture
def option_history(market_clock):
    def build(now):
        boundary = pd.Timestamp(now).floor("min")
        return pd.DataFrame([
            {"timestamp": boundary - pd.Timedelta(minutes=minute) + pd.Timedelta(seconds=second),
             "bid_price": price - 0.1, "ask_price": price + 0.1}
            for minute in range(15, 0, -1)
            for second, price in ((1, 100.0), (30, 104.0))
        ])
    return build




@pytest.fixture
def market_clock(monkeypatch, request):
    """Stable intraday clock: ATR tests must not depend on the host's midnight."""
    from datetime import UTC
    from datetime import datetime as real_datetime

    import trading_engine.intelligence.monitor as monitor

    class DateTimeMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, real_datetime)

    class MarketDateTime(real_datetime, metaclass=DateTimeMeta):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 11, 12, 6, 0, tzinfo=UTC)
            return value.astimezone(tz) if tz is not None else value.astimezone().replace(tzinfo=None)

    monkeypatch.setattr(request.module, "datetime", MarketDateTime)
    monkeypatch.setattr(monitor, "datetime", MarketDateTime)
    return MarketDateTime
