"""Current contract and warmed option history fixtures for analytical tests."""
import pandas as pd
import pytest


@pytest.fixture
def current_option_chain():
    def complete(frame, now):
        frame = frame.copy()
        frame["expiry"] = (pd.Timestamp(now).tz_convert("Asia/Kolkata") + pd.Timedelta(days=7)).date().isoformat()
        frame["open_interest"] = 1000.0
        other_side = frame.copy()
        other_side["option_type"] = other_side["option_type"].map({"CE": "PE", "PE": "CE"})
        if "instrument_token" in other_side:
            other_side["instrument_token"] += 100000
        result = pd.concat([frame, other_side], ignore_index=True)
        result["open_interest"] = result.option_type.map({"CE": 1000., "PE": 1600.})
        result["delta"] = result.option_type.map({"CE": .7, "PE": -.7})
        return result
    return complete
