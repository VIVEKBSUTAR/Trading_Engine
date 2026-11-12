"""Execution desk with IST telemetry, lifecycle alerts, and active positions.

Shared renderers format UTC instants as twelve-hour Asia/Kolkata timestamps.
"""
import streamlit as st

from trading_engine.dashboard.shared import execution_desk, inject_styles


def main() -> None:
    st.set_page_config(page_title="Execution Desk · NIFTY", layout="wide")
    inject_styles()
    st.fragment(run_every="5s")(execution_desk)()


if __name__ == "__main__":
    main()
