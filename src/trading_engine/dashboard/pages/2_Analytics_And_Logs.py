"""Secondary terminal page: historical performance, equity, and signal audits."""
import streamlit as st

from trading_engine.dashboard.shared import analytics_desk, inject_styles


def main() -> None:
    st.set_page_config(page_title="Analytics & Logs · NIFTY", layout="wide")
    inject_styles()
    st.fragment(run_every="5s")(analytics_desk)()


if __name__ == "__main__":
    main()
