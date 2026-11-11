"""Default terminal page: fresh signals, active positions, and system health."""
import streamlit as st

from trading_engine.dashboard.shared import execution_desk, inject_styles


def main() -> None:
    st.set_page_config(page_title="Execution Desk · NIFTY", layout="wide")
    inject_styles()
    st.fragment(run_every="5s")(execution_desk)()


if __name__ == "__main__":
    main()
