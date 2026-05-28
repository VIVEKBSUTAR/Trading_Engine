"""Streamlit dashboard for live NIFTY options intelligence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from config.settings import AppSettings
from trading_engine.intelligence.models import IntelligenceReport, MarketRegime, SessionPhase, TradeGrade
from trading_engine.intelligence.runtime import LiveIntelligenceRuntime


def main() -> None:
    """Render the live intelligence dashboard."""
    st.set_page_config(page_title="Trading Engine Intelligence", layout="wide")
    settings = AppSettings.from_env()
    st.title("NIFTY Options Intelligence Dashboard")
    st.caption("Probabilistic live market intelligence for short-horizon directional expansion")

    if "runtime" not in st.session_state:
        st.session_state.runtime = None
        st.session_state.last_report = None

    runtime = _get_runtime(settings)
    if runtime is None:
        st.error("Kite Connect credentials are missing. Populate .env and reload the app.")
        return

    st.caption(f"Manual refresh recommended every {settings.intelligence.dashboard_refresh_seconds} seconds while live mode is active.")

    if st.button("Refresh Now") or st.session_state.last_report is None:
        try:
            st.session_state.last_report = runtime.refresh_once()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to refresh live report: {exc}")
            return

    report: IntelligenceReport = st.session_state.last_report

    _render_header(report)
    col1, col2 = st.columns([3.2, 1.25], gap="large")

    with col1:
        _render_state_panel(report)
        _render_price_chart(report)
        _render_probability_panel(report)
        _render_option_chain_panel(report)

    with col2:
        _render_signal_ticket(report)
        _render_trade_monitor(report)
        _render_alerts(report)


def _get_runtime(settings: AppSettings) -> LiveIntelligenceRuntime | None:
    if not settings.kite.api_key or not settings.kite.access_token and not settings.kite.request_token:
        return None

    runtime = st.session_state.runtime
    if runtime is None:
        runtime = LiveIntelligenceRuntime(settings)
        st.session_state.runtime = runtime
    return runtime


def _render_header(report: IntelligenceReport) -> None:
    spot = report.snapshot.spot_price or 0.0
    vix = report.snapshot.vix_value or 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"{spot:,.2f}", delta=f"{report.probabilities.bullish_probability - report.probabilities.bearish_probability:+.1%}")
    c2.metric("VIX", f"{vix:,.2f}", delta=f"{report.state.volatility_state.value}")
    c3.metric("Regime", report.state.regime.value.replace("_", " ").title(), delta=f"{report.state.regime_confidence:.0%}")
    c4.metric("Signal", report.signal.action.value.replace("_", " ").title(), delta=f"Grade {report.signal.trade_grade.value}")


def _render_state_panel(report: IntelligenceReport) -> None:
    st.subheader("Market State")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trend", report.state.trend_state.value.replace("_", " ").title(), delta=f"{report.state.trend_strength:+.3f}")
    c2.metric("Volatility", report.state.volatility_state.value.replace("_", " ").title(), delta=f"{report.state.session_quality:.0%}")
    c3.metric("Session", report.state.session_state.value.replace("_", " ").title())
    c4.metric("Trade Grade", report.state.trade_grade.value)
    c5.metric("Quality", f"{report.state.quality_score:.0%}", delta=report.state.transition.stage.value.replace("_", " ").title())
    st.write(
        f"Transition: **{report.transition.stage.value.replace('_', ' ').title()}** | "
        f"Directional bias: **{report.state.directional_bias.value.replace('_', ' ').title()}** | "
        f"Trap probability: **{report.state.trap_probability:.0%}**"
    )


def _render_price_chart(report: IntelligenceReport) -> None:
    st.subheader("Live Market Structure")
    candles = report.snapshot.candles_1m.tail(120)
    if candles.empty:
        st.info("Waiting for live candle data.")
        return

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=candles["timestamp"],
                open=candles["open"],
                high=candles["high"],
                low=candles["low"],
                close=candles["close"],
                name="NIFTY",
            )
        ]
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_probability_panel(report: IntelligenceReport) -> None:
    st.subheader("Directional Probabilities")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bullish", f"{report.probabilities.bullish_probability:.1%}")
    c2.metric("Bearish", f"{report.probabilities.bearish_probability:.1%}")
    c3.metric("Sideways", f"{report.probabilities.sideways_probability:.1%}")
    st.write("Evidence:", ", ".join(report.probabilities.evidence) if report.probabilities.evidence else "None")
    st.write("Trap evidence:", ", ".join(report.trap.evidence) if report.trap.evidence else "None")


def _render_option_chain_panel(report: IntelligenceReport) -> None:
    st.subheader("Option Chain Snapshot")
    chain = report.snapshot.option_chain_frame
    if chain.empty:
        st.info("No option-chain snapshot available yet.")
        return

    display_cols = [column for column in ["timestamp", "expiry", "strike", "option_type", "open_interest", "implied_volatility", "last_price", "underlying_price"] if column in chain.columns]
    st.dataframe(chain.loc[:, display_cols].tail(40), use_container_width=True, height=240)


def _render_signal_ticket(report: IntelligenceReport) -> None:
    st.subheader("Signal Ticket")
    st.write(f"Action: **{report.signal.action.value.upper()}**")
    st.write(f"Confidence: **{report.signal.confidence:.1%}**")
    st.write(f"Trade Grade: **{report.signal.trade_grade.value}**")
    st.write(f"Strike: **{report.strike.strike if report.strike else 'N/A'}**")
    st.write(f"Expiry: **{report.strike.expiry or 'N/A'}**")
    st.write(f"Reasoning: {', '.join(report.signal.reasoning) if report.signal.reasoning else 'No trade filter'}")
    st.write(f"Entry: {report.signal.entry_reference if report.signal.entry_reference is not None else 'N/A'}")
    st.write(f"Stop Loss: {report.signal.stop_loss if report.signal.stop_loss is not None else 'N/A'}")
    st.write(f"Target: {report.signal.target if report.signal.target is not None else 'N/A'}")
    st.write(f"Trap Probability: **{report.trap.trap_score:.1%}**")
    st.write(f"Signal Expires: **{_expiry_text(report.signal.expires_at)}**")


def _render_trade_monitor(report: IntelligenceReport) -> None:
    st.subheader("Open Trades")
    open_trades = report.trade_update.open_trades
    if not open_trades:
        st.info("No open live trades.")
        return

    trade_frame = pd.DataFrame([asdict(trade) for trade in open_trades])
    st.dataframe(trade_frame, use_container_width=True, height=260)
    st.caption(f"Total closed PnL: {report.trade_update.total_pnl:,.2f}")
    if report.trade_update.expired_trades:
        st.caption(f"Expired trades: {len(report.trade_update.expired_trades)}")


def _render_alerts(report: IntelligenceReport) -> None:
    st.subheader("Alerts")
    if not report.trade_update.alert_messages:
        st.info("No alerts yet.")
        return
    for item in report.trade_update.alert_messages[-10:]:
        st.warning(item)


def _expiry_text(expiry: datetime | None) -> str:
    if expiry is None:
        return "N/A"
    remaining = expiry - datetime.now(UTC)
    minutes = max(int(remaining.total_seconds() // 60), 0)
    return f"{expiry.isoformat()} ({minutes}m left)"


if __name__ == "__main__":
    main()
