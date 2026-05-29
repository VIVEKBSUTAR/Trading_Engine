"""Streamlit dashboard for live NIFTY options intelligence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import AppSettings
from trading_engine.intelligence.models import IntelligenceReport, SignalAction, TradeGrade
from trading_engine.intelligence.runtime import LiveIntelligenceRuntime


def main() -> None:
    """Render the live intelligence dashboard."""
    st.set_page_config(page_title="NIFTY Intelligence Terminal", layout="wide", initial_sidebar_state="collapsed")
    _inject_styles()
    settings = AppSettings.from_env()
    st.markdown("## NIFTY Intelligence Terminal")
    st.caption("Institutional probabilistic market-state terminal for selective intraday options participation")

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

    _render_header_strip(report)
    left, center, right = st.columns([1.65, 1.45, 1.05], gap="large")

    with left:
        _render_market_intelligence_panel(report)
        _render_session_panel(report)
        _render_option_chain_panel(report)

    with center:
        _render_signal_terminal(report)
        _render_structure_panel(report)

    with right:
        _render_trade_monitor(report)
        _render_alerts(report)
        _render_analytics_panel(report)


def _get_runtime(settings: AppSettings) -> LiveIntelligenceRuntime | None:
    if not settings.kite.api_key or not settings.kite.access_token and not settings.kite.request_token:
        return None

    runtime = st.session_state.runtime
    if runtime is None:
        runtime = LiveIntelligenceRuntime(settings)
        st.session_state.runtime = runtime
    return runtime


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1680px;
        }
        div[data-testid="stMetric"] {
            background: rgba(8, 15, 32, 0.92);
            border: 1px solid rgba(66, 71, 84, 0.85);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
        }
        div[data-testid="stMetric"] label {
            color: #8ca0c8 !important;
            font-size: 0.70rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
        }
        div[data-testid="stMetric"] div[data-testid="metric-container"] {
            gap: 0.2rem;
        }
        section[data-testid="stSidebar"] {
            background: #060e20;
        }
        .terminal-note {
            color: #8ca0c8;
            font-size: 0.88rem;
            margin-top: -0.3rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header_strip(report: IntelligenceReport) -> None:
    spot = report.snapshot.spot_price or 0.0
    vix = report.snapshot.vix_value or 0.0
    r1 = st.columns(4)
    r1[0].metric("NIFTY Spot", f"{spot:,.2f}", delta=f"{report.state.directional_bias.value.replace('_', ' ').title()}")
    r1[1].metric("India VIX", f"{vix:,.2f}", delta=report.state.volatility_state.value.replace("_", " ").title())
    r1[2].metric("Session Phase", report.state.session_state.value.replace("_", " ").title())
    r1[3].metric("Market Regime", report.state.regime.value.replace("_", " ").title(), delta=f"{report.state.regime_confidence:.0%}")

    r2 = st.columns(4)
    r2[0].metric("Transition State", report.state.transition.stage.value.replace("_", " ").title(), delta=f"{report.state.transition.confidence:.0%}")
    r2[1].metric("Trend Strength", f"{report.state.trend_strength:+.3f}")
    r2[2].metric("Volatility State", report.state.volatility_state.value.replace("_", " ").title())
    r2[3].metric("Market Quality", report.state.trade_grade.value, delta=f"Chop {report.state.chop_probability:.0%}")
    st.markdown(
        f"<div class='terminal-note'>Trap {report.state.trap_probability:.0%} | Persistence {report.state.persistence_probability:.0%} | Exhaustion {report.state.exhaustion_probability:.0%} | Instability {report.state.instability_probability:.0%}</div>",
        unsafe_allow_html=True,
    )


def _render_market_intelligence_panel(report: IntelligenceReport) -> None:
    st.subheader("Market Intelligence")
    prob_cols = st.columns(3)
    _meter(prob_cols[0], "Bullish Probability", report.probabilities.bullish_probability)
    _meter(prob_cols[1], "Bearish Probability", report.probabilities.bearish_probability)
    _meter(prob_cols[2], "Sideways Probability", report.probabilities.sideways_probability)

    risk_cols = st.columns(2)
    _meter(risk_cols[0], "Chop Risk", report.state.chop_probability, inverse=True)
    _meter(risk_cols[1], "Trap Probability", report.state.trap_probability, inverse=True)

    stats = st.columns(4)
    stats[0].metric("Momentum Quality", f"{report.state.persistence_probability:.0%}")
    stats[1].metric("Liquidity State", report.state.liquidity_state.value.replace("_", " ").title())
    stats[2].metric("Breakout Quality", f"{abs(report.state.breakout_pressure):.3f}")
    stats[3].metric("Trend Quality", report.state.trend_state.value.replace("_", " ").title())

    st.caption(
        "Evidence: " + (", ".join(report.probabilities.evidence[-4:]) if report.probabilities.evidence else "None")
    )
    st.caption("Trap evidence: " + (", ".join(report.trap.evidence[-4:]) if report.trap.evidence else "None"))


def _render_session_panel(report: IntelligenceReport) -> None:
    st.subheader("Session Intelligence")
    phase = report.state.session_state.value.replace("_", " ").title()
    session_map = {
        "Opening Expansion": "Opening expansion: wider confirmation windows and fast breakout validation.",
        "Trend Window": "Trend development: favor continuation with clean alignment.",
        "Midday Compression": "Midday compression: expect chop, reduce participation, confirm twice.",
        "Transitional Noise": "Transitional noise: be selective, lower size, avoid forcing momentum.",
        "Afternoon Expansion": "Afternoon expansion: watch for orderly trend continuation or breakout resumption.",
        "Closing Volatility": "Closing volatility: faster decay, tighter risk, quicker exits.",
    }
    st.metric("Current Session", phase, delta=f"Quality {report.state.session_quality:.0%}")
    st.write(session_map.get(phase, "Session context unavailable."))
    c1, c2 = st.columns(2)
    c1.metric("Time From Open", f"{report.features.feature_map.get('time_from_open_minutes', 0):.0f} min")
    c2.metric("Time To Close", f"{report.features.feature_map.get('time_to_close_minutes', 0):.0f} min")


def _render_structure_panel(report: IntelligenceReport) -> None:
    st.subheader("Market Structure")
    candles = report.snapshot.candles_1m.tail(40)
    if not candles.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=candles["timestamp"], y=candles["close"], mode="lines", line=dict(color="#adc6ff", width=2), name="NIFTY"))
        fig.update_layout(height=220, margin=dict(l=8, r=8, t=20, b=8), xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Waiting for structure data.")

    bars = st.columns(2)
    _meter(bars[0], "Compression", float(report.state.transition.instability_probability if report.state.transition else 0.0), inverse=True)
    _meter(bars[1], "Trend Persistence", report.state.persistence_probability)
    st.write(f"Transition: **{report.state.transition.stage.value.replace('_', ' ').title()}** → **{report.transition.to_regime.value.replace('_', ' ').title()}**")
    st.write(f"Transition path: {' → '.join(report.state.transition.path)}")


def _meter(container: st.delta_generator.DeltaGenerator, label: str, value: float, *, inverse: bool = False) -> None:
    value = max(0.0, min(float(value), 1.0))
    container.markdown(f"**{label}**")
    container.progress(int(value * 100))
    suffix = "low" if inverse and value < 0.35 else "high" if value > 0.68 else "moderate"
    container.caption(f"{value:.0%} | {suffix}")


def _render_option_chain_panel(report: IntelligenceReport) -> None:
    st.subheader("Option Chain Snapshot")
    chain = report.snapshot.option_chain_frame
    if chain.empty:
        st.info("No option-chain snapshot available yet.")
        return

    display_cols = [column for column in ["timestamp", "expiry", "strike", "option_type", "open_interest", "implied_volatility", "last_price", "underlying_price"] if column in chain.columns]
    summary_cols = st.columns(4)
    summary_cols[0].metric("PCR", f"{report.features.feature_map.get('put_call_ratio', float('nan')):.2f}" if pd.notna(report.features.feature_map.get('put_call_ratio')) else "N/A")
    summary_cols[1].metric("Call Wall", f"{report.features.feature_map.get('call_wall', float('nan')):.0f}" if pd.notna(report.features.feature_map.get('call_wall')) else "N/A")
    summary_cols[2].metric("Put Wall", f"{report.features.feature_map.get('put_wall', float('nan')):.0f}" if pd.notna(report.features.feature_map.get('put_wall')) else "N/A")
    summary_cols[3].metric("IV Skew", f"{report.features.feature_map.get('iv_skew', float('nan')):.2f}" if pd.notna(report.features.feature_map.get('iv_skew')) else "N/A")

    top_ce = chain.loc[chain["option_type"].astype(str).str.upper().eq("CE")].sort_values("open_interest", ascending=False).head(1)
    top_pe = chain.loc[chain["option_type"].astype(str).str.upper().eq("PE")].sort_values("open_interest", ascending=False).head(1)
    c1, c2 = st.columns(2)
    c1.write(f"Strongest CE OI: **{_top_option_text(top_ce)}**")
    c2.write(f"Strongest PE OI: **{_top_option_text(top_pe)}**")
    st.caption(f"OI bias: {report.features.feature_map.get('option_chain_bias', 0.0):+.3f} | IV spread: {report.features.feature_map.get('iv_realized_spread', float('nan')):.2f}" if pd.notna(report.features.feature_map.get('iv_realized_spread')) else f"OI bias: {report.features.feature_map.get('option_chain_bias', 0.0):+.3f}")
    st.dataframe(chain.loc[:, display_cols].tail(12), use_container_width=True, height=220)


def _render_signal_terminal(report: IntelligenceReport) -> None:
    st.subheader("Live Signal Terminal")
    direction = _direction_label(report)
    action = report.signal.action.value.replace("_", " ").upper()
    signal_cols = st.columns(2)
    signal_cols[0].metric("Direction", direction)
    signal_cols[1].metric("Trade Type", action)

    ticket_cols = st.columns(2)
    ticket_cols[0].write(f"Suggested Strike: **{_strike_text(report)}**")
    ticket_cols[1].write(f"Entry Zone: **{_entry_zone(report)}**")
    ticket_cols[0].write(f"Stoploss: **{_price_text(report.signal.stop_loss)}**")
    ticket_cols[1].write(f"Targets: **{_target_ladder(report)}**")
    ticket_cols[0].write(f"Signal TTL: **{report.signal.signal_ttl_candles or report.state.signal_ttl_candles} candles**")
    ticket_cols[1].write(f"Trade Grade: **{report.signal.trade_grade.value}** | Risk Profile: **{_risk_profile(report)}**")
    st.write(f"Expected Move Horizon: **{report.probabilities.horizon_minutes}-{report.probabilities.horizon_minutes + 11} minutes**")
    st.markdown(f"**Action:** { _action_text(report) }")
    st.markdown("**Reason**")
    for reason in _signal_reasons(report):
        st.write(f"• {reason}")
    st.caption(f"Expires: {_expiry_text(report.signal.expires_at)}")


def _render_trade_monitor(report: IntelligenceReport) -> None:
    st.subheader("Open Trade Monitor")
    open_trades = report.trade_update.open_trades
    if not open_trades:
        st.info("No open live trades.")
    else:
        trade_frame = pd.DataFrame([asdict(trade) for trade in open_trades])
        st.dataframe(trade_frame, use_container_width=True, height=220)

    c1, c2, c3 = st.columns(3)
    c1.metric("Open", len(report.trade_update.open_trades))
    c2.metric("Closed", len(report.trade_update.closed_trades))
    c3.metric("Expired", len(report.trade_update.expired_trades))
    st.caption(f"Total closed P&L: {report.trade_update.total_pnl:,.2f}")
    if report.trade_update.alert_messages:
        st.caption(f"Lifecycle notes: {report.trade_update.alert_messages[-1]}")


def _render_alerts(report: IntelligenceReport) -> None:
    st.subheader("Alerts / No-Trade Warnings")
    warnings = _warning_lines(report)
    if not warnings:
        st.success("Current state is not forcing a no-trade warning.")
        return
    for item in warnings:
        st.warning(item)


def _render_analytics_panel(report: IntelligenceReport) -> None:
    st.subheader("Performance & Analytics")
    closed = report.trade_update.closed_trades
    closed_count = len(closed)
    wins = sum(1 for trade in closed if float(getattr(trade, "realized_pnl", 0.0) or 0.0) > 0)
    win_rate = wins / closed_count if closed_count else 0.0
    avg_pnl = sum(float(getattr(trade, "realized_pnl", 0.0) or 0.0) for trade in closed) / closed_count if closed_count else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Win Rate", f"{win_rate:.0%}" if closed_count else "N/A")
    c2.metric("Expectancy", f"{avg_pnl:,.2f}" if closed_count else "N/A")
    c3.metric("Drawdown", f"{min(0.0, report.trade_update.total_pnl):,.2f}")
    c4, c5 = st.columns(2)
    c4.metric("Regime-Wise", report.state.regime.value.replace("_", " ").title())
    c5.metric("Confidence Reliability", f"{report.probabilities.confidence:.0%}")


def _expiry_text(expiry: datetime | None) -> str:
    if expiry is None:
        return "N/A"
    remaining = expiry - datetime.now(UTC)
    minutes = max(int(remaining.total_seconds() // 60), 0)
    return f"{expiry.isoformat()} ({minutes}m left)"


def _top_option_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "N/A"
    row = frame.iloc[0]
    strike = row.get("strike")
    oi = row.get("open_interest")
    expiry = row.get("expiry")
    return f"{strike} ({int(oi) if pd.notna(oi) else 'N/A'} OI, {expiry})"


def _direction_label(report: IntelligenceReport) -> str:
    if report.signal.action == SignalAction.BUY_CE:
        return "BULLISH"
    if report.signal.action == SignalAction.BUY_PE:
        return "BEARISH"
    return "NO TRADE"


def _strike_text(report: IntelligenceReport) -> str:
    if report.strike and report.strike.strike is not None:
        return f"{report.strike.strike:,.0f} {report.strike.option_side.value}"
    return "N/A"


def _entry_zone(report: IntelligenceReport) -> str:
    entry = report.signal.entry_reference
    if entry is None:
        return "N/A"
    band = max(abs(entry) * 0.0015, 1.0)
    return f"₹{entry - band:,.2f} – ₹{entry + band:,.2f}"


def _price_text(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"₹{value:,.2f}"


def _target_ladder(report: IntelligenceReport) -> str:
    target = report.signal.target
    entry = report.signal.entry_reference
    if target is None:
        return "N/A"
    if entry is None:
        return f"₹{target:,.2f}"
    step = max(abs(target - entry), max(abs(entry) * 0.0015, 1.0))
    if target >= entry:
        targets = [target, target + 0.5 * step, target + step]
    else:
        targets = [target, target - 0.5 * step, target - step]
    return ", ".join(_price_text(value) for value in targets)


def _risk_profile(report: IntelligenceReport) -> str:
    if report.signal.action == SignalAction.NO_TRADE:
        return "IDLE"
    if report.signal.trade_grade.value == "A+":
        return "LOW"
    if report.signal.trade_grade.value == "A":
        return "MEDIUM"
    return "ELEVATED"


def _action_text(report: IntelligenceReport) -> str:
    if report.signal.action == SignalAction.NO_TRADE:
        return "HOLD / NO TRADE"
    if report.signal.action == SignalAction.BUY_CE:
        return "BUY ON PULLBACK"
    if report.signal.action == SignalAction.BUY_PE:
        return "SELL RALLIES / BUY PE ON FAILURE"
    return report.signal.action.value.replace("_", " ").upper()


def _signal_reasons(report: IntelligenceReport) -> list[str]:
    reasons = list(report.signal.reasoning or [])
    reasons.extend(report.probabilities.evidence[-3:])
    reasons.extend(report.trap.evidence[-2:])
    if report.state.transition.reasons:
        reasons.extend(report.state.transition.reasons[-2:])
    cleaned: list[str] = []
    for reason in reasons:
        if reason and reason not in cleaned:
            cleaned.append(reason)
    return cleaned[:6] or ["No trade filter"]


def _warning_lines(report: IntelligenceReport) -> list[str]:
    warnings: list[str] = []
    if report.state.chop_probability >= 0.68 or report.state.transition.stage.value == "sideways":
        warnings.append("High overlap or compression detected. Range behavior dominates. Prefer no trade.")
    if report.state.instability_probability >= 0.65:
        warnings.append("Directional instability elevated. Confirmation should be stricter.")
    if report.state.trap_probability >= 0.60:
        warnings.append("Trap probability is elevated. Breakout reliability is reduced.")
    if report.state.trade_grade == TradeGrade.AVOID:
        warnings.append("Market state is graded Avoid. Capital preservation should take priority.")
    if report.state.session_state.value in {"midday_compression", "transitional_noise"}:
        warnings.append("Session conditions are less favorable for momentum participation.")
    return warnings


if __name__ == "__main__":
    main()
