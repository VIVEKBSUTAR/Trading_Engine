"""Streamlit dashboard for live NIFTY options intelligence."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import AppSettings
from storage.duckdb_manager import DuckDBManager
from trading_engine.intelligence.models import (
    IntelligenceReport,
    OpenTrade,
    SignalAction,
    TradeGrade,
)
from trading_engine.intelligence.monitor import NIFTY_LOT_SIZE, LiveTradeMonitor
from trading_engine.intelligence.runtime import LiveIntelligenceRuntime


def compute_performance_metrics(trade_outcomes: pd.DataFrame) -> dict[str, Any]:
    """Calculate total realized profit, total closed trades, win rate %, and profit factor."""
    if trade_outcomes.empty or "realized_pnl" not in trade_outcomes.columns:
        return {
            "total_realized_profit": 0.0,
            "total_closed_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
        }

    pnl = pd.to_numeric(trade_outcomes["realized_pnl"], errors="coerce").fillna(0.0)
    total_realized_profit = float(pnl.sum())
    total_closed_trades = len(pnl)

    winning_trades = pnl[pnl > 0]
    losing_trades = pnl[pnl < 0]

    win_rate_pct = (len(winning_trades) / total_closed_trades * 100.0) if total_closed_trades > 0 else 0.0

    gross_profit = float(winning_trades.sum())
    gross_loss = abs(float(losing_trades.sum()))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "total_realized_profit": total_realized_profit,
        "total_closed_trades": total_closed_trades,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
    }


def compute_equity_curve(trade_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative profit chronologically across closed trades."""
    if trade_outcomes.empty or "realized_pnl" not in trade_outcomes.columns or "exit_timestamp" not in trade_outcomes.columns:
        return pd.DataFrame(columns=["exit_timestamp", "cumulative_profit"])

    df = trade_outcomes.copy()
    df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"], errors="coerce")
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["exit_timestamp"]).sort_values("exit_timestamp", ascending=True)
    df["cumulative_profit"] = df["realized_pnl"].cumsum()
    return df[["exit_timestamp", "cumulative_profit"]]


def compute_unrealized_pnl(trade: OpenTrade) -> float:
    """Calculate unrealized profit: (current_bid - entry_ask) * 65 * quantity."""
    bid = float(trade.current_bid_price or 0.0)
    ask = float(trade.entry_ask_price or 0.0)
    qty = int(trade.quantity or 0)
    return (bid - ask) * float(NIFTY_LOT_SIZE) * float(qty)


def query_recent_signals(duckdb_mgr: DuckDBManager, limit: int = 50) -> pd.DataFrame:
    """Query DuckDB signals table for recent records."""
    sql = f"""
    SELECT
        timestamp,
        spot_price,
        strike,
        option_type,
        action,
        confidence,
        trap_score,
        filter_passed,
        filter_rejection_reason AS rejection_reason
    FROM signals
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    df = duckdb_mgr.query(sql)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "spot_price",
                "strike",
                "option_type",
                "action",
                "confidence",
                "trap_score",
                "filter_passed",
                "rejection_reason",
            ]
        )
    return df


def get_dashboard_duckdb_manager(settings: AppSettings) -> DuckDBManager:
    """Initialize or reuse a thread-safe read-only DuckDBManager."""
    if "duckdb_mgr" in st.session_state and st.session_state["duckdb_mgr"] is not None:
        return st.session_state["duckdb_mgr"]

    # If runtime is in session state, reuse its duckdb_manager if possible
    if "runtime" in st.session_state and st.session_state["runtime"] is not None:
        runtime_eng = getattr(st.session_state["runtime"], "engine", None)
        if runtime_eng is not None and getattr(runtime_eng, "duckdb_manager", None) is not None:
            st.session_state["duckdb_mgr"] = runtime_eng.duckdb_manager
            return runtime_eng.duckdb_manager

    mgr = DuckDBManager(settings.storage.duckdb_path, read_only=True)
    st.session_state["duckdb_mgr"] = mgr
    return mgr


def main() -> None:
    """Render the live intelligence dashboard."""
    st.set_page_config(page_title="NIFTY Intelligence Terminal", layout="wide", initial_sidebar_state="collapsed")
    _inject_styles()
    settings = AppSettings.from_env()

    # Step 5: Automatic refresh controls in sidebar
    refresh_seconds = getattr(settings.intelligence, "dashboard_refresh_seconds", 5)
    st.sidebar.markdown("### Terminal Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        st.sidebar.caption(f"Refresh interval: {refresh_seconds}s")

    st.markdown("## NIFTY Intelligence Terminal")
    st.caption("Institutional probabilistic market-state terminal for selective intraday options participation")

    # Step 1: Initialize DuckDBManager and render summary performance banner
    duckdb_mgr = get_dashboard_duckdb_manager(settings)
    trade_outcomes = duckdb_mgr.get_trade_outcomes()
    perf_metrics = compute_performance_metrics(trade_outcomes)
    _render_performance_banner(perf_metrics)

    # Step 2: Render equity curve section below performance banner
    _render_equity_curve_section(trade_outcomes)

    if "runtime" not in st.session_state:
        st.session_state.runtime = None
        st.session_state.last_report = None

    runtime = _get_runtime(settings)
    if runtime is None:
        st.info("Kite Connect credentials unconfigured. Displaying DuckDB historical analytics and database logs.")
    else:
        st.caption(f"Manual refresh recommended every {refresh_seconds} seconds while live mode is active.")
        if st.button("Refresh Now") or st.session_state.last_report is None:
            try:
                st.session_state.last_report = runtime.refresh_once()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to refresh live report: {exc}")

    report: IntelligenceReport | None = st.session_state.last_report

    if report is not None:
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
            if runtime is not None:
                _render_api_health_panel(runtime)
            _render_active_positions(runtime.monitor if runtime else None, report)
            _render_alerts(report)
            _render_analytics_panel(report)
    else:
        # Step 3: Rewrite active positions display even when offline
        _render_active_positions(runtime.monitor if runtime else None, None)

    # Step 4: Signal log section displaying recent 50 records from DuckDB
    _render_signal_log_section(duckdb_mgr)

    # Step 5: Automatic refresh rerun timer
    if auto_refresh and st.runtime.exists():
        time.sleep(refresh_seconds)
        st.rerun()


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


def _render_performance_banner(metrics: dict[str, Any]) -> None:
    """Step 1: Summary performance banner at the top of the dashboard with 4 metric cards."""
    st.markdown("### Historical Performance Overview (DuckDB Analytics)")
    cols = st.columns(4)
    total_pnl = metrics["total_realized_profit"]
    delta_str = f"{'+' if total_pnl > 0 else ''}₹{total_pnl:,.2f}" if total_pnl != 0 else None
    cols[0].metric(
        label="Total Realized Profit",
        value=f"₹{total_pnl:,.2f}",
        delta=delta_str,
    )
    cols[1].metric(
        label="Total Closed Trades",
        value=f"{metrics['total_closed_trades']:,}",
    )
    cols[2].metric(
        label="Win Rate",
        value=f"{metrics['win_rate_pct']:.1f}%",
    )
    pf = metrics["profit_factor"]
    pf_display = "∞" if pf == float("inf") else (f"{pf:.2f}" if pf > 0 else "0.00")
    cols[3].metric(
        label="Profit Factor",
        value=pf_display,
    )


def _render_equity_curve_section(trade_outcomes: pd.DataFrame) -> None:
    """Step 2: Equity curve section below performance banner using Streamlit native line chart."""
    st.markdown("#### Realized Cumulative Equity Curve")
    equity_df = compute_equity_curve(trade_outcomes)
    if equity_df.empty:
        st.info("No closed trade records found in DuckDB yet. Equity curve will populate as trades complete.")
        return

    chart_data = equity_df.set_index("exit_timestamp")[["cumulative_profit"]]
    st.line_chart(chart_data, y="cumulative_profit", use_container_width=True)


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

    # Latency and Trap/Risk metadata strip
    lat_text = "N/A"
    if report.latency is not None:
        lat_text = f"Total: {report.latency.total_latency_ms:.1f}ms (Feat: {report.latency.feature_latency_ms:.1f}ms | Sig: {report.latency.signal_latency_ms:.1f}ms)"
    
    # Classify trap risk label
    trap_prob = report.state.trap_probability
    if trap_prob >= 0.65:
        trap_label = "HIGH RISK"
    elif trap_prob >= 0.35:
        trap_label = "MEDIUM RISK"
    else:
        trap_label = "LOW RISK"

    st.markdown(
        f"<div class='terminal-note'>Processing Latency: <b>{lat_text}</b> | Trap: <b>{trap_label} ({trap_prob:.0%})</b> | Persistence {report.state.persistence_probability:.0%} | Exhaustion {report.state.exhaustion_probability:.0%} | Instability {report.state.instability_probability:.0%}</div>",
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

    # Premium Velocity indicators
    def get_velocity_label(vel: float) -> str:
        if pd.isna(vel) or vel == 0.0:
            return "FLAT"
        if vel > 0.005:
            return "ACCELERATING" if vel > 0.012 else "RISING"
        if vel < -0.005:
            return "DECELERATING"
        return "FLAT"

    ce_vel_label = get_velocity_label(report.ce_premium_velocity)
    pe_vel_label = get_velocity_label(report.pe_premium_velocity)

    vel_cols = st.columns(2)
    vel_cols[0].metric("ATM CE Velocity", ce_vel_label, f"Score: {report.ce_score:.1f}/10")
    vel_cols[1].metric("ATM PE Velocity", pe_vel_label, f"Score: {report.pe_score:.1f}/10")

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
        fig.add_trace(go.Scatter(x=candles["timestamp"], y=candles["close"], mode="lines", line={"color": "#adc6ff", "width": 2}, name="NIFTY"))
        fig.update_layout(height=220, margin={"l": 8, "r": 8, "t": 20, "b": 8}, xaxis_rangeslider_visible=False, showlegend=False)
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
    
    # Signal countdown in seconds
    age_sec = report.signal.signal_age_seconds()
    val_sec = report.signal.signal_validity_seconds
    time_str = "N/A"
    if pd.notna(age_sec) and pd.notna(val_sec):
        rem_sec = max(0, int(val_sec - age_sec))
        time_str = f"{rem_sec}s remaining (Age: {int(age_sec)}s)"
        
    ticket_cols[0].write(f"Signal Validity: **{time_str}**")
    ticket_cols[1].write(f"Trade Grade: **{report.signal.trade_grade.value}** | Risk Profile: **{_risk_profile(report)}**")
    st.write(f"Expected Move Horizon: **{report.probabilities.horizon_minutes}-{report.probabilities.horizon_minutes + 11} minutes**")
    
    if report.signal.score is not None:
        st.write(f"Composite Score: **{report.signal.score.total:.2f}/10** ({report.signal.score.label()})")

    st.markdown(f"**Action:** { _action_text(report) }")
    st.markdown("**Reason**")
    for reason in _signal_reasons(report):
        st.write(f"• {reason}")
    st.caption(f"Expires: {_expiry_text(report.signal.expires_at)}")


def _render_active_positions(monitor: LiveTradeMonitor | None, report: IntelligenceReport | None) -> None:
    """Step 3: Rewritten active positions display connecting to LiveTradeMonitor."""
    st.subheader("Active Positions (Live Trade Monitor)")
    open_trades: list[OpenTrade] = []
    if monitor is not None and hasattr(monitor, "open_trades") and monitor.open_trades:
        open_trades = list(monitor.open_trades.values()) if isinstance(monitor.open_trades, dict) else list(monitor.open_trades)
    elif report is not None and report.trade_update and report.trade_update.open_trades:
        open_trades = report.trade_update.open_trades

    if not open_trades:
        st.info("No active open positions currently tracked by LiveTradeMonitor.")
        return

    # Alert badge for frozen trades due to stale ticks
    frozen_trades = [t for t in open_trades if getattr(t, "is_frozen", False) or t.status == "frozen"]
    for ft in frozen_trades:
        st.error(
            f"🚨 **UNRESOLVABLE DATA ALERT:** Position **{ft.trade_id}** "
            f"({ft.strike:,.0f} {ft.option_type}) is marked as **FROZEN** due to stale ticks (>2.0s gap or missing quotes). "
            f"Evaluation and trailing stop adjustments are suspended."
        )

    rows = []
    total_unrealized_pnl = 0.0
    for trade in open_trades:
        unrealized = compute_unrealized_pnl(trade)
        total_unrealized_pnl += unrealized
        is_frozen = getattr(trade, "is_frozen", False) or trade.status == "frozen"
        status_label = "⚠️ FROZEN (UNRESOLVABLE DATA)" if is_frozen else "🟢 LIVE"
        stop_val = trade.signal.stop_loss if trade.signal else None
        stop_str = f"₹{stop_val:,.2f}" if stop_val is not None else "N/A"

        rows.append(
            {
                "Trade ID": trade.trade_id,
                "Strike": f"{trade.strike:,.0f}",
                "Option Type": trade.option_type,
                "Qty (Lots)": trade.quantity,
                "Entry Ask": f"₹{trade.entry_ask_price:,.2f}",
                "Current Bid": f"₹{trade.current_bid_price:,.2f}",
                "Trailing Stop": stop_str,
                "Unrealized P&L (₹)": f"{'+' if unrealized > 0 else ''}₹{unrealized:,.2f}",
                "Data Health": status_label,
            }
        )

    positions_df = pd.DataFrame(rows)
    st.dataframe(positions_df, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Open Contracts", len(open_trades))
    c2.metric(
        "Total Unrealized P&L",
        f"₹{total_unrealized_pnl:,.2f}",
        delta=f"{'+' if total_unrealized_pnl > 0 else ''}₹{total_unrealized_pnl:,.2f}" if total_unrealized_pnl != 0 else None,
    )


def _render_trade_monitor(report: IntelligenceReport) -> None:
    """Backward compatible wrapper for active positions display."""
    _render_active_positions(None, report)


def _render_signal_log_section(duckdb_mgr: DuckDBManager) -> None:
    """Step 4: Query DuckDB signals table for the 50 most recent records and display with distinct rejection styling."""
    st.markdown("### Signal Log (Last 50 Evaluated Setups from DuckDB)")
    signals_df = query_recent_signals(duckdb_mgr, limit=50)

    if signals_df.empty:
        st.info("No signal evaluations recorded in DuckDB yet.")
        return

    total = len(signals_df)
    passed_mask = signals_df["filter_passed"].astype(bool)
    passed_count = int(passed_mask.sum())
    rejected_count = total - passed_count

    c1, c2 = st.columns([2, 1])
    c1.caption(f"Evaluations: **{total}** total | ✅ **{passed_count}** Passed | ⛔ **{rejected_count}** Filter-Rejected")

    display_df = signals_df.copy()
    display_df["Status"] = display_df["filter_passed"].apply(lambda p: "✅ PASSED" if bool(p) else "⛔ REJECTED")

    if "spot_price" in display_df.columns:
        display_df["spot_price"] = display_df["spot_price"].apply(lambda v: f"₹{v:,.2f}" if pd.notna(v) else "-")
    if "confidence" in display_df.columns:
        display_df["confidence"] = display_df["confidence"].apply(lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    if "trap_score" in display_df.columns:
        display_df["trap_score"] = display_df["trap_score"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    if "rejection_reason" in display_df.columns:
        display_df["rejection_reason"] = display_df["rejection_reason"].fillna("-")

    col_order = [
        "timestamp",
        "Status",
        "spot_price",
        "strike",
        "option_type",
        "action",
        "confidence",
        "trap_score",
        "filter_passed",
        "rejection_reason",
    ]
    existing_cols = [c for c in col_order if c in display_df.columns]
    ordered_df = display_df[existing_cols]

    def _style_signal_row(row: pd.Series) -> list[str]:
        if not bool(row.get("filter_passed", True)):
            return ["background-color: rgba(239, 68, 68, 0.18); color: #fca5a5"] * len(row)
        return ["background-color: rgba(34, 197, 94, 0.10); color: #86efac"] * len(row)

    styled_table = ordered_df.style.apply(_style_signal_row, axis=1)
    st.dataframe(styled_table, use_container_width=True, height=300)

    # Rejection review callout to help the trader inspect stay-out rationale
    rejected_setups = signals_df[~passed_mask]
    if not rejected_setups.empty:
        with st.expander(f"🔍 Trader Review: Why Did the Engine Say To Stay Out? ({rejected_count} Setups)"):
            for _, r in rejected_setups.head(10).iterrows():
                st.markdown(
                    f"• **{r['timestamp']}** | Strike: **{r.get('strike', 'N/A')} {r.get('option_type', '')}** | Action: `{r.get('action', 'NO_TRADE')}` | Trap Score: `{r.get('trap_score', 0.0):.2f}`<br>"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;⚠️ **Stay-Out Rationale:** `{r.get('rejection_reason') or 'Filter blocked trade'}`",
                    unsafe_allow_html=True,
                )


def _render_api_health_panel(runtime: LiveIntelligenceRuntime) -> None:
    st.subheader("API Health")
    health = runtime.api_health
    columns = st.columns(2)
    columns[0].metric("Environment", health.environment.title(), delta="Safe Mode" if health.safe_mode else "Active")
    columns[1].metric("Execution", "Allowed" if health.execution_allowed else "Blocked")

    metrics = st.columns(3)
    metrics[0].metric("REST Requests", health.rest_requests)
    metrics[1].metric("REST Failures", health.rest_failures)
    metrics[2].metric("Rate Limit Hits", health.rate_limit_hits)

    stream_metrics = st.columns(3)
    stream_metrics[0].metric("Reconnects", health.reconnect_attempts)
    stream_metrics[1].metric("Duplicate Ticks", health.duplicate_ticks)
    stream_metrics[2].metric("Out of Order", health.out_of_order_ticks)

    if health.last_heartbeat_at is not None:
        st.caption(f"Last heartbeat: {health.last_heartbeat_at.isoformat()}")
    if health.notes:
        for note in health.notes[-3:]:
            st.warning(note)


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
