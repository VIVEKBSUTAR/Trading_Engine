"""Read-only terminal data access, analytics, and focused page components."""
from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import AppSettings
from storage.duckdb_manager import DuckDBManager
from trading_engine.intelligence.models import OpenTrade
from trading_engine.intelligence.monitor import NIFTY_LOT_SIZE

LIVE_WINDOW_SECONDS = 60
PUBLICATION_STALE_SECONDS = 30


def format_ist_timestamps(values: pd.Series) -> pd.Series:
    """Treat naive inputs as UTC; preserve aware instants before IST display."""
    timestamps = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    return timestamps.dt.tz_convert("Asia/Kolkata").dt.strftime("%d %b %Y %I:%M:%S %p IST").fillna("—")


def format_ist_timestamp(value: Any) -> str:
    return str(format_ist_timestamps(pd.Series([value])).iloc[0])


def render_telemetry(payload: dict[str, Any]) -> None:
    """Compact feed timestamps; freshness calculations continue to use UTC."""
    st.caption(f"Live Engine Telemetry · Updated {format_ist_timestamp(payload.get('timestamp'))}"
               f" · Last tick {format_ist_timestamp(payload.get('websocket', {}).get('last_tick_at'))}")


def inject_styles() -> None:
    st.markdown("""<style>
    .block-container {max-width:1440px;padding-top:1.4rem;padding-bottom:2rem;}
    [data-testid="stMetric"] {padding:0.75rem 1rem;background:#161B22;
        border:1px solid #262D36;border-radius:8px;}
    [data-testid="stMetricLabel"] {color:#98A2B3;font-size:0.78rem;}
    [data-testid="stMetricValue"] {font-size:1.55rem;font-variant-numeric:tabular-nums;}
    h1 {font-size:1.65rem!important;letter-spacing:-0.04rem;}
    h3 {font-size:1.1rem!important;}
    </style>""", unsafe_allow_html=True)
    st.sidebar.caption("NIFTY · ANALYTICAL TERMINAL")


def compute_performance_metrics(trade_outcomes: pd.DataFrame) -> dict[str, Any]:
    pnl = (pd.to_numeric(trade_outcomes["realized_pnl"], errors="coerce").fillna(0.0)
           if "realized_pnl" in trade_outcomes else pd.Series(dtype=float))
    gross_profit, gross_loss = float(pnl[pnl > 0].sum()), abs(float(pnl[pnl < 0].sum()))
    return {
        "total_realized_profit": float(pnl.sum()), "total_closed_trades": len(pnl),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
    }


def compute_equity_curve(trade_outcomes: pd.DataFrame) -> pd.DataFrame:
    if trade_outcomes.empty or not {"exit_timestamp", "realized_pnl"}.issubset(trade_outcomes):
        return pd.DataFrame(columns=["exit_timestamp", "cumulative_profit"])
    frame = trade_outcomes.copy()
    frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], errors="coerce", utc=True)
    frame["realized_pnl"] = pd.to_numeric(frame["realized_pnl"], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["exit_timestamp"]).sort_values("exit_timestamp", kind="stable")
    frame["cumulative_profit"] = frame["realized_pnl"].cumsum()
    return frame[["exit_timestamp", "cumulative_profit"]]


def compute_unrealized_pnl(trade: OpenTrade) -> float:
    return (float(trade.current_bid_price or 0) - float(trade.entry_ask_price or 0)) * NIFTY_LOT_SIZE * int(trade.quantity or 0)


def query_recent_signals(manager: DuckDBManager, limit: int = 50) -> pd.DataFrame:
    return manager.query("""SELECT signal_id, timestamp, spot_price, strike, option_type,
        action, entry_price, confidence, trap_score, filter_passed,
        filter_rejection_reason AS rejection_reason FROM signals
        ORDER BY timestamp DESC LIMIT ?""", [limit])


def query_live_signals(manager: DuckDBManager, now: datetime | None = None) -> pd.DataFrame:
    return manager.get_live_signals(now=now)


def get_dashboard_duckdb_manager(settings: AppSettings) -> DuckDBManager:
    existing = st.session_state.get("duckdb_mgr")
    if existing is not None and getattr(existing, "_tmp_copy", None) is not None:
        existing.close()
        st.session_state["duckdb_mgr"] = None
    if st.session_state.get("duckdb_mgr") is None:
        st.session_state["duckdb_mgr"] = DuckDBManager(settings.storage.duckdb_path, read_only=True)
    return st.session_state["duckdb_mgr"]


def load_terminal_data(*, analytics: bool = False) -> dict[str, Any]:
    """Deployment reads the sole writer's publication; offline mode can query DB."""
    directory = os.getenv("TE_DASHBOARD_SNAPSHOT_DIR")
    path = Path(directory or "data/dashboard") / "latest.json"
    if directory or path.exists():
        if not path.exists():
            return {"source": "snapshot", "waiting": True}
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("Invalid publication")
            return {**payload, "source": "snapshot"}
        except (OSError, ValueError):
            return {"source": "snapshot", "waiting": True, "error": "Engine publication is unavailable or invalid."}
    settings = AppSettings.from_env()
    if not settings.storage.duckdb_path.exists():
        return {"source": "database", "waiting": True}
    try:
        manager = get_dashboard_duckdb_manager(settings)
        return {
            "source": "database", "timestamp": datetime.now(UTC).isoformat(),
            "lifecycle_alerts": manager.get_lifecycle_alerts().to_dict("records"),
            "live_signals": query_live_signals(manager).to_dict("records"),
            "signals": query_recent_signals(manager).to_dict("records") if analytics else [],
            "outcomes": manager.get_trade_outcomes().to_dict("records") if analytics else [],
            "positions": [],
        }
    except Exception:
        return {"source": "database", "waiting": True,
                "error": "Database unavailable. Use the engine publication while the daemon is running."}


def publication_age(payload: dict[str, Any], now: datetime | None = None) -> float:
    value = pd.to_datetime(payload.get("timestamp"), errors="coerce", utc=True)
    if pd.isna(value) or value is None:
        return float("inf")
    return (pd.Timestamp(now or datetime.now(UTC)) - value).total_seconds()


def recent_passed_signals(frame: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Recheck the database window at render time; snapshots can outlive it."""
    if frame.empty or not {"timestamp", "action", "filter_passed"}.issubset(frame):
        return pd.DataFrame()
    now = pd.Timestamp(now or datetime.now(UTC))
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    passed = frame["filter_passed"].fillna(False).eq(True)
    action = frame["action"].astype(str).str.upper().isin(["BUY_CE", "BUY_PE"])
    mask = passed & action & timestamps.between(now - timedelta(seconds=LIVE_WINDOW_SECONDS), now)
    result = frame.loc[mask].copy()
    result["timestamp"] = timestamps.loc[mask]
    return result.sort_values("timestamp", ascending=False)


def render_live_alert(payload: dict[str, Any]) -> None:
    # Never present a cached signal as live after engine publication stops.
    if payload.get("source") == "snapshot" and not 0 <= publication_age(payload) <= PUBLICATION_STALE_SECONDS:
        return
    events = pd.DataFrame(payload.get("lifecycle_alerts", []))
    if events.empty:
        st.caption("Monitoring · No fresh lifecycle alert.")
        return
    stamps = pd.to_datetime(events.timestamp, utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    events = events.loc[stamps.between(now - pd.Timedelta(seconds=60), now)].copy()
    if events.empty:
        return
    events = events.sort_values("timestamp", ascending=False)
    # Surface exits before entries when both occur during the same refresh window.
    events["_exit"] = events.alert_type.str.startswith("EXIT_")
    events = events.sort_values("_exit", ascending=False, kind="stable")
    seen = st.session_state.get("seen_lifecycle_alerts", {})
    seen = {key: value for key, value in seen.items() if value >= now.timestamp() - 60}
    for _, event in events.head(10).iterrows():
        message = f"{event.alert_type} · {str(event.get('action', '')).upper()} · Strike {_number(event.get('strike'), decimals=0)} · Premium {_number(event.get('premium'), prefix='₹')} · {format_ist_timestamp(event.get('timestamp'))}"
        (st.error if event["_exit"] else st.success)(message, icon="🔔")
        if event.event_id not in seen:
            st.toast(message, icon="🔔")
        seen[event.event_id] = now.timestamp()
    st.session_state["seen_lifecycle_alerts"] = seen


def render_health(payload: dict[str, Any]) -> None:
    health = payload.get("websocket", {})
    age = publication_age(payload)
    tick = pd.to_datetime(health.get("last_tick_at"), errors="coerce", utc=True)
    tick_age = (pd.Timestamp.now(tz="UTC") - tick).total_seconds() if tick is not None and not pd.isna(tick) else float("inf")
    if not health:
        status = "Unavailable"
    elif not health.get("connected"):
        status = "Disconnected"
    elif health.get("safe_mode") or not 0 <= age <= PUBLICATION_STALE_SECONDS or not 0 <= tick_age <= 20:
        status = "Stale / paused"
    else:
        status = "Connected"
    columns = st.columns(3)
    columns[0].metric("Fyers WebSocket", status)
    columns[1].metric("NIFTY Spot", _number(payload.get("spot_price")))
    columns[2].metric("India VIX", _number(payload.get("vix_value")))


def positions_frame(positions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for position in positions:
        quantity = position.get("quantity")
        pnl = position.get("unrealized_pnl")
        if pnl is None and quantity is not None:
            pnl = (float(position["current_bid"]) - float(position["entry_ask"])) * NIFTY_LOT_SIZE * quantity
        rows.append({
            "Position": position.get("trade_id"), "Strike": position.get("strike"),
            "Side": position.get("option_type"), "Lots": quantity,
            "Entry Ask": position.get("entry_ask"), "Live Premium": position.get("current_bid"),
            "Entry Time": format_ist_timestamp(position.get("opened_at")),
            "Signal Time": format_ist_timestamp(position.get("signal_timestamp")),
            "Time in Trade": position.get("time_in_trade"), "Live IV": position.get("live_iv"),
            "Trailing Stop": position.get("stop_loss"), "Unrealized PnL": pnl,
            "Data": "FROZEN" if position.get("is_frozen") else "Live",
        })
    return pd.DataFrame(rows)


def style_positions(frame: pd.DataFrame):
    def color(value: float) -> str:
        if pd.isna(value) or value == 0:
            return "color: #98A2B3"
        return "color: #00FF00" if value > 0 else "color: #FF5252"
    return (frame.style.map(color, subset=["Unrealized PnL"])
        .set_properties(subset=["Trailing Stop"], **{"font-weight": "700"})
        .format({"Strike": "{:,.0f}", "Entry Ask": "₹{:,.2f}", "Live Premium": "₹{:,.2f}", "Time in Trade": "{:.1f} min", "Live IV": "{:.2f}%",
                 "Trailing Stop": "₹{:,.2f}", "Unrealized PnL": "₹{:+,.2f}"}, na_rep="—"))


def render_positions(positions: list[dict[str, Any]], *, stale: bool = False) -> None:
    st.subheader("Active Positions")
    if not positions:
        st.info("No active analytical positions.")
        return
    frame = positions_frame(positions)
    if stale:
        frame["Data"] = "STALE"
    if stale or frame["Data"].eq("FROZEN").any():
        st.error("UNRESOLVABLE DATA ALERT · Quotes are stale or missing. PnL and stops are frozen.")
    st.dataframe(style_positions(frame), width="stretch", hide_index=True)


def _render_active_positions(monitor, report) -> None:
    trades = (monitor.open_trades if monitor is not None else report.trade_update.open_trades if report else [])
    if isinstance(trades, dict):
        trades = trades.values()
    render_positions([{
        "trade_id": trade.trade_id, "strike": trade.strike, "option_type": trade.option_type,
        "quantity": trade.quantity, "entry_ask": trade.entry_ask_price, "current_bid": trade.current_bid_price,
        "stop_loss": trade.signal.stop_loss, "unrealized_pnl": compute_unrealized_pnl(trade),
        "is_frozen": trade.is_frozen, "live_iv": trade.live_iv,
        "time_in_trade": max(0, (datetime.now(UTC) - trade.opened_at).total_seconds() / 60),
        "opened_at": trade.opened_at, "signal_timestamp": trade.signal_timestamp,
    } for trade in trades])


def execution_desk() -> None:
    payload = load_terminal_data()
    render_health(payload)
    render_telemetry(payload)
    st.title("Execution Desk")
    if payload.get("waiting"):
        st.info(payload.get("error", "Waiting for the engine's first analytical snapshot."))
    else:
        if payload["source"] == "snapshot" and not 0 <= publication_age(payload) <= PUBLICATION_STALE_SECONDS:
            st.warning("Engine publication is stale. Live alerts are paused.")
        render_live_alert(payload)
    render_positions(payload.get("positions", []), stale=publication_age(payload) > PUBLICATION_STALE_SECONDS)
    if payload.get("source") == "database" and not payload.get("waiting"):
        st.caption("Database view · Connect the engine publication for live health and positions.")


def _render_performance_banner(metrics: dict[str, Any]) -> None:
    columns = st.columns(4)
    columns[0].metric("Total Profit", f"₹{metrics['total_realized_profit']:,.2f}")
    columns[1].metric("Closed Trades", f"{metrics['total_closed_trades']:,}")
    columns[2].metric("Win Rate", f"{metrics['win_rate_pct']:.1f}%")
    factor = metrics["profit_factor"]
    columns[3].metric("Profit Factor", "∞" if math.isinf(factor) else f"{factor:.2f}")


def equity_figure(trade_outcomes: pd.DataFrame) -> go.Figure:
    equity = compute_equity_curve(trade_outcomes)
    figure = go.Figure(go.Scatter(
        x=equity["exit_timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None), y=equity["cumulative_profit"], mode="lines", fill="tozeroy",
        line={"color": "#00FF00", "width": 2}, fillcolor="rgba(0,255,0,0.09)",
        customdata=format_ist_timestamps(equity["exit_timestamp"]),
        hovertemplate="%{customdata}<br>Realized PnL ₹%{y:,.2f}<extra></extra>",
    ))
    figure.update_layout(template="plotly_dark", height=340, showlegend=False,
        margin={"l": 0, "r": 0, "t": 12, "b": 0}, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font={"color": "#98A2B3"}, hovermode="x unified",
        xaxis={"title": None, "tickformat": "%I:%M:%S %p IST", "showgrid": False, "zeroline": False, "showline": False},
        yaxis={"title": None, "gridcolor": "rgba(152,162,179,0.10)", "zerolinecolor": "#3B4350", "tickprefix": "₹"})
    return figure


def _render_equity_curve_section(trade_outcomes: pd.DataFrame) -> None:
    st.subheader("Realized Equity Curve")
    if compute_equity_curve(trade_outcomes).empty:
        st.info("The equity curve will appear after the first analytical position closes.")
        return
    st.plotly_chart(equity_figure(trade_outcomes), config={"displayModeBar": False})


def render_audit_log(signals: pd.DataFrame) -> None:
    st.subheader("Signal Audit Log · Last 50")
    if signals.empty:
        st.info("No signal evaluations recorded yet.")
        return
    signals = signals.copy().head(50)
    if "timestamp" in signals:
        signals["timestamp"] = format_ist_timestamps(signals["timestamp"])
    signals = signals.rename(columns={"filter_rejection_reason": "rejection_reason"})
    if "filter_passed" not in signals:
        signals["filter_passed"] = False
    signals["filter_passed"] = signals["filter_passed"].fillna(False).eq(True)
    signals["Status"] = signals["filter_passed"].map({True: "PASSED", False: "REJECTED"})
    columns = [key for key in ["timestamp", "Status", "action", "strike", "option_type", "entry_price",
        "confidence", "trap_score", "rejection_reason"] if key in signals]
    st.dataframe(signals[columns], width="stretch", hide_index=True)
    rejected = signals.loc[~signals["filter_passed"]]
    with st.expander(f"Rejection Reason · {len(rejected)} filtered setups"):
        if rejected.empty:
            st.caption("No rejected setups in this window.")
        for _, row in rejected.iterrows():
            st.write(f"{row.get('timestamp', '—')} · {str(row.get('action', '')).upper()} · {row.get('strike', '—')}")
            st.caption(str(row.get("rejection_reason") or "Filter blocked setup"))


def _render_signal_log_section(manager: DuckDBManager) -> None:
    render_audit_log(query_recent_signals(manager))


def analytics_desk() -> None:
    payload = load_terminal_data(analytics=True)
    outcomes = pd.DataFrame(payload.get("outcomes", []))
    _render_performance_banner(compute_performance_metrics(outcomes))
    st.title("Analytics & Logs")
    if payload.get("waiting"):
        st.info(payload.get("error", "Waiting for the engine's first analytical snapshot."))
    if payload.get("source") == "snapshot":
        st.caption("Most recent 1,000 analytical outcomes · Last 50 signal evaluations")
        if publication_age(payload) > PUBLICATION_STALE_SECONDS:
            st.warning("Engine publication is stale. Historical data remains available.")
    _render_equity_curve_section(outcomes)
    render_audit_log(pd.DataFrame(payload.get("signals", [])))


def _number(value: Any, *, prefix: str = "", decimals: int = 2) -> str:
    try:
        number = float(value)
        if not math.isfinite(number):
            return "—"
        return f"{prefix}{number:,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"
