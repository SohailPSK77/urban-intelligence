"""
SIH26124: Custom UI Header & Command KPI Metrics Component
"""

import streamlit as st
from config import SIMULATION_LABEL


def _render_identity_bar() -> str:
    if "authenticated" in st.session_state and st.session_state.authenticated:
        role = st.session_state.get("user_role", "UNKNOWN")
        user_id = st.session_state.get("user_id", "N/A")
        bus_id = st.session_state.get("bus_id")
        route_id = st.session_state.get("route_id")
        login_time = st.session_state.get("login_time", "")

        if role == "BUS":
            identity_str = f"AUTHENTICATED &nbsp;|&nbsp; ROLE: <b>ONBOARD BUS</b> &nbsp;|&nbsp; BUS ID: <b>{bus_id}</b> &nbsp;|&nbsp; ROUTE: <b>{route_id}</b> &nbsp;|&nbsp; LOGIN: <b>{login_time}</b>"
            badge_color = "#38BDF8"
            bg_color = "rgba(56, 189, 248, 0.1)"
            border_color = "rgba(56, 189, 248, 0.3)"
        else:
            identity_str = f"AUTHENTICATED &nbsp;|&nbsp; ROLE: <b>OFFICIAL COMMAND CENTER</b> &nbsp;|&nbsp; COMMANDER ID: <b>{user_id}</b> &nbsp;|&nbsp; LOGIN: <b>{login_time}</b>"
            badge_color = "#C084FC"
            bg_color = "rgba(192, 132, 252, 0.1)"
            border_color = "rgba(192, 132, 252, 0.3)"

        return f"""
        <div style="
            margin-top: 14px;
            padding: 8px 16px;
            background: {bg_color};
            border: 1px solid {border_color};
            border-radius: 8px;
            color: {badge_color};
            font-size: 0.82rem;
            letter-spacing: 0.5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>🟢 {identity_str}</div>
            <div style="font-size: 0.75rem; color: #94A3B8;">SECURITY: AUTHENTICATED SESSION ACTIVE</div>
        </div>
        """
    return ""


def render_header(buses: list, raw_events: list, persistent_issues: list):
    """
    Renders the top banner and KPI summary cards.
    """
    st.markdown(
        f"""<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #334155; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
<div>
<span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">SIH26124 • PROBLEM STATEMENT</span>
<h1 style="color: #F8FAFC; font-size: 1.85rem; font-weight: 800; margin: 8px 0 4px 0; letter-spacing: -0.5px; background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI-Powered Mobile Urban Intelligence Platform</h1>
<p style="color: #94A3B8; margin: 0; font-size: 0.95rem;">Transforming Public Transport Bus Fleets into Continuous Mobile AI Sensing Networks</p>
</div>
<div style="text-align: right;">
<span style="background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 6px 14px; border-radius: 8px; font-size: 0.8rem; font-weight: 700;">⚡ {SIMULATION_LABEL}</span>
<div style="color: #64748B; font-size: 0.78rem; margin-top: 6px;">SYSTEM STATUS: <span style="color: #34D399; font-weight: 600;">● ONLINE & CONNECTED</span></div>
</div>
</div>
{_render_identity_bar()}
</div>""",
        unsafe_allow_html=True
    )

    # Calculate metrics
    active_buses = len(buses)
    total_events = len(raw_events)
    high_priority = len([e for e in raw_events if e.get("priority") in ["high", "critical", "HIGH", "CRITICAL"]])
    fused_issues_count = len(persistent_issues)

    # 6 Metric KPI Columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    metrics = [
        ("ACTIVE BUSES", f"{active_buses}", "Public Sensing Units", "#38BDF8"),
        ("AI EVENTS TODAY", f"{total_events}", "Edge Observations", "#34D399"),
        ("HIGH PRIORITY", f"{high_priority}", "Urgent Alerts", "#F87171"),
        ("FUSED PERSISTENT", f"{fused_issues_count}", "Multi-Bus Clusters", "#C084FC"),
        ("NETWORK DENSITY", "HIGH", "Peak Corridor Traffic", "#FBBF24"),
        ("EDGE BUFFER", "0 QUEUED", "100% Synced to Cloud", "#38BDF8")
    ]

    cols = [col1, col2, col3, col4, col5, col6]
    for col, (title, val, sub, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: #1E293B;
                    border: 1px solid #334155;
                    border-radius: 10px;
                    padding: 14px 12px;
                    text-align: center;
                    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
                ">
                    <div style="color: #94A3B8; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 4px;">{title}</div>
                    <div style="color: {color}; font-size: 1.45rem; font-weight: 800; line-height: 1.2;">{val}</div>
                    <div style="color: #64748B; font-size: 0.68rem; margin-top: 4px;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
