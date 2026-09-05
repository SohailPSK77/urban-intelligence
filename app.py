"""
SIH26124: AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Command Center Dashboard Application (Visakhapatnam - Phase 7 Role-Based Auth Integrated)
"""

import streamlit as st
import os
import sys

# Ensure current directory and components directory are in Python path for Streamlit Cloud deployment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "components"))

# Set Streamlit Page Config MUST be the first streamlit call
st.set_page_config(
    page_title="SIH26124 - Urban Intelligence Platform",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Imports from core modules with flat-layout fallback support
from config import ROUTES, SIMULATION_LABEL, CITY_NAME
from data_simulator import generate_bus_fleet, generate_raw_ai_events
from fusion_engine import MultiBusFusionEngine
from central_api import start_local_central_api

try:
    from components.login import render_login_screen
    from components.header import render_header
    from components.gis_map import render_gis_map
    from components.video_analytics import render_video_analytics
    from components.fusion_lab import render_fusion_lab
    from components.traffic_analytics import render_traffic_analytics
    from components.anpr_incidents import render_anpr_incidents
    from components.event_table import render_event_table
    from components.edge_sensing_pipeline import render_edge_sensing_pipeline
except ModuleNotFoundError:
    from login import render_login_screen
    from header import render_header
    from gis_map import render_gis_map
    from video_analytics import render_video_analytics
    from fusion_lab import render_fusion_lab
    from traffic_analytics import render_traffic_analytics
    from anpr_incidents import render_anpr_incidents
    from event_table import render_event_table
    from edge_sensing_pipeline import render_edge_sensing_pipeline


def main():
    # Start Phase 6 Local Central Ingestion API Service
    start_local_central_api()

    # Session State Initialization for Authentication & City Data
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.bus_id = None
        st.session_state.route_id = None
        st.session_state.operator_id = None
        st.session_state.login_time = None

    if "city_name" not in st.session_state or st.session_state.get("city_name") != CITY_NAME:
        st.session_state.city_name = CITY_NAME
        st.session_state.buses = generate_bus_fleet()
        st.session_state.raw_events = generate_raw_ai_events()

    # If user is not authenticated, render Login Screen only and stop
    if not st.session_state.authenticated:
        render_login_screen()
        return

    buses = st.session_state.buses
    raw_events = st.session_state.raw_events

    # Execute Multi-Bus Event Fusion Engine
    fusion_engine = MultiBusFusionEngine(distance_threshold_m=20.0)
    persistent_issues, isolated_events = fusion_engine.fuse_events(raw_events)

    role = st.session_state.user_role
    user_id = st.session_state.user_id
    bus_id = st.session_state.get("bus_id", "BUS-07")
    route_id = st.session_state.get("route_id", "ROUTE-101")

    # -------------------------------------------------------------
    # SIDEBAR CONTROLS & SESSION IDENTITY
    # -------------------------------------------------------------
    st.sidebar.image("https://img.icons8.com/isometric-folders/100/bus.png", width=64)
    st.sidebar.title("🚍 FLEET COMMAND")
    
    # Render Authenticated Session Card
    if role == "BUS":
        st.sidebar.success(
            f"**🚌 AUTHENTICATED BUS UNIT**\n\n"
            f"• **Bus ID**: `{bus_id}`\n"
            f"• **Corridor**: `{route_id}`\n"
            f"• **Operator**: `{st.session_state.get('operator_id', 'APSRTC')}`\n"
            f"• **Login Time**: `{st.session_state.login_time}`"
        )
    else:
        st.sidebar.info(
            f"**🏢 OFFICIAL COMMAND CENTER**\n\n"
            f"• **Commander**: `{user_id}`\n"
            f"• **Role**: `CHIEF FLEET COMMAND`\n"
            f"• **Authority**: `APSRTC-HQ`\n"
            f"• **Login Time**: `{st.session_state.login_time}`"
        )

    if st.sidebar.button("🚪 Logout Session", type="secondary", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.bus_id = None
        st.session_state.route_id = None
        st.session_state.login_time = None
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**City Scope:** `{CITY_NAME}`")
    st.sidebar.markdown(f"**Mode:** `<span style='color:#FBBF24;'>{SIMULATION_LABEL}</span>`", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Route Filter
    route_options = ["ALL"] + list(ROUTES.keys())
    selected_route = st.sidebar.selectbox("Filter by Transit Corridor", route_options, index=0)

    # Manual Refresh / Simulate Telemetry Tick
    if st.sidebar.button("🔄 Refresh Fleet Telemetry Tick"):
        st.session_state.buses = generate_bus_fleet()
        st.session_state.raw_events = generate_raw_ai_events()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 SIH26124 Mantra")
    st.sidebar.info(
        "**BUS → SEE → UNDERSTAND → LOCATE → SHARE → FUSE → VISUALIZE → DECIDE**\n\n"
        "Public transport buses act as continuously moving mobile AI sensing units."
    )

    # -------------------------------------------------------------
    # MAIN DASHBOARD RENDER & ROLE ROUTING
    # -------------------------------------------------------------
    render_header(buses, raw_events, persistent_issues)

    if role == "BUS":
        # BUS ROLE VIEW: Focus on single-bus edge sensing pipeline, live dashcam POV, and edge diagnostics
        st.markdown(f"### 🚌 Onboard Bus System Interface — Unit `{bus_id}` ({route_id})")
        
        tab_edge, tab_video, tab_traffic, tab_logs = st.tabs([
            "📡 Bus Edge Sensing Unit & Event Pipeline",
            "📹 Live Bus Camera View & AI Vision",
            "🚦 Corridor Traffic Intelligence",
            "📋 Onboard Recorded Issue Logs"
        ])

        with tab_edge:
            render_edge_sensing_pipeline()

        with tab_video:
            render_video_analytics()

        with tab_traffic:
            render_traffic_analytics()

        with tab_logs:
            # Filter raw events for authenticated bus identity
            bus_events = [e for e in raw_events if e.get("bus_id") == bus_id]
            render_event_table(bus_events if bus_events else raw_events, persistent_issues)

    else:
        # OFFICIAL ROLE VIEW: Access to cross-fleet GIS command map, multi-bus fusion lab, ANPR alerts, and central analytics
        st.markdown("### 🏢 Official Command Center Interface — Full Transit Fleet Overview")

        tab_map, tab_video, tab_fusion, tab_traffic, tab_anpr, tab_logs = st.tabs([
            "🗺️ GIS Command Map",
            "📹 Fleet Video Stream Overview",
            "🧪 Multi-Bus Fusion Lab",
            "🚦 Traffic Intelligence & Bottlenecks",
            "🚔 ANPR & Rash Driving (Real AI Verified)",
            "📋 Central Event Store & Fleet Logs"
        ])

        with tab_map:
            st.markdown("#### 🗺️ Live GIS Urban Sensing Map & Multi-Bus Event Fusion Clusters")
            st.caption("Layer Toggle: Active Buses (Blue Circles) | Single-Bus AI Detections (Markers) | Fused Persistent Issues (Purple Stars)")
            render_gis_map(buses, raw_events, persistent_issues, selected_route)

        with tab_video:
            render_video_analytics()

        with tab_fusion:
            render_fusion_lab()

        with tab_traffic:
            render_traffic_analytics()

        with tab_anpr:
            render_anpr_incidents(raw_events)

        with tab_logs:
            render_event_table(raw_events, persistent_issues)


if __name__ == "__main__":
    main()
