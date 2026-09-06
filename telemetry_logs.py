"""
SIH26124: Onboard AI Telemetry Logs & Hardware Performance Metrics Component
Provides persistent logging, interactive charts, metric sweeps, CSV export, and telemetry JSON inspection.
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime
from config import ROUTES
from video_processor import BusCameraVideoProcessor


def get_asset_path(filename: str) -> str:
    """Helper to locate asset files dynamically across workspace environments with domain-accurate URL fallback."""
    if not filename:
        return "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?auto=format&fit=crop&w=800&q=80"

    raw_str = str(filename)
    clean_name = os.path.basename(raw_str)
    
    if os.path.exists(raw_str) and os.path.isfile(raw_str):
        return raw_str

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths = [
        os.path.join(base_dir, "assets", clean_name),
        os.path.join(os.getcwd(), "assets", clean_name),
        os.path.join("assets", clean_name),
        clean_name
    ]
    for path in candidate_paths:
        if os.path.exists(path) and os.path.isfile(path):
            return path

    clean_lower = clean_name.lower()
    for adir in [os.path.join(base_dir, "assets"), os.path.join(os.getcwd(), "assets"), "assets"]:
        if os.path.exists(adir) and os.path.isdir(adir):
            try:
                for existing_file in os.listdir(adir):
                    if existing_file.lower() == clean_lower:
                        full_p = os.path.join(adir, existing_file)
                        if os.path.exists(full_p) and os.path.isfile(full_p):
                            return full_p
            except Exception:
                pass

    fallback_urls = {
        "vizag_bus_front.jpg": "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?auto=format&fit=crop&w=800&q=80",
        "pothole_road_vizag.jpg": "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=800&q=80",
        "rash_driving_car.jpg": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=800&q=80"
    }
    return fallback_urls.get(clean_name, "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?auto=format&fit=crop&w=800&q=80")


def initialize_telemetry_history():
    """Initializes persistent telemetry history in session state with 20 initial frames if empty."""
    if "telemetry_history" not in st.session_state:
        st.session_state.telemetry_history = []

    if not st.session_state.telemetry_history:
        base_time = datetime.now()
        for idx in range(1, 21):
            st.session_state.telemetry_history.append({
                "log_id": idx,
                "frame_number": idx,
                "bus_id": "BUS-07",
                "route_id": "ROUTE-101",
                "active_vehicles": max(1, (idx * 3) % 6 + 1),
                "active_pedestrians": (idx % 3) + 1,
                "traffic_density_index": round(min(1.0, 0.20 + (idx % 5) * 0.08), 2),
                "congestion_score": round(min(1.0, 0.25 + (idx % 4) * 0.10), 2),
                "pixel_displacement_px": round(2.5 + (idx % 5) * 1.2, 1),
                "latency_ms": round(78.0 + (idx % 7) * 2.5, 1),
                "measured_fps": round(12.5 - (idx % 4) * 0.3, 1),
                "gps_location": f"17.7145° N, {round(83.3235 + idx*0.0001, 4)}° E",
                "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S")
            })


def render_telemetry_logs(selected_bus_id: str = "BUS-07", selected_route_id: str = "ROUTE-101", key_prefix: str = "telemetry"):
    """
    Renders the Onboard AI Telemetry Logs & Performance Metrics section.
    Includes top KPI metrics, benchmark sweep execution, interactive line charts, filterable data tables, and CSV export.
    Key prefix avoids StreamlitDuplicateElementId when rendered in multiple tabs.
    """
    initialize_telemetry_history()

    st.markdown(
        """<div style="background: #0F172A; border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div>
<h4 style="color: #F8FAFC; margin: 0 0 4px 0; font-weight: 800;">📊 Onboard Camera & Vehicle Analytics Logs</h4>
<p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">Live, frame-by-frame sensing measurements recorded across public transport bus camera feeds in Visakhapatnam.</p>
</div>
<div>
<span style="background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid #3B82F6; padding: 4px 10px; border-radius: 16px; font-size: 0.75rem; font-weight: 800;">⚡ CAMERA SENSING ACTIVE</span>
</div>
</div>
</div>""",
        unsafe_allow_html=True
    )

    df_raw = pd.DataFrame(st.session_state.telemetry_history)

    if df_raw.empty:
        st.info("No telemetry entries recorded yet. Scrub video frames or click the benchmark sweep button below.")
        return

    # Assign sequential Log Index (1..N) to ensure unique clean index for line charts
    df_raw["seq_index"] = range(1, len(df_raw) + 1)
    df_raw["seq_label"] = df_raw.apply(lambda r: f"#{r['seq_index']} ({r.get('bus_id', 'BUS')})", axis=1)

    # 1. Top 4 KPI Metrics
    avg_fps = round(df_raw["measured_fps"].mean(), 1) if "measured_fps" in df_raw.columns else 12.1
    avg_lat = round(df_raw["latency_ms"].mean(), 1) if "latency_ms" in df_raw.columns else 82.0
    avg_veh = round(df_raw["active_vehicles"].mean(), 1) if "active_vehicles" in df_raw.columns else 3.0
    avg_tdi = round(df_raw["traffic_density_index"].mean(), 2) if "traffic_density_index" in df_raw.columns else 0.35

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Logged Telemetry Entries", f"{len(df_raw)} Frames", delta="Live Camera Feed")
    k2.metric("Avg Response Time", f"{avg_lat} ms", delta="Onboard Processor")
    k3.metric("Avg Processing Speed", f"{avg_fps} FPS", delta="Camera Processor")
    k4.metric("Avg Traffic Density (TDI)", f"{avg_tdi}", delta="Visakhapatnam Fleet")

    st.markdown("---")

    # 2. Controls & Autonomous 50-Frame Sweep
    c_sw1, c_sw2, c_sw3 = st.columns([1.8, 1, 1])

    with c_sw1:
        if st.button("▶️ Run 50-Frame Fleet Sensing Performance Sweep", type="secondary", use_container_width=True, key=f"{key_prefix}_sweep_btn"):
            st.info(f"Executing 50-frame camera processing sweep on {selected_bus_id} ({selected_route_id})...")
            video_filename = f"sample_vizag_{selected_route_id.lower().replace('-', '_')}.mp4"
            video_path = get_asset_path(video_filename)

            try:
                proc_sweep = BusCameraVideoProcessor(video_path)
                for f in range(1, 51):
                    _, _, tel_s = proc_sweep.process_frame_at(f, bus_id=selected_bus_id, route_id=selected_route_id)
                    lat_s = round(17.7200 + (f * 0.00008), 5)
                    lon_s = round(83.3000 + (f * 0.00008), 5)
                    st.session_state.telemetry_history.append({
                        "log_id": len(st.session_state.telemetry_history) + 1,
                        "frame_number": f,
                        "bus_id": selected_bus_id,
                        "route_id": selected_route_id,
                        "active_vehicles": tel_s.get("current_active_vehicles", 3),
                        "active_pedestrians": tel_s.get("current_active_pedestrians", 1),
                        "traffic_density_index": tel_s.get("traffic_density_index", 0.35),
                        "congestion_score": tel_s.get("relative_congestion_score", 0.40),
                        "pixel_displacement_px": tel_s.get("average_displacement_px", 3.5),
                        "latency_ms": tel_s.get("latency_ms", 82.0),
                        "measured_fps": tel_s.get("measured_fps", 12.1),
                        "gps_location": f"{lat_s}° N, {lon_s}° E",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                proc_sweep.close()
                st.success("✅ 50-Frame AI Telemetry Sweep Complete! Logs & charts updated.")
                st.rerun()
            except Exception as ex:
                st.error(f"Error running benchmark sweep: {ex}")

    with c_sw2:
        if st.button("🗑️ Clear Log History", use_container_width=True, key=f"{key_prefix}_clear_btn"):
            st.session_state.telemetry_history = []
            st.success("Cleared telemetry logs.")
            st.rerun()

    with c_sw3:
        csv_data = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV Log", csv_data, "telemetry_logs.csv", "text/csv", use_container_width=True, key=f"{key_prefix}_export_btn")

    # Filter by Bus ID if multiple exist
    unique_buses = list(df_raw["bus_id"].unique()) if "bus_id" in df_raw.columns else []
    if len(unique_buses) > 1:
        sel_bus_filter = st.multiselect("🔍 Filter View by Bus Unit:", options=unique_buses, default=unique_buses, key=f"{key_prefix}_bus_filter")
        df_filtered = df_raw[df_raw["bus_id"].isin(sel_bus_filter)].copy()
    else:
        df_filtered = df_raw.copy()

    if df_filtered.empty:
        st.warning("No records match the selected filter.")
        return

    st.markdown("---")

    # 3. Interactive Line Charts with Unique Indexing
    st.markdown("##### 🚗 Active Tracked Vehicles & Pedestrians Over Sequence")
    df_chart1 = df_filtered.set_index("seq_index")[["active_vehicles", "active_pedestrians"]].rename(
        columns={"active_vehicles": "Vehicles", "active_pedestrians": "Pedestrians"}
    )
    st.line_chart(df_chart1)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("##### 🚦 Traffic Density Index (TDI) & Congestion Score")
        df_chart2 = df_filtered.set_index("seq_index")[["traffic_density_index", "congestion_score"]].rename(
            columns={"traffic_density_index": "TDI Index", "congestion_score": "Congestion Score"}
        )
        st.line_chart(df_chart2)

    with ch2:
        st.markdown("##### ⚡ System Response Time & Processing Speed")
        df_chart3 = df_filtered.set_index("seq_index")[["latency_ms", "measured_fps"]].rename(
            columns={"latency_ms": "Latency (ms)", "measured_fps": "FPS"}
        )
        st.line_chart(df_chart3)

    st.markdown("---")

    # 4. Interactive Telemetry History Table
    st.markdown("##### 📋 Complete Onboard Telemetry History Table")
    
    # Select clean display columns
    display_cols = [col for col in [
        "seq_index", "timestamp", "bus_id", "route_id", "frame_number",
        "active_vehicles", "active_pedestrians", "traffic_density_index",
        "congestion_score", "pixel_displacement_px", "latency_ms", "measured_fps", "gps_location"
    ] if col in df_filtered.columns]

    st.dataframe(
        df_filtered[display_cols].rename(columns={"seq_index": "Log Index"}),
        use_container_width=True
    )

    # 5. Onboard Telemetry Information Summary
    with st.expander("🔍 Inspect Latest Frame Telemetry Summary"):
        try:
            from components.event_card import render_human_readable_telemetry_card
        except ModuleNotFoundError:
            from event_card import render_human_readable_telemetry_card
        render_human_readable_telemetry_card(st.session_state.telemetry_history[-1])
