"""
SIH26124: Event Intelligence Data Table & Raw Edge Payload Inspector
"""

import streamlit as st
import pandas as pd
import json


try:
    from components.telemetry_logs import render_telemetry_logs
    from components.event_card import render_human_readable_event_card
except ModuleNotFoundError:
    from telemetry_logs import render_telemetry_logs
    from event_card import render_human_readable_event_card


def render_event_table(raw_events: list, persistent_issues: list):
    """
    Renders filterable data tables for AI detections, persistent issues,
    Onboard AI Telemetry Logs & Hardware Performance Metrics, and human-readable event information cards.
    """
    st.markdown(
        """<div style="background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
<h3 style="color: #F8FAFC; margin: 0 0 4px 0; font-weight: 800;">📋 Urban Intelligence Data Logs & Event Inspector</h3>
<p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">Inspect compact geotagged edge events, fused persistent issue records, real-time onboard AI hardware telemetry logs, and event information.</p>
</div>""",
        unsafe_allow_html=True
    )

    tab0, tab1, tab2, tab3 = st.tabs([
        "📊 Onboard Camera & Vehicle Analytics Logs",
        "🔗 Fused Persistent Issues",
        "⚠️ Single-Bus Detections",
        "📡 Geotagged Event Information Viewer"
    ])

    with tab0:
        render_telemetry_logs(key_prefix="logs_tab_telemetry")

    with tab1:
        if not persistent_issues:
            st.info("No multi-bus fused persistent issues detected yet.")
        else:
            df_fused = pd.DataFrame(persistent_issues)
            
            # Ensure mandatory columns exist safely
            if "issue_id" not in df_fused.columns:
                df_fused["issue_id"] = df_fused.get("cluster_id", "PR-0001")
            if "first_observed" not in df_fused.columns:
                df_fused["first_observed"] = df_fused.get("first_observed_timestamp", "2026-09-05 10:00:00")
            if "last_observed" not in df_fused.columns:
                df_fused["last_observed"] = df_fused.get("last_observed_timestamp", "2026-09-05 10:05:00")
            if "observing_buses" not in df_fused.columns:
                df_fused["observing_buses"] = df_fused.get("buses_observed", [[]])
            if "severity" not in df_fused.columns:
                df_fused["severity"] = "critical"
            if "status" not in df_fused.columns:
                df_fused["status"] = "CONFIRMED"
            if "fused_confidence" not in df_fused.columns:
                df_fused["fused_confidence"] = 0.95

            df_display = df_fused[[
                "issue_id", "event_type", "observation_count", "observing_buses",
                "fused_confidence", "severity", "status", "first_observed", "last_observed"
            ]].copy()

            df_display["fused_confidence"] = df_display["fused_confidence"].apply(lambda c: f"{round(float(c)*100, 1)}%")
            df_display["observing_buses"] = df_display["observing_buses"].apply(lambda b: ", ".join(b) if isinstance(b, (list, set)) else str(b))
            df_display["event_type"] = df_display["event_type"].apply(lambda t: str(t).replace("_", " ").title())
            df_display["status"] = df_display["status"].apply(lambda s: str(s).replace("_", " ").title())

            df_display = df_display.rename(columns={
                "issue_id": "Issue Reference",
                "event_type": "What Was Detected",
                "observation_count": "Number of Observations",
                "observing_buses": "Buses Reporting the Issue",
                "fused_confidence": "Combined Confidence",
                "severity": "Severity",
                "status": "Current Status",
                "first_observed": "First Detected",
                "last_observed": "Latest Detection"
            })

            st.dataframe(df_display, use_container_width=True)

    with tab2:
        df_raw = pd.DataFrame(raw_events)
        
        # Filter controls
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            evt_type_filter = st.multiselect("Filter by Hazard Type", options=df_raw["event_type"].unique(), default=df_raw["event_type"].unique())
        with c_filter2:
            bus_filter = st.multiselect("Filter by Reporting Bus", options=df_raw["bus_id"].unique(), default=df_raw["bus_id"].unique())

        filtered_df = df_raw[
            (df_raw["event_type"].isin(evt_type_filter)) &
            (df_raw["bus_id"].isin(bus_filter))
        ]

        df_render = filtered_df[["event_id", "event_type", "bus_id", "route_id", "timestamp", "confidence", "status", "details"]].copy()
        df_render["event_type"] = df_render["event_type"].apply(lambda t: str(t).replace("_", " ").title())
        df_render["confidence"] = df_render["confidence"].apply(lambda c: f"{int(float(c)*100)}%")
        df_render["status"] = df_render["status"].apply(lambda s: str(s).replace("_", " ").title())

        df_render = df_render.rename(columns={
            "event_id": "Event Reference",
            "event_type": "What Was Detected",
            "bus_id": "Detecting Bus",
            "route_id": "Bus Route",
            "timestamp": "Time Recorded",
            "confidence": "Detection Confidence",
            "status": "Current Status",
            "details": "Observation Description"
        })

        st.dataframe(df_render, use_container_width=True)

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Filtered Events CSV", csv, "sih26124_urban_events.csv", "text/csv")

    with tab3:
        st.markdown("#### 📡 Geotagged Event Information Viewer")
        st.markdown("Every bus onboard edge computer transmits compact geotagged event packages rather than streaming heavy raw video.")
        
        if raw_events:
            for evt in raw_events[:2]:
                render_human_readable_event_card(evt)
