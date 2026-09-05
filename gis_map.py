"""
SIH26124: GIS Command Map Component (Folium Integration)
"""

import folium
from folium import plugins
from streamlit_folium import st_folium
import streamlit as st
from config import CITY_CENTER, DEFAULT_ZOOM, ROUTES


def render_gis_map(buses: list, raw_events: list, persistent_issues: list, selected_route: str = "ALL"):
    """
    Renders the interactive GIS Command Map with public bus nodes,
    single-bus AI hazard detections, and fused multi-bus persistent issue clusters.
    """
    # Create Folium Map centered on Metro City with clean real OpenStreetMap tiles (No API key needed)
    m = folium.Map(
        location=CITY_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # 0. Add Real Map Tile Layers (100% Free Public Tiles — NO API Key / NO Watermarks)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        name="🛰️ Real Satellite View (Esri High-Res)",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr='Tiles &copy; Esri',
        name="🏙️ Real City Streets (Esri World Street)",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr='Tiles &copy; Esri',
        name="🏔️ Real Topographic Map (Esri Topo)",
        overlay=False,
        control=True
    ).add_to(m)

    # 1. Add Bus Route Polyline Layers
    for route_id, route_info in ROUTES.items():
        if selected_route != "ALL" and selected_route != route_id:
            continue

        folium.PolyLine(
            locations=route_info["waypoints"],
            color=route_info["color"],
            weight=4,
            opacity=0.6,
            popup=f"Route: {route_id} - {route_info['name']}"
        ).add_to(m)

    # 2. Add Active Public Bus Nodes
    bus_group = folium.FeatureGroup(name="🚍 Active Bus Fleet")
    for bus in buses:
        if selected_route != "ALL" and bus["route_id"] != selected_route:
            continue

        popup_html = f"""
        <div style="font-family: sans-serif; width: 220px; color: #1E293B;">
            <h4 style="margin: 0 0 6px 0; color: #0284C7; font-weight: 800;">🚍 {bus['bus_id']}</h4>
            <div style="font-size: 0.8rem; line-height: 1.4;">
                <b>Route:</b> {bus['route_id']}<br/>
                <b>Speed:</b> {bus['speed_kmh']} km/h<br/>
                <b>Driver ID:</b> {bus['driver_id']}<br/>
                <b>Cameras:</b> {bus['active_cameras']} Active (Front HD + Rear)<br/>
                <b>Processor Load:</b> {bus['edge_gpu_utilization']}<br/>
                <b>Connection Status:</b> <span style="color:#059669; font-weight:600;">{bus['network_status']}</span>
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[bus["latitude"], bus["longitude"]],
            radius=8,
            color="#38BDF8",
            fill=True,
            fill_color="#0284C7",
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{bus['bus_id']} ({bus['speed_kmh']} km/h)"
        ).add_to(bus_group)

    bus_group.add_to(m)

    # 3. Add Single-Bus AI Hazard Detections
    single_evt_group = folium.FeatureGroup(name="⚠️ Single-Bus AI Detections")
    
    event_icon_colors = {
        "pothole": "red",
        "waterlogging": "blue",
        "damaged_signage": "orange",
        "pedestrian_hazard": "purple",
        "rash_driving_anpr": "darkred",
        "garbage_litter": "green"
    }

    for evt in raw_events:
        if selected_route != "ALL" and evt.get("route_id") != selected_route:
            continue

        icon_color = event_icon_colors.get(evt["event_type"], "gray")

        evt_title_map = {
            "garbage_litter": "🗑️ Garbage / Litter Detected",
            "pothole": "🕳️ Road Damage Detected",
            "waterlogging": "🌊 Waterlogging Hazard",
            "traffic_congestion": "🚦 Traffic Congestion",
            "pedestrian_hazard": "🚶 Pedestrian Hazard",
            "damaged_signage": "⚠️ Damaged Signage"
        }
        human_title = evt_title_map.get(evt["event_type"], evt["event_type"].replace("_", " ").title())
        human_status = str(evt["status"]).replace("_", " ").title()

        popup_html = f"""
        <div style="font-family: sans-serif; width: 240px; color: #0F172A;">
            <h4 style="margin: 0 0 6px 0; color: #0369A1; font-size: 0.95rem; font-weight: 800;">{human_title}</h4>
            <div style="font-size: 0.8rem; line-height: 1.4;">
                <b>Detected By:</b> {evt['bus_id']} ({evt['route_id']})<br/>
                <b>Location:</b> {evt['route_id']} Corridor, Visakhapatnam<br/>
                <b>Confidence:</b> {int(evt['confidence']*100)}%<br/>
                <b>Status:</b> <span style="font-weight:700; color:#D97706;">{human_status}</span>
            </div>
        </div>
        """

        folium.Marker(
            location=[evt["latitude"], evt["longitude"]],
            icon=folium.Icon(color=icon_color, icon="info-sign"),
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{human_title} - {evt['bus_id']} ({int(evt['confidence']*100)}%)"
        ).add_to(single_evt_group)

    single_evt_group.add_to(m)

    # 4. Add Multi-Bus Event Fused Clusters (The Innovation Layer!)
    fused_group = folium.FeatureGroup(name="🔗 Fused Multi-Bus Persistent Issues")

    for issue in persistent_issues:
        buses_list = issue.get("observing_buses", issue.get("buses_observed", []))
        buses_str = ", ".join(buses_list)
        unique_buses = issue.get("unique_bus_count", len(set(buses_list)))
        obs_count = issue.get("observation_count", len(buses_list))
        conf_pct = round(issue.get("fused_confidence", 0.95) * 100, 1)
        first_obs = issue.get("first_observed", issue.get("first_seen", "2026-09-05 10:00:00"))
        last_obs = issue.get("last_observed", issue.get("last_seen", "2026-09-05 10:05:00"))
        iss_status = issue.get("status", "CONFIRMED")
        lat_val = issue.get("latitude", issue.get("centroid_latitude", 17.7145))
        lon_val = issue.get("longitude", issue.get("centroid_longitude", 83.3235))

        evt_type_clean = issue.get('event_type', 'hazard').replace('_', ' ').title()
        popup_html = f"""
        <div style="font-family: sans-serif; width: 260px; color: #0F172A; background: #FAF5FF; padding: 10px; border-radius: 8px;">
            <div style="background: #7E22CE; color: #FFFFFF; padding: 4px 8px; border-radius: 4px; font-weight: 800; font-size: 0.8rem;">
                🔗 Multi-Bus Confirmed Issue
            </div>
            <h4 style="margin: 8px 0 4px 0; color: #6B21A8;">{evt_type_clean}</h4>
            <div style="font-size: 0.8rem; line-height: 1.4;">
                <b>Buses Reporting:</b> <span style="color:#7E22CE; font-weight:700;">{buses_str}</span><br/>
                <b>Unique Buses:</b> {unique_buses} Buses<br/>
                <b>Total Observations:</b> {obs_count}<br/>
                <b>Combined Confidence:</b> <span style="color:#15803D; font-weight:800;">{conf_pct}%</span><br/>
                <b>Status:</b> <span style="color:#B91C1C; font-weight:800;">{iss_status.title()}</span>
            </div>
        </div>
        """

        folium.Marker(
            location=[lat_val, lon_val],
            icon=folium.Icon(color="darkpurple", icon="star", prefix="fa"),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"🔗 MULTI-BUS CONFIRMED: {evt_type_clean} ({unique_buses} Buses)"
        ).add_to(fused_group)

    fused_group.add_to(m)


    # Layer Control
    folium.LayerControl().add_to(m)

    # Render Map in Streamlit
    st_data = st_folium(m, width="100%", height=520)

    # Render Urban Cleanliness KPIs Section Below Map (Phase 8)
    st.markdown("---")
    st.markdown("#### 🧹 Urban Cleanliness & Municipal Garbage KPIs (Phase 8)")
    
    garbage_events = [e for e in raw_events if e.get("event_type") == "garbage_litter"]
    total_garbage = len(garbage_events)
    dumpster_overflows = sum(1 for e in garbage_events if "overflow" in e.get("details", "").lower() or "dumpster" in e.get("details", "").lower() or "overflow" in str(e.get("garbage_class", "")).lower())
    high_severity_waste = sum(1 for e in garbage_events if e.get("severity") in ["high", "critical"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Garbage Incidents", f"{total_garbage} Detections", delta="Active Route Detections")
    k2.metric("Dumpster Overflow Hotspots", f"{dumpster_overflows} Locations", delta="High Waste Density")
    k3.metric("High Severity Waste", f"{high_severity_waste} Critical", delta="Priority Cleanup")
    k4.metric("Garbage AI Engine Mode", "Demonstration", delta="Demonstration Event")

    return st_data
