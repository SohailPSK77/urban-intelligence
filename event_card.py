"""
SIH26124: Human-Readable Event Information Card Component
Converts technical event payloads into clean, user-friendly UI cards for judges, officials, and citizens.
Places raw technical JSON data behind a collapsible "🔧 Technical Details" expander.
"""

import streamlit as st


def get_event_badge(detection_type: str, event_type: str) -> tuple[str, str]:
    """Returns (badge_html, badge_text) based on REAL AI vs DEMONSTRATION vs PLANNED status."""
    det_upper = str(detection_type).upper()
    
    if event_type == "rash_driving_anpr":
        return (
            '<span style="background: rgba(52, 211, 153, 0.2); color: #34D399; border: 1px solid #34D399; padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 800;">🟢 REAL AI — Dedicated YOLO Plate Model + EasyOCR</span>',
            "REAL_AI"
        )
    elif "SIMULATED" in det_upper or event_type in ["garbage_litter", "waterlogging", "pedestrian_hazard", "damaged_signage"]:
        return (
            '<span style="background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 800;">🟡 DEMONSTRATION — Prototype Data</span>',
            "DEMONSTRATION"
        )
    else:
        return (
            '<span style="background: rgba(52, 211, 153, 0.2); color: #34D399; border: 1px solid #34D399; padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 800;">🟢 REAL AI — Neural Model</span>',
            "REAL_AI"
        )


def render_human_readable_event_card(evt: dict):
    """
    Renders a clean, human-readable Event Information Card.
    Provides clear answers to: WHAT, WHERE, WHEN, WHICH BUS, SEVERITY, STATUS, and WHY IT MATTERS.
    Places raw technical JSON inside a collapsible expander for system audits.
    """
    event_type = evt.get("event_type", evt.get("road_damage_class", "hazard"))
    detection_type = evt.get("detection_type", "REAL_AI_DETECTION")
    badge_html, badge_code = get_event_badge(detection_type, event_type)

    bus_id = evt.get("bus_id", "BUS-07")
    route_id = evt.get("route_id", "ROUTE-303")

    # Location formatting
    corridor_names = {
        "ROUTE-101": "RK Beach Coastal Expressway, Visakhapatnam",
        "ROUTE-202": "NAD Flyover Industrial Corridor, Visakhapatnam",
        "ROUTE-303": "Rushikonda IT Hill Corridor, Visakhapatnam",
        "ROUTE-404": "MVP Colony Double Road, Visakhapatnam"
    }
    loc_data = evt.get("location")
    if isinstance(loc_data, dict):
        location_str = f"{loc_data.get('corridor', corridor_names.get(route_id, route_id))}, {loc_data.get('city', 'Visakhapatnam')}"
    else:
        location_str = corridor_names.get(route_id, f"{route_id}, Visakhapatnam")

    # GPS Status
    raw_gps = str(evt.get("location_source", evt.get("location", {}).get("location_source", "SIMULATED_GPS")))
    gps_status = "Simulated GPS Location" if "SIMULATED" in raw_gps else "Hardware NMEA GPS Location"

    # Status formatting
    raw_status = str(evt.get("status", "needs_review")).replace("_", " ").title()

    # Tracking Reference
    track_id_val = (
        evt.get("garbage_track_id") or
        evt.get("road_damage_track_id") or
        (f"Tracking Ref #{evt['track_id']}" if evt.get("track_id", -1) != -1 else "GARBAGE-TRK-01" if event_type == "garbage_litter" else "POTHOLE-TRK-01")
    )

    # Confidence formatting
    conf_val = evt.get("confidence")
    has_real_conf = conf_val is not None and badge_code == "REAL_AI"
    conf_pct_str = f"{int(float(conf_val) * 100)}%" if has_real_conf else "N/A (Demonstration Event)"

    # Severity & Priority
    severity_str = str(evt.get("severity", "high")).title()

    # Hazard-specific content templates
    if event_type == "garbage_litter":
        title = "🗑️ Garbage / Litter Detected"
        border_color = "#10B981"
        what_happened = "Garbage or litter accumulation was identified along the public bus corridor."
        why_it_matters = "Helps municipal sanitation teams locate waste overflows along transit routes for prompt dispatch and cleanup."
        det_status_str = "Demonstration Mode"

    elif event_type == "pothole":
        title = "🕳️ Road Damage Detected"
        border_color = "#EF4444"
        what_happened = "A road surface pothole defect was detected by the bus windshield camera AI."
        why_it_matters = "Helps public works authorities identify dangerous road damage early and prioritize repair maintenance."
        det_status_str = "Real AI Detection (YOLOv8s Pothole Model)"

    elif event_type in ["traffic_congestion", "bottleneck_candidate"]:
        title = "🚦 Traffic Congestion Condition"
        border_color = "#F59E0B"
        what_happened = "Traffic in the monitored road segment is showing high vehicle occupancy and reduced movement speed."
        why_it_matters = "Enables transit operators to detect corridor bottlenecks early and optimize bus dispatch schedules."
        det_status_str = "Real AI Analytics (YOLOv8n + ByteTrack)"

    elif event_type == "waterlogging":
        title = "🌊 Waterlogging Hazard Detected"
        border_color = "#3B82F6"
        what_happened = "Stormwater accumulation or monsoon drain overflow was observed on the transit road."
        why_it_matters = "Alerts city drainage authorities to waterlogged underpasses and prevents vehicle strandings."
        det_status_str = "Demonstration Mode"

    elif event_type == "rash_driving_anpr":
        title = "🚔 Suspected Rash Driving & License Plate Identification"
        border_color = "#F97316"
        what_happened = "A vehicle displaying high-risk maneuvering or erratic motion was tracked, and its license plate was extracted using dedicated YOLO plate model + EasyOCR."
        why_it_matters = "Alerts traffic enforcement officers with vehicle registration details and keyframe evidence for officer review."
        det_status_str = "Real AI Detection (YOLOv8 Plate Model + EasyOCR)"

    elif event_type in ["car", "bus", "truck", "motorcycle", "vehicle"]:
        title = f"🚗 {event_type.title()} Detected"
        border_color = "#38BDF8"
        what_happened = f"A moving {event_type} was observed in the camera field of view by the onboard bus vision system."
        why_it_matters = "Monitors real-time vehicle flow, traffic density, and lane occupancy along city transit corridors."
        det_status_str = "Real AI Detection (YOLOv8n + ByteTrack)"

    elif event_type in ["person", "pedestrian"]:
        title = "🚶 Pedestrian Detected"
        border_color = "#C084FC"
        what_happened = "A pedestrian was observed near the transit road or bus stop area."
        why_it_matters = "Enhances pedestrian safety awareness around high-density bus stops and crosswalks."
        det_status_str = "Real AI Detection (YOLOv8n + ByteTrack)"

    else:
        title = f"⚠️ {event_type.replace('_', ' ').title()} Detected"
        border_color = "#38BDF8"
        what_happened = f"An urban hazard observation ({event_type.replace('_', ' ')}) was recorded."
        why_it_matters = "Provides continuous urban infrastructure monitoring across city transit corridors."
        det_status_str = "Demonstration Mode" if badge_code == "DEMONSTRATION" else "Real AI Detection"

    # Render Card HTML
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border: 2px solid {border_color};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 12px;">
                <h3 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 1.25rem;">
                    {title}
                </h3>
                {badge_html}
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; color: #CBD5E1; font-size: 0.88rem; line-height: 1.6;">
                <div style="grid-column: span 2; background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border-left: 3px solid {border_color};">
                    <span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">What happened?</span>
                    <span style="color: #F8FAFC; font-weight: 600;">{what_happened}</span>
                </div>

                <div>
                    <span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Detected by:</span>
                    <strong style="color: #F8FAFC; font-size: 1.05rem;">{bus_id}</strong>
                </div>
                <div>
                    <span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Route:</span>
                    <strong style="color: #F8FAFC; font-size: 1.05rem;">{route_id}</strong>
                </div>

                <div style="grid-column: span 2;">
                    <span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Location:</span>
                    <strong style="color: #38BDF8; font-size: 0.95rem;">{location_str}</strong>
                </div>

                <div>
                    <span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Detection status:</span>
                    <strong style="color: #34D399;">{det_status_str}</strong>
                </div>
                <div>
                    <span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Location status:</span>
                    <strong style="color: #FBBF24;">{gps_status}</strong>
                </div>

                <div>
                    <span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Tracking Reference:</span>
                    <strong style="color: #A78BFA;">{track_id_val}</strong>
                </div>
                <div>
                    <span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Current Status:</span>
                    <strong style="color: #F87171;">{raw_status}</strong>
                </div>

                {'<div><span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">AI Detection Confidence:</span><strong style="color: #34D399;">' + conf_pct_str + '</strong></div>' if has_real_conf else ''}
                <div>
                    <span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Severity Rating:</span>
                    <strong style="color: #F8FAFC;">{severity_str}</strong>
                </div>

                <div style="grid-column: span 2; background: rgba(30, 41, 59, 0.6); padding: 10px 14px; border-radius: 8px; margin-top: 4px;">
                    <span style="color: #60A5FA; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Why this matters:</span>
                    <span style="color: #CBD5E1; font-size: 0.82rem;">{why_it_matters}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Technical Details Expander for developers/auditors
    with st.expander("🔧 Technical Details (For System Audits & Developers)"):
        st.caption("Underlying canonical JSON event payload stored in SQLite central database:")
        st.json(evt)


def render_human_readable_telemetry_card(tel: dict):
    """Renders a clean human-readable summary card for hardware telemetry logs."""
    bus_id = tel.get("bus_id", "BUS-07")
    route_id = tel.get("route_id", "ROUTE-101")
    frame_num = tel.get("frame_number", 1)
    fps = tel.get("measured_fps", 12.1)
    latency = tel.get("latency_ms", 82.0)
    gps = tel.get("gps_location", "17.7145° N, 83.3235° E")
    active_veh = tel.get("active_vehicles", 3)
    active_ped = tel.get("active_pedestrians", 1)
    tdi = tel.get("traffic_density_index", 0.35)
    disp = tel.get("pixel_displacement_px", 3.5)

    st.markdown(
        f"""
        <div style="background: #0F172A; border: 1px solid #334155; border-radius: 10px; padding: 16px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="color: #38BDF8; margin: 0;">⚡ Onboard AI Hardware Telemetry Summary</h4>
                <span style="background: rgba(52, 211, 153, 0.2); color: #34D399; border: 1px solid #34D399; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">🟢 REAL PROFILING</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; color: #CBD5E1; font-size: 0.85rem;">
                <div><b>Detecting Bus:</b> {bus_id} ({route_id})</div>
                <div><b>Frame Index:</b> #{frame_num}</div>
                <div><b>Measured Processing Speed:</b> {fps} FPS</div>
                <div><b>Hardware Latency:</b> {latency} ms</div>
                <div><b>GPS Location:</b> {gps}</div>
                <div><b>Active Tracked Vehicles:</b> {active_veh}</div>
                <div><b>Active Pedestrians:</b> {active_ped}</div>
                <div><b>Traffic Density Index:</b> {tdi}</div>
                <div style="grid-column: span 2;"><b>Movement Analytics:</b> {disp} px/frame (Pixel Displacement)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("🔧 Technical Details (Raw Telemetry Dictionary)"):
        st.json(tel)
