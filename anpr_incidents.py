"""
SIH26124: Hit-and-Run & Rash Driving ANPR Incident Review Portal
Enhanced Human-in-the-Loop Verification Dashboard
"""

import os
import textwrap
import streamlit as st
import pandas as pd


def get_asset_path(filename: str) -> str:
    """Helper to locate asset files dynamically across workspace environments."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths = [
        os.path.join(base_dir, "assets", filename),
        os.path.join(os.getcwd(), "assets", filename),
        filename
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return filename


def render_anpr_incidents(raw_events: list):
    """
    ANPR / OCR Module — Phase 9 Real AI Verified Capability.
    """
    st.markdown(
        textwrap.dedent("""
        <div style="
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 2px solid #34D399;
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <h2 style="color: #F8FAFC; margin: 0; font-weight: 900; font-size: 1.5rem;">
                        🚔 Rash Driving & Vehicle Identification (Phase 9 Real AI Verified)
                    </h2>
                    <p style="color: #CBD5E1; font-size: 0.88rem; margin: 4px 0 0 0;">
                        <b>Purpose:</b> Real-time YOLOv8 license plate detection, EasyOCR plate recognition, multi-frame temporal validation, and human officer review.
                    </p>
                </div>
                <span style="background: rgba(52, 211, 153, 0.2); color: #34D399; border: 1px solid #34D399; padding: 6px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 800;">
                    🟢 REAL AI — Dedicated YOLO Plate Model + EasyOCR
                </span>
            </div>

            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 16px; border-radius: 8px; font-size: 0.85rem; color: #CBD5E1; margin-top: 10px;">
                <b>Verified Capabilities:</b>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px;">
                    <div>• Dedicated license plate detection (license_plate_yolov8n.pt)</div>
                    <div>• Real EasyOCR plate character extraction</div>
                    <div>• Temporal multi-frame OCR agreement validation</div>
                    <div>• Risky driving behavior image-space analysis</div>
                    <div style="grid-column: span 2;">• Human officer decision workflow (AI-generated alert requiring human review)</div>
                </div>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )
    
    # 1. Initialize State for Human-in-the-Loop Decisions
    if "anpr_statuses" not in st.session_state:
        st.session_state.anpr_statuses = {}

    # Extract ANPR events from raw_events
    anpr_events = [e for e in raw_events if e.get("event_type") == "rash_driving_anpr" or "anpr_data" in e]

    # Ensure all events have a default state in session state
    for evt in anpr_events:
        eid = evt["event_id"]
        if eid not in st.session_state.anpr_statuses:
            st.session_state.anpr_statuses[eid] = "pending"

    # 2. Portal Header & Workflow Visualizer
    st.markdown(
        textwrap.dedent("""
        <div style="
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <h2 style="color: #F8FAFC; margin: 0; font-weight: 900; font-size: 1.6rem; letter-spacing: -0.5px;">
                        🚔 Hit-and-Run & Rash Driving ANPR Portal
                    </h2>
                    <p style="color: #94A3B8; font-size: 0.9rem; margin: 4px 0 0 0;">
                        Visakhapatnam Public Transit Mobile Edge AI Sensing — Automated Plate Recognition & Human-in-the-Loop Sign-off
                    </p>
                </div>
                <span style="background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 800;">
                    🛡️ HUMAN-IN-THE-LOOP ACTIVE
                </span>
            </div>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 14px 0 18px 0;"/>

            <!-- 4-Step Visual Workflow -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px;">
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #475569; border-radius: 10px; padding: 12px; text-align: center;">
                    <div style="font-size: 1.4rem; margin-bottom: 4px;">📸 STEP 1</div>
                    <div style="color: #F8FAFC; font-weight: 700; font-size: 0.85rem;">Bus Camera Capture</div>
                    <div style="color: #94A3B8; font-size: 0.75rem; margin-top: 2px;">Onboard camera captures high-speed trailing vehicle video</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #475569; border-radius: 10px; padding: 12px; text-align: center;">
                    <div style="font-size: 1.4rem; margin-bottom: 4px;">🧠 STEP 2</div>
                    <div style="color: #F8FAFC; font-weight: 700; font-size: 0.85rem;">Vehicle Movement Sensor</div>
                    <div style="color: #94A3B8; font-size: 0.75rem; margin-top: 2px;">Detects overspeeding, rash maneuvers, or hit-and-run evasion</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #475569; border-radius: 10px; padding: 12px; text-align: center;">
                    <div style="font-size: 1.4rem; margin-bottom: 4px;">🔤 STEP 3</div>
                    <div style="color: #F8FAFC; font-weight: 700; font-size: 0.85rem;">License Plate Capture</div>
                    <div style="color: #94A3B8; font-size: 0.75rem; margin-top: 2px;">Extracts vehicle registration plate number</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #F59E0B; border-radius: 10px; padding: 12px; text-align: center; background: rgba(245, 158, 11, 0.1);">
                    <div style="font-size: 1.4rem; margin-bottom: 4px;">👮 STEP 4</div>
                    <div style="color: #FBBF24; font-weight: 800; font-size: 0.85rem;">Officer Verification</div>
                    <div style="color: #CBD5E1; font-size: 0.75rem; margin-top: 2px;">Authorized officer reviews keyframe evidence before confirming incident</div>
                </div>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    if not anpr_events:
        st.info("No active ANPR incidents found.")
        return

    # 3. Compute Summary KPI Statistics
    pending_count = sum(1 for e in anpr_events if st.session_state.anpr_statuses[e["event_id"]] == "pending")
    verified_count = sum(1 for e in anpr_events if st.session_state.anpr_statuses[e["event_id"]] == "verified")
    flagged_count = sum(1 for e in anpr_events if st.session_state.anpr_statuses[e["event_id"]] == "flagged")

    ocr_confs = [e.get("anpr_data", {}).get("ocr_confidence", 0.90) for e in anpr_events]
    avg_ocr = (sum(ocr_confs) / len(ocr_confs)) * 100 if ocr_confs else 92.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("🚨 Pending Officer Review", f"{pending_count} Incidents", delta=f"{len(anpr_events)} Total Tracked")
    with kpi2:
        st.metric("✅ Verified & Forwarded", f"{verified_count} Confirmed", delta=f"{flagged_count} Flagged Clips")
    with kpi3:
        st.metric("🎯 Avg OCR Confidence", f"{avg_ocr:.1f}%", delta="High Precision")
    with kpi4:
        st.metric("⚡ Peak Motion Delta", "+38 px/sec", delta="Gajuwaka Truck Corridor", delta_color="inverse")

    st.markdown("---")

    # 4. Filters & Queue Navigation Tabs
    tab_pending, tab_verified, tab_flagged, tab_analytics = st.tabs([
        f"⏳ Pending Officer Review ({pending_count})",
        f"✅ Verified & Forwarded Incidents ({verified_count})",
        f"⚠️ Flagged for Secondary Review ({flagged_count})",
        "📊 ANPR Telemetry & OCR Analytics"
    ])

    # Filter controls in expander
    with st.expander("🔍 Filter & Search ANPR Incidents", expanded=True):
        fcol1, fcol2 = st.columns([1.5, 1])
        with fcol1:
            search_query = st.text_input("🔎 Search by License Plate or Incident ID", placeholder="e.g. AP 39, AP 31, EVT-VZG-1008...")
        with fcol2:
            route_filter = st.selectbox("📍 Filter by Corridor / Route", ["All Corridors", "ROUTE-101 (Beach Road)", "ROUTE-202 (NAD Flyover)", "ROUTE-303 (Rushikonda)", "ROUTE-404 (MVP Colony)"])

    def filter_events(events_list, target_status):
        filtered = []
        for e in events_list:
            # Check status matching
            if st.session_state.anpr_statuses[e["event_id"]] != target_status:
                continue

            # Check route filter
            if route_filter != "All Corridors" and route_filter.split()[0] not in e.get("route_id", ""):
                continue

            # Check text search
            if search_query:
                q = search_query.strip().upper()
                plate = e.get("anpr_data", {}).get("plate_number", "").upper()
                eid = e.get("event_id", "").upper()
                if q not in plate and q not in eid:
                    continue

            filtered.append(e)
        return filtered

    # Helper function to render an incident review card
    def render_incident_card(evt: dict, idx: int):
        anpr = evt.get("anpr_data", {})
        plate = anpr.get("plate_number", "UNKNOWN")
        ocr_conf = int(anpr.get("ocr_confidence", 0.88) * 100)
        eid = evt["event_id"]
        status_state = st.session_state.anpr_statuses[eid]

        # Resolve evidence image path
        img_ref = evt.get("evidence_reference", "")
        if isinstance(img_ref, str) and not img_ref.startswith("http"):
            img_path = get_asset_path(os.path.basename(img_ref))
        else:
            img_path = img_ref

        # Card Container
        with st.container():
            # Header Row with Indian License Plate Graphic
            st.markdown(
                textwrap.dedent(f"""
                <div style="
                    background: #1E293B;
                    border: 1px solid #334155;
                    border-radius: 12px 12px 0 0;
                    padding: 14px 18px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div style="display: flex; align-items: center; gap: 14px;">
                        <!-- Authentic Indian License Plate UI Badge -->
                        <div style="
                            background: #FACC15;
                            color: #0F172A;
                            border: 2px solid #000;
                            border-radius: 6px;
                            padding: 4px 12px;
                            font-family: 'Courier New', monospace;
                            font-weight: 900;
                            font-size: 1.1rem;
                            letter-spacing: 2px;
                            display: flex;
                            align-items: center;
                            gap: 8px;
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
                        ">
                            <span style="background: #1D4ED8; color: #FFF; font-size: 0.65rem; padding: 2px 4px; border-radius: 3px;">IND</span>
                            {plate}
                        </div>
                        <div>
                            <span style="color: #F8FAFC; font-weight: 800; font-size: 1rem;">Incident #{idx+1}: {evt.get('details', '').split('.')[0]}</span>
                            <div style="color: #94A3B8; font-size: 0.78rem;">ID: <code>{eid}</code> | Sensing Unit: <b>{evt.get('bus_id', 'BUS')}</b> ({evt.get('route_id', 'ROUTE')})</div>
                        </div>
                    </div>
                    <div>
                        <span style="
                            background: {'rgba(52, 211, 153, 0.2)' if status_state == 'verified' else ('rgba(248, 113, 113, 0.2)' if status_state == 'flagged' else 'rgba(245, 158, 11, 0.2)')};
                            color: {'#34D399' if status_state == 'verified' else ('#F87171' if status_state == 'flagged' else '#F59E0B')};
                            border: 1px solid {'#34D399' if status_state == 'verified' else ('#F87171' if status_state == 'flagged' else '#F59E0B')};
                            font-size: 0.8rem;
                            font-weight: 800;
                            padding: 4px 10px;
                            border-radius: 6px;
                        ">
                            {'✅ VERIFIED & DISPATCHED' if status_state == 'verified' else ('⚠️ FLAGGED LOW CONFIDENCE' if status_state == 'flagged' else '⏳ PENDING OFFICER REVIEW')}
                        </span>
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )

            c_img, c_details = st.columns([1.1, 1.2])

            with c_img:
                st.image(
                    img_path,
                    caption=f"📸 AI Windshield Camera Evidence Capture ({evt.get('timestamp', '')})",
                    use_container_width=True
                )
                st.caption(f"📍 GPS Location: **{evt.get('latitude')}° N, {evt.get('longitude')}° E** — [Open Location on Google Maps](https://www.google.com/maps/search/?api=1&query={evt.get('latitude')},{evt.get('longitude')})")

            with c_details:
                st.markdown(
                    textwrap.dedent(f"""
                    <div style="
                        background: #0F172A;
                        border: 1px solid #334155;
                        border-radius: 0 0 12px 12px;
                        padding: 16px;
                    ">
                        <!-- Key Telemetry Cards -->
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px;">
                            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 10px;">
                                <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">OCR Plate Match</div>
                                <div style="color: #34D399; font-size: 1.2rem; font-weight: 900;">{ocr_conf}% Match</div>
                                <div style="color: #CBD5E1; font-size: 0.75rem;">Confidence Score</div>
                            </div>
                            <div style="background: #1E293B; border: 1px solid #F87171; border-radius: 8px; padding: 10px;">
                                <div style="color: #F87171; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">Pixel Motion Delta</div>
                                <div style="color: #F87171; font-size: 1.2rem; font-weight: 900;">{anpr.get('speed_delta_kmh', '+25 px/frame')}</div>
                                <div style="color: #CBD5E1; font-size: 0.75rem;">Above Motion Baseline</div>
                            </div>
                        </div>

                        <!-- Incident Details Table -->
                        <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.7;">
                            • <b>Vehicle Type:</b> <span style="color:#F8FAFC; font-weight:700;">{anpr.get('vehicle_type', 'Passenger Vehicle')}</span><br/>
                            • <b>Recorded Infraction:</b> <span style="color:#F59E0B; font-weight:700;">{evt.get('details', '')}</span><br/>
                            • <b>Transit Corridor:</b> {evt.get('route_id', 'ROUTE')} (Visakhapatnam Transit Network)<br/>
                            • <b>AI Detection Severity:</b> <span style="color:#F87171; font-weight:800; text-transform:uppercase;">{evt.get('severity', 'HIGH')}</span>
                        </div>

                        <hr style="border: 0; border-top: 1px solid #334155; margin: 14px 0 12px 0;"/>
                    """),
                    unsafe_allow_html=True
                )

                # Human-in-the-Loop Verification Action Buttons
                st.markdown("##### 👮 Human Officer Verification Actions:")
                bcol1, bcol2, bcol3 = st.columns([1.2, 1.2, 1])

                with bcol1:
                    if st.button("✅ Confirm & Forward Incident", key=f"verify_btn_{eid}_{idx}"):
                        st.session_state.anpr_statuses[eid] = "verified"
                        st.success(f"Incident {eid} verified by officer and forwarded to Traffic Control Center.")
                        st.rerun()

                with bcol2:
                    if st.button("❌ Flag Review", key=f"flag_btn_{eid}_{idx}"):
                        st.session_state.anpr_statuses[eid] = "flagged"
                        st.warning(f"Incident {eid} flagged for secondary manual video review.")
                        st.rerun()

                with bcol3:
                    if st.button("🔄 Reset", key=f"reset_btn_{eid}_{idx}"):
                        st.session_state.anpr_statuses[eid] = "pending"
                        st.info(f"Reset {eid} status to pending.")
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

    # RENDER TAB 1: PENDING QUEUE
    with tab_pending:
        pending_events = filter_events(anpr_events, "pending")
        if not pending_events:
            st.success("🎉 All pending ANPR incidents have been reviewed by the traffic officer!")
        else:
            st.info(f"📋 Showing **{len(pending_events)}** pending incidents awaiting human officer sign-off.")
            for idx, evt in enumerate(pending_events):
                render_incident_card(evt, idx)

    # RENDER TAB 2: VERIFIED QUEUE
    with tab_verified:
        verified_events = filter_events(anpr_events, "verified")
        if not verified_events:
            st.info("No incidents verified yet. Click '✅ Verify & Fine' on pending incidents above.")
        else:
            st.success(f"✅ Showing **{len(verified_events)}** officer-verified incidents dispatched to Traffic Control Center.")
            for idx, evt in enumerate(verified_events):
                render_incident_card(evt, idx)

    # RENDER TAB 3: FLAGGED QUEUE
    with tab_flagged:
        flagged_events = filter_events(anpr_events, "flagged")
        if not flagged_events:
            st.info("No flagged incidents. Click '❌ Flag Review' if an ANPR image or license plate read is unclear.")
        else:
            st.warning(f"⚠️ Showing **{len(flagged_events)}** incidents flagged for manual video clip inspection.")
            for idx, evt in enumerate(flagged_events):
                render_incident_card(evt, idx)

    # RENDER TAB 4: TELEMETRY & OCR ANALYTICS
    with tab_analytics:
        st.markdown("#### 📊 ANPR Optical Character Recognition & Infraction Telemetry")
        
        analytics_data = []
        for e in anpr_events:
            anpr = e.get("anpr_data", {})
            analytics_data.append({
                "Incident ID": e.get("event_id"),
                "License Plate": anpr.get("plate_number"),
                "Corridor": e.get("route_id"),
                "Vehicle Type": anpr.get("vehicle_type"),
                "OCR Conf (%)": int(anpr.get("ocr_confidence", 0.9) * 100),
                "Speed Excess": anpr.get("speed_delta_kmh"),
                "Status": st.session_state.anpr_statuses[e.get("event_id")].upper()
            })

        df = pd.DataFrame(analytics_data)
        st.dataframe(df, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 🚗 Vehicle Category Breakdown")
            st.bar_chart(df["Vehicle Type"].value_counts())
        with col_b:
            st.markdown("##### 🎯 OCR Confidence Distribution")
            st.line_chart(df["OCR Conf (%)"])
