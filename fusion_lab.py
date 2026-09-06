"""
SIH26124: Multi-Bus Event Fusion Interactive Demonstration Lab
Dedicated visualizer for mentors/judges demonstrating spatial proximity
clustering and joint probabilistic confidence updates.
"""

import streamlit as st
from fusion_engine import calculate_joint_confidence, haversine_distance


def render_fusion_lab():
    """
    Renders an interactive demonstration workspace where mentors can simulate
    multiple buses detecting the same pothole or hazard sequentially.
    """
    st.markdown(
        """<div style="background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
<h3 style="color: #F8FAFC; margin: 0 0 6px 0; font-weight: 800;">🧪 Multi-Bus Event Fusion Interactive Lab</h3>
<p style="color: #94A3B8; font-size: 0.88rem; margin: 0;">Test how independent bus detections are automatically verified and combined when multiple buses pass the same location.</p>
</div>""",
        unsafe_allow_html=True
    )

    if "lab_bus2_active" not in st.session_state:
        st.session_state.lab_bus2_active = True
    if "lab_bus3_active" not in st.session_state:
        st.session_state.lab_bus3_active = True
    if "lab_distance_m" not in st.session_state:
        st.session_state.lab_distance_m = 4.2

    col_ctrl, col_viz = st.columns([1, 1.2])

    with col_ctrl:
        st.subheader("⚙️ Simulation Controls")


        if st.button("⚡ SIMULATED MULTI-BUS DEMONSTRATION (BUS-07, BUS-12, BUS-18)", key="btn_fusion_lab_multibus", use_container_width=True):
            from fusion_engine import MultiBusFusionEngine
            fusion_engine = MultiBusFusionEngine(distance_threshold_m=20.0)
            demo_events = fusion_engine.simulate_multi_bus_demonstration()

            if "raw_events" in st.session_state:
                for devt in demo_events:
                    lat_val = devt.get("latitude", devt.get("location", {}).get("latitude", 17.7145))
                    lon_val = devt.get("longitude", devt.get("location", {}).get("longitude", 83.3235))
                    flat_item = {
                        "event_id": devt["event_id"],
                        "event_type": devt["event_type"],
                        "bus_id": devt["bus_id"],
                        "route_id": devt["route_id"],
                        "timestamp": devt["timestamp"],
                        "latitude": float(lat_val),
                        "longitude": float(lon_val),
                        "confidence": devt["confidence"],
                        "severity": devt["severity"],
                        "priority": devt["priority"],
                        "details": devt.get("details", f"[SIMULATED MULTI-BUS DEMONSTRATION] Pothole on RK Beach Rd by {devt['bus_id']}"),
                        "evidence_reference": devt.get("evidence_reference", f"{devt['bus_id']}_KEYFRAME"),
                        "status": "under_review"
                    }

                    if not any(e["event_id"] == flat_item["event_id"] for e in st.session_state.raw_events):
                        st.session_state.raw_events.insert(0, flat_item)

                persistent_issues, isolated = fusion_engine.fuse_events(st.session_state.raw_events)
                st.session_state.persistent_issues = persistent_issues

            st.session_state.lab_bus2_active = True
            st.session_state.lab_bus3_active = True
            st.session_state.lab_distance_m = 4.2
            st.success("✅ Multi-bus demonstration events (BUS-07, BUS-12, BUS-18) injected into live engine!")
            st.rerun()

        st.markdown("---")

        event_type = st.selectbox(
            "Hazard Event Type",
            ["Pothole / Surface Damage", "Waterlogging", "Damaged Traffic Sign", "Debris / Hazard"],
            index=0
        )

        st.markdown("#### 🚌 Bus Detections Pipeline")

        # Bus 1 Input
        bus1_active = st.checkbox("Bus-07 Observation (Initial Detector)", value=True, disabled=True)
        conf1 = st.slider("BUS-07 Vision Confidence", 0.50, 0.99, 0.91, 0.01)

        # Bus 2 Input
        bus2_active = st.checkbox("Bus-12 Observation (20 mins later)", key="lab_bus2_active")
        conf2 = st.slider("BUS-12 Vision Confidence", 0.50, 0.99, 0.88, 0.01) if bus2_active else 0.0

        # Bus 3 Input
        bus3_active = st.checkbox("Bus-18 Observation (45 mins later)", key="lab_bus3_active")
        conf3 = st.slider("BUS-18 Vision Confidence", 0.50, 0.99, 0.94, 0.01) if bus3_active else 0.0

        st.markdown("#### 📏 Spatial Offset Distance")
        distance_m = st.slider("Distance between detections (Meters)", 0.0, 50.0, step=0.5, key="lab_distance_m")




    with col_viz:
        st.subheader("📊 Live Fusion Engine Output")

        # Determine fusion state
        active_confs = [conf1]
        active_buses = ["BUS-07"]

        if bus2_active and distance_m <= 20.0:
            active_confs.append(conf2)
            active_buses.append("BUS-12")

        if bus3_active and distance_m <= 20.0:
            active_confs.append(conf3)
            active_buses.append("BUS-15")

        is_fused = len(active_buses) > 1
        joint_conf = calculate_joint_confidence(active_confs)

        if distance_m > 20.0:
            st.warning(f"⚠️ Spatial Distance ({distance_m:.1f}m) exceeds fusion threshold (20.0m). Observations treated as SEPARATE issues.")
            is_fused = False
            joint_conf = conf1
            active_buses = ["BUS-07"]

        # Card Rendering
        status_color = "#34D399" if is_fused else "#F59E0B"
        status_text = "CONFIRMED (MULTI-BUS VERIFIED)" if len(active_buses) >= 3 else ("VERIFIED PERSISTENT (2 BUSES)" if len(active_buses) == 2 else "SINGLE OBSERVATION (UNVERIFIED)")

        st.markdown(
            f"""<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 2px solid {status_color}; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
<h3 style="color: #F8FAFC; margin: 0; font-weight: 800;">🔗 Multi-Bus Confirmation</h3>
<span style="background: rgba(52, 211, 153, 0.15); color: {status_color}; border: 1px solid {status_color}; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;">{status_text}</span>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; color: #CBD5E1; font-size: 0.88rem; line-height: 1.6;">
<div style="grid-column: span 2; background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border-left: 3px solid #38BDF8;">
<span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">What happened?</span>
<span style="color: #F8FAFC; font-weight: 600;">Multiple buses independently reported the same road issue along the transit corridor.</span>
</div>
<div>
<span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Buses reporting the issue:</span>
<strong style="color: #38BDF8; font-size: 1.1rem;">{", ".join(active_buses)}</strong>
</div>
<div>
<span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Total observations:</span>
<strong style="color: #F8FAFC; font-size: 1.1rem;">{len(active_buses)} Independent Passes</strong>
</div>
<div>
<span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Combined Verification Confidence:</span>
<strong style="color: #34D399; font-size: 1.1rem;">{round(joint_conf * 100, 1)}%</strong>
</div>
<div>
<span style="color: #94A3B8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Issue Status:</span>
<strong style="color: #34D399; font-size: 1.1rem;">Confirmed</strong>
</div>
<div style="grid-column: span 2; background: rgba(30, 41, 59, 0.6); padding: 10px 14px; border-radius: 8px;">
<span style="color: #60A5FA; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: block;">Result & Why this matters:</span>
<span style="color: #CBD5E1; font-size: 0.84rem;">Instead of creating separate duplicate tickets from every passing bus, the platform automatically combines matching observations into <b>one single high-confidence persistent issue</b> for city authorities.</span>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )

        with st.expander("🔧 Technical Details (Bayesian Joint Confidence Formulation)"):
            st.markdown(
                f"""
                <b>Mathematical Formulation:</b><br/>
                $$C_{{fused}} = 1 - \\prod_{{i=1}}^{{k}} (1 - c_i)$$<br/>
                Calculated probabilistic evidence: <code>1 - {round(1-joint_conf, 5)} = {round(joint_conf, 4)}</code><br/>
                Spatial Gap Distance: <b>{distance_m:.1f} meters</b> (Threshold: 20.0m)
                """,
                unsafe_allow_html=True
            )
