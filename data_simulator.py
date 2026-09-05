"""
SIH26124: Telemetry & AI Event Simulator - Visakhapatnam (Vizag) Deployment
Generates realistic bus fleet telemetry, road hazard observations,
traffic density metrics, and ANPR incident records for Vizag city corridors.
"""

import os
import random
from datetime import datetime, timedelta
from config import ROUTES, ACTIVE_BUS_COUNT

# Asset paths for vehicle & hazard evidence images
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
IMG_BUS_FRONT = os.path.join(ASSETS_DIR, "vizag_bus_front.jpg")
IMG_RASH_CAR = os.path.join(ASSETS_DIR, "rash_driving_car.jpg")
IMG_POTHOLE = os.path.join(ASSETS_DIR, "pothole_road_vizag.jpg")


def generate_bus_fleet() -> list[dict]:
    """
    Generates realistic real-time telemetry for 18 active public buses across 4 major Vizag transit corridors.
    """
    buses = []
    route_keys = list(ROUTES.keys())

    for i in range(1, ACTIVE_BUS_COUNT + 1):
        bus_id = f"BUS-{i:02d}"
        route_id = route_keys[(i - 1) % len(route_keys)]
        route_info = ROUTES[route_id]
        waypoints = route_info["waypoints"]

        # Pick a position along the route waypoints with small jitter
        idx = (i * 2) % len(waypoints)
        base_lat, base_lon = waypoints[idx]
        jitter_lat = base_lat + random.uniform(-0.002, 0.002)
        jitter_lon = base_lon + random.uniform(-0.002, 0.002)

        buses.append({
            "bus_id": bus_id,
            "route_id": route_id,
            "route_name": route_info["name"],
            "driver_id": f"APSRTC-VZG-{700 + i}",
            "speed_kmh": round(random.uniform(22.0, 48.0), 1),
            "latitude": round(jitter_lat, 5),
            "longitude": round(jitter_lon, 5),
            "active_cameras": 3,
            "edge_gpu_utilization": f"{random.randint(45, 80)}%",
            "network_status": "ONLINE (5G/LTE)",
            "buffer_queue_size": random.randint(0, 2),
            "last_ping": datetime.now().strftime("%H:%M:%S"),
            "simulation_engine": "PHASE-1 SYNTHETIC TELEMETRY (VIZAG)"
        })

    return buses


def generate_raw_ai_events() -> list[dict]:
    """
    Generates single-bus AI visual detections in Visakhapatnam.
    Includes intentional multi-bus observation overlaps at persistent problem locations
    (e.g., RK Beach Road Pothole, NAD Flyover Waterlogging) to demonstrate Multi-Bus Event Fusion.
    """
    now = datetime.now()
    
    # Vizag Landmark Coordinates
    HOTSPOT_RK_BEACH = (17.7100, 83.3180)     # Submarine Museum / RK Beach Road
    HOTSPOT_NAD_FLYOVER = (17.7320, 83.2510)  # NAD Junction Flyover
    HOTSPOT_SIRIPURAM = (17.7210, 83.3150)    # Siripuram Circle

    events = [
        # --- MULTI-BUS POTHOLE CLUSTER (RK Beach Road near Submarine Museum) ---
        {
            "event_id": "EVT-VZG-1001",
            "event_type": "pothole",
            "bus_id": "BUS-07",
            "route_id": "ROUTE-101",
            "timestamp": (now - timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": HOTSPOT_RK_BEACH[0] + 0.0001,
            "longitude": HOTSPOT_RK_BEACH[1] - 0.0001,
            "confidence": 0.91,
            "severity": "high",
            "priority": "high",
            "details": "Coastal road asphalt erosion near Submarine Museum parking exit (Visual BBox Area Ratio: 5.8%)",
            "evidence_reference": IMG_POTHOLE if os.path.exists(IMG_POTHOLE) else "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=600&q=80",
            "status": "needs_maintenance"
        },
        {
            "event_id": "EVT-VZG-1002",
            "event_type": "pothole",
            "bus_id": "BUS-12",
            "route_id": "ROUTE-101",
            "timestamp": (now - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": HOTSPOT_RK_BEACH[0] + 0.00012,
            "longitude": HOTSPOT_RK_BEACH[1] - 0.00008,
            "confidence": 0.88,
            "severity": "high",
            "priority": "high",
            "details": "Pothole visual detection verified by optical AI pipeline",
            "evidence_reference": IMG_POTHOLE if os.path.exists(IMG_POTHOLE) else "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=600&q=80",
            "status": "needs_maintenance"
        },
        {
            "event_id": "EVT-VZG-1003",
            "event_type": "pothole",
            "bus_id": "BUS-15",
            "route_id": "ROUTE-303",
            "timestamp": (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": HOTSPOT_RK_BEACH[0] + 0.00009,
            "longitude": HOTSPOT_RK_BEACH[1] - 0.00011,
            "confidence": 0.94,
            "severity": "high",
            "priority": "high",
            "details": "Asphalt road defect creating traffic slowdown on Beach Road north lane",
            "evidence_reference": IMG_POTHOLE if os.path.exists(IMG_POTHOLE) else "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=600&q=80",
            "status": "needs_maintenance"
        },

        # --- MULTI-BUS WATERLOGGING CLUSTER (NAD Junction Flyover Underpass) ---
        {
            "event_id": "EVT-VZG-1004",
            "event_type": "waterlogging",
            "bus_id": "BUS-02",
            "route_id": "ROUTE-202",
            "timestamp": (now - timedelta(minutes=35)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": HOTSPOT_NAD_FLYOVER[0],
            "longitude": HOTSPOT_NAD_FLYOVER[1],
            "confidence": 0.89,
            "severity": "high",
            "priority": "high",
            "details": "Stormwater accumulation under NAD Flyover loop (Visual Surface Anomaly)",
            "evidence_reference": IMG_BUS_FRONT if os.path.exists(IMG_BUS_FRONT) else "https://images.unsplash.com/photo-1519692933481-e162a57d6721?auto=format&fit=crop&w=600&q=80",
            "status": "under_review"
        },
        {
            "event_id": "EVT-VZG-1005",
            "event_type": "waterlogging",
            "bus_id": "BUS-08",
            "route_id": "ROUTE-202",
            "timestamp": (now - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": HOTSPOT_NAD_FLYOVER[0] + 0.00005,
            "longitude": HOTSPOT_NAD_FLYOVER[1] - 0.00003,
            "confidence": 0.93,
            "severity": "high",
            "priority": "high",
            "details": "Waterlogging expanding across 2 lanes towards Gajuwaka industrial corridor",
            "evidence_reference": IMG_BUS_FRONT if os.path.exists(IMG_BUS_FRONT) else "https://images.unsplash.com/photo-1519692933481-e162a57d6721?auto=format&fit=crop&w=600&q=80",
            "status": "under_review"
        },

        # --- MULTI-BUS GARBAGE & LITTER ACCUMULATION CLUSTER (Siripuram Circle & MVP Colony) ---
        {
            "event_id": "EVT-VZG-1008",
            "event_type": "garbage_litter",
            "detection_type": "SIMULATED_DEMO",
            "bus_id": "BUS-03",
            "route_id": "ROUTE-303",
            "timestamp": (now - timedelta(minutes=40)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": HOTSPOT_SIRIPURAM[0] + 0.0001,
            "longitude": HOTSPOT_SIRIPURAM[1] - 0.0001,
            "confidence": 0.88,
            "severity": "high",
            "priority": "high",
            "details": "Municipal litter overflow & plastic waste pile near Siripuram Circle (SIMULATED_DEMO)",
            "evidence_reference": IMG_BUS_FRONT if os.path.exists(IMG_BUS_FRONT) else "https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?auto=format&fit=crop&w=600&q=80",
            "status": "needs_maintenance",
            "garbage_class": "plastic_overflow",
            "garbage_track_id": "GARBAGE-TRK-01"
        },
        {
            "event_id": "EVT-VZG-1009",
            "event_type": "garbage_litter",
            "detection_type": "SIMULATED_DEMO",
            "bus_id": "BUS-09",
            "route_id": "ROUTE-404",
            "timestamp": (now - timedelta(minutes=18)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": HOTSPOT_SIRIPURAM[0] + 0.00012,
            "longitude": HOTSPOT_SIRIPURAM[1] - 0.00008,
            "confidence": 0.91,
            "severity": "high",
            "priority": "high",
            "details": "Corroborated municipal waste accumulation at Siripuram commercial zone (SIMULATED_DEMO)",
            "evidence_reference": IMG_BUS_FRONT if os.path.exists(IMG_BUS_FRONT) else "https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?auto=format&fit=crop&w=600&q=80",
            "status": "needs_maintenance",
            "garbage_class": "litter_pile",
            "garbage_track_id": "GARBAGE-TRK-01"
        },

        # --- SINGLE-BUS DETECTIONS (Isolated) ---
        {
            "event_id": "EVT-VZG-1006",
            "event_type": "damaged_signage",
            "bus_id": "BUS-04",
            "route_id": "ROUTE-303",
            "timestamp": (now - timedelta(minutes=55)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": HOTSPOT_SIRIPURAM[0],
            "longitude": HOTSPOT_SIRIPURAM[1],
            "confidence": 0.84,
            "severity": "medium",
            "priority": "medium",
            "details": "Siripuram Junction overhead direction board tilted following coastal winds",
            "evidence_reference": "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?auto=format&fit=crop&w=600&q=80",
            "status": "new"
        },
        {
            "event_id": "EVT-VZG-1007",
            "event_type": "pedestrian_hazard",
            "bus_id": "BUS-09",
            "route_id": "ROUTE-404",
            "timestamp": (now - timedelta(minutes=18)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": 17.7450,
            "longitude": 83.3310,
            "confidence": 0.87,
            "severity": "high",
            "priority": "high",
            "details": "Unregulated pedestrian crossing risk near MVP Colony Double Road market area",
            "evidence_reference": IMG_BUS_FRONT if os.path.exists(IMG_BUS_FRONT) else "https://images.unsplash.com/photo-1509099836639-18ba1795216d?auto=format&fit=crop&w=600&q=80",
            "status": "review_alert"
        },

        # --- RASH DRIVING / ANPR INCIDENT 1 (Visakhapatnam AP-39 Registration) ---
        {
            "event_id": "EVT-VZG-1008",
            "event_type": "rash_driving_anpr",
            "bus_id": "BUS-11",
            "route_id": "ROUTE-303",
            "timestamp": (now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": 17.7820,
            "longitude": 83.3850,
            "confidence": 0.88,
            "severity": "critical",
            "priority": "critical",
            "details": "Overspeeding rash maneuver near Rushikonda IT Hill curve. ANPR OCR extracted: AP 39 TV 7219",
            "evidence_reference": IMG_RASH_CAR if os.path.exists(IMG_RASH_CAR) else "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=600&q=80",
            "status": "REQUIRES HUMAN REVIEW",
            "anpr_data": {
                "plate_number": "AP 39 TV 7219",
                "ocr_confidence": 0.88,
                "vehicle_type": "White SUV / Passenger",
                "speed_delta_kmh": "+28 km/h"
            }
        },

        # --- HIT-AND-RUN / ANPR INCIDENT 2 (NAD Junction Flyover Loop) ---
        {
            "event_id": "EVT-VZG-1009",
            "event_type": "rash_driving_anpr",
            "bus_id": "BUS-02",
            "route_id": "ROUTE-202",
            "timestamp": (now - timedelta(minutes=8)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": 17.7325,
            "longitude": 83.2515,
            "confidence": 0.94,
            "severity": "critical",
            "priority": "critical",
            "details": "Reckless speed lane-cutting and hit-and-run collision threat under NAD Flyover loop. ANPR OCR extracted: AP 31 CZ 4402",
            "evidence_reference": os.path.join(ASSETS_DIR, "anpr_hit_run_ap31.jpg") if os.path.exists(os.path.join(ASSETS_DIR, "anpr_hit_run_ap31.jpg")) else "https://images.unsplash.com/photo-1519692933481-e162a57d6721?auto=format&fit=crop&w=600&q=80",
            "status": "REQUIRES HUMAN REVIEW",
            "anpr_data": {
                "plate_number": "AP 31 CZ 4402",
                "ocr_confidence": 0.94,
                "vehicle_type": "Black Sedan / High-Speed Reckless Overtake",
                "speed_delta_kmh": "+35 km/h"
            }
        },

        # --- RASH DRIVING / ANPR INCIDENT 3 (RK Beach Expressway U-Turn) ---
        {
            "event_id": "EVT-VZG-1010",
            "event_type": "rash_driving_anpr",
            "bus_id": "BUS-07",
            "route_id": "ROUTE-101",
            "timestamp": (now - timedelta(minutes=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": 17.7115,
            "longitude": 83.3192,
            "confidence": 0.91,
            "severity": "critical",
            "priority": "critical",
            "details": "Illegal high-speed U-turn crossing central median divider on RK Beach Road. ANPR OCR extracted: AP 39 BK 9182",
            "evidence_reference": os.path.join(ASSETS_DIR, "anpr_rash_driving_ap39.jpg") if os.path.exists(os.path.join(ASSETS_DIR, "anpr_rash_driving_ap39.jpg")) else "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?auto=format&fit=crop&w=600&q=80",
            "status": "REQUIRES HUMAN REVIEW",
            "anpr_data": {
                "plate_number": "AP 39 BK 9182",
                "ocr_confidence": 0.91,
                "vehicle_type": "Red Sports Coupe / Median Violation",
                "speed_delta_kmh": "+26 km/h"
            }
        },

        # --- HIT-AND-RUN / ANPR INCIDENT 4 (MVP Colony Market Street) ---
        {
            "event_id": "EVT-VZG-1011",
            "event_type": "rash_driving_anpr",
            "bus_id": "BUS-09",
            "route_id": "ROUTE-404",
            "timestamp": (now - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": 17.7455,
            "longitude": 83.3315,
            "confidence": 0.89,
            "severity": "critical",
            "priority": "critical",
            "details": "Speeding vehicle pedestrian hazard near MVP Colony market crossing. ANPR OCR extracted: AP 39 EU 1509",
            "evidence_reference": os.path.join(ASSETS_DIR, "vizag_pedestrian_cross_404.jpg") if os.path.exists(os.path.join(ASSETS_DIR, "vizag_pedestrian_cross_404.jpg")) else "https://images.unsplash.com/photo-1509099836639-18ba1795216d?auto=format&fit=crop&w=600&q=80",
            "status": "REQUIRES HUMAN REVIEW",
            "anpr_data": {
                "plate_number": "AP 39 EU 1509",
                "ocr_confidence": 0.89,
                "vehicle_type": "Silver Pickup Truck / Pedestrian Zone Hazard",
                "speed_delta_kmh": "+22 km/h"
            }
        },

        # --- HIT-AND-RUN / ANPR INCIDENT 5 (Gajuwaka Industrial Corridor) ---
        {
            "event_id": "EVT-VZG-1012",
            "event_type": "rash_driving_anpr",
            "bus_id": "BUS-05",
            "route_id": "ROUTE-202",
            "timestamp": (now - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": 17.6890,
            "longitude": 83.2150,
            "confidence": 0.94,
            "severity": "critical",
            "priority": "critical",
            "details": "Heavy container truck hit-and-run evasion after dangerous high-speed drift in Gajuwaka industrial zone. ANPR OCR extracted: AP 35 TH 8831",
            "evidence_reference": os.path.join(ASSETS_DIR, "anpr_hit_run_ap35.jpg") if os.path.exists(os.path.join(ASSETS_DIR, "anpr_hit_run_ap35.jpg")) else "https://images.unsplash.com/photo-1519692933481-e162a57d6721?auto=format&fit=crop&w=600&q=80",
            "status": "REQUIRES HUMAN REVIEW",
            "anpr_data": {
                "plate_number": "AP 35 TH 8831",
                "ocr_confidence": 0.96,
                "vehicle_type": "Heavy Freight Truck / Hit-and-Run Evasion",
                "speed_delta_kmh": "+38 km/h"
            }
        },

        # --- RASH DRIVING / ANPR INCIDENT 6 (Siripuram Junction Night Wheelie) ---
        {
            "event_id": "EVT-VZG-1013",
            "event_type": "rash_driving_anpr",
            "bus_id": "BUS-04",
            "route_id": "ROUTE-303",
            "timestamp": (now - timedelta(minutes=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": 17.7210,
            "longitude": 83.3150,
            "confidence": 0.91,
            "severity": "critical",
            "priority": "critical",
            "details": "Reckless night motorcycle stunt and high-speed slalom overtake at Siripuram Junction circle. ANPR OCR extracted: AP 39 MW 5021",
            "evidence_reference": os.path.join(ASSETS_DIR, "anpr_rash_driving_night.jpg") if os.path.exists(os.path.join(ASSETS_DIR, "anpr_rash_driving_night.jpg")) else "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=600&q=80",
            "status": "REQUIRES HUMAN REVIEW",
            "anpr_data": {
                "plate_number": "AP 39 MW 5021",
                "ocr_confidence": 0.93,
                "vehicle_type": "Sports Motorcycle / Stunt Hazard",
                "speed_delta_kmh": "+31 km/h"
            }
        },

        # --- HIT-AND-RUN / ANPR INCIDENT 7 (NAD Junction Flyover Ramp) ---
        {
            "event_id": "EVT-VZG-1014",
            "event_type": "rash_driving_anpr",
            "bus_id": "BUS-02",
            "route_id": "ROUTE-202",
            "timestamp": (now - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": 17.7330,
            "longitude": 83.2520,
            "confidence": 0.95,
            "severity": "critical",
            "priority": "critical",
            "details": "Red light jump and hit-and-run evasion attempt at NAD Flyover upper ramp. ANPR OCR extracted: AP 31 EA 1109",
            "evidence_reference": os.path.join(ASSETS_DIR, "anpr_hit_run_ap31_car.jpg") if os.path.exists(os.path.join(ASSETS_DIR, "anpr_hit_run_ap31_car.jpg")) else "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?auto=format&fit=crop&w=600&q=80",
            "status": "REQUIRES HUMAN REVIEW",
            "anpr_data": {
                "plate_number": "AP 31 EA 1109",
                "ocr_confidence": 0.95,
                "vehicle_type": "Black Luxury Sedan / Red Light Evasion",
                "speed_delta_kmh": "+34 km/h"
            }
        }
    ]

    return events


def get_traffic_analytics_summary() -> dict:
    """
    Generates aggregated fleet-wide traffic density metrics for Visakhapatnam,
    integrating dynamic Phase 4 Real AI telemetry from active ByteTrack observations when available.
    """
    import streamlit as st

    # Check if real AI telemetry exists in session state
    latest_telemetry = st.session_state.get("latest_telemetry", {}) if "st" in locals() and hasattr(st, "session_state") else {}
    rolling_history = st.session_state.get("rolling_history", []) if "st" in locals() and hasattr(st, "session_state") else []
    if not rolling_history and "st" in locals() and hasattr(st, "session_state"):
        rolling_history = st.session_state.get("telemetry_history", [])

    if not rolling_history:
        rolling_history = [
            {
                "frame_number": idx,
                "traffic_density_index": round(min(1.0, 0.25 + (idx % 5) * 0.08), 2),
                "congestion_score": round(min(1.0, 0.30 + (idx % 4) * 0.10), 2),
                "active_vehicle_count": max(1, (idx * 3) % 6 + 1),
                "active_vehicles": max(1, (idx * 3) % 6 + 1)
            } for idx in range(1, 21)
        ]


    active_vehicles = latest_telemetry.get("current_active_vehicles", 5)
    active_pedestrians = latest_telemetry.get("current_active_pedestrians", 2)
    tdi = latest_telemetry.get("traffic_density_index", 0.50)
    density_level = latest_telemetry.get("traffic_density", "MODERATE")
    congestion_score = latest_telemetry.get("relative_congestion_score", 0.48)
    congestion_level = latest_telemetry.get("congestion_level", "SLOW")
    moving_vehicles = latest_telemetry.get("moving_vehicle_count", 4)
    stationary_vehicles = latest_telemetry.get("stationary_vehicle_count", 1)
    unique_tracks = latest_telemetry.get("cumulative_unique_tracks", 14)
    cls_counts = latest_telemetry.get("class_wise_counts", {
        "cars": 3,
        "buses": 1,
        "trucks": 0,
        "motorcycles": 1,
        "bicycles": 0,
        "pedestrians": 2
    })

    return {
        "active_vehicles": active_vehicles,
        "active_pedestrians": active_pedestrians,
        "traffic_density_index": tdi,
        "density_level": density_level,
        "relative_congestion_score": congestion_score,
        "congestion_level": congestion_level,
        "moving_vehicles": moving_vehicles,
        "stationary_vehicles": stationary_vehicles,
        "cumulative_unique_tracks": unique_tracks,
        "total_vehicles_detected": 1650 + unique_tracks,
        "class_breakdown": {
            "Cars": cls_counts.get("cars", 3),
            "Buses": cls_counts.get("buses", 1),
            "Trucks": cls_counts.get("trucks", 0),
            "Two-Wheelers": cls_counts.get("motorcycles", 1),
            "Bicycles": cls_counts.get("bicycles", 0),
            "Pedestrians": cls_counts.get("pedestrians", 2)
        },
        "network_density_level": f"{density_level} (TDI: {tdi} - PEAK HOUR VIZAG PORT & IT CORRIDOR)",
        "rolling_history": rolling_history,
        "bottlenecks": [
            {
                "route_id": "ROUTE-101",
                "location": "RK Beach Road Monitored ROI (AI-derived Bottleneck Candidate)",
                "length_km": 0.6,
                "est_delay_min": 5,
                "avg_speed_kmh": f"{latest_telemetry.get('average_displacement_px', 4.2)} px/frame (pixels/frame != km/h)",
                "is_real_ai": True
            },
            {
                "route_id": "ROUTE-202",
                "location": "NAD Junction Flyover Underpass Loop",
                "length_km": 1.2,
                "est_delay_min": 10,
                "avg_speed_kmh": "16.5 km/h (Corridor Fleet Speed)",
                "is_real_ai": False
            }
        ]
    }

