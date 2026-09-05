"""
SIH26124: Multi-Bus Event Fusion Engine (Phase 6 Upgraded)
Implements deterministic spatial-temporal clustering ($dist \le 20\text{m}$, $\Delta t \le 300\text{s}$),
same-bus deduplication, unique bus corroboration count, joint fused confidence scoring,
and full issue lifecycle management (NEW, CONFIRMED, ASSIGNED, IN_PROGRESS, RESOLVED).
"""

import math
from datetime import datetime
from typing import Dict, List, Tuple, Set, Any
from config import FUSION_DISTANCE_THRESHOLD_METERS


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points on Earth (in meters).
    Uses the Haversine formula.
    """
    R = 6371000.0  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c


def parse_timestamp_sec(ts_input: Any) -> float:
    """Parses timestamp string or float into epoch seconds."""
    if isinstance(ts_input, (int, float)):
        return float(ts_input)
    if isinstance(ts_input, str) and ts_input:
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
            try:
                dt = datetime.strptime(ts_input, fmt)
                return dt.timestamp()
            except ValueError:
                pass
    return datetime.now().timestamp()


def calculate_fused_confidence(confidences: List[float]) -> float:
    """
    Calculates fused confidence score across k independent observations:
    C_fused = 1 - PROD_{i=1..k} (1 - c_i)
    Explicitly labeled as Fused Confidence Score (not calibrated probability).
    """
    if not confidences:
        return 0.0
    
    prod_unlikely = 1.0
    for c in confidences:
        clamped_c = max(0.01, min(0.99, float(c)))
        prod_unlikely *= (1.0 - clamped_c)
        
    fused = 1.0 - prod_unlikely
    return round(min(0.9999, fused), 4)


# Alias for backward compatibility with fusion_lab component
calculate_joint_confidence = calculate_fused_confidence



class MultiBusFusionEngine:
    def __init__(self, distance_threshold_m: float = 20.0, time_threshold_sec: float = 300.0):
        self.distance_threshold_m = distance_threshold_m  # Max 20 meters
        self.time_threshold_sec = time_threshold_sec      # Max 300 seconds (5 minutes)

    def deduplicate_same_bus_events(self, raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicates consecutive frame observations from the SAME bus unit for the same track/hazard
        into a single consolidated event candidate before central ingestion/fusion.
        """
        deduped = []
        track_groups: Dict[str, List[Dict[str, Any]]] = {}

        for evt in raw_events:
            bus_id = evt.get("bus_id", "BUS-07")
            evt_type = evt.get("event_type", "pothole")
            trk_id = evt.get("garbage_track_id", evt.get("road_damage_track_id", evt.get("track_id", "TRK-01")))
            
            # Key for same bus & same object track
            group_key = f"{bus_id}_{evt_type}_{trk_id}"
            
            if group_key not in track_groups:
                track_groups[group_key] = []
            track_groups[group_key].append(evt)

        for group_key, evts in track_groups.items():
            # Pick highest confidence event as primary representative
            primary = max(evts, key=lambda x: float(x.get("confidence", 0.0)))
            consolidated = dict(primary)
            consolidated["observation_count"] = len(evts)
            
            # Record video timeline bounds
            v_times = [float(e.get("video_time_sec", 0.0)) for e in evts if "video_time_sec" in e]
            if v_times:
                consolidated["first_observed_video_time_sec"] = min(v_times)
                consolidated["last_observed_video_time_sec"] = max(v_times)

            deduped.append(consolidated)

        return deduped

    def fuse_events(self, raw_events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Processes a list of single-bus AI event payloads and clusters them into
        Persistent Urban Issues.

        Strict Spatial-Temporal Rules:
        1. Spatial Proximity: Distance <= 20.0 meters
        2. Temporal Window: Timestamp Difference <= 300.0 seconds
        BOTH conditions MUST be satisfied to cluster events.

        Returns:
            (persistent_issues, isolated_events)
        """
        clusters: List[Dict[str, Any]] = []

        # Sort raw events by timestamp
        sorted_events = sorted(
            raw_events, 
            key=lambda x: parse_timestamp_sec(x.get("timestamp", ""))
        )

        for event in sorted_events:
            evt_lat = float(event.get("latitude", 17.7200))
            evt_lon = float(event.get("longitude", 83.3000))
            evt_type = event.get("event_type", "pothole")
            evt_ts_sec = parse_timestamp_sec(event.get("timestamp", ""))
            evt_bus = event.get("bus_id", "BUS-07")

            matched_cluster = None

            for cluster in clusters:
                # 1. Event Type Match
                if cluster["event_type"] != evt_type:
                    continue

                # 2. Spatial Compatibility Check (dist <= 20m)
                dist = haversine_distance(
                    cluster["centroid_latitude"], cluster["centroid_longitude"],
                    evt_lat, evt_lon
                )
                if dist > self.distance_threshold_m:
                    continue

                # 3. Temporal Compatibility Check (dt <= 300s)
                cluster_last_ts_sec = parse_timestamp_sec(cluster["last_observed_timestamp"])
                time_diff = abs(evt_ts_sec - cluster_last_ts_sec)
                if time_diff > self.time_threshold_sec:
                    continue

                # Both spatial AND temporal conditions satisfied!
                matched_cluster = cluster
                break

            if matched_cluster:
                # Merge observation into cluster
                matched_cluster["observations"].append(event)
                matched_cluster["contributing_event_ids"].append(event.get("event_id", f"EVT-{len(matched_cluster['observations'])}"))
                matched_cluster["observing_buses"].add(evt_bus)
                
                matched_cluster["observation_count"] = sum(e.get("observation_count", 1) for e in matched_cluster["observations"])
                matched_cluster["unique_bus_count"] = len(matched_cluster["observing_buses"])

                # Recalculate geographic centroid
                lats = [float(e["latitude"]) for e in matched_cluster["observations"]]
                lons = [float(e["longitude"]) for e in matched_cluster["observations"]]
                matched_cluster["centroid_latitude"] = round(sum(lats) / len(lats), 5)
                matched_cluster["centroid_longitude"] = round(sum(lons) / len(lons), 5)
                matched_cluster["latitude"] = matched_cluster["centroid_latitude"]
                matched_cluster["longitude"] = matched_cluster["centroid_longitude"]

                # Recalculate fused confidence
                conf_list = [float(e.get("confidence", 0.90)) for e in matched_cluster["observations"]]
                matched_cluster["fused_confidence"] = calculate_fused_confidence(conf_list)

                # Update timestamp bounds
                matched_cluster["last_observed_timestamp"] = event.get("timestamp", matched_cluster["last_observed_timestamp"])

                # Issue Lifecycle & Status Update
                if matched_cluster["unique_bus_count"] >= 3:
                    matched_cluster["status"] = "CONFIRMED"
                    matched_cluster["priority"] = "CRITICAL"
                elif matched_cluster["unique_bus_count"] == 2:
                    matched_cluster["status"] = "CONFIRMED"
                    matched_cluster["priority"] = "HIGH"
                else:
                    matched_cluster["status"] = "NEW"

            else:
                # Create a new persistent issue cluster
                cid = f"PR-{len(clusters) + 1:04d}"
                new_cluster = {
                    "issue_id": cid,
                    "cluster_id": cid,
                    "event_type": evt_type,
                    "centroid_latitude": round(evt_lat, 5),
                    "centroid_longitude": round(evt_lon, 5),
                    "latitude": round(evt_lat, 5),
                    "longitude": round(evt_lon, 5),
                    "first_observed_timestamp": event.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "last_observed_timestamp": event.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "observing_buses": {evt_bus},
                    "buses_observed": [evt_bus],
                    "unique_bus_count": 1,
                    "observations": [event],
                    "observation_count": event.get("observation_count", 1),
                    "fused_confidence": float(event.get("confidence", 0.90)),
                    "severity": event.get("severity", "medium"),
                    "priority": event.get("priority", "medium").upper(),
                    "contributing_event_ids": [event.get("event_id", f"EVT-{cid}-1")],
                    "status": "NEW",
                    "route_id": event.get("route_id", "ROUTE-101"),
                    "evidence_reference": event.get("evidence_reference", "")
                }
                clusters.append(new_cluster)

        persistent_issues = []
        isolated_events = []

        for c in clusters:
            c["observing_buses"] = sorted(list(c["observing_buses"]))
            c["buses_observed"] = c["observing_buses"]
            c["first_observed"] = c.get("first_observed_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            c["last_observed"] = c.get("last_observed_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if c["unique_bus_count"] >= 2 or c["observation_count"] >= 3:
                persistent_issues.append(c)
            else:
                isolated_events.append(c)


        return persistent_issues, isolated_events

    def simulate_multi_bus_demonstration(self) -> List[Dict[str, Any]]:
        """Method wrapper for generate_simulated_multibus_demonstration()."""
        return generate_simulated_multibus_demonstration()


def generate_simulated_multibus_demonstration() -> List[Dict[str, Any]]:

    """
    Generates controlled demonstration observations from BUS-07, BUS-12, and BUS-18
    observing the SAME pothole at RK Beach Coastal Expressway within 12 meters and 120 seconds.
    Used for SIMULATED_MULTI_BUS_DEMONSTRATION mode.
    """
    base_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    evt1 = {
        "event_id": "EVT-DEMO-BUS07-POT-001",
        "event_type": "pothole",
        "detection_type": "REAL_AI_ROAD_DAMAGE",
        "bus_id": "BUS-07",
        "route_id": "ROUTE-101",
        "timestamp": base_time,
        "video_time_sec": 14.2,
        "latitude": 17.7145,
        "longitude": 83.3235,
        "location_source": "SIMULATED_GPS",
        "location_accuracy_m": 5.0,
        "location": {
            "city": "Visakhapatnam",
            "corridor": "RK Beach Coastal Expressway",
            "latitude": 17.7145,
            "longitude": 83.3235,
            "location_source": "SIMULATED_GPS"
        },
        "confidence": 0.92,
        "severity": "critical",
        "priority": "critical",
        "source_frame": 28,
        "evidence_reference": "BUS-07_POTHOLE_KEYFRAME_28",
        "model_name": "pothole_yolov8s.pt (YOLOv8s)",
        "model_version": "v1.0",
        "status": "NEW",
        "road_damage_class": "pothole",
        "road_damage_track_id": "POTHOLE-TRK-01",
        "bbox": [580, 480, 920, 680],
        "bbox_area": 68000.0,
        "severity_method": "AI-assisted visual severity heuristic (Area Ratio)"
    }

    evt2 = {
        "event_id": "EVT-DEMO-BUS12-POT-002",
        "event_type": "pothole",
        "detection_type": "REAL_AI_ROAD_DAMAGE",
        "bus_id": "BUS-12",
        "route_id": "ROUTE-101",
        "timestamp": base_time,
        "video_time_sec": 48.5,
        "latitude": 17.71454,  # ~5 meters offset
        "longitude": 83.32353,
        "location_source": "SIMULATED_GPS",
        "location_accuracy_m": 5.0,
        "location": {
            "city": "Visakhapatnam",
            "corridor": "RK Beach Coastal Expressway",
            "latitude": 17.71454,
            "longitude": 83.32353,
            "location_source": "SIMULATED_GPS"
        },
        "confidence": 0.89,
        "severity": "high",
        "priority": "high",
        "source_frame": 45,
        "evidence_reference": "BUS-12_POTHOLE_KEYFRAME_45",
        "model_name": "pothole_yolov8s.pt (YOLOv8s)",
        "model_version": "v1.0",
        "status": "NEW",
        "road_damage_class": "pothole",
        "road_damage_track_id": "POTHOLE-TRK-02",
        "bbox": [560, 470, 910, 660],
        "bbox_area": 66500.0,
        "severity_method": "AI-assisted visual severity heuristic (Area Ratio)"
    }

    evt3 = {
        "event_id": "EVT-DEMO-BUS18-POT-003",
        "event_type": "pothole",
        "detection_type": "REAL_AI_ROAD_DAMAGE",
        "bus_id": "BUS-18",
        "route_id": "ROUTE-101",
        "timestamp": base_time,
        "video_time_sec": 92.0,
        "latitude": 17.71446,  # ~6 meters offset
        "longitude": 83.32347,
        "location_source": "SIMULATED_GPS",
        "location_accuracy_m": 5.0,
        "location": {
            "city": "Visakhapatnam",
            "corridor": "RK Beach Coastal Expressway",
            "latitude": 17.71446,
            "longitude": 83.32347,
            "location_source": "SIMULATED_GPS"
        },
        "confidence": 0.94,
        "severity": "critical",
        "priority": "critical",
        "source_frame": 72,
        "evidence_reference": "BUS-18_POTHOLE_KEYFRAME_72",
        "model_name": "pothole_yolov8s.pt (YOLOv8s)",
        "model_version": "v1.0",
        "status": "NEW",

        "road_damage_class": "pothole",
        "road_damage_track_id": "POTHOLE-TRK-03",
        "bbox": [590, 485, 930, 690],
        "bbox_area": 69700.0,
        "severity_method": "AI-assisted visual severity heuristic (Area Ratio)"
    }


    return [evt1, evt2, evt3]
