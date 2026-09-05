"""
SIH26124: Canonical UrbanEvent Schema Module (Phase 6)
Defines the standardized, canonical UrbanEvent schema used across the edge sensing pipeline,
durable edge SQLite buffer, local FastAPI ingestion, central event store, multi-bus fusion engine, and GIS dashboard.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class UrbanEvent:
    event_id: str
    event_type: str                            # pothole, waterlogging, pedestrian_hazard, traffic_congestion, bottleneck_candidate
    detection_type: str                        # REAL_AI_ROAD_DAMAGE, REAL_AI_TRAFFIC_ANALYTICS, SIMULATED_DEMONSTRATION_EVENT
    bus_id: str                                # e.g. BUS-07
    route_id: str                              # e.g. ROUTE-101
    timestamp: str                             # ISO 8601 string
    video_time_sec: float                      # Elapsed video timeline position in seconds

    latitude: float                            # Geotagged latitude
    longitude: float                           # Geotagged longitude
    location_source: str = "SIMULATED_GPS"     # SIMULATED_GPS or REAL_HARDWARE_GPS
    location_accuracy_m: float = 5.0           # GPS accuracy radius in meters

    confidence: float = 0.90                   # Detection confidence score (0.0 to 1.0)
    severity: str = "medium"                   # critical, high, medium, low
    priority: str = "medium"                   # critical, high, medium, low

    source_frame: int = 1                      # Video frame index (1..100)
    evidence_reference: str = ""               # Onboard keyframe / snapshot identifier

    model_name: str = "YOLOv8s"                # YOLOv8s for pothole, YOLOv8n + ByteTrack for traffic
    model_version: str = "v1.0"                # Model release version

    status: str = "NEW"                        # NEW, CONFIRMED, ASSIGNED, IN_PROGRESS, RESOLVED

    # Pothole & Road Damage Extensions
    road_damage_class: Optional[str] = None    # pothole, crack, waterlogging
    road_damage_track_id: Optional[str] = None # e.g. POTHOLE-TRK-01
    bbox: Optional[List[int]] = None           # [xmin, ymin, xmax, ymax] in pixels
    bbox_area: Optional[float] = None          # Bounding box area in pixels
    severity_method: Optional[str] = None     # Area Ratio visual heuristic or Sensor threshold
    observation_count: int = 1                 # Total observations
    first_observed_video_time_sec: Optional[float] = None
    last_observed_video_time_sec: Optional[float] = None

    # Garbage & Urban Cleanliness Extensions (Phase 8)
    garbage_class: Optional[str] = None        # plastic_overflow, litter_pile, municipal_waste
    garbage_track_id: Optional[str] = None     # e.g. GARBAGE-TRK-01

    # Traffic Intelligence Extensions
    track_id: Optional[int] = None             # ByteTrack object persistent ID
    object_class: Optional[str] = None         # car, bus, truck, motorcycle, bicycle, person
    pixel_displacement: Optional[float] = None  # Pixel displacement (px/frame)
    tdi: Optional[float] = None                # Traffic Density Index (0.0 to 1.0)
    congestion_score: Optional[float] = None   # Relative congestion score (0.0 to 1.0)
    bottleneck_candidate: bool = False         # True if sustained congestion exceeds threshold
    persistence_duration_sec: float = 0.0      # Sustained video-time persistence duration in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Converts the UrbanEvent dataclass to a standard clean dictionary."""
        d = asdict(self)
        # Clean null values if necessary or preserve structure
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UrbanEvent":
        """Instantiates an UrbanEvent object from a dictionary, ensuring type casting & field defaults."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Apply schema defaults for missing mandatory fields
        if "location_source" not in filtered_data or not filtered_data["location_source"]:
            filtered_data["location_source"] = "SIMULATED_GPS"
        if "location_accuracy_m" not in filtered_data:
            filtered_data["location_accuracy_m"] = 5.0
        if "status" not in filtered_data or not filtered_data["status"]:
            filtered_data["status"] = "NEW"
        if "timestamp" not in filtered_data or not filtered_data["timestamp"]:
            filtered_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Map legacy keys if present
        if "road_damage_track_id" not in filtered_data or not filtered_data["road_damage_track_id"]:
            if "track_id" in data and data["track_id"] != -1:
                filtered_data["road_damage_track_id"] = f"TRK-{data['track_id']:02d}"

        if "bbox_area" not in filtered_data or filtered_data["bbox_area"] is None:
            if "detection_area_px" in data:
                filtered_data["bbox_area"] = float(data["detection_area_px"])

        if "pixel_displacement" not in filtered_data or filtered_data["pixel_displacement"] is None:
            if "pixel_displacement_px" in data:
                filtered_data["pixel_displacement"] = float(data["pixel_displacement_px"])

        if "tdi" not in filtered_data or filtered_data["tdi"] is None:
            if "traffic_density_index" in data:
                filtered_data["tdi"] = float(data["traffic_density_index"])

        return cls(**filtered_data)


def validate_urban_event_schema(payload: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validates whether an incoming JSON payload meets canonical UrbanEvent schema requirements.
    Returns (is_valid, error_message).
    """
    required_fields = ["event_id", "event_type", "bus_id", "route_id", "latitude", "longitude"]
    missing = [f for f in required_fields if f not in payload or payload[f] is None]
    if missing:
        return False, f"Missing required canonical fields: {', '.join(missing)}"

    try:
        lat = float(payload["latitude"])
        lon = float(payload["longitude"])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return False, f"Invalid GPS coordinates: lat={lat}, lon={lon}"
    except (ValueError, TypeError):
        return False, "Latitude and longitude must be numeric numbers"

    try:
        conf = float(payload.get("confidence", 0.90))
        if not (0.0 <= conf <= 1.0):
            return False, f"Confidence score out of range [0.0, 1.0]: {conf}"
    except (ValueError, TypeError):
        return False, "Confidence must be a numeric float"

    return True, "Valid UrbanEvent schema"
