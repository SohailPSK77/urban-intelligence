"""
SIH26124: GPS Abstraction Layer (Phase 6)
Provides explicit separation between Simulated GPS (deterministic route waypoints synced with video time)
and Real Hardware NMEA GPS stub interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

from config import ROUTES


class GPSProvider(ABC):
    """Abstract base class for location providers."""

    @abstractmethod
    def get_location(self, route_id: str, frame_number: int, video_time_sec: float) -> Dict[str, Any]:
        """Returns location dict with latitude, longitude, location_source, and accuracy_m."""
        pass


class SimulatedGPSProvider(GPSProvider):
    """
    Simulated GPS Provider: Maps video timeline frame index to deterministic route trajectory waypoints.
    Explicitly tags location_source = "SIMULATED_GPS".
    """
    def __init__(self):
        self.location_source = "SIMULATED_GPS"
        self.accuracy_m = 5.0

    def get_location(self, route_id: str, frame_number: int = 1, video_time_sec: float = 0.0) -> Dict[str, Any]:
        route = ROUTES.get(route_id, ROUTES["ROUTE-101"])
        waypoints = route.get("waypoints", [[17.7200, 83.3000]])
        
        # Calculate progress along route waypoints (1 to 100 frame timeline)
        progress = max(0.0, min(1.0, (frame_number - 1) / 99.0))
        num_segments = len(waypoints) - 1

        if num_segments <= 0:
            lat, lon = waypoints[0]
        else:
            scaled_idx = progress * num_segments
            seg_idx = min(int(scaled_idx), num_segments - 1)
            seg_t = scaled_idx - seg_idx
            
            p1 = waypoints[seg_idx]
            p2 = waypoints[seg_idx + 1]

            lat = round(p1[0] + (p2[0] - p1[0]) * seg_t, 5)
            lon = round(p1[1] + (p2[1] - p1[1]) * seg_t, 5)

        return {
            "latitude": lat,
            "longitude": lon,
            "location_source": self.location_source,
            "location_accuracy_m": self.accuracy_m,
            "video_time_sec": round(video_time_sec, 2),
            "corridor_name": route.get("name", "Visakhapatnam Transit Corridor")
        }


class RealGPSProvider(GPSProvider):
    """
    Hardware GPS Provider Stub: Interface for future USB/Serial NMEA 0183 GPS hardware receiver integration.
    """
    def __init__(self, serial_port: str = "/dev/ttyUSB0", baud_rate: int = 9600):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.connected = False
        self.location_source = "REAL_HARDWARE_GPS"
        self.accuracy_m = 2.5

    def connect(self) -> bool:

        """Stub method to initiate serial connection to hardware NMEA receiver."""
        # Hardware connection logic will be implemented when physical NMEA receiver is attached.
        self.connected = False
        return False

    def get_location(self, route_id: str, frame_number: int = 1, video_time_sec: float = 0.0) -> Dict[str, Any]:
        """Returns NMEA satellite lock location if connected, or raises error."""
        if not self.connected:
            raise RuntimeError("Hardware NMEA GPS receiver not connected. Fallback to SimulatedGPSProvider.")
        
        return {
            "latitude": 17.7200,
            "longitude": 83.3000,
            "location_source": self.location_source,
            "location_accuracy_m": self.accuracy_m,
            "video_time_sec": round(video_time_sec, 2),
            "satellite_count": 0
        }
