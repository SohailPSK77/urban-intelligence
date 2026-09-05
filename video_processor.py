"""
SIH26124: Phase 2 & Phase 3 Bus Camera Video Stream Processing & Ingestion Engine
Handles frame-by-frame video ingestion, real PyTorch/YOLOv8 edge neural network inference,
measured hardware execution telemetry, and dynamic SIH geotagged event payload generation.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from yolo_detector import EdgeYOLOv8Detector

try:
    from anpr_engine import ANPRPipelineEngine
    ANPR_ENGINE_AVAILABLE = True
except ImportError:
    ANPR_ENGINE_AVAILABLE = False


class BusCameraVideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 20.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.duration_sec = round(self.total_frames / self.fps, 1)

        # Initialize Real PyTorch / Ultralytics YOLO Engine & ANPR Engine
        self.detector = EdgeYOLOv8Detector(model_name="yolov8n.pt", conf_threshold=0.35)
        self.anpr_engine = ANPRPipelineEngine() if ANPR_ENGINE_AVAILABLE else None

    def get_metadata(self) -> dict:
        """Returns video file properties & camera resolution info."""
        return {
            "file_name": os.path.basename(self.video_path),
            "resolution": f"{self.frame_width} x {self.frame_height}",
            "fps": round(self.fps, 1),
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
            "file_size_mb": round(os.path.getsize(self.video_path) / (1024 * 1024), 2)
        }

    def process_frame_at(self, frame_number: int, bus_id: str = "BUS-07", route_id: str = "ROUTE-101") -> tuple[np.ndarray, list[dict], dict]:
        """
        Extracts a specific frame by index (1..100), executes REAL PyTorch/YOLOv8 inference,
        renders real bounding boxes & measured HUD metrics, and returns (processed_frame_rgb, generated_events, telemetry).
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_number - 1, self.total_frames - 1)))
        ret, frame = self.cap.read()

        if not ret or frame is None:
            # Fallback canvas if frame read fails
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Dynamic Route-Aware GPS Trajectory Progress
        base_coords = {
            "ROUTE-101": (17.7100, 83.3180), # RK Beach
            "ROUTE-202": (17.7320, 83.2510), # NAD Flyover
            "ROUTE-303": (17.7820, 83.3850), # Rushikonda IT Hill
            "ROUTE-404": (17.7450, 83.3310)  # MVP Colony Market
        }
        b_lat, b_lon = base_coords.get(route_id, (17.7200, 83.3000))
        lat_prog = round(b_lat + (frame_number / 100.0) * 0.0015, 5)
        lon_prog = round(b_lon + (frame_number / 100.0) * 0.0012, 5)

        # Execute REAL YOLOv8 Inference with source video FPS & GPS coordinates
        frame_rgb, raw_detections, telemetry = self.detector.process_frame(
            frame=frame,
            frame_number=frame_number,
            route_id=route_id,
            video_fps=self.fps,
            lat=lat_prog,
            lon=lon_prog
        )

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        generated_events = []

        # Convert raw detections into SIH event records
        for idx, det in enumerate(raw_detections):
            det_type = det.get("detection_type", "REAL_AI_DETECTION")
            det_cls = det.get("class", det.get("road_damage_class", "pothole"))
            if det_type == "REAL_AI_DETECTION":
                event_id_prefix = "REAL"
            elif det_type == "REAL_AI_ROAD_DAMAGE":
                event_id_prefix = "REAL-POT"
            elif det_type == "REAL_AI_TRAFFIC_ANALYTICS":
                event_id_prefix = "REAL-TRA"
            else:
                event_id_prefix = "DEMO"
            
            event_item = {
                "event_id": det.get("event_id", f"EVT-VZG-{event_id_prefix}-{det_cls[:3].upper()}-{frame_number:04d}-{idx+1}"),
                "event_type": det_cls,
                "detection_type": det_type,
                "road_damage_class": det.get("road_damage_class", det_cls),
                "road_damage_track_id": det.get("road_damage_track_id", f"TRK-{det.get('track_id', -1)}"),
                "track_id": det.get("track_id", -1),
                "pixel_displacement_px": det.get("pixel_displacement_px", 0.0),
                "in_roi": det.get("in_roi", True),
                "bus_id": bus_id,
                "route_id": route_id,
                "timestamp": now_str,
                "latitude": lat_prog,
                "longitude": lon_prog,
                "location_source": det.get("location_source", "SIMULATED_GPS"),
                "confidence": det["confidence"], # Real YOLO detection confidence
                "bbox": det["bbox"],
                "detection_area_px": det.get("detection_area_px", 0),
                "bbox_width_px": det.get("bbox_width_px", 0),
                "bbox_height_px": det.get("bbox_height_px", 0),
                "severity_method": det.get("severity_method", "Standard Threshold"),
                "model_name": det.get("model_name", telemetry.get("model_name", "YOLOv8n")),
                "model_version": det.get("model_version", "v1.0"),
                "active_vehicle_count": telemetry.get("current_active_vehicles", 0),
                "active_pedestrian_count": telemetry.get("current_active_pedestrians", 0),
                "traffic_density_index": telemetry.get("traffic_density_index", 0.0),
                "congestion_score": telemetry.get("relative_congestion_score", 0.0),
                "video_time_sec": telemetry.get("video_time_sec", 0.0),
                "processing_fps": telemetry.get("processing_fps", 20.0),
                "elapsed_congestion_video_sec": telemetry.get("elapsed_congestion_video_sec", 0.0),
                "persistence_time_source": "VIDEO_TIMESTAMP",
                "is_bottleneck_candidate": telemetry.get("is_bottleneck_candidate", False),
                "severity": det.get("severity", "medium"),
                "priority": det.get("priority", "medium"),
                "details": det["details"],
                "evidence_reference": det.get("evidence_reference", f"FRAME_KEYFRAME_{frame_number}"),
                "status": "needs_maintenance" if det_type == "REAL_AI_ROAD_DAMAGE" or det.get("severity") in ["high", "critical"] else "monitored",
                "source_frame": frame_number
            }
            generated_events.append(event_item)

        # Process real ANPR & Rash Driving events if ANPR engine is active
        if self.anpr_engine:
            try:
                video_time_sec = round(float(frame_number / self.fps), 2)
                anpr_events = self.anpr_engine.process_anpr_frame(
                    frame_bgr=frame,
                    vehicle_tracks=raw_detections,
                    bus_id=bus_id,
                    route_id=route_id,
                    frame_idx=frame_number,
                    video_time_sec=video_time_sec,
                    lat=lat_prog,
                    lon=lon_prog
                )
                if anpr_events:
                    generated_events.extend(anpr_events)
            except Exception as ex:
                print(f"[ANPR VIDEO PROCESSOR NOTICE] ANPR frame processing notice: {ex}")

        return frame_rgb, generated_events, telemetry

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
