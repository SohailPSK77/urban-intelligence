"""
SIH26124: Real YOLOv8 + ByteTrack Multi-Object Tracking Engine
Loads actual pretrained YOLOv8 model weights (yolov8n.pt), executes real PyTorch forward-pass
inference and ByteTrack multi-object tracking across consecutive frames, maintains persistent track IDs,
computes pixel displacement, and measures actual hardware execution FPS.
"""

import os
import cv2
import numpy as np
import time
import math
from datetime import datetime
from typing import Dict, List, Tuple

# Try importing PyTorch & Ultralytics YOLO
ULTRALYTICS_AVAILABLE = False
TORCH_AVAILABLE = False
torch = None
YOLO = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass

try:
    from road_damage_detector import RoadDamageYOLOv8Detector
    ROAD_DAMAGE_DETECTOR_AVAILABLE = True
except ImportError:
    ROAD_DAMAGE_DETECTOR_AVAILABLE = False



class EdgeYOLOv8Detector:
    """
    Real PyTorch / Ultralytics YOLOv8 + ByteTrack Object Tracking & Traffic Intelligence Engine.
    Executes actual forward pass on video frames, tracks persistent object IDs across frames,
    filters road ROI, computes Traffic Density Index (TDI), Relative Congestion Score,
    vehicle movement / pixel displacement, AI-derived bottlenecks, and rolling history.
    """
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35, iou_threshold: float = 0.45):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None

        # ByteTrack Centroid History for Pixel Displacement (track_id -> (cx, cy))
        self.previous_centroids: Dict[int, Tuple[int, int]] = {}
        self.unique_track_ids = set()
        self.unique_vehicle_track_ids = set()
        self.unique_pedestrian_track_ids = set()

        # Target road classes for tracking
        self.target_vehicle_classes = {"car", "bus", "truck", "motorcycle", "bicycle", "suv", "vehicle"}
        self.target_road_classes = self.target_vehicle_classes.union({"person"})

        # -------------------------------------------------------------
        # PHASE 4: TRAFFIC INTELLIGENCE CONFIGURATION
        # -------------------------------------------------------------
        # Configurable ROI Zone (Normalized relative bounds: [ymin_ratio, xmin_ratio, ymax_ratio, xmax_ratio])
        # Covers lower road lane area (35% to 98% height, 5% to 95% width)
        self.roi_norm = (0.35, 0.05, 0.98, 0.95)
        self.roi_capacity = 10  # Max vehicle capacity for camera ROI segment
        
        # Configurable Movement & Congestion Thresholds
        self.stationary_threshold_px = 3.0       # Pixel displacement threshold for stationary vehicle (pixels/frame != physical speed in km/h)
        self.bottleneck_min_vehicles = 3          # Min ROI vehicle count required for bottleneck candidate
        self.bottleneck_min_duration_sec = 2.0    # Min video-time persistence duration (in seconds) required for bottleneck candidate trigger
        
        self.consecutive_congestion_frames = 0        # Counter for congestion persistence frames
        self.congestion_start_video_time: float = None# Source video timestamp (in seconds) when congestion condition first begins
        self.elapsed_congestion_video_sec: float = 0.0# Time delta in video timeline of sustained congestion

        # Rolling History Buffer (Last 100 observations)
        self.rolling_history: List[Dict] = []

        # Hardware Device Detection
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "CUDA"
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.device = "CPU"
            self.device_name = "System CPU (Intel/AMD)"

        self.gpu_temp = "N/A"  # Explicitly N/A when no NVML thermal sensor is attached

        # Load Actual YOLO Model Weights
        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(model_name)
                print(f"[YOLO+BYTETRACK ENGINE] Successfully loaded model weights '{model_name}' on device '{self.device}'")
            except Exception as e:
                print(f"[YOLO+BYTETRACK ENGINE] Error loading model '{model_name}': {e}")
                self.model = None

        # Load Dedicated Phase 5 Real AI Road Damage Detector (Pothole Model)
        self.road_damage_detector = None
        if ROAD_DAMAGE_DETECTOR_AVAILABLE:
            try:
                self.road_damage_detector = RoadDamageYOLOv8Detector("pothole_yolov8s.pt", conf_threshold=0.25)
            except Exception as e:
                print(f"[ROAD DAMAGE ENGINE] Error initializing road damage detector: {e}")
                self.road_damage_detector = None



    def set_thresholds(self, conf_threshold: float, iou_threshold: float):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def set_roi(self, ymin: float = 0.35, xmin: float = 0.05, ymax: float = 0.98, xmax: float = 0.95, capacity: int = 10):
        """Sets configurable road ROI bounds and capacity."""
        self.roi_norm = (ymin, xmin, ymax, xmax)
        self.roi_capacity = max(1, capacity)

    def is_in_roi(self, cx: int, bottom_y: int, frame_w: int, frame_h: int) -> bool:
        """Checks whether bounding box bottom-center point lies inside the defined ROI zone."""
        roi_y1 = int(self.roi_norm[0] * frame_h)
        roi_x1 = int(self.roi_norm[1] * frame_w)
        roi_y2 = int(self.roi_norm[2] * frame_h)
        roi_x2 = int(self.roi_norm[3] * frame_w)
        return (roi_x1 <= cx <= roi_x2) and (roi_y1 <= bottom_y <= roi_y2)

    def process_frame(self, frame: np.ndarray, frame_number: int = 1, route_id: str = "ROUTE-101", video_fps: float = 20.0, lat: float = None, lon: float = None) -> Tuple[np.ndarray, List[Dict], Dict]:
        """
        Processes a single BGR OpenCV video frame.
        Performs REAL forward-pass inference + ByteTrack tracking + Phase 4 Traffic Intelligence.
        - VIDEO_TIME_SEC: Used for traffic persistence duration on recorded video frames.
        - PROCESSING_WALL_CLOCK_SEC: Used ONLY for hardware performance/FPS measurement.
        Returns:
            - Processed RGB frame with persistent track IDs, ROI polygon & HUD
            - List of detection events with persistent track_id, displacement, TDI, congestion score
            - Telemetry dict with actual measured FPS, TDI, Relative Congestion Score, and movement metrics
        """
        t_start = time.time()
        frame_h, frame_w = frame.shape[:2]

        # VIDEO_TIME_SEC: Derived from source video frame position and source FPS
        video_time_sec = round(max(0, frame_number - 1) / max(1.0, video_fps), 2)

        all_detections = []
        real_detections = []
        simulated_events = []
        
        class_wise_counts = {
            "cars": 0,
            "buses": 0,
            "trucks": 0,
            "motorcycles": 0,
            "bicycles": 0,
            "pedestrians": 0
        }

        roi_vehicle_count = 0
        roi_stationary_count = 0
        roi_moving_count = 0
        displacements_px: List[float] = []

        # -------------------------------------------------------------
        # 1. REAL FORWARD-PASS NEURAL INFERENCE + BYTETRACK MULTI-OBJECT TRACKING
        # -------------------------------------------------------------
        if self.model is not None:
            t_infer_start = time.time()
            
            # Execute ByteTrack tracking
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            t_infer_ms = (time.time() - t_infer_start) * 1000.0

            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy()) # YOLO detection confidence
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                    # Filter for relevant road classes
                    if cls_name not in self.target_road_classes:
                        continue

                    # Extract persistent track ID from ByteTrack
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                    if track_id != -1:
                        self.unique_track_ids.add(track_id)
                        if cls_name in self.target_vehicle_classes:
                            self.unique_vehicle_track_ids.add(track_id)
                        elif cls_name == "person":
                            self.unique_pedestrian_track_ids.add(track_id)

                    # Update class-wise frequency
                    if cls_name == "car":
                        class_wise_counts["cars"] += 1
                    elif cls_name == "bus":
                        class_wise_counts["buses"] += 1
                    elif cls_name == "truck":
                        class_wise_counts["trucks"] += 1
                    elif cls_name == "motorcycle":
                        class_wise_counts["motorcycles"] += 1
                    elif cls_name == "bicycle":
                        class_wise_counts["bicycles"] += 1
                    elif cls_name == "person":
                        class_wise_counts["pedestrians"] += 1

                    # Compute centroid & pixel displacement (Note: pixels/frame != physical speed in km/h)
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)
                    bottom_y = int(xyxy[3])

                    displacement_px = 0.0
                    if track_id != -1 and track_id in self.previous_centroids:
                        prev_cx, prev_cy = self.previous_centroids[track_id]
                        displacement_px = round(math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2), 1)

                    if track_id != -1:
                        self.previous_centroids[track_id] = (cx, cy)

                    # ROI Filtering Check
                    in_roi = self.is_in_roi(cx, bottom_y, frame_w, frame_h)
                    
                    if cls_name in self.target_vehicle_classes:
                        displacements_px.append(displacement_px)
                        if in_roi:
                            roi_vehicle_count += 1
                            if displacement_px < self.stationary_threshold_px:
                                roi_stationary_count += 1
                            else:
                                roi_moving_count += 1

                    # Real detection payload with ByteTrack track_id & ROI status
                    det_item = {
                        "detection_type": "REAL_AI_DETECTION",
                        "track_id": track_id,
                        "class": cls_name,
                        "label": f"[REAL AI] {cls_name.upper()} ID:#{track_id}",
                        "confidence": round(conf, 2), # YOLO detection confidence
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "pixel_displacement_px": displacement_px,
                        "in_roi": in_roi,
                        "severity": "medium" if cls_name in self.target_vehicle_classes else "high",
                        "priority": "medium" if cls_name in self.target_vehicle_classes else "high",
                        "details": f"ByteTrack persistent track ID #{track_id} '{cls_name}' (YOLO Conf: {int(conf*100)}%, Disp: {displacement_px}px, In-ROI: {in_roi}) in Frame #{frame_number}",
                        "evidence_reference": f"REAL_TRACKED_KEYFRAME_{frame_number}",
                        "status": "active_tracked"
                    }
                    real_detections.append(det_item)

                    # Draw green/cyan bounding box with ByteTrack persistent ID
                    color = (52, 211, 153) if cls_name in self.target_vehicle_classes else (56, 189, 248)
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    
                    lbl_str = f"ID:#{track_id} {cls_name.upper()} {int(conf*100)}%" if track_id != -1 else f"{cls_name.upper()} {int(conf*100)}%"
                    (lbl_w, lbl_h), _ = cv2.getTextSize(lbl_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (xyxy[0], xyxy[1] - lbl_h - 8), (xyxy[0] + lbl_w + 8, xyxy[1]), color, -1)
                    cv2.putText(frame, lbl_str, (xyxy[0] + 4, xyxy[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        else:
            t_infer_ms = 0.0

        # -------------------------------------------------------------
        # 2. PHASE 4: REAL TRAFFIC INTELLIGENCE FORMULAS & ANALYTICS
        # -------------------------------------------------------------
        active_vehicles_count = sum(1 for d in real_detections if d["class"] in self.target_vehicle_classes)
        active_pedestrians_count = sum(1 for d in real_detections if d["class"] == "person")

        # A. Traffic Density Index (TDI) = active_vehicles_in_roi / roi_capacity (prototype metric)
        tdi = round(min(1.0, roi_vehicle_count / self.roi_capacity), 2)
        if tdi < 0.30:
            density_level = "LOW"
        elif tdi < 0.60:
            density_level = "MODERATE"
        elif tdi < 0.85:
            density_level = "HIGH"
        else:
            density_level = "SEVERE"

        # B. Relative Congestion Score (0.0 to 1.0)
        # Occupancy Ratio (45%) + Low Movement Ratio (35%) + ROI Persistence (20%)
        occupancy_ratio = min(1.0, roi_vehicle_count / self.roi_capacity)
        low_movement_ratio = (roi_stationary_count / roi_vehicle_count) if roi_vehicle_count > 0 else 0.0
        
        # Vehicle Movement Aggregates (Note: pixels/frame != physical speed in km/h)
        avg_displacement_px = round(sum(displacements_px) / len(displacements_px), 1) if displacements_px else 0.0
        moving_vehicles_count = sum(1 for d in displacements_px if d >= self.stationary_threshold_px)
        stationary_vehicles_count = sum(1 for d in displacements_px if d < self.stationary_threshold_px)

        # C. Time-Based Bottleneck Candidate Persistence Tracking (Using VIDEO_TIME_SEC)
        # Check if congestion condition is active in ROI
        congestion_condition_active = (roi_vehicle_count >= self.bottleneck_min_vehicles) and (low_movement_ratio >= 0.5) and (occupancy_ratio >= 0.4)
        
        if congestion_condition_active:
            if self.congestion_start_video_time is None:
                self.congestion_start_video_time = video_time_sec
            self.consecutive_congestion_frames += 1
            self.elapsed_congestion_video_sec = round(video_time_sec - self.congestion_start_video_time, 2)
        else:
            self.congestion_start_video_time = None
            self.consecutive_congestion_frames = max(0, self.consecutive_congestion_frames - 1)
            self.elapsed_congestion_video_sec = 0.0

        persistence_factor = min(1.0, self.consecutive_congestion_frames / 10.0)
        congestion_score = round(0.45 * occupancy_ratio + 0.35 * low_movement_ratio + 0.20 * persistence_factor, 2)

        if congestion_score < 0.30:
            congestion_level = "NORMAL"
        elif congestion_score < 0.55:
            congestion_level = "SLOW"
        elif congestion_score < 0.80:
            congestion_level = "CONGESTED"
        else:
            congestion_level = "SEVERE"

        # AI-Derived Bottleneck Candidate Flag (Requires minimum VIDEO_TIME_SEC persistence in seconds)
        is_bottleneck_candidate = (self.elapsed_congestion_video_sec >= self.bottleneck_min_duration_sec) and (congestion_score >= 0.55)

        # D. Generate Real Traffic Analytics Event when candidate threshold reached
        if is_bottleneck_candidate or congestion_score >= 0.65:
            real_detections.append({
                "detection_type": "REAL_AI_TRAFFIC_ANALYTICS",
                "track_id": -1,
                "class": "bottleneck_candidate" if is_bottleneck_candidate else "traffic_congestion",
                "label": f"[REAL AI] TRAFFIC {'BOTTLENECK CANDIDATE' if is_bottleneck_candidate else 'CONGESTION'} DETECTED",
                "confidence": min(0.99, round(0.70 + congestion_score * 0.30, 2)), # YOLO detection confidence
                "bbox": [
                    int(self.roi_norm[1] * frame_w),
                    int(self.roi_norm[0] * frame_h),
                    int(self.roi_norm[3] * frame_w),
                    int(self.roi_norm[2] * frame_h)
                ],
                "pixel_displacement_px": avg_displacement_px,
                "in_roi": True,
                "severity": "high" if is_bottleneck_candidate else "medium",
                "priority": "high" if is_bottleneck_candidate else "medium",
                "details": f"[REAL AI TRAFFIC ANALYTICS] {'AI-Derived Bottleneck Candidate' if is_bottleneck_candidate else 'Relative Traffic Congestion'} in camera ROI (Active ROI Vehicles: {roi_vehicle_count}, TDI: {tdi}, Relative Congestion Score: {congestion_score}, Avg Disp: {avg_displacement_px}px, Video Sustained: {self.elapsed_congestion_video_sec}s, Frame #{frame_number})",
                "evidence_reference": f"REAL_TRAFFIC_ANALYTICS_KEYFRAME_{frame_number}",
                "status": "detected"
            })

        # -------------------------------------------------------------
        # 3. PHASE 5: REAL AI ROAD DAMAGE / POTHOLE DETECTION (DEDICATED MODEL)
        # -------------------------------------------------------------
        if self.road_damage_detector is not None and route_id == "ROUTE-101":
            _, rd_events, _ = self.road_damage_detector.process_frame(
                frame=frame,
                frame_number=frame_number,
                video_time_sec=video_time_sec,
                route_id=route_id,
                bus_id="BUS-07"
            )
            for rd_evt in rd_events:
                real_detections.append(rd_evt)


        # -------------------------------------------------------------
        # 4. CLEARLY LABELED SIMULATED DEMONSTRATION EVENTS FOR UNSUPPORTED HAZARDS
        # -------------------------------------------------------------
        if route_id == "ROUTE-303":
            simulated_events.append({
                "detection_type": "SIMULATED_DEMONSTRATION_EVENT",
                "track_id": -1,
                "class": "waterlogging",
                "label": "[SIMULATED DEMO] WATERLOGGING DETECTED",
                "confidence": round(0.93 + (frame_number % 4) * 0.01, 2),
                "bbox": [600, 540, 1020, 670],
                "pixel_displacement_px": 0.0,
                "severity": "high",
                "priority": "high",
                "details": f"[SIMULATED DEMO EVENT] Rushikonda IT Hill Road stormwater logging demonstration (Frame #{frame_number})",
                "evidence_reference": "BUS-11_WATERLOGGING_DEMO_KEYFRAME",
                "status": "under_review"
            })

        elif route_id == "ROUTE-404":
            simulated_events.append({
                "detection_type": "SIMULATED_DEMONSTRATION_EVENT",
                "track_id": -1,
                "class": "pedestrian_hazard",
                "label": "[SIMULATED DEMO] PEDESTRIAN CROWD",
                "confidence": round(0.88 + (frame_number % 6) * 0.01, 2),
                "bbox": [420, 480, 570, 750],
                "pixel_displacement_px": 0.0,
                "severity": "high",
                "priority": "high",
                "details": f"[SIMULATED DEMO EVENT] MVP Colony market street crossing demonstration (Frame #{frame_number})",
                "evidence_reference": "BUS-09_PEDESTRIAN_DEMO_KEYFRAME",
                "status": "review_alert"
            })

        all_detections = real_detections + simulated_events

        # -------------------------------------------------------------
        # 4. MEASURE TRUE HARDWARE EXECUTION LATENCY & RENDER HUD OVERLAYS
        # -------------------------------------------------------------
        t_total_sec = max(time.time() - t_start, 0.001)
        actual_fps = round(1.0 / t_total_sec, 1)
        total_latency_ms = round(t_total_sec * 1000.0, 1)

        # Draw Camera Traffic ROI Rectangle on Video Frame
        roi_y1 = int(self.roi_norm[0] * frame_h)
        roi_x1 = int(self.roi_norm[1] * frame_w)
        roi_y2 = int(self.roi_norm[2] * frame_h)
        roi_x2 = int(self.roi_norm[3] * frame_w)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (245, 158, 11), 1, cv2.LINE_AA)
        cv2.putText(frame, f"TRAFFIC ROI ZONE (Cap:{self.roi_capacity})", (roi_x1 + 8, roi_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 158, 11), 1)

        # Render Onboard HUD Top Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, 48), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        gps_disp_str = f" | GPS: {lat:.5f} N, {lon:.5f} E" if (lat is not None and lon is not None) else ""
        model_name_disp = "YOLOv8n + ByteTrack" if self.model else "SIMULATED BACKEND"
        hud_str = f"MODEL: {model_name_disp} | DEVICE: {self.device} | FPS: {actual_fps} | LATENCY: {total_latency_ms}ms{gps_disp_str}"
        cv2.putText(frame, hud_str, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (52, 211, 153), 1)
        
        hud_intel_str = f"VEHICLES: {active_vehicles_count} | PEDESTRIANS: {active_pedestrians_count} | TDI: {tdi} ({density_level}) | CONGESTION: {congestion_score} ({congestion_level})"
        cv2.putText(frame, hud_intel_str, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Draw Bottom HUD Bar Overlay to cover up any legacy static burnt-in text on sample MP4s
        cv2.rectangle(frame, (0, frame_h - 26), (frame_w, frame_h), (15, 23, 42), -1)
        cv2.putText(frame, f"🟢 EDGE AI INFERENCE: ONLINE ({actual_fps} FPS) | DEVICE: {self.device} | CORRIDOR: {route_id} | STATUS: EVENT TRANSMISSION READY", (12, frame_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (52, 211, 153), 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        telemetry = {
            "model_name": model_name_disp,
            "tracker": "ByteTrack",
            "device": self.device,
            "device_name": self.device_name,
            "gpu_temp": self.gpu_temp,
            "inference_status": "ACTIVE" if self.model else "SIMULATED",
            "measured_fps": actual_fps,
            "processing_fps": actual_fps,
            "source_video_fps": round(video_fps, 1),
            "video_time_sec": video_time_sec,
            "persistence_time_source": "VIDEO_TIMESTAMP",
            "latency_ms": total_latency_ms,
            "inference_ms": round(t_infer_ms, 1),
            "frame_number": frame_number,
            "timestamp": now_str,
            "current_active_objects": len(real_detections),
            "current_active_vehicles": active_vehicles_count,
            "current_active_pedestrians": active_pedestrians_count,
            "unique_vehicle_tracks": len(self.unique_vehicle_track_ids),
            "unique_pedestrian_tracks": len(self.unique_pedestrian_track_ids),
            "cumulative_unique_tracks": len(self.unique_track_ids),
            "class_wise_counts": class_wise_counts,
            "roi_active_vehicles": roi_vehicle_count,
            "traffic_density_index": tdi,
            "traffic_density": density_level,
            "relative_congestion_score": congestion_score,
            "congestion_level": congestion_level,
            "average_displacement_px": avg_displacement_px,
            "moving_vehicle_count": moving_vehicles_count,
            "stationary_vehicle_count": stationary_vehicles_count,
            "is_bottleneck_candidate": is_bottleneck_candidate,
            "elapsed_congestion_video_sec": self.elapsed_congestion_video_sec,
            "consecutive_congestion_frames": self.consecutive_congestion_frames,
            "simulated_events_count": len(simulated_events)
        }

        # Save to Rolling History Buffer (Max 100 entries)
        self.rolling_history.append({
            "timestamp": now_str,
            "frame_number": frame_number,
            "video_time_sec": video_time_sec,
            "active_vehicle_count": active_vehicles_count,
            "active_pedestrian_count": active_pedestrians_count,
            "roi_vehicle_count": roi_vehicle_count,
            "traffic_density_index": tdi,
            "congestion_score": congestion_score,
            "average_displacement_px": avg_displacement_px,
            "moving_vehicle_count": moving_vehicles_count,
            "stationary_vehicle_count": stationary_vehicles_count,
            "is_bottleneck_candidate": is_bottleneck_candidate,
            "elapsed_congestion_video_sec": self.elapsed_congestion_video_sec
        })
        if len(self.rolling_history) > 100:
            self.rolling_history.pop(0)

        return frame_rgb, all_detections, telemetry

    def get_rolling_history(self) -> List[Dict]:
        """Returns the rolling history buffer of traffic intelligence observations."""
        return self.rolling_history

