"""
SIH26124: Phase 5 - Real AI Road-Damage / Pothole Detection Engine
Integrates a dedicated lightweight PyTorch/Ultralytics YOLO model (pothole_yolov8n.pt)
to perform genuine forward-pass inference on video frames, bounding box extraction,
detection confidence reporting, AI-assisted heuristic severity estimation,
and temporal IoU object tracking across consecutive frames.
"""

import os
import cv2
import numpy as np
import time
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

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


class PotholeTrack:
    """Represents a persistent road-damage / pothole track across consecutive video frames."""
    def __init__(self, track_id: int, bbox: List[int], video_time_sec: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.first_seen_video_time = video_time_sec
        self.last_seen_video_time = video_time_sec
        self.observation_count = 1
        self.last_updated_frame = 0

    def update(self, bbox: List[int], video_time_sec: float, frame_number: int):
        self.bbox = bbox
        self.last_seen_video_time = video_time_sec
        self.observation_count += 1
        self.last_updated_frame = frame_number

    @property
    def centroid(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return max(0.0, min(1.0, iou))


class RoadDamageYOLOv8Detector:
    """
    Dedicated Real AI Road-Damage & Pothole Detection Model Engine.
    Executes actual neural network forward-pass on video frames using 'pothole_yolov8n.pt' weights,
    derived from HuggingFace dataset 'peterhdd/pothole-detection-yolov8'.
    """

    def __init__(self, model_path: str = "pothole_yolov8s.pt", conf_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_name = "pothole_yolov8s.pt (YOLOv8s)"
        self.model_version = "YOLOv8s-Pothole-v1"

        self.dataset_source = "HuggingFace peterhdd/pothole-detection-yolov8"

        # Active Pothole Tracks for Temporal Persistence (track_id -> PotholeTrack)
        self.active_tracks: Dict[int, PotholeTrack] = {}
        self.next_track_id = 1

        # Hardware Device Detection
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "CUDA"
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.device = "CPU"
            self.device_name = "System CPU (Intel/AMD)"

        # Search candidate paths for model weights
        resolved_weights = self._resolve_model_path(model_path)

        if ULTRALYTICS_AVAILABLE and resolved_weights:
            try:
                self.model = YOLO(resolved_weights)
                print(f"[ROAD DAMAGE ENGINE] Successfully loaded dedicated pothole weights '{resolved_weights}' on device '{self.device}'")
            except Exception as e:
                print(f"[ROAD DAMAGE ENGINE] Error loading weights '{resolved_weights}': {e}")
                self.model = None
        else:
            print(f"[ROAD DAMAGE ENGINE] Model weights '{model_path}' not found or Ultralytics unavailable.")

    def _resolve_model_path(self, path: str) -> Optional[str]:
        """Resolves absolute path to pothole model weights file."""
        if os.path.exists(path):
            return path
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cand1 = os.path.join(base_dir, path)
        if os.path.exists(cand1):
            return cand1
        
        cand2 = os.path.join(os.getcwd(), path)
        if os.path.exists(cand2):
            return cand2

        return None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 1,
        video_time_sec: float = 0.0,
        route_id: str = "ROUTE-101",
        bus_id: str = "BUS-07",
        save_evidence: bool = False
    ) -> Tuple[np.ndarray, List[Dict], Dict]:
        """
        Executes REAL forward-pass inference using the dedicated pothole detection model.
        Performs bounding-box extraction, YOLO detection confidence calculation,
        temporal IoU tracking across consecutive frames, and AI-assisted heuristic severity estimation.
        """
        t_start = time.time()
        frame_h, frame_w = frame.shape[:2]
        total_frame_area = max(1, frame_h * frame_w)

        raw_detections = []
        events_payload = []

        if self.model is not None:
            t_infer_start = time.time()
            # Perform REAL forward-pass inference on BGR video frame
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
            t_infer_ms = (time.time() - t_infer_start) * 1000.0

            current_frame_boxes = []

            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())  # YOLO detection confidence
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.model.names.get(cls_id, "pothole")
                    
                    if cls_name in ["0", "1"]:
                        cls_name = "pothole"

                    current_frame_boxes.append({
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "conf": conf,
                        "class": cls_name
                    })

            # -------------------------------------------------------------
            # TEMPORAL IOU OBJECT TRACKING FOR POTHOLES
            # -------------------------------------------------------------
            assigned_track_ids = set()
            frame_detections_with_track = []

            for det in current_frame_boxes:
                bbox = det["bbox"]
                best_iou = 0.0
                best_track_id = None

                # Find best matching existing active track
                for tid, track in self.active_tracks.items():
                    if tid in assigned_track_ids:
                        continue
                    iou = compute_iou(bbox, track.bbox)
                    if iou > 0.25 and iou > best_iou:
                        best_iou = iou
                        best_track_id = tid

                if best_track_id is not None:
                    # Update existing track
                    track = self.active_tracks[best_track_id]
                    track.update(bbox, video_time_sec, frame_number)
                    assigned_track_ids.add(best_track_id)
                    assigned_tid = best_track_id
                else:
                    # Create new track
                    new_tid = self.next_track_id
                    self.next_track_id += 1
                    track = PotholeTrack(new_tid, bbox, video_time_sec)
                    track.last_updated_frame = frame_number
                    self.active_tracks[new_tid] = track
                    assigned_track_ids.add(new_tid)
                    assigned_tid = new_tid

                frame_detections_with_track.append((det, assigned_tid, self.active_tracks[assigned_tid]))

            # Prune stale tracks older than 30 frames
            stale_tids = [tid for tid, trk in self.active_tracks.items() if (frame_number - trk.last_updated_frame) > 30]
            for tid in stale_tids:
                del self.active_tracks[tid]

            # -------------------------------------------------------------
            # BUILD REAL ROAD-DAMAGE DETECTION PAYLOADS & RENDER OVERLAYS
            # -------------------------------------------------------------
            for det, track_id, track in frame_detections_with_track:
                xyxy = det["bbox"]
                conf = det["conf"]
                cls_name = det["class"]

                bw = max(1, xyxy[2] - xyxy[0])
                bh = max(1, xyxy[3] - xyxy[1])
                det_area_px = bw * bh
                area_ratio = det_area_px / float(total_frame_area)

                # Explicitly Documented Heuristic Severity (Bounding-Box Area Ratio)
                if area_ratio >= 0.08:
                    severity = "critical"
                    priority = "critical"
                elif area_ratio >= 0.03:
                    severity = "high"
                    priority = "high"
                else:
                    severity = "medium"
                    priority = "medium"

                track_label_id = f"POTHOLE-TRK-{track_id:02d}"

                evidence_ref = f"REAL_ROAD_DAMAGE_KEYFRAME_{frame_number}"

                evt_item = {
                    "event_id": f"EVT-VZG-REAL-POT-{frame_number:04d}-{track_id}",
                    "event_type": "pothole",
                    "detection_type": "REAL_AI_ROAD_DAMAGE",
                    "road_damage_class": cls_name,
                    "road_damage_track_id": track_label_id,
                    "confidence": round(conf, 2),  # YOLO detection confidence
                    "bbox": xyxy,
                    "detection_area_px": det_area_px,
                    "bbox_width_px": bw,
                    "bbox_height_px": bh,
                    "severity_method": "AI-assisted heuristic severity (bounding-box area ratio)",
                    "severity": severity,
                    "priority": priority,
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "source_frame": frame_number,
                    "video_time_sec": video_time_sec,
                    "first_seen_video_time": round(track.first_seen_video_time, 2),
                    "last_seen_video_time": round(track.last_seen_video_time, 2),
                    "observation_count": track.observation_count,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "bus_id": bus_id,
                    "route_id": route_id,
                    "location_source": "SIMULATED_GPS",
                    "details": f"[REAL AI ROAD DAMAGE] Real model detection '{cls_name}' ({track_label_id}) with YOLO confidence {int(conf*100)}% (Box Area: {det_area_px} px², Heuristic Severity: {severity.upper()}, Observed {track.observation_count}x) in Frame #{frame_number}",
                    "evidence_reference": evidence_ref,
                    "status": "needs_maintenance"
                }

                events_payload.append(evt_item)

                # Draw Distinct Crimson Bounding Box & Yellow Text Header on Video Frame
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 239), 3)

                lbl_str = f"🟢 REAL AI POTHOLE #{track_id} ({int(conf*100)}%)"
                (lbl_w, lbl_h), _ = cv2.getTextSize(lbl_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (xyxy[0], xyxy[1] - lbl_h - 8), (xyxy[0] + lbl_w + 8, xyxy[1]), (0, 0, 239), -1)
                cv2.putText(frame, lbl_str, (xyxy[0] + 4, xyxy[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        else:
            t_infer_ms = 0.0

        # Save keyframe evidence if requested and detections exist
        if save_evidence and events_payload:
            assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
            os.makedirs(assets_dir, exist_ok=True)
            evidence_file = os.path.join(assets_dir, f"REAL_ROAD_DAMAGE_KEYFRAME_{frame_number}.jpg")
            cv2.imwrite(evidence_file, frame)

        t_total_sec = max(time.time() - t_start, 0.001)
        actual_fps = round(1.0 / t_total_sec, 1)

        telemetry = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_source": self.dataset_source,
            "device": self.device,
            "device_name": self.device_name,
            "inference_status": "ACTIVE" if self.model is not None else "UNAVAILABLE",
            "potholes_detected": len(events_payload),
            "highest_confidence": max([e["confidence"] for e in events_payload], default=0.0),
            "measured_fps": actual_fps,
            "latency_ms": round(t_total_sec * 1000.0, 1),
            "inference_ms": round(t_infer_ms, 1),
            "active_pothole_tracks_count": len(self.active_tracks)
        }

        return frame, events_payload, telemetry
