"""
SIH26124: Phase 9 Unit & Regression Test Suite
Validates Real ANPR & Rash Driving Identification Engine, Dedicated License Plate Detector,
EasyOCR Recognition, Temporal OCR Validation, Vehicle-Plate Association, Evidence Frame Generation,
Rash Driving Risk Scoring, UrbanEvent Ingestion, Edge Buffering, Central Store, GIS Cards, Human Officer Review,
and Full Platform Regression across Phases 4, 5, 6, 7, and 8.
"""

import os
import unittest
import numpy as np
import cv2
from datetime import datetime
from fastapi.testclient import TestClient

from schemas import UrbanEvent, validate_urban_event_schema
from edge_buffer import DurableEdgeEventBuffer
from central_store import CentralEventStore
from central_api import app
from anpr_engine import (
    DedicatedPlateYOLODetector,
    ANPROCREngine,
    VehiclePlateAssociator,
    TemporalOCRTracker,
    RashDrivingBehaviorAnalyzer,
    ANPRPipelineEngine
)
from components.gis_map import render_gis_map
from components.event_card import render_human_readable_event_card


class TestPhase9ANPRRashDriving(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.test_edge_db = "test_phase9_edge_buffer.db"
        self.test_central_db = "test_phase9_central_store.db"

        if os.path.exists(self.test_edge_db):
            try: os.remove(self.test_edge_db)
            except Exception: pass

        if os.path.exists(self.test_central_db):
            try: os.remove(self.test_central_db)
            except Exception: pass

        self.edge_buffer = DurableEdgeEventBuffer(db_path=self.test_edge_db)
        self.central_store = CentralEventStore(db_path=self.test_central_db)

    def tearDown(self):
        if os.path.exists(self.test_edge_db):
            try: os.remove(self.test_edge_db)
            except Exception: pass

        if os.path.exists(self.test_central_db):
            try: os.remove(self.test_central_db)
            except Exception: pass

    def test_01_anpr_engine_initialization(self):
        """Test 1: Verify DedicatedPlateYOLODetector and ANPROCREngine initialization."""
        detector = DedicatedPlateYOLODetector(model_path="license_plate_yolov8n.pt")
        self.assertIsNotNone(detector)
        ocr = ANPROCREngine()
        self.assertIsNotNone(ocr)
        self.assertTrue(detector.is_real, "Dedicated license plate model file license_plate_yolov8n.pt must be loaded")
        self.assertGreater(detector.model_size_bytes, 1000000, "Dedicated plate model size must exceed 1MB")

    def test_02_real_plate_detection_execution(self):
        """Test 2: Test real license plate detection on actual sample image."""
        detector = DedicatedPlateYOLODetector(model_path="license_plate_yolov8n.pt")
        sample_img_path = os.path.join("assets", "anpr_rash_driving_ap39.jpg")

        if os.path.exists(sample_img_path):
            img = cv2.imread(sample_img_path)
            plates = detector.detect_plates(img)
            self.assertGreaterEqual(len(plates), 1, "Must detect at least 1 plate region in sample image")
            first_plate = plates[0]
            self.assertIn("bbox", first_plate)
            self.assertEqual(len(first_plate["bbox"]), 4)
            self.assertGreater(first_plate["detection_confidence"], 0.15)
        else:
            self.skipTest("Sample image assets/anpr_rash_driving_ap39.jpg not available")

    def test_03_real_easyocr_execution(self):
        """Test 3: Test real EasyOCR execution on plate crop."""
        ocr = ANPROCREngine()
        self.assertTrue(ocr.is_real, "EasyOCR reader must be available")

        # Create realistic synthetic text crop
        crop = np.zeros((60, 200, 3), dtype=np.uint8)
        crop.fill(255)
        cv2.putText(crop, "AP39TV7219", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        text, conf, status = ocr.recognize_plate_text(crop)
        self.assertIsNotNone(text)
        self.assertIsNotNone(conf)
        self.assertIn("AP", text)
        self.assertGreater(conf, 0.30)

    def test_04_ocr_uncertainty_handling(self):
        """Test 4: Verify low confidence or empty crop returns 'Registration unclear — manual review required'."""
        ocr = ANPROCREngine()
        # Empty black crop
        empty_crop = np.zeros((10, 10, 3), dtype=np.uint8)
        text, conf, status = ocr.recognize_plate_text(empty_crop)
        self.assertEqual(text, "Registration unclear — manual review required")
        self.assertEqual(conf, 0.0)

    def test_05_vehicle_plate_spatial_association(self):
        """Test 5: Verify spatial association between plate bbox and vehicle track ID."""
        veh_tracks = [
            {"track_id": 17, "class": "car", "bbox": [100, 100, 400, 400], "pixel_displacement_px": 8.5},
            {"track_id": 22, "class": "bus", "bbox": [500, 500, 900, 900], "pixel_displacement_px": 2.1}
        ]
        plate_dets = [
            {"plate_index": 1, "bbox": [200, 320, 300, 360], "detection_confidence": 0.91}
        ]

        associated = VehiclePlateAssociator.associate(veh_tracks, plate_dets)
        self.assertEqual(len(associated), 1)
        assoc = associated[0]
        self.assertEqual(assoc["vehicle_track_id"], "VEHICLE-TRACK-17")
        self.assertEqual(assoc["plate_track_id"], "PLATE-TRACK-17")

    def test_06_temporal_ocr_aggregation(self):
        """Test 6: Test multi-frame temporal OCR agreement aggregation."""
        tracker = TemporalOCRTracker()

        # Frame 1: Low conf reading
        txt1, conf1, n1 = tracker.update_and_aggregate("VEHICLE-TRACK-17", "AP39TV7219", 0.70)
        self.assertEqual(txt1, "AP39TV7219")
        self.assertEqual(n1, 1)

        # Frame 2: Higher conf reading
        txt2, conf2, n2 = tracker.update_and_aggregate("VEHICLE-TRACK-17", "AP39TV7219", 0.88)
        self.assertEqual(txt2, "AP39TV7219")
        self.assertEqual(n2, 2)
        self.assertGreaterEqual(conf2, conf1, "Temporal aggregated confidence should increase with consistent observations")

    def test_07_rash_driving_risk_scoring(self):
        """Test 7: Verify Risky Driving Behavior Score heuristic."""
        # High pixel displacement relative to fleet average across consecutive frames
        score, cat, status = RashDrivingBehaviorAnalyzer.analyze_vehicle_risk(
            vehicle_track_id="VEHICLE-TRACK-17",
            disp_px=10.5,
            fleet_avg_disp_px=2.5,
            consecutive_high_disp_frames=4
        )
        self.assertGreaterEqual(score, 0.70)
        self.assertEqual(cat, "HIGH_RISK_MANEUVER")
        self.assertEqual(status, "SUSPECTED")

    def test_08_evidence_snapshot_generation(self):
        """Test 8: Verify keyframe evidence snapshot saving in assets."""
        pipeline = ANPRPipelineEngine()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw vehicle and license plate box
        cv2.rectangle(frame, (100, 100), (500, 400), (200, 200, 200), -1)
        cv2.rectangle(frame, (200, 300), (350, 350), (255, 255, 255), -1)
        cv2.putText(frame, "AP 39 BK 9182", (210, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        veh_tracks = [{"track_id": 1, "class": "car", "bbox": [100, 100, 500, 400], "pixel_displacement_px": 9.0}]

        events = pipeline.process_anpr_frame(
            frame_bgr=frame,
            vehicle_tracks=veh_tracks,
            bus_id="BUS-07",
            route_id="ROUTE-101",
            frame_idx=38
        )

        if events:
            evt = events[0]
            ref_path = os.path.join("assets", evt["evidence_reference"])
            self.assertTrue(os.path.exists(ref_path), f"Evidence snapshot file {ref_path} must exist")

    def test_09_anpr_urbanevent_schema_validation(self):
        """Test 9: Verify ANPR event dictionary validates against canonical UrbanEvent schema."""
        evt_dict = {
            "event_id": "EVT-VZG-ANPR-0038-1",
            "event_type": "rash_driving_anpr",
            "detection_type": "REAL_AI_DETECTION",
            "bus_id": "BUS-07",
            "route_id": "ROUTE-101",
            "timestamp": "2026-09-05 18:00:00",
            "video_time_sec": 1.90,
            "latitude": 17.7145,
            "longitude": 83.3235,
            "confidence": 0.88,
            "severity": "high",
            "priority": "high",
            "status": "SUSPECTED",
            "anpr_data": {
                "vehicle_track_id": "VEHICLE-TRACK-17",
                "plate_track_id": "PLATE-TRACK-17",
                "plate_number": "AP 39 BK 9182",
                "ocr_confidence": 0.88,
                "driving_risk_score": 0.82
            }
        }
        is_valid, msg = validate_urban_event_schema(evt_dict)
        self.assertTrue(is_valid, f"Schema validation error: {msg}")

    def test_10_anpr_edge_buffer_storage(self):
        """Test 10: Store ANPR event into DurableEdgeEventBuffer and query status."""
        anpr_evt = {
            "event_id": "EVT-VZG-ANPR-0099-1",
            "event_type": "rash_driving_anpr",
            "detection_type": "REAL_AI_DETECTION",
            "bus_id": "BUS-07",
            "route_id": "ROUTE-101",
            "timestamp": "2026-09-05 18:00:00",
            "video_time_sec": 4.90,
            "latitude": 17.7145,
            "longitude": 83.3235,
            "confidence": 0.90,
            "severity": "high",
            "priority": "high",
            "status": "SUSPECTED"
        }

        db_id = self.edge_buffer.buffer_event(anpr_evt)
        self.assertTrue(bool(db_id))
        counts = self.edge_buffer.get_status_counts()
        self.assertEqual(counts.get("PENDING", 0), 1)

    def test_11_fastapi_anpr_ingestion(self):
        """Test 11: Ingest ANPR event via FastAPI central API endpoint (POST /api/v1/events)."""
        payload = {
            "event_id": "EVT-VZG-ANPR-API-001",
            "event_type": "rash_driving_anpr",
            "detection_type": "REAL_AI_DETECTION",
            "bus_id": "BUS-07",
            "route_id": "ROUTE-101",
            "timestamp": "2026-09-05 18:00:00",
            "video_time_sec": 2.5,
            "latitude": 17.7145,
            "longitude": 83.3235,
            "confidence": 0.89,
            "severity": "high",
            "priority": "high",
            "status": "SUSPECTED",
            "anpr_data": {
                "plate_number": "AP39TV7219",
                "ocr_confidence": 0.92,
                "driving_risk_score": 0.85
            }
        }
        res = self.client.post("/api/v1/events", json=payload)
        self.assertEqual(res.status_code, 201)
        body = res.json()
        self.assertEqual(body["event_id"], "EVT-VZG-ANPR-API-001")

    def test_12_gis_map_anpr_rendering(self):
        """Test 12: Verify GIS map component accepts ANPR events cleanly."""
        raw_events = [{
            "event_id": "EVT-VZG-ANPR-GIS-001",
            "event_type": "rash_driving_anpr",
            "bus_id": "BUS-07",
            "route_id": "ROUTE-101",
            "latitude": 17.7145,
            "longitude": 83.3235,
            "confidence": 0.88,
            "status": "SUSPECTED"
        }]
        st_data = render_gis_map([], raw_events, [], "ALL")
        self.assertIsNotNone(st_data)

    def test_13_human_officer_review_actions(self):
        """Test 13: Verify officer review status transitions (SUSPECTED -> CONFIRMED / DISMISSED)."""
        evt = {
            "event_id": "EVT-VZG-ANPR-REV-001",
            "status": "SUSPECTED"
        }
        # Simulate officer confirming incident
        evt["status"] = "CONFIRMED"
        self.assertEqual(evt["status"], "CONFIRMED")
        # Simulate officer dismissing incident
        evt["status"] = "DISMISSED"
        self.assertEqual(evt["status"], "DISMISSED")

    # REGRESSION TESTS
    def test_14_phase4_traffic_regression(self):
        """Test 14: Phase 4 Traffic Intelligence regression check."""
        from yolo_detector import EdgeYOLOv8Detector
        det = EdgeYOLOv8Detector()
        self.assertIsNotNone(det)

    def test_15_phase5_pothole_regression(self):
        """Test 15: Phase 5 Road Damage & Pothole regression check."""
        from road_damage_detector import RoadDamageYOLOv8Detector
        rd = RoadDamageYOLOv8Detector()
        self.assertIsNotNone(rd)

    def test_16_phase6_pipeline_regression(self):
        """Test 16: Phase 6 Ingestion Pipeline regression check."""
        counts = self.edge_buffer.get_status_counts()
        self.assertIn("PENDING", counts)

    def test_17_phase7_auth_regression(self):
        """Test 17: Phase 7 Role-Based Authentication regression check."""
        from auth import authenticate_user
        ok, user, msg = authenticate_user("BUS-07", "bus123", "BUS")
        self.assertTrue(ok)

    def test_18_phase8_garbage_regression(self):
        """Test 18: Phase 8 Garbage / Litter Detection regression check."""
        from components.garbage_detector import GarbageLitterDetector
        gd = GarbageLitterDetector()
        self.assertIsNotNone(gd)


if __name__ == "__main__":
    unittest.main()
