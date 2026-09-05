"""
SIH26124: Phase 6 Automated Test Suite
Executes 12 mandatory integration and regression tests for Phase 6:
Canonical UrbanEvent Schema, Edge SQLite Buffer Restart Persistence, FastAPI Local Central Ingestion,
Central SQLite Store, Multi-Bus Spatial-Temporal Fusion, Unique Bus Corroboration, Deduplication,
Idempotent Ingestion, Phase 4 & Phase 5 Regressions, and GIS Map Rendering.
"""

import unittest
import os
import sqlite3
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from schemas import UrbanEvent, validate_urban_event_schema
from gps_provider import SimulatedGPSProvider
from edge_buffer import DurableEdgeEventBuffer
from central_store import CentralEventStore
from central_api import app as fastapi_app
from fusion_engine import MultiBusFusionEngine, generate_simulated_multibus_demonstration
from video_processor import BusCameraVideoProcessor


class TestPhase6Pipeline(unittest.TestCase):

    def setUp(self):
        self.edge_db_path = "test_edge_buffer.db"
        self.central_db_path = "test_central_store.db"
        
        # Clean up any leftover test databases
        for db in [self.edge_db_path, self.central_db_path]:
            if os.path.exists(db):
                try:
                    os.remove(db)
                except Exception:
                    pass

        self.edge_buffer = DurableEdgeEventBuffer(db_path=self.edge_db_path)
        self.central_store = CentralEventStore(db_path=self.central_db_path)
        import central_api
        central_api.central_store = self.central_store
        self.fusion_engine = MultiBusFusionEngine(distance_threshold_m=20.0, time_threshold_sec=300.0)
        self.api_client = TestClient(fastapi_app)


    def tearDown(self):
        for db in [self.edge_db_path, self.central_db_path]:
            if os.path.exists(db):
                try:
                    os.remove(db)
                except Exception:
                    pass

    def test_01_canonical_schema(self):
        """TEST 1: Validates conversion of raw AI detection to canonical UrbanEvent schema."""
        evt = UrbanEvent(
            event_id="EVT-TEST-001",
            event_type="pothole",
            detection_type="REAL_AI_ROAD_DAMAGE",
            bus_id="BUS-07",
            route_id="ROUTE-101",
            timestamp="2026-09-05 10:00:00",
            video_time_sec=12.5,
            latitude=17.7145,
            longitude=83.3235,
            confidence=0.94,
            severity="critical",
            priority="critical",
            model_name="pothole_yolov8s.pt (YOLOv8s)",
            road_damage_track_id="POTHOLE-TRK-01"
        )
        evt_dict = evt.to_dict()
        is_valid, msg = validate_urban_event_schema(evt_dict)
        self.assertTrue(is_valid, f"Schema validation failed: {msg}")
        self.assertEqual(evt_dict["event_type"], "pothole")
        self.assertEqual(evt_dict["model_name"], "pothole_yolov8s.pt (YOLOv8s)")

    def test_02_edge_buffer_persistence(self):
        """TEST 2: Tests SQLite edge buffer enqueuing and persistence across DB close/restart."""
        evt_dict = {
            "event_id": "EVT-BUFFER-001",
            "event_type": "pothole",
            "detection_type": "REAL_AI_ROAD_DAMAGE",
            "bus_id": "BUS-07",
            "route_id": "ROUTE-101",
            "timestamp": "2026-09-05 10:00:00",
            "video_time_sec": 15.0,
            "latitude": 17.7145,
            "longitude": 83.3235,
            "confidence": 0.93,
            "location_source": "SIMULATED_GPS"
        }
        ok, msg = self.edge_buffer.enqueue_event(evt_dict)
        self.assertTrue(ok)

        # Verify state is PENDING
        pending = self.edge_buffer.get_pending_events()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["event_id"], "EVT-BUFFER-001")

        # Simulate process restart by creating a new buffer instance on same DB file
        reopened_buffer = DurableEdgeEventBuffer(db_path=self.edge_db_path)
        pending_after_restart = reopened_buffer.get_pending_events()
        self.assertEqual(len(pending_after_restart), 1, "Event must survive process restart!")

        # Mark transmitted
        reopened_buffer.mark_transmitted("EVT-BUFFER-001")
        counts = reopened_buffer.get_status_counts()
        self.assertEqual(counts["TRANSMITTED"], 1)
        self.assertEqual(counts["PENDING"], 0)

    def test_03_fastapi_ingestion(self):
        """TEST 3: Tests local FastAPI POST /api/v1/events endpoint."""
        evt_dict = {
            "event_id": "EVT-API-001",
            "event_type": "waterlogging",
            "detection_type": "SIMULATED_DEMONSTRATION_EVENT",
            "bus_id": "BUS-11",
            "route_id": "ROUTE-303",
            "timestamp": "2026-09-05 10:05:00",
            "video_time_sec": 20.0,
            "latitude": 17.7820,
            "longitude": 83.3850,
            "confidence": 0.91,
            "location_source": "SIMULATED_GPS"
        }
        response = self.api_client.post("/api/v1/events", json=evt_dict)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertTrue(data["accepted"])
        self.assertEqual(data["event_id"], "EVT-API-001")

    def test_04_central_sqlite_store(self):
        """TEST 4: Verifies central SQLite database insertion, indexing, and count methods."""
        evt_dict = {
            "event_id": "EVT-CENTRAL-001",
            "event_type": "pothole",
            "detection_type": "REAL_AI_ROAD_DAMAGE",
            "bus_id": "BUS-07",
            "route_id": "ROUTE-101",
            "timestamp": "2026-09-05 10:10:00",
            "video_time_sec": 10.0,
            "latitude": 17.7145,
            "longitude": 83.3235,
            "confidence": 0.95
        }
        ok, msg, is_dup = self.central_store.insert_event(evt_dict)
        self.assertTrue(ok)
        self.assertFalse(is_dup)
        self.assertEqual(self.central_store.count_events(), 1)

    def test_05_multibus_positive_fusion(self):
        """TEST 5: Three simulated buses (BUS-07, BUS-12, BUS-18) observing same pothole -> ONE fused issue (unique_bus_count = 3)."""
        demo_events = generate_simulated_multibus_demonstration()
        fused_issues, isolated = self.fusion_engine.fuse_events(demo_events)
        
        self.assertEqual(len(fused_issues), 1, "Should merge 3 close bus observations into exactly ONE persistent issue!")
        issue = fused_issues[0]
        self.assertEqual(issue["unique_bus_count"], 3)
        self.assertIn("BUS-07", issue["observing_buses"])
        self.assertIn("BUS-12", issue["observing_buses"])
        self.assertIn("BUS-18", issue["observing_buses"])
        self.assertEqual(issue["status"], "CONFIRMED")

    def test_06_samebus_deduplication(self):
        """TEST 6: Multiple frame observations from same bus -> unique_bus_count = 1."""
        base_time = "2026-09-05 10:15:00"
        same_bus_events = [
            {
                "event_id": f"EVT-BUS07-{i}",
                "event_type": "pothole",
                "bus_id": "BUS-07",
                "route_id": "ROUTE-101",
                "timestamp": base_time,
                "video_time_sec": float(i),
                "latitude": 17.7145,
                "longitude": 83.3235,
                "confidence": 0.90,
                "road_damage_track_id": "POTHOLE-TRK-01"
            } for i in range(1, 6)
        ]
        
        # Apply deduplication first
        deduped = self.fusion_engine.deduplicate_same_bus_events(same_bus_events)
        self.assertEqual(len(deduped), 1, "Same bus track should be deduplicated to 1 candidate!")
        
        fused_issues, isolated = self.fusion_engine.fuse_events(deduped)
        cluster = (fused_issues + isolated)[0]
        self.assertEqual(cluster["unique_bus_count"], 1)

    def test_07_spatial_separation_negative(self):
        """TEST 7: Events > 20 meters apart -> NOT fused into same issue."""
        ts = "2026-09-05 10:20:00"
        evt1 = {"event_id": "E1", "event_type": "pothole", "bus_id": "BUS-07", "timestamp": ts, "latitude": 17.7145, "longitude": 83.3235, "confidence": 0.90}
        evt2 = {"event_id": "E2", "event_type": "pothole", "bus_id": "BUS-12", "timestamp": ts, "latitude": 17.7500, "longitude": 83.3500, "confidence": 0.90} # > 5 km away
        
        fused, isolated = self.fusion_engine.fuse_events([evt1, evt2])
        self.assertEqual(len(fused), 0, "Events 5km apart MUST NOT be fused!")
        self.assertEqual(len(isolated), 2)

    def test_08_temporal_separation_negative(self):
        """TEST 8: Events > 300 seconds apart -> NOT fused into same issue."""
        lat, lon = 17.7145, 83.3235
        evt1 = {"event_id": "E1", "event_type": "pothole", "bus_id": "BUS-07", "timestamp": "2026-09-05 10:00:00", "latitude": lat, "longitude": lon, "confidence": 0.90}
        evt2 = {"event_id": "E2", "event_type": "pothole", "bus_id": "BUS-12", "timestamp": "2026-09-05 11:00:00", "latitude": lat, "longitude": lon, "confidence": 0.90} # 1 hour difference (3600s > 300s)

        fused, isolated = self.fusion_engine.fuse_events([evt1, evt2])
        self.assertEqual(len(fused), 0, "Events 1 hour apart MUST NOT be fused!")

    def test_09_idempotent_ingestion(self):
        """TEST 9: Re-sending duplicate event_id -> idempotent rejection without duplicate storage."""
        evt_dict = {
            "event_id": "EVT-DUP-001",
            "event_type": "pothole",
            "bus_id": "BUS-07",
            "route_id": "ROUTE-101",
            "latitude": 17.7145,
            "longitude": 83.3235,
            "confidence": 0.90
        }
        res1 = self.api_client.post("/api/v1/events", json=evt_dict)
        self.assertEqual(res1.status_code, 201)
        self.assertFalse(res1.json()["duplicate_ignored"])

        res2 = self.api_client.post("/api/v1/events", json=evt_dict)
        self.assertEqual(res2.status_code, 201)
        self.assertTrue(res2.json()["duplicate_ignored"], "Duplicate event MUST be marked duplicate_ignored!")
        self.assertEqual(self.central_store.count_events(), 1, "Store count MUST remain 1!")

    def test_10_phase4_yolo_bytetrack_regression(self):
        """TEST 10: Phase 4 YOLOv8n + ByteTrack traffic intelligence regression test."""
        video_path = "assets/sample_vizag_route_101.mp4"
        if os.path.exists(video_path):
            processor = BusCameraVideoProcessor(video_path)
            frame_rgb, events, telemetry = processor.process_frame_at(1, bus_id="BUS-07", route_id="ROUTE-101")
            processor.close()
            self.assertIn("measured_fps", telemetry)
            self.assertIn("traffic_density_index", telemetry)
            self.assertIn("relative_congestion_score", telemetry)

    def test_11_phase5_pothole_yolov8s_regression(self):
        """TEST 11: Phase 5 YOLOv8s pothole detection regression test with pothole_yolov8s.pt."""
        weights_path = "pothole_yolov8s.pt"
        self.assertTrue(os.path.exists(weights_path), "Weight file pothole_yolov8s.pt must exist!")
        
        video_path = "assets/sample_vizag_route_101.mp4"
        if os.path.exists(video_path):
            processor = BusCameraVideoProcessor(video_path)
            frame_rgb, events, telemetry = processor.process_frame_at(1, bus_id="BUS-07", route_id="ROUTE-101")
            processor.close()
            # Verify events payload contains model name
            for e in events:
                if e.get("detection_type") == "REAL_AI_ROAD_DAMAGE":
                    self.assertIn("YOLOv8s", e.get("model_name", ""))

    def test_12_gis_fusion_rendering(self):
        """TEST 12: GIS map cluster formatting test."""
        demo_events = generate_simulated_multibus_demonstration()
        fused_issues, isolated = self.fusion_engine.fuse_events(demo_events)
        self.assertTrue(len(fused_issues) >= 1)
        issue = fused_issues[0]
        self.assertIn("centroid_latitude", issue)
        self.assertIn("centroid_longitude", issue)
        self.assertIn("unique_bus_count", issue)


if __name__ == "__main__":
    unittest.main()
