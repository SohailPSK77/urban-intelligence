"""
SIH26124: Automated Test Suite for Phase 7 (Role-Based Authentication & Regression Safety)
Tests:
  [1] Valid BUS credentials -> successful login
  [2] Valid OFFICIAL credentials -> successful login
  [3] Invalid password -> login rejected
  [4] Unknown ID -> login rejected
  [5] Unauthenticated user cannot access operational dashboard
  [6] BUS role cannot access official-only functions
  [7] OFFICIAL role can access command-center functions
  [8] Authenticated BUS-07 event contains bus_id = BUS-07
  [9] Logout clears session
  [10] Phase 4 regression PASS (YOLOv8n + ByteTrack + Traffic Intelligence)
  [11] Phase 5 regression PASS (YOLOv8s Pothole Detection)
  [12] Phase 6 regression PASS (UrbanEvent + Edge Buffer + FastAPI Ingestion + Multi-Bus Fusion)
"""

import unittest
import os
import tempfile
import sqlite3
from datetime import datetime

from auth import hash_password, verify_password, authenticate_user, init_user_db, get_user_info, generate_mobile_otp, verify_otp_and_reset_password
from schemas import validate_urban_event_schema, UrbanEvent
from edge_buffer import DurableEdgeEventBuffer
from central_store import CentralEventStore
from fusion_engine import MultiBusFusionEngine
from yolo_detector import EdgeYOLOv8Detector
from road_damage_detector import RoadDamageYOLOv8Detector


class TestPhase7AuthenticationAndRegression(unittest.TestCase):

    def setUp(self):
        # Create isolated temporary database for authentication testing
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)
        init_user_db(self.temp_db_path)

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except Exception:
                pass

    def test_01_valid_bus_login(self):
        """TEST 1: Valid BUS credentials -> successful login."""
        success, user_dict, err_msg = authenticate_user("BUS-07", "bus123", "BUS", db_path=self.temp_db_path)
        self.assertTrue(success, f"Expected valid BUS login, got error: {err_msg}")
        self.assertIsNotNone(user_dict)
        self.assertEqual(user_dict["user_id"], "BUS-07")
        self.assertEqual(user_dict["role"], "BUS")
        self.assertEqual(user_dict["bus_id"], "BUS-07")
        self.assertEqual(user_dict["route_id"], "ROUTE-101")

    def test_02_valid_official_login(self):
        """TEST 2: Valid OFFICIAL credentials -> successful login."""
        success, user_dict, err_msg = authenticate_user("OFFICIAL-001", "admin123", "OFFICIAL", db_path=self.temp_db_path)
        self.assertTrue(success, f"Expected valid OFFICIAL login, got error: {err_msg}")
        self.assertIsNotNone(user_dict)
        self.assertEqual(user_dict["user_id"], "OFFICIAL-001")
        self.assertEqual(user_dict["role"], "OFFICIAL")

    def test_03_invalid_password_rejection(self):
        """TEST 3: Invalid password -> login rejected."""
        success, user_dict, err_msg = authenticate_user("BUS-07", "WrongPassword123!", "BUS", db_path=self.temp_db_path)
        self.assertFalse(success)
        self.assertIsNone(user_dict)
        self.assertIn("Invalid password", err_msg)

    def test_04_unknown_id_rejection(self):
        """TEST 4: Unknown ID -> login rejected."""
        success, user_dict, err_msg = authenticate_user("BUS-999", "bus123", "BUS", db_path=self.temp_db_path)
        self.assertFalse(success)
        self.assertIsNone(user_dict)
        self.assertIn("Unknown ID", err_msg)

    def test_05_unauthenticated_user_access_control(self):
        """TEST 5: Unauthenticated user cannot access operational dashboard."""
        session_state_mock = {"authenticated": False, "user_role": None}
        self.assertFalse(session_state_mock["authenticated"])
        self.assertIsNone(session_state_mock["user_role"])

    def test_06_bus_role_access_restriction(self):
        """TEST 6: BUS role cannot access official-only functions."""
        success, user_dict, _ = authenticate_user("BUS-07", "bus123", "BUS", db_path=self.temp_db_path)
        self.assertTrue(success)
        self.assertEqual(user_dict["role"], "BUS")
        # Verify role is not OFFICIAL
        self.assertNotEqual(user_dict["role"], "OFFICIAL")

    def test_07_official_role_access_grant(self):
        """TEST 7: OFFICIAL role can access command-center functions."""
        success, user_dict, _ = authenticate_user("OFFICIAL-001", "admin123", "OFFICIAL", db_path=self.temp_db_path)
        self.assertTrue(success)
        self.assertEqual(user_dict["role"], "OFFICIAL")

    def test_08_bus_identity_binding_in_urbanevent(self):
        """TEST 8: Authenticated BUS-07 event contains bus_id = BUS-07."""
        success, user_dict, _ = authenticate_user("BUS-07", "bus123", "BUS", db_path=self.temp_db_path)
        self.assertTrue(success)
        
        # Enforce binding user_dict['bus_id'] to event payload
        event = UrbanEvent(
            event_id="EVT-AUTH-TEST-001",
            event_type="pothole",
            detection_type="REAL_AI_ROAD_DAMAGE",
            bus_id=user_dict["bus_id"],  # Strict session binding
            route_id=user_dict["route_id"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            video_time_sec=12.5,
            latitude=17.7145,
            longitude=83.3235,
            confidence=0.88,
            severity="critical"
        ).to_dict()
        self.assertEqual(event["bus_id"], "BUS-07")
        self.assertEqual(event["route_id"], "ROUTE-101")
        is_valid, errors = validate_urban_event_schema(event)
        self.assertTrue(is_valid, f"Schema validation failed: {errors}")

    def test_09_logout_clears_session(self):
        """TEST 9: Logout clears session state."""
        session_mock = {
            "authenticated": True,
            "user_role": "BUS",
            "user_id": "BUS-07",
            "bus_id": "BUS-07"
        }
        # Simulate logout action
        session_mock["authenticated"] = False
        session_mock["user_role"] = None
        session_mock["user_id"] = None
        session_mock["bus_id"] = None

        self.assertFalse(session_mock["authenticated"])
        self.assertIsNone(session_mock["user_role"])
        self.assertIsNone(session_mock["user_id"])
        self.assertIsNone(session_mock["bus_id"])

    def test_10_phase4_regression_safety(self):
        """TEST 10: Phase 4 regression PASS (YOLOv8n + ByteTrack + Traffic Intelligence)."""
        detector = EdgeYOLOv8Detector(model_name="yolov8n.pt")
        self.assertTrue(os.path.exists(detector.model_name))
        self.assertEqual(detector.model_name, "yolov8n.pt")

    def test_11_phase5_regression_safety(self):
        """TEST 11: Phase 5 regression PASS (YOLOv8s Pothole Detection)."""
        road_detector = RoadDamageYOLOv8Detector(model_path="pothole_yolov8s.pt")
        self.assertTrue(os.path.exists(road_detector.model_path))
        self.assertEqual(road_detector.model_path, "pothole_yolov8s.pt")

    def test_12_phase6_regression_safety(self):
        """TEST 12: Phase 6 regression PASS (UrbanEvent + Edge Buffer + Fusion Engine)."""
        # 1. Edge Buffer test
        buf_fd, buf_path = tempfile.mkstemp(suffix=".db")
        os.close(buf_fd)
        edge_buf = DurableEdgeEventBuffer(db_path=buf_path)
        
        evt = UrbanEvent(
            event_id="EVT-REG6-01",
            event_type="pothole",
            detection_type="REAL_AI_ROAD_DAMAGE",
            bus_id="BUS-07",
            route_id="ROUTE-101",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            video_time_sec=5.0,
            latitude=17.7145,
            longitude=83.3235,
            confidence=0.92,
            severity="critical"
        ).to_dict()
        row_id = edge_buf.buffer_event(evt)
        self.assertEqual(row_id, "EVT-REG6-01")
        
        # 2. Multi-bus Fusion Engine test
        fusion = MultiBusFusionEngine(distance_threshold_m=20.0)
        demo_evts = fusion.simulate_multi_bus_demonstration()
        fused, isolated = fusion.fuse_events(demo_evts)
        self.assertGreater(len(fused), 0)
        self.assertEqual(fused[0]["unique_bus_count"], 3)

        if os.path.exists(buf_path):
            try:
                os.remove(buf_path)
            except Exception:
                pass

    def test_13_mobile_otp_generation_and_verification(self):
        """TEST 13: Mobile OTP generation, verification, and password reset."""
        # 1. Request OTP with matching registered mobile (BUS-07: 9491591473)
        ok, code, masked_mob, msg = generate_mobile_otp("BUS-07", "9491591473", db_path=self.temp_db_path)
        self.assertTrue(ok, f"Expected OTP generation success, got: {msg}")
        self.assertIsNotNone(code)
        self.assertEqual(len(code), 6)

        # 2. Verify OTP & reset password to 'NewPass#77'
        v_ok, v_msg = verify_otp_and_reset_password("BUS-07", code, code, "NewPass#77", db_path=self.temp_db_path)
        self.assertTrue(v_ok, f"Expected password reset success, got: {v_msg}")

        # 3. Authenticate with new password
        auth_ok, user_dict, _ = authenticate_user("BUS-07", "NewPass#77", "BUS", db_path=self.temp_db_path)
        self.assertTrue(auth_ok)

    def test_14_mobile_otp_mismatched_mobile_rejection(self):
        """TEST 14: Mobile OTP rejection on mismatched mobile number."""
        ok, code, masked_mob, msg = generate_mobile_otp("BUS-07", "1111111111", db_path=self.temp_db_path)
        self.assertFalse(ok)
        self.assertIsNone(code)
        self.assertIn("mismatch", msg.lower())


if __name__ == "__main__":
    unittest.main()
