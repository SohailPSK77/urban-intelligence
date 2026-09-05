"""
SIH26124: Persistent Central SQLite Event Store (Phase 6)
Implements central event persistence backed by SQLite with indexing and idempotent ingestion.
Used by the local FastAPI ingestion endpoint and MultiBusFusionEngine.
"""

import sqlite3
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from schemas import validate_urban_event_schema


class CentralEventStore:
    def __init__(self, db_path: str = "central_store.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='central_events'")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS central_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    detection_type TEXT NOT NULL,
                    bus_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    video_time_sec REAL NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    confidence REAL NOT NULL,
                    severity TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    ingestion_timestamp TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'NEW'
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_central_ts ON central_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_central_type ON central_events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_central_bus ON central_events(bus_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_central_route ON central_events(route_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_central_latlon ON central_events(latitude, longitude)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_central_status ON central_events(status)")
            conn.commit()
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_connection()
            conn.close()

    def insert_event(self, evt_dict: Dict[str, Any]) -> Tuple[bool, str, bool]:
        """
        Idempotently inserts an incoming canonical UrbanEvent JSON into central SQLite store.
        Returns (success, message, is_duplicate).
        """
        is_valid, msg = validate_urban_event_schema(evt_dict)
        if not is_valid:
            return False, f"Schema validation failed: {msg}", False

        evt_id = evt_dict["event_id"]
        server_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if event_id already exists (idempotency check)
            cursor.execute("SELECT event_id FROM central_events WHERE event_id = ?", (evt_id,))
            if cursor.fetchone() is not None:
                conn.close()
                return True, f"Event {evt_id} already exists in central store. Duplicate ignored idempotently.", True

            try:
                cursor.execute("""
                    INSERT INTO central_events (
                        event_id, event_type, detection_type, bus_id, route_id,
                        timestamp, video_time_sec, latitude, longitude, confidence,
                        severity, priority, payload_json, ingestion_timestamp, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evt_id,
                    evt_dict["event_type"],
                    evt_dict.get("detection_type", "REAL_AI_DETECTION"),
                    evt_dict["bus_id"],
                    evt_dict["route_id"],
                    evt_dict.get("timestamp", server_ts),
                    float(evt_dict.get("video_time_sec", 0.0)),
                    float(evt_dict["latitude"]),
                    float(evt_dict["longitude"]),
                    float(evt_dict.get("confidence", 0.90)),
                    evt_dict.get("severity", "medium"),
                    evt_dict.get("priority", "medium"),
                    json.dumps(evt_dict),
                    server_ts,
                    evt_dict.get("status", "NEW")
                ))
                conn.commit()
                return True, f"Successfully persisted event {evt_id} into central store.", False
            except sqlite3.IntegrityError:
                return True, f"Event {evt_id} already exists in central store (IntegrityError).", True
            except Exception as e:
                return False, f"Central database error: {e}", False
            finally:
                conn.close()

    def get_all_events(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Retrieves central event records sorted by ingestion timestamp."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT payload_json FROM central_events ORDER BY ingestion_timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [json.loads(r["payload_json"]) for r in rows]

    def count_events(self) -> int:
        """Returns total count of events in central store."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM central_events")
            cnt = cursor.fetchone()["cnt"]
            conn.close()
            return cnt

    def clear(self):
        """Clears all records in central store."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM central_events")
            conn.commit()
            conn.close()
