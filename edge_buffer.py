"""
SIH26124: Real Durable Edge Event SQLite Buffer (Phase 6)
Implements a persistent SQLite database on the onboard bus edge unit to store canonical UrbanEvents,
queue event transmission, manage PENDING / TRANSMITTED / FAILED states, and survive process restarts.
"""

import sqlite3
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from schemas import UrbanEvent, validate_urban_event_schema


class DurableEdgeEventBuffer:
    def __init__(self, db_path: str = "edge_buffer.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edge_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    bus_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    video_time_sec REAL NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    transmission_status TEXT NOT NULL DEFAULT 'PENDING',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edge_status ON edge_events(transmission_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edge_bus ON edge_events(bus_id)")
            conn.commit()
            conn.close()

    def enqueue_event(self, event_input: Any) -> Tuple[bool, str]:
        """
        Validates and enqueues a canonical UrbanEvent into the local SQLite edge buffer.
        State is initialized as PENDING.
        """
        if isinstance(event_input, UrbanEvent):
            evt_dict = event_input.to_dict()
        elif isinstance(event_input, dict):
            evt_dict = event_input
        else:
            return False, "Invalid event input type. Must be dict or UrbanEvent."

        is_valid, msg = validate_urban_event_schema(evt_dict)
        if not is_valid:
            return False, f"Schema validation failed: {msg}"

        evt_id = evt_dict["event_id"]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO edge_events (
                        event_id, event_type, bus_id, route_id, timestamp,
                        video_time_sec, latitude, longitude, payload_json,
                        transmission_status, retry_count, error_message, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 0, '', ?, ?)
                """, (
                    evt_id,
                    evt_dict["event_type"],
                    evt_dict["bus_id"],
                    evt_dict["route_id"],
                    evt_dict.get("timestamp", now_str),
                    float(evt_dict.get("video_time_sec", 0.0)),
                    float(evt_dict["latitude"]),
                    float(evt_dict["longitude"]),
                    json.dumps(evt_dict),
                    now_str,
                    now_str
                ))
                conn.commit()
                return True, f"Enqueued event {evt_id} into edge buffer (PENDING)."
            except Exception as e:
                return False, f"Database insert error: {e}"
            finally:
                conn.close()

    def buffer_event(self, event_input: Any) -> str:
        """Helper alias that enqueues event and returns the event_id string."""
        if isinstance(event_input, UrbanEvent):
            evt_id = event_input.event_id
        elif isinstance(event_input, dict):
            evt_id = event_input.get("event_id", "EVT-UNKNOWN")
        else:
            evt_id = "EVT-UNKNOWN"
        self.enqueue_event(event_input)
        return evt_id


    def get_pending_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetches up to `limit` PENDING events from the local buffer."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT payload_json FROM edge_events 
                WHERE transmission_status = 'PENDING' 
                ORDER BY created_at ASC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [json.loads(row["payload_json"]) for row in rows]

    def get_all_events(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Returns all events stored in local buffer with their transmission_status."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_id, event_type, bus_id, route_id, timestamp, 
                       transmission_status, retry_count, payload_json, updated_at
                FROM edge_events ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            conn.close()
            
            result = []
            for r in rows:
                p = json.loads(r["payload_json"])
                p["transmission_status"] = r["transmission_status"]
                p["retry_count"] = r["retry_count"]
                result.append(p)
            return result

    def mark_transmitted(self, event_id: str) -> bool:
        """Marks an event as TRANSMITTED in the edge buffer."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE edge_events 
                SET transmission_status = 'TRANSMITTED', updated_at = ?
                WHERE event_id = ?
            """, (now_str, event_id))
            conn.commit()
            rows_affected = cursor.rowcount
            conn.close()
            return rows_affected > 0

    def mark_failed(self, event_id: str, error_message: str = "") -> bool:
        """Marks an event as FAILED and increments its retry count."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE edge_events 
                SET transmission_status = 'FAILED', 
                    retry_count = retry_count + 1, 
                    error_message = ?, 
                    updated_at = ?
                WHERE event_id = ?
            """, (error_message, now_str, event_id))
            conn.commit()
            rows_affected = cursor.rowcount
            conn.close()
            return rows_affected > 0

    def retry_pending_events(self, transmit_fn) -> Dict[str, int]:
        """
        Attempts to re-transmit all PENDING and FAILED events via provided `transmit_fn`.
        Returns count of successfully transmitted vs failed events.
        """
        pending = self.get_pending_events()
        # Also grab failed events
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT payload_json FROM edge_events WHERE transmission_status = 'FAILED'")
            rows = cursor.fetchall()
            conn.close()
            failed_list = [json.loads(r["payload_json"]) for r in rows]

        all_to_retry = pending + failed_list
        success_count = 0
        fail_count = 0

        for evt in all_to_retry:
            evt_id = evt["event_id"]
            try:
                ok, err = transmit_fn(evt)
                if ok:
                    self.mark_transmitted(evt_id)
                    success_count += 1
                else:
                    self.mark_failed(evt_id, err)
                    fail_count += 1
            except Exception as ex:
                self.mark_failed(evt_id, str(ex))
                fail_count += 1

        return {"transmitted": success_count, "failed": fail_count}

    def get_status_counts(self) -> Dict[str, int]:
        """Returns exact count of PENDING, TRANSMITTED, and FAILED events in local SQLite buffer."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT transmission_status, COUNT(*) as cnt FROM edge_events GROUP BY transmission_status")
            rows = cursor.fetchall()
            conn.close()

            counts = {"PENDING": 0, "TRANSMITTED": 0, "FAILED": 0}
            for r in rows:
                st_name = r["transmission_status"]
                if st_name in counts:
                    counts[st_name] = r["cnt"]
            return counts

    def clear(self):
        """Clears all records in the local edge buffer."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM edge_events")
            conn.commit()
            conn.close()
