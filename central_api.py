"""
SIH26124: Local Central Ingestion FastAPI Service (Phase 6)
Implements a real local FastAPI REST API service for central urban event ingestion.
Endpoint: POST /api/v1/events
Persists validated canonical UrbanEvents into central SQLite store idempotently.
"""

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any, List, Optional

import uvicorn
import threading
from schemas import validate_urban_event_schema, UrbanEvent
from central_store import CentralEventStore

app = FastAPI(
    title="SIH26124 Local Central Ingestion API",
    description="Local Central Event Ingestion Service for Public Transport Fleet Mobile Sensing Units",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

central_store = CentralEventStore()


@app.get("/api/v1/health")
def health_check():
    """Health check endpoint for Local Central Ingestion API."""
    return {
        "status": "ONLINE",
        "service": "LOCAL CENTRAL INGESTION API",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_events_stored": central_store.count_events()
    }


@app.post("/api/v1/events", status_code=201)
async def ingest_event(payload: Dict[str, Any]):
    """
    Ingests, validates, and persists a canonical UrbanEvent JSON payload into the central SQLite event store.
    Idempotent: Duplicate event_ids are accepted without error and marked duplicate_ignored.
    """
    is_valid, err_msg = validate_urban_event_schema(payload)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UrbanEvent schema: {err_msg}"
        )

    evt_id = payload["event_id"]
    success, msg, is_duplicate = central_store.insert_event(payload)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist event: {msg}"
        )

    server_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "event_id": evt_id,
        "accepted": True,
        "duplicate_ignored": is_duplicate,
        "server_timestamp": server_ts,
        "message": msg
    }


@app.get("/api/v1/events")
def get_central_events(limit: int = 200):
    """Retrieves all persisted central urban events."""
    return {
        "count": central_store.count_events(),
        "events": central_store.get_all_events(limit=limit)
    }


# Background Server Manager for Streamlit Application Integration
_server_thread: Optional[threading.Thread] = None
_server_running: bool = False


def start_local_central_api(host: str = "127.0.0.1", port: int = 8000):
    """Launches the local FastAPI central ingestion server in a background thread if not already running."""
    global _server_thread, _server_running
    if _server_running:
        return

    def run_server():
        global _server_running
        _server_running = True
        try:
            config = uvicorn.Config(app, host=host, port=port, log_level="warning")
            server = uvicorn.Server(config)
            server.run()
        except Exception:
            pass
        finally:
            _server_running = False

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
