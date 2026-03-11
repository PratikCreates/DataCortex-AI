"""
DatacortexAI — API Gateway

FastAPI server with WebSocket support for real-time pipeline streaming.
Central entry point for the DatacortexAI platform.

FIX: Added event buffering + replay to handle the race condition where
the pipeline emits events before the WebSocket client connects.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.core.config import Config
from src.core.checkpoint import CheckpointManager
from src.agents.planner import PlannerAgent
from src.api.models import (
    AnalyzeResponse,
    HealthResponse,
    SessionStatus,
    ReportResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────

app = FastAPI(
    title="DatacortexAI",
    description="AI-Orchestrated Unified Data Science Lifecycle Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Global state
config = Config.get()
checkpoint_manager = CheckpointManager(persist_dir="./checkpoints")
active_websockets: dict[str, list[WebSocket]] = {}
pipeline_results: dict[str, dict] = {}
pending_approvals: dict[str, asyncio.Event] = {}
approval_decisions: dict[str, bool] = {}

# ─── Event Buffer for replay ─────────────────────────
# Buffers events per session so a late-connecting WebSocket can replay them.
event_buffers: dict[str, list[dict]] = {}


# ─────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend dashboard."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>DatacortexAI API</h1><p>Frontend not found. API is running.</p>")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with credential pool status."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        credential_pool=config.credential_pool.status_summary,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    user_query: str = Form(default=""),
    target_column: Optional[str] = Form(default=None),
):
    """
    Start a new data analysis pipeline.

    Upload a data file (CSV, JSON, Excel, Parquet) and optionally
    specify a natural language question and target column.
    """
    session_id = str(uuid.uuid4())

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    max_size = config.settings.max_file_size_mb * 1024 * 1024
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {config.settings.max_file_size_mb}MB",
        )

    logger.info(
        f"Analysis requested: session={session_id}, file={file.filename}, "
        f"size={len(file_content)} bytes, query='{user_query}'"
    )

    # Initialize event buffer for this session
    event_buffers[session_id] = []

    # Create event emitter that both buffers AND broadcasts
    async def on_event(event: dict):
        event["session_id"] = session_id
        event.setdefault("timestamp", time.time())
        # Always buffer
        event_buffers.setdefault(session_id, []).append(event)
        # Also broadcast if any WebSocket is connected
        await broadcast_to_session(session_id, event)

    # Launch pipeline in background with a small delay
    # to give the frontend time to establish the WebSocket
    asyncio.create_task(
        _run_pipeline_delayed(
            session_id=session_id,
            file_content=file_content,
            file_name=file.filename,
            user_query=user_query,
            target_column=target_column,
            on_event=on_event,
        )
    )

    return AnalyzeResponse(
        session_id=session_id,
        status="started",
        message=f"Analysis pipeline started for {file.filename}",
    )


@app.get("/status/{session_id}", response_model=SessionStatus)
async def get_status(session_id: str):
    """Get the current status of a pipeline session."""
    checkpoint = checkpoint_manager.get(session_id)
    if not checkpoint:
        # Return initializing state if polled before first checkpoint completes
        return SessionStatus(
            session_id=session_id,
            status="starting",
            progress_pct=0,
            current_phase="initializing",
            phases_completed=[],
            error=None,
        )

    return SessionStatus(
        session_id=session_id,
        status=checkpoint.status,
        progress_pct=checkpoint.progress_pct,
        current_phase=checkpoint.current_phase,
        phases_completed=checkpoint.get_completed_phases(),
        error=None,
    )


@app.get("/report/{session_id}", response_model=ReportResponse)
async def get_report(session_id: str):
    """Get the final report for a completed session."""
    result = pipeline_results.get(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Report not found")

    registry = checkpoint_manager.get_registry(session_id)

    return ReportResponse(
        session_id=session_id,
        report_markdown=result.get("report", ""),
        sections=len(result.get("phases_completed", [])),
        artifacts_count=registry.summary.get("total_artifacts", 0),
    )


@app.get("/artifacts/{session_id}")
async def get_artifacts(session_id: str):
    """Get all artifacts for a session."""
    registry = checkpoint_manager.get_registry(session_id)
    return registry.to_dict()


@app.get("/events/{session_id}")
async def get_events(session_id: str):
    """Get all buffered events for a session (for late joiners / polling fallback)."""
    return event_buffers.get(session_id, [])


@app.post("/approve/{session_id}")
async def approve_action(session_id: str, approved: bool = True):
    """Approve or reject a pending pipeline action (HITL Gateway)."""
    if session_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="No pending approval for this session")
    
    approval_decisions[session_id] = approved
    pending_approvals[session_id].set()
    
    return {"session_id": session_id, "approved": approved}


# ─────────────────────────────────────────────────────
# WebSocket for Real-Time Streaming
# ─────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time pipeline event streaming."""
    await websocket.accept()

    if session_id not in active_websockets:
        active_websockets[session_id] = []
    active_websockets[session_id].append(websocket)

    logger.info(f"WebSocket connected: session={session_id}")

    # REPLAY buffered events the client missed
    buffered = event_buffers.get(session_id, [])
    for event in buffered:
        try:
            await websocket.send_text(json.dumps(event, default=str))
        except Exception:
            break

    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        if session_id in active_websockets and websocket in active_websockets[session_id]:
            active_websockets[session_id].remove(websocket)
        logger.info(f"WebSocket disconnected: session={session_id}")


async def broadcast_to_session(session_id: str, event: dict):
    """Broadcast an event to all WebSocket clients for a session."""
    if session_id not in active_websockets:
        return

    message = json.dumps(event, default=str)
    disconnected = []

    for ws in active_websockets[session_id]:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        active_websockets[session_id].remove(ws)


# ─────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────

async def _run_pipeline_delayed(
    session_id: str,
    file_content: bytes,
    file_name: str,
    user_query: str,
    target_column: Optional[str],
    on_event,
):
    """Run the pipeline after a short delay to let WebSocket connect."""
    # Give the frontend ~1.5 seconds to establish its WebSocket connection
    await asyncio.sleep(1.5)
    await _run_pipeline(session_id, file_content, file_name, user_query, target_column, on_event)


async def _run_pipeline(
    session_id: str,
    file_content: bytes,
    file_name: str,
    user_query: str,
    target_column: Optional[str],
    on_event,
):
    """Run the full pipeline in the background."""
    async def approval_callback(action: str, reasoning: str) -> bool:
        """Wait for human approval via the /approve endpoint."""
        event = asyncio.Event()
        pending_approvals[session_id] = event
        try:
            # Wait for user input or timeout after 10 minutes
            await asyncio.wait_for(event.wait(), timeout=600)
            return approval_decisions.get(session_id, False)
        except asyncio.TimeoutError:
            logger.warning(f"Approval timeout for session {session_id}")
            return False
        finally:
            pending_approvals.pop(session_id, None)
            approval_decisions.pop(session_id, None)

    try:
        planner = PlannerAgent(
            config=config,
            checkpoint_manager=checkpoint_manager,
            on_event=on_event,
            approval_callback=approval_callback,
        )

        result = await planner.run_pipeline(
            file_content=file_content,
            file_name=file_name,
            user_query=user_query,
            target_column=target_column,
            session_id=session_id,
        )

        pipeline_results[session_id] = result

    except Exception as e:
        logger.error(f"Pipeline error for session {session_id}: {e}", exc_info=True)
        await on_event({
            "type": "pipeline_failed",
            "error": str(e),
            "session_id": session_id,
        })


# ─────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
