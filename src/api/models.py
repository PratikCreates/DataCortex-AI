"""
DatacortexAI — API Data Models

Pydantic models for all API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum


class PipelineStatus(str, Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class AnalyzeRequest(BaseModel):
    """Request to start a new analysis pipeline."""
    user_query: str = Field(default="", description="Natural language analysis question")
    target_column: Optional[str] = Field(default=None, description="Target column for supervised learning")


class AnalyzeResponse(BaseModel):
    """Response after starting analysis."""
    session_id: str
    status: str
    message: str


class SessionStatus(BaseModel):
    """Current status of a pipeline session."""
    session_id: str
    status: str
    progress_pct: float
    current_phase: Optional[str]
    phases_completed: list[str]
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


class ReportResponse(BaseModel):
    """Final report response."""
    session_id: str
    report_markdown: str
    sections: int
    artifacts_count: int


class PipelineEvent(BaseModel):
    """WebSocket event for real-time streaming."""
    type: str
    phase: Optional[str] = None
    message: Optional[str] = None
    data: Optional[Any] = None
    timestamp: float = 0.0
    progress_pct: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    credential_pool: Optional[dict] = None
