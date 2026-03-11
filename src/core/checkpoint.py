"""
DatacortexAI — Checkpoint Serialization Module

Stateless Lifecycle Resumability via Checkpoint Serialization

Serializes the complete state of a data science lifecycle execution at each
phase boundary, enabling the pipeline to resume from any checkpoint after
interruption without re-executing completed phases.
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from pathlib import Path

from src.core.artifact_registry import ArtifactRegistry

logger = logging.getLogger(__name__)


@dataclass
class PhaseCheckpoint:
    """Checkpoint for a single lifecycle phase."""
    phase_name: str
    status: str  # "pending", "running", "completed", "failed", "skipped"
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    output_summary: Optional[dict] = None


@dataclass
class PipelineCheckpoint:
    """
    Complete serializable state of a DatacortexAI pipeline execution.
    """
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "initialized"  # "initialized", "running", "completed", "failed", "paused"

    # Phase states
    phases: dict[str, PhaseCheckpoint] = field(default_factory=dict)

    # Current execution pointer
    current_phase: Optional[str] = None
    execution_plan: list[str] = field(default_factory=list)

    # User configuration
    user_query: str = ""
    target_column: Optional[str] = None
    file_name: Optional[str] = None

    # Metrics
    total_llm_calls: int = 0
    total_sandbox_executions: int = 0
    total_tokens_used: int = 0

    def mark_phase_started(self, phase_name: str):
        self.phases[phase_name] = PhaseCheckpoint(
            phase_name=phase_name,
            status="running",
            started_at=time.time(),
        )
        self.current_phase = phase_name
        self.status = "running"
        self.updated_at = time.time()

    def mark_phase_completed(self, phase_name: str, output_summary: Optional[dict] = None):
        if phase_name in self.phases:
            self.phases[phase_name].status = "completed"
            self.phases[phase_name].completed_at = time.time()
            self.phases[phase_name].output_summary = output_summary
        self.updated_at = time.time()

    def mark_phase_failed(self, phase_name: str, error: str):
        if phase_name in self.phases:
            self.phases[phase_name].status = "failed"
            self.phases[phase_name].completed_at = time.time()
            self.phases[phase_name].error = error
        self.updated_at = time.time()

    def get_completed_phases(self) -> list[str]:
        return [
            name for name, cp in self.phases.items()
            if cp.status == "completed"
        ]

    def get_next_phase(self) -> Optional[str]:
        completed = set(self.get_completed_phases())
        for phase in self.execution_plan:
            if phase not in completed:
                return phase
        return None

    @property
    def progress_pct(self) -> float:
        if not self.execution_plan:
            return 0.0
        completed = len(self.get_completed_phases())
        return completed / len(self.execution_plan) * 100

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "phases": {k: asdict(v) for k, v in self.phases.items()},
            "current_phase": self.current_phase,
            "execution_plan": self.execution_plan,
            "user_query": self.user_query,
            "target_column": self.target_column,
            "file_name": self.file_name,
            "total_llm_calls": self.total_llm_calls,
            "total_sandbox_executions": self.total_sandbox_executions,
            "total_tokens_used": self.total_tokens_used,
            "progress_pct": self.progress_pct,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "PipelineCheckpoint":
        data = json.loads(json_str)
        phases = {}
        for name, phase_data in data.pop("phases", {}).items():
            phases[name] = PhaseCheckpoint(**phase_data)
        data.pop("progress_pct", None)
        checkpoint = cls(**data)
        checkpoint.phases = phases
        return checkpoint


class CheckpointManager:
    """
    Manages checkpoint persistence for pipeline sessions.
    Uses in-memory store + optional filesystem persistence.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._checkpoints: dict[str, PipelineCheckpoint] = {}
        self._registries: dict[str, ArtifactRegistry] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    def create(self, session_id: str, **kwargs) -> PipelineCheckpoint:
        checkpoint = PipelineCheckpoint(session_id=session_id, **kwargs)
        self._checkpoints[session_id] = checkpoint
        self._registries[session_id] = ArtifactRegistry()
        self._registries[session_id].set_session(session_id)
        self._persist(session_id)
        return checkpoint

    def get(self, session_id: str) -> Optional[PipelineCheckpoint]:
        if session_id not in self._checkpoints and self.persist_dir:
            self._load(session_id)
        return self._checkpoints.get(session_id)

    def get_registry(self, session_id: str) -> ArtifactRegistry:
        if session_id not in self._registries:
            self._registries[session_id] = ArtifactRegistry()
            self._registries[session_id].set_session(session_id)
        return self._registries[session_id]

    def save(self, session_id: str):
        """Persist checkpoint + registry to disk."""
        self._persist(session_id)

    def list_sessions(self) -> list[str]:
        return list(self._checkpoints.keys())

    def _persist(self, session_id: str):
        if not self.persist_dir:
            return
        try:
            cp = self._checkpoints.get(session_id)
            reg = self._registries.get(session_id)
            if cp:
                cp_path = self.persist_dir / f"{session_id}_checkpoint.json"
                cp_path.write_text(cp.to_json(), encoding="utf-8")
            if reg:
                reg_path = self.persist_dir / f"{session_id}_artifacts.json"
                reg_path.write_text(reg.to_json(), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist checkpoint {session_id}: {e}")

    def _load(self, session_id: str):
        if not self.persist_dir:
            return
        try:
            cp_path = self.persist_dir / f"{session_id}_checkpoint.json"
            if cp_path.exists():
                self._checkpoints[session_id] = PipelineCheckpoint.from_json(
                    cp_path.read_text(encoding="utf-8")
                )
            reg_path = self.persist_dir / f"{session_id}_artifacts.json"
            if reg_path.exists():
                self._registries[session_id] = ArtifactRegistry.from_json(
                    reg_path.read_text(encoding="utf-8")
                )
        except Exception as e:
            logger.error(f"Failed to load checkpoint {session_id}: {e}")
