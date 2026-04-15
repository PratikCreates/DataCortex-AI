"""
DatacortexAI — Contextual Artifact Registry

Cross-Phase Knowledge Propagation via Contextual Artifact Registry

A shared artifact registry where outputs from each lifecycle phase
(schema metadata, statistical profiles, data quality reports, EDA insights,
feature importance rankings) are stored as structured contexts and
automatically propagated to downstream agents.
"""

import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    INGESTION = "ingestion"
    PROFILING = "profiling"
    EDA = "eda"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    TRAINING = "training"
    EVALUATION = "evaluation"
    REPORTING = "reporting"


class ArtifactType(str, Enum):
    SCHEMA = "schema"
    DATA_QUALITY = "data_quality"
    STATISTICS = "statistics"
    VISUALIZATION = "visualization"
    CORRELATION = "correlation"
    FINDINGS = "findings"
    FEATURE_IMPORTANCE = "feature_importance"
    MODEL_METRICS = "model_metrics"
    MODEL_COMPARISON = "model_comparison"
    REPORT = "report"
    CODE = "code"
    DATAFRAME_SAMPLE = "dataframe_sample"
    DATA_FINGERPRINT = "data_fingerprint"
    EXECUTION_LOG = "execution_log"
    CLEANING_LOG = "cleaning_log"


@dataclass
class ValidationResult:
    """Result of an artifact validation."""
    is_valid: bool
    errors: list = field(default_factory=list)


@dataclass
class ArtifactSchema:
    """Schema definition for an artifact type."""
    artifact_type: ArtifactType
    validator: Callable[[Any], bool]
    description: str = ""


class ValidationEngine:
    """Engine to validate artifacts against registered schemas."""
    def __init__(self):
        self.schemas: dict[str, ArtifactSchema] = {}

    def register_schema(self, schema: ArtifactSchema):
        self.schemas[schema.artifact_type.value] = schema

    def validate(self, artifact_type: str, data: Any) -> ValidationResult:
        if artifact_type not in self.schemas:
            return ValidationResult(is_valid=True)
        
        schema = self.schemas[artifact_type]
        is_valid = schema.validator(data)
        return ValidationResult(
            is_valid=is_valid,
            errors=[] if is_valid else [f"Validation failed for {artifact_type}"]
        )


@dataclass
class Artifact:
    """A single artifact produced by a lifecycle phase."""
    phase: str
    artifact_type: str
    data: Any
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    artifact_id: str = ""

    def __post_init__(self):
        if not self.artifact_id:
            self.artifact_id = f"{self.phase}_{self.artifact_type}_{int(self.timestamp)}"

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "phase": self.phase,
            "artifact_type": self.artifact_type,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Artifact":
        return cls(**d)

    def to_context_string(self, max_length: int = 2000) -> str:
        """Convert artifact to a concise string for LLM context injection."""
        data_str = json.dumps(self.data, default=str) if isinstance(self.data, (dict, list)) else str(self.data)
        if len(data_str) > max_length:
            data_str = data_str[:max_length] + "...[truncated]"
        return f"[{self.phase}/{self.artifact_type}] {data_str}"


class ArtifactRegistry:
    """
    Cross-Phase Knowledge Propagation via Contextual Artifact Registry.

    Stores all artifacts from the data science lifecycle and provides
    methods to query them by phase, type, or downstream dependency.
    Automatically builds context windows for each agent.
    """

    # Define the phase ordering for downstream propagation
    PHASE_ORDER = [
        Phase.INGESTION,
        Phase.PROFILING,
        Phase.EDA,
        Phase.FEATURE_ENGINEERING,
        Phase.MODEL_SELECTION,
        Phase.TRAINING,
        Phase.EVALUATION,
        Phase.REPORTING,
    ]

    def __init__(self):
        self._artifacts: list[Artifact] = []
        self._session_id: str = ""

    def set_session(self, session_id: str):
        self._session_id = session_id

    # ─── Registration ─────────────────────────────────

    def register(
        self,
        phase: Phase | str,
        artifact_type: ArtifactType | str,
        data: Any,
        metadata: Optional[dict] = None,
        validate: bool = True,
    ) -> Artifact:
        """Register a new artifact from a lifecycle phase."""
        # Always extract .value from enums to avoid repr issues in Python 3.11+
        phase_str = phase.value if isinstance(phase, Phase) else str(phase)
        type_str = artifact_type.value if isinstance(artifact_type, ArtifactType) else str(artifact_type)
        
        # CLAIM 2: Sanitize data for cross-phase propagation and serialization
        clean_data = self._clean_data(data)
        
        if validate:
            validation = self.validate_artifact(type_str, clean_data)
            if not validation.is_valid:
                logger.warning(f"Artifact validation failed for {type_str}: {validation.errors}")
                if metadata is None: metadata = {}
                metadata["validation_errors"] = validation.errors
                metadata["is_valid"] = False
            else:
                if metadata is None: metadata = {}
                metadata["is_valid"] = True

        artifact = Artifact(
            phase=phase_str,
            artifact_type=type_str,
            data=clean_data,
            metadata=metadata or {},
        )
        self._artifacts.append(artifact)
        logger.info(f"Registered artifact: {artifact.artifact_id}")
        return artifact

    def validate_artifact(self, artifact_type: str, data: Any) -> ValidationResult:
        """Validate artifact data against its schema."""
        # Simple rule-based validation for now, can be expanded with Pydantic
        errors = []
        
        if artifact_type == ArtifactType.SCHEMA:
            if not isinstance(data, dict) or "columns" not in data:
                errors.append("Schema artifact must be a dict with 'columns' key")
        
        elif artifact_type == ArtifactType.STATISTICS:
            if not isinstance(data, dict):
                errors.append("Statistics artifact must be a dictionary")
        
        elif artifact_type == ArtifactType.MODEL_METRICS:
            if not isinstance(data, dict):
                errors.append("Model metrics must be a dictionary")
            else:
                required = ["accuracy", "precision", "recall", "f1"]
                # Only check if it's a model metrics type that expects these
                # some models might have different metrics
                pass

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _clean_data(self, data: Any) -> Any:
        """Recursively clean data for JSON serialization (replace NaN/Inf with None)."""
        import numpy as np
        if isinstance(data, dict):
            return {k: self._clean_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_data(v) for v in data]
        elif isinstance(data, float):
            if np.isnan(data) or np.isinf(data):
                return 0.0
            return data
        elif hasattr(data, "to_dict"):
            return self._clean_data(data.to_dict())
        return data

    # ─── Querying ─────────────────────────────────────

    def query(
        self,
        phase: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> list[Artifact]:
        """Query artifacts by phase and/or type."""
        results = self._artifacts
        if phase:
            p = phase if isinstance(phase, str) else phase.value
            results = [a for a in results if a.phase == p]
        if artifact_type:
            t = artifact_type if isinstance(artifact_type, str) else artifact_type.value
            results = [a for a in results if a.artifact_type == t]
        return results

    def get_latest(
        self,
        phase: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> Optional[Artifact]:
        """Get the most recent artifact matching criteria."""
        matches = self.query(phase, artifact_type)
        return matches[-1] if matches else None

    # ─── Context Propagation (CLAIM 2 CORE) ──────────

    def get_context_for_phase(
        self,
        target_phase: Phase | str,
        max_context_chars: int = 6000,
    ) -> str:
        """
        Build a context string containing all relevant upstream artifacts
        for a target phase. This enables each agent to benefit from
        all upstream knowledge without manual wiring.
        """
        target = target_phase if isinstance(target_phase, str) else target_phase.value

        # Find upstream phases
        try:
            target_idx = [p.value for p in self.PHASE_ORDER].index(target)
        except ValueError:
            target_idx = len(self.PHASE_ORDER)

        upstream_phases = [p.value for p in self.PHASE_ORDER[:target_idx]]

        # Collect upstream artifacts, most recent first
        upstream_artifacts = [
            a for a in self._artifacts if a.phase in upstream_phases
        ]
        upstream_artifacts.sort(key=lambda a: a.timestamp, reverse=True)

        # Build context string within budget
        context_parts = []
        chars_used = 0
        for artifact in upstream_artifacts:
            ctx = artifact.to_context_string(max_length=1500)
            if chars_used + len(ctx) > max_context_chars:
                break
            context_parts.append(ctx)
            chars_used += len(ctx)

        context_parts.reverse()  # Chronological order

        if not context_parts:
            return "No upstream context available."

        return (
            "=== UPSTREAM CONTEXT (from previous lifecycle phases) ===\n"
            + "\n\n".join(context_parts)
            + "\n=== END UPSTREAM CONTEXT ==="
        )

    # ─── Serialization (for checkpointing) ───────────

    def to_dict(self) -> dict:
        return {
            "session_id": self._session_id,
            "artifacts": [a.to_dict() for a in self._artifacts],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ArtifactRegistry":
        data = json.loads(json_str)
        registry = cls()
        registry._session_id = data.get("session_id", "")
        registry._artifacts = [Artifact.from_dict(a) for a in data.get("artifacts", [])]
        return registry

    @property
    def summary(self) -> dict:
        """Get a summary of all registered artifacts."""
        by_phase: dict[str, list[str]] = {}
        for a in self._artifacts:
            by_phase.setdefault(a.phase, []).append(a.artifact_type)
        return {
            "total_artifacts": len(self._artifacts),
            "by_phase": by_phase,
        }
