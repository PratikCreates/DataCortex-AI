"""
Test — Artifact Registry

Tests register, query, propagation of cross-phase artifacts.
"""

import json
import pytest

from src.core.artifact_registry import ArtifactRegistry, Phase, ArtifactType, Artifact


@pytest.fixture
def registry():
    reg = ArtifactRegistry()
    reg.set_session("test-session-001")
    return reg


class TestArtifactRegistration:

    def test_register_artifact(self, registry):
        art = registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"columns": ["a", "b"]})
        assert art.phase == "ingestion"
        assert art.artifact_type == "schema"
        assert art.data == {"columns": ["a", "b"]}

    def test_register_multiple_phases(self, registry):
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"cols": 5})
        registry.register(Phase.EDA, ArtifactType.STATISTICS, {"mean": 3.5})
        registry.register(Phase.TRAINING, ArtifactType.MODEL_METRICS, {"score": 0.95})
        assert registry.summary["total_artifacts"] == 3

    def test_query_by_phase(self, registry):
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"a": 1})
        registry.register(Phase.INGESTION, ArtifactType.DATA_QUALITY, {"b": 2})
        registry.register(Phase.EDA, ArtifactType.STATISTICS, {"c": 3})

        results = registry.query(phase="ingestion")
        assert len(results) == 2

    def test_query_by_type(self, registry):
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"a": 1})
        registry.register(Phase.EDA, ArtifactType.SCHEMA, {"b": 2})
        results = registry.query(artifact_type="schema")
        assert len(results) == 2

    def test_get_latest(self, registry):
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"v": 1})
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"v": 2})
        latest = registry.get_latest(phase="ingestion", artifact_type="schema")
        assert latest.data["v"] == 2


class TestCrossPhaseKnowledgePropagation:
    """Tests for PATENT CLAIM 2."""

    def test_context_for_eda_includes_ingestion(self, registry):
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"cols": ["a", "b", "target"]})
        registry.register(Phase.INGESTION, ArtifactType.DATA_QUALITY, {"score": 0.95})

        context = registry.get_context_for_phase(Phase.EDA)
        assert "UPSTREAM CONTEXT" in context
        assert "ingestion" in context
        assert "schema" in context

    def test_context_for_modeling_includes_ingestion_and_eda(self, registry):
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"cols": 5})
        registry.register(Phase.EDA, ArtifactType.FINDINGS, {"key_findings": ["Finding 1"]})

        context = registry.get_context_for_phase(Phase.TRAINING)
        assert "ingestion" in context
        assert "eda" in context

    def test_context_for_reporting_includes_all(self, registry):
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"cols": 5})
        registry.register(Phase.EDA, ArtifactType.FINDINGS, {"f": 1})
        registry.register(Phase.TRAINING, ArtifactType.MODEL_METRICS, {"score": 0.9})
        registry.register(Phase.EVALUATION, ArtifactType.FEATURE_IMPORTANCE, {"a": 0.5})

        context = registry.get_context_for_phase(Phase.REPORTING)
        assert "ingestion" in context
        assert "eda" in context

    def test_no_upstream_context_for_ingestion(self, registry):
        context = registry.get_context_for_phase(Phase.INGESTION)
        assert "No upstream context" in context

    def test_context_respects_max_chars(self, registry):
        # Register a large artifact
        registry.register(Phase.INGESTION, ArtifactType.DATAFRAME_SAMPLE, {"data": "x" * 10000})
        context = registry.get_context_for_phase(Phase.EDA, max_context_chars=500)
        assert len(context) < 2000  # Should be truncated


class TestSerialization:

    def test_json_roundtrip(self, registry):
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, {"cols": ["a", "b"]})
        registry.register(Phase.EDA, ArtifactType.STATISTICS, {"mean": 3.5})

        json_str = registry.to_json()
        restored = ArtifactRegistry.from_json(json_str)

        assert restored.summary["total_artifacts"] == 2
        assert len(restored.query(phase="ingestion")) == 1

    def test_artifact_to_dict(self):
        art = Artifact(
            phase="ingestion",
            artifact_type="schema",
            data={"columns": ["a"]},
        )
        d = art.to_dict()
        assert d["phase"] == "ingestion"
        assert d["data"]["columns"] == ["a"]

    def test_artifact_from_dict(self):
        d = {
            "artifact_id": "test_1",
            "phase": "eda",
            "artifact_type": "statistics",
            "data": {"mean": 5},
            "metadata": {},
            "timestamp": 1000,
        }
        art = Artifact.from_dict(d)
        assert art.phase == "eda"
        assert art.data["mean"] == 5
