"""
Test — Checkpoint Module

Tests checkpoint creation, serialization, and pipeline resumability.
"""

import json
import pytest

from src.core.checkpoint import PipelineCheckpoint, CheckpointManager


class TestPipelineCheckpoint:

    def test_create_checkpoint(self):
        cp = PipelineCheckpoint(session_id="test-001")
        assert cp.session_id == "test-001"
        assert cp.status == "initialized"
        assert cp.progress_pct == 0.0

    def test_phase_lifecycle(self):
        cp = PipelineCheckpoint(session_id="test-002")
        cp.execution_plan = ["ingestion", "eda", "modeling", "reporting"]

        cp.mark_phase_started("ingestion")
        assert cp.phases["ingestion"].status == "running"
        assert cp.current_phase == "ingestion"

        cp.mark_phase_completed("ingestion", {"rows": 100})
        assert cp.phases["ingestion"].status == "completed"
        assert cp.phases["ingestion"].output_summary == {"rows": 100}
        assert cp.progress_pct == 25.0

    def test_phase_failure(self):
        cp = PipelineCheckpoint(session_id="test-003")
        cp.execution_plan = ["ingestion", "eda"]

        cp.mark_phase_started("ingestion")
        cp.mark_phase_failed("ingestion", "Parse error")
        assert cp.phases["ingestion"].status == "failed"
        assert cp.phases["ingestion"].error == "Parse error"

    def test_get_next_phase(self):
        cp = PipelineCheckpoint(session_id="test-004")
        cp.execution_plan = ["ingestion", "eda", "modeling"]

        assert cp.get_next_phase() == "ingestion"

        cp.mark_phase_started("ingestion")
        cp.mark_phase_completed("ingestion")
        assert cp.get_next_phase() == "eda"

        cp.mark_phase_started("eda")
        cp.mark_phase_completed("eda")
        assert cp.get_next_phase() == "modeling"

    def test_json_roundtrip(self):
        cp = PipelineCheckpoint(session_id="test-005", user_query="Analyze data")
        cp.execution_plan = ["ingestion", "eda"]
        cp.mark_phase_started("ingestion")
        cp.mark_phase_completed("ingestion", {"rows": 50})

        json_str = cp.to_json()
        restored = PipelineCheckpoint.from_json(json_str)

        assert restored.session_id == "test-005"
        assert restored.user_query == "Analyze data"
        assert "ingestion" in restored.phases
        assert restored.phases["ingestion"].status == "completed"
        assert restored.progress_pct == 50.0

    def test_progress_calculation(self):
        cp = PipelineCheckpoint(session_id="test-006")
        cp.execution_plan = ["a", "b", "c", "d"]

        assert cp.progress_pct == 0.0

        cp.mark_phase_started("a")
        cp.mark_phase_completed("a")
        assert cp.progress_pct == 25.0

        cp.mark_phase_started("b")
        cp.mark_phase_completed("b")
        assert cp.progress_pct == 50.0


class TestCheckpointManager:

    def test_create_and_get(self, tmp_path):
        mgr = CheckpointManager(persist_dir=str(tmp_path))
        cp = mgr.create("session-001", user_query="test")
        assert cp.session_id == "session-001"

        retrieved = mgr.get("session-001")
        assert retrieved is not None
        assert retrieved.user_query == "test"

    def test_registry_access(self, tmp_path):
        mgr = CheckpointManager(persist_dir=str(tmp_path))
        mgr.create("session-002")

        reg = mgr.get_registry("session-002")
        assert reg is not None
        assert reg.summary["total_artifacts"] == 0

    def test_list_sessions(self, tmp_path):
        mgr = CheckpointManager(persist_dir=str(tmp_path))
        mgr.create("s1")
        mgr.create("s2")
        assert "s1" in mgr.list_sessions()
        assert "s2" in mgr.list_sessions()
