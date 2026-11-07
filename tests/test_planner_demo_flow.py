import pytest

from src.agents.planner import PlannerAgent
from src.core.checkpoint import CheckpointManager
from src.core.config import Config


@pytest.mark.asyncio
async def test_planner_uses_auto_detected_target_for_demo_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("DATACORTEX_ENABLE_REMOTE_LLM", "0")
    Config.reset()

    rows = ["feature_a,feature_b,target"]
    for i in range(20):
        rows.append(f"{1 + i * 0.05:.2f},{0.1 + i * 0.01:.2f},0")
    for i in range(20):
        rows.append(f"{3 + i * 0.05:.2f},{1.0 + i * 0.01:.2f},1")
    csv_data = ("\n".join(rows) + "\n").encode("utf-8")
    events = []

    async def on_event(event):
        events.append(event)

    planner = PlannerAgent(
        config=Config.get(),
        checkpoint_manager=CheckpointManager(persist_dir=str(tmp_path)),
        on_event=on_event,
    )

    result = await planner.run_pipeline(
        file_content=csv_data,
        file_name="demo.csv",
        user_query="demo",
        target_column=None,
        session_id="demo-test",
    )

    assert result["status"] == "completed"
    assert "modeling" in result["phases_completed"]
    assert result["report"]
    assert any(event.get("type") == "pipeline_completed" for event in events)
