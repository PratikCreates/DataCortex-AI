import asyncio
import logging
import json
import os
from src.core.config import Config
from src.core.artifact_registry import ArtifactRegistry, Phase, ArtifactType
from src.agents.planner import PlannerAgent
from src.core.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_audit_logs():
    """Verify that Observe-Reflect-Act logs are being registered."""
    config = Config.get()
    checkpoint_mgr = CheckpointManager()
    
    # Mock file content
    content = b"a,b,target\n1,2,0\n3,4,1\n" * 5 # Small dataset
    
    planner = PlannerAgent(config=config, checkpoint_manager=checkpoint_mgr)
    
    logger.info("Running small test pipeline...")
    result = await planner.run_pipeline(
        file_content=content,
        file_name="safety_test.csv",
        user_query="Train a model",
        target_column="target",
        session_id="test_safety_session"
    )
    
    registry = checkpoint_mgr.get_registry("test_safety_session")
    artifacts = registry.query()
    
    audit_logs = [a for a in artifacts if a.artifact_type == ArtifactType.EXECUTION_LOG.value]
    
    print("\n" + "="*50)
    print("SAFETY FEATURE VERIFICATION: AUDIT LOGS")
    print("="*50)
    if audit_logs:
        print(f"SUCCESS: Found {len(audit_logs)} audit log entries.")
        for log in audit_logs:
            data = log.data
            print(f"\nPhase: {log.phase}")
            print(f"Observation: {data.get('observation', 'N/A')}")
            print(f"Reflection: {data.get('reflection', 'N/A')}")
            print(f"Action:      {data.get('action', 'N/A')}")
    else:
        print("FAILED: No audit logs found.")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(test_audit_logs())
