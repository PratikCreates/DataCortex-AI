import io

import pandas as pd
import pytest

from src.agents.modeling_agent import ModelingAgent
from src.core.artifact_registry import ArtifactRegistry, Phase, ArtifactType
from src.core.config import Config
from src.core.data_fingerprint import DataFingerprinter


class DummyLLM:
    async def analyze(self, *args, **kwargs):
        raise RuntimeError("LLM should not be needed for first-pass local training")

    async def validate(self, *args, **kwargs):
        return False


@pytest.mark.asyncio
async def test_local_training_registers_audit_log_and_model_results():
    csv_data = """f1,f2,target
1.0,0.1,0
1.2,0.2,0
1.4,0.1,0
2.8,1.0,1
3.0,1.1,1
3.2,1.2,1
"""
    registry = ArtifactRegistry()
    registry.register(Phase.INGESTION, ArtifactType.DATA_QUALITY, {"full_csv": csv_data})
    df = pd.read_csv(io.StringIO(csv_data))
    fp = DataFingerprinter().compute(df, target_col="target")

    agent = ModelingAgent(Config(), DummyLLM())
    result = await agent.run(fp, "target", registry)

    assert result["models"]
    assert result["best_model"]
    assert registry.query(Phase.TRAINING.value, ArtifactType.EXECUTION_LOG.value)


@pytest.mark.asyncio
async def test_local_training_reports_single_class_target_cleanly():
    csv_data = """f1,f2,target
1.0,0.1,1
1.2,0.2,1
1.4,0.1,1
2.8,1.0,1
3.0,1.1,1
"""
    registry = ArtifactRegistry()
    registry.register(Phase.INGESTION, ArtifactType.DATA_QUALITY, {"full_csv": csv_data})
    df = pd.read_csv(io.StringIO(csv_data))
    fp = DataFingerprinter().compute(df, target_col="target")

    agent = ModelingAgent(Config(), DummyLLM())
    result = await agent._train_locally(fp, "target", fp.suggested_models, registry)

    assert result["models"] == []
    assert "fewer than two classes" in result["error"]
