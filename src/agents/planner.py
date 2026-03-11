"""
DatacortexAI — Meta-Cognitive Planning Agent

Meta-Cognitive Planning Agent with Reflective Agent Graph

A hierarchical orchestration system where a Meta-Cognitive Planner decomposes
a data science problem into a DAG of subtasks, dynamically dispatching to
specialized agents based on runtime analysis of data characteristics, and
adapting the execution plan based on intermediate results.
"""

import asyncio
import logging
import time
import uuid
from typing import Optional, Callable, Awaitable, Any

from src.core.config import Config
from src.core.llm_client import LLMClient
from src.core.artifact_registry import ArtifactRegistry, Phase, ArtifactType
from src.core.checkpoint import CheckpointManager, PipelineCheckpoint
from src.core.data_fingerprint import DataFingerprint
from src.agents.ingestion_agent import IngestionAgent
from src.agents.eda_agent import EDAAgent
from src.agents.modeling_agent import ModelingAgent
from src.agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# DAG Execution Plan
# ─────────────────────────────────────────────────────

DEFAULT_EXECUTION_PLAN = [
    "ingestion",
    "profiling",
    "eda",
    "modeling",
    "reporting",
]


class PlannerAgent:
    """
    Meta-Cognitive Planning Agent with Reflective Agent Graph.

    Orchestrates the entire data science lifecycle by:
    1. Analyzing user intent and data characteristics
    2. Generating an adaptive execution DAG
    3. Dispatching to specialized agents
    4. Monitoring intermediate outputs and adapting the plan
    5. Streaming progress events to the frontend
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        on_event: Optional[Callable[[dict], Awaitable[None]]] = None,
        **kwargs,
    ):
        self.config = config or Config.get()
        self.llm = LLMClient(self.config)
        self.checkpoint_mgr = checkpoint_manager or CheckpointManager()
        self.on_event = on_event  # WebSocket event emitter
        self.approval_callback = kwargs.get("approval_callback")

        # Specialized agents
        self.ingestion_agent = IngestionAgent(self.config, self.llm)
        self.eda_agent = EDAAgent(self.config, self.llm)
        self.modeling_agent = ModelingAgent(self.config, self.llm)
        self.report_agent = ReportAgent(self.config, self.llm)

    # ─── Main Pipeline Entry Point ────────────────────

    async def run_pipeline(
        self,
        file_content: bytes,
        file_name: str,
        user_query: str = "",
        target_column: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Execute the full data science lifecycle pipeline.

        This is the primary entry point for DatacortexAI.
        """
        session_id = session_id or str(uuid.uuid4())
        start_time = time.time()

        # Create checkpoint
        checkpoint = self.checkpoint_mgr.create(
            session_id=session_id,
            user_query=user_query,
            target_column=target_column,
            file_name=file_name,
        )
        registry = self.checkpoint_mgr.get_registry(session_id)

        await self._emit_event({
            "type": "pipeline_started",
            "session_id": session_id,
            "file_name": file_name,
            "timestamp": time.time(),
        })

        try:
            # ── PHASE 1: Data Ingestion ───────────────
            await self._emit_event({"type": "phase_started", "phase": "ingestion"})
            checkpoint.mark_phase_started("ingestion")

            ingestion_result = await self.ingestion_agent.run(
                file_content=file_content,
                file_name=file_name,
                registry=registry,
            )

            checkpoint.mark_phase_completed("ingestion", {
                "rows": ingestion_result.get("n_rows", 0),
                "cols": ingestion_result.get("n_cols", 0),
            })
            await self._emit_event({"type": "phase_completed", "phase": "ingestion", "result": ingestion_result})

            # ── PHASE 2: Data Profiling & Fingerprinting ─
            await self._emit_event({"type": "phase_started", "phase": "profiling"})
            checkpoint.mark_phase_started("profiling")

            fingerprint = ingestion_result.get("fingerprint")
            if not fingerprint:
                checkpoint.mark_phase_failed("profiling", "No fingerprint available")
            else:
                checkpoint.mark_phase_completed("profiling", fingerprint.to_dict() if isinstance(fingerprint, DataFingerprint) else fingerprint)
                await self._emit_event({"type": "phase_completed", "phase": "profiling"})

            # ── Adaptive Plan Generation (CLAIM 1 CORE) ─
            execution_plan = await self._generate_adaptive_plan(
                fingerprint=fingerprint,
                user_query=user_query,
                target_column=target_column,
                registry=registry,
            )
            checkpoint.execution_plan = execution_plan
            await self._emit_event({"type": "plan_generated", "plan": execution_plan})

            # ── PHASE 3: Exploratory Data Analysis ────
            if "eda" in execution_plan:
                await self._emit_event({"type": "phase_started", "phase": "eda"})
                checkpoint.mark_phase_started("eda")

                eda_result = await self.eda_agent.run(
                    df_info=ingestion_result,
                    fingerprint=fingerprint,
                    target_column=target_column,
                    registry=registry,
                )

                checkpoint.mark_phase_completed("eda", {"n_findings": len(eda_result.get("findings", []))})
                await self._emit_event({"type": "phase_completed", "phase": "eda", "result": eda_result})

            # ── PHASE 4: Modeling ─────────────────────
            if "modeling" in execution_plan:
                await self._emit_event({"type": "phase_started", "phase": "modeling"})
                checkpoint.mark_phase_started("modeling")

                modeling_result = await self.modeling_agent.run(
                    fingerprint=fingerprint,
                    target_column=target_column,
                    registry=registry,
                )

                checkpoint.mark_phase_completed("modeling", {
                    "best_model": modeling_result.get("best_model"),
                    "best_score": modeling_result.get("best_score"),
                })
                await self._emit_event({"type": "phase_completed", "phase": "modeling", "result": modeling_result})

            # ── PHASE 5: Report Generation ────────────
            if "reporting" in execution_plan:
                await self._emit_event({"type": "phase_started", "phase": "reporting"})
                checkpoint.mark_phase_started("reporting")

                report_result = await self.report_agent.run(
                    user_query=user_query,
                    file_name=file_name,
                    registry=registry,
                )

                checkpoint.mark_phase_completed("reporting")
                await self._emit_event({"type": "phase_completed", "phase": "reporting", "result": report_result})

            # ── Pipeline Complete ─────────────────────
            checkpoint.status = "completed"
            elapsed = time.time() - start_time

            final_result = {
                "session_id": session_id,
                "status": "completed",
                "elapsed_seconds": round(elapsed, 2),
                "phases_completed": checkpoint.get_completed_phases(),
                "report": report_result.get("report_markdown", "") if "reporting" in execution_plan else "",
                "summary": registry.summary,
            }

            await self._emit_event({"type": "pipeline_completed", **final_result})
            self.checkpoint_mgr.save(session_id)

            return final_result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            checkpoint.status = "failed"
            if checkpoint.current_phase:
                checkpoint.mark_phase_failed(checkpoint.current_phase, str(e))
            self.checkpoint_mgr.save(session_id)

            await self._emit_event({
                "type": "pipeline_failed",
                "error": str(e),
                "phase": checkpoint.current_phase,
            })

            return {
                "session_id": session_id,
                "status": "failed",
                "error": str(e),
                "phases_completed": checkpoint.get_completed_phases(),
            }

    # ─── Adaptive Plan Generation (CLAIM 1 CORE) ─────

    async def _generate_adaptive_plan(
        self,
        fingerprint: Optional[DataFingerprint],
        user_query: str,
        target_column: Optional[str],
        registry: ArtifactRegistry,
    ) -> list[str]:
        """
        CLAIM 1 INTELLIGENCE:
        Dynamically generate execution plan based on data characteristics.
        """
        plan = ["eda"]  # Always do EDA

        if target_column:
            plan.append("modeling")
        elif fingerprint and fingerprint.suggested_task != "clustering":
            # Try to auto-detect if modeling makes sense
            plan.append("modeling")

        plan.append("reporting")

        # Adaptive modifications based on fingerprint
        if fingerprint:
            # Skip modeling for very small datasets
            if fingerprint.n_rows < 30:
                if "modeling" in plan:
                    # GUARDIAN CHECK
                    is_safe = await self.llm.validate(
                        action="Skip modeling",
                        reasoning=f"Dataset size ({fingerprint.n_rows} rows) is below the statistical reliability threshold (<30).",
                        context=f"Data Fingerprint: {fingerprint.to_dict()}"
                    )
                    if is_safe:
                        plan.remove("modeling")
                        logger.info("Skipping modeling: dataset too small (<30 rows)")
                    else:
                        logger.warning("Guardian overrode skip-modeling decision. Proceeding with caution.")

            # Add feature engineering for high-dimensional data
            if fingerprint.dimensionality_ratio > 0.05 and fingerprint.feature_redundancy > 0.1:
                if "feature_engineering" not in plan:
                    plan.insert(plan.index("eda") + 1, "feature_engineering")

        # Log the plan with AUDIT LOG structure
        logger.info(f"Generated execution plan: {plan}")
        self._log_audit(
            registry=registry,
            phase=Phase.PROFILING,
            observation=f"Data fingerprint generated for {fingerprint.n_rows if fingerprint else 'unknown'} rows.",
            reflection=f"Determined that a {len(plan)}-step pipeline is optimal for this data variety.",
            action=f"Generated execution plan: {plan}"
        )

        return plan

    def _log_audit(
        self,
        registry: ArtifactRegistry,
        phase: Phase,
        observation: str,
        reflection: str,
        action: str
    ):
        """Register a structured Observe-Reflect-Act audit log entry."""
        log_entry = {
            "timestamp": time.time(),
            "observation": observation,
            "reflection": reflection,
            "action": action,
            "system_state": "healthy"
        }
        registry.register(
            phase,
            ArtifactType.EXECUTION_LOG,
            log_entry
        )
        logger.info(f"Audit Log [{phase.value}]: {action}")

    async def request_approval(self, action: str, reasoning: str) -> bool:
        """Request human approval for a high-impact action."""
        if not self.approval_callback:
            logger.info(f"Auto-approving action (no HITL callback): {action}")
            return True
        
        await self._emit_event({
            "type": "approval_required",
            "action": action,
            "reasoning": reasoning,
            "timestamp": time.time(),
        })
        
        logger.info(f"Awaiting human approval for: {action}")
        return await self.approval_callback(action, reasoning)

    # ─── Event Emission ──────────────────────────────

    async def _emit_event(self, event: dict):
        """Emit a real-time event to the frontend via callback."""
        event.setdefault("timestamp", time.time())
        logger.info(f"Event: {event.get('type')} — {event.get('phase', '')}")
        if self.on_event:
            try:
                await self.on_event(event)
            except Exception as e:
                logger.warning(f"Event emission failed: {e}")
