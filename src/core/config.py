"""
DatacortexAI Configuration Module

Distributed Agent Workload Balancing
Across Credential Pools - automatic rotation based on rate limits,
budget constraints, and account-level quota monitoring.
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────
# Blaxel Account Configuration
# ─────────────────────────────────────────────────────

@dataclass
class BlaxelAccount:
    """Single Blaxel account with budget and usage tracking."""
    key_id: int
    api_key: str
    workspace_name: str
    budget: float
    spent: float = 0.0
    request_count: int = 0
    last_used: float = 0.0
    rate_limit_until: float = 0.0

    @property
    def remaining_budget(self) -> float:
        return self.budget - self.spent

    @property
    def is_available(self) -> bool:
        return (
            self.remaining_budget > 0.5
            and time.time() > self.rate_limit_until
        )

    def record_usage(self, cost: float = 0.01):
        self.spent += cost
        self.request_count += 1
        self.last_used = time.time()

    def mark_rate_limited(self, cooldown_seconds: float = 60.0):
        self.rate_limit_until = time.time() + cooldown_seconds


@dataclass
class CredentialPool:
    """
    Distributed credential pool with automatic rotation.

    Manages multiple Blaxel accounts and selects the optimal one
    for each request based on remaining budget, rate limit status,
    and round-robin fairness.
    """
    accounts: list[BlaxelAccount] = field(default_factory=list)
    _rotation_index: int = 0

    def get_next_account(self) -> BlaxelAccount:
        """Select next available account using weighted round-robin."""
        available = [a for a in self.accounts if a.is_available]
        if not available:
            raise RuntimeError(
                "All Blaxel accounts exhausted or rate-limited. "
                f"Accounts: {[(a.key_id, a.remaining_budget, a.is_available) for a in self.accounts]}"
            )

        # Weighted selection: prefer accounts with more remaining budget
        weights = [a.remaining_budget for a in available]
        total = sum(weights)
        if total == 0:
            return random.choice(available)

        selected = random.choices(available, weights=weights, k=1)[0]
        selected.record_usage()
        return selected

    def get_account_by_id(self, key_id: int) -> Optional[BlaxelAccount]:
        for a in self.accounts:
            if a.key_id == key_id:
                return a
        return None

    @property
    def total_remaining_budget(self) -> float:
        return sum(a.remaining_budget for a in self.accounts)

    @property
    def status_summary(self) -> dict:
        return {
            "total_budget_remaining": self.total_remaining_budget,
            "accounts": [
                {
                    "key_id": a.key_id,
                    "workspace": a.workspace_name,
                    "remaining": round(a.remaining_budget, 2),
                    "requests": a.request_count,
                    "available": a.is_available,
                }
                for a in self.accounts
            ],
        }


# ─────────────────────────────────────────────────────
# AI Inference Configuration
# ─────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    """Configuration for AI model inference endpoints.
    
    Multi-key pool rotation for OpenRouter
    to distribute requests across multiple free-tier accounts.
    """
    openrouter_api_keys: list[str] = field(default_factory=list)
    nvidia_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # ── NVIDIA NIM Model Assignments (integrate.api.nvidia.com) ────────────
    # Each model is chosen for its specific strength in the pipeline.
    # All use the same NVIDIA API key and base URL.
    #
    # PLANNING — Nemotron Super 49B
    #   Purpose-built for agentic multistep reasoning and DAG construction.
    #   Has a thinking-budget toggle. Best for adaptive plan generation.
    planning_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    #
    # CODE GENERATION — DeepSeek-R1
    #   Best open-source coder. Native chain-of-thought reasoning.
    #   Dominates HumanEval + SWE-Bench. Used for all generated Python.
    code_gen_model: str = "deepseek-ai/deepseek-r1"
    #
    # ANALYSIS — Llama 3.3 70B Instruct
    #   Fast, reliable JSON-mode output. Used for EDA interpretation,
    #   fingerprint analysis, and structured artifact generation.
    analysis_model: str = "meta/llama-3.3-70b-instruct"
    #
    # NARRATION — Llama 3.3 70B Instruct
    #   Strong instruction-following + fluent prose output.
    #   Used for executive summaries and recommendations in reports.
    narration_model: str = "meta/llama-3.3-70b-instruct"
    #
    # GUARDIAN — Nemotron Nano 8B
    #   Fast, low-latency safety judge. Separate from planning model
    #   to avoid monoculture (different model family = stronger
    #   multi-judge independence for Claim 3).
    guardian_model: str = "nvidia/llama-3.1-nemotron-nano-8b-v1"
    #
    # FALLBACK — Nemotron Nano 8B
    #   Lightweight always-available fallback if larger models time out.
    fallback_model: str = "nvidia/llama-3.1-nemotron-nano-8b-v1"

    _key_index: int = 0

    @property
    def openrouter_api_key(self) -> str:
        """Return current OpenRouter key."""
        if not self.openrouter_api_keys:
            return ""
        return self.openrouter_api_keys[self._key_index % len(self.openrouter_api_keys)]

    def rotate_openrouter_key(self) -> str:
        """Rotate to next OpenRouter key (on 429 rate limit)."""
        if len(self.openrouter_api_keys) > 1:
            self._key_index = (self._key_index + 1) % len(self.openrouter_api_keys)
        return self.openrouter_api_key


# ─────────────────────────────────────────────────────
# Application Settings
# ─────────────────────────────────────────────────────

@dataclass
class AppSettings:
    """Global application settings."""
    max_sandbox_retries: int = 3
    sandbox_timeout_seconds: int = 300  # 5 minutes
    sandbox_memory_mb: int = 4096  # 4GB (Tier 0 max)
    max_concurrent_sandboxes: int = 10  # Tier 0 limit
    checkpoint_enabled: bool = True
    streaming_enabled: bool = True
    max_file_size_mb: int = 100


# ─────────────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────────────

class Config:
    """Central configuration singleton."""

    _instance: Optional["Config"] = None

    def __init__(self):
        self.credential_pool = self._load_credential_pool()
        self.inference = self._load_inference_config()
        self.settings = AppSettings()

    @classmethod
    def get(cls) -> "Config":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def _load_credential_pool(self) -> CredentialPool:
        """Load Blaxel accounts 2, 3, 4 from environment."""
        pool = CredentialPool()
        for key_id in [2, 3, 4]:
            api_key = os.getenv(f"BLAXEL_ACCOUNT_{key_id}_API_KEY", "")
            budget = float(os.getenv(f"BLAXEL_ACCOUNT_{key_id}_BUDGET", "0"))
            workspace = os.getenv(f"BLAXEL_ACCOUNT_{key_id}_WORKSPACE_NAME", "")
            if api_key and workspace:
                pool.accounts.append(
                    BlaxelAccount(
                        key_id=key_id,
                        api_key=api_key,
                        workspace_name=workspace,
                        budget=budget,
                    )
                )
        return pool

    def _load_inference_config(self) -> InferenceConfig:
        # Load all OpenRouter keys (primary + fallbacks)
        or_keys = []
        primary = os.getenv("OPENROUTER_API_KEY", "")
        if primary:
            or_keys.append(primary)
        for i in range(2, 10):
            key = os.getenv(f"OPENROUTER_API_KEY_{i}", "")
            if key:
                or_keys.append(key)
        return InferenceConfig(
            openrouter_api_keys=or_keys,
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
        )
