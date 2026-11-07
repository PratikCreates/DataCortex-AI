"""
DatacortexAI — Unified LLM Client

Provides a single interface to OpenRouter (free router) and NVIDIA API.
Both use OpenAI-compatible API format. Includes retry logic, model fallback,
and structured output (JSON mode) support.

FIX: Added per-call timeout (15s) and reduced retries (1) to prevent
pipeline hangs when LLM providers are unavailable (e.g. NVIDIA 504).
"""

import json
import logging
import asyncio
from typing import Optional, Any
from dataclasses import dataclass, field
from openai import AsyncOpenAI

from src.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM inference."""
    content: str
    model_used: str
    provider: str  # "nvidia" | "openrouter" | "fallback"
    usage: dict = field(default_factory=dict)
    reasoning: Optional[str] = None
    tool_calls: Optional[list] = None


class LLMClient:
    """
    Unified LLM client for DatacortexAI — NVIDIA NIM Primary.

    All requests route through NVIDIA integrate.api.nvidia.com as primary.
    OpenRouter is kept as a last-resort fallback (NVIDIA fully down).

    Model assignments (Patent Claim 3 — Multi-Judge Independence):
      Planning  → Kimi-K2-Thinking    (256K context reasoning SOTA)
      Code Gen  → DeepSeek-V3.2       (685B thinking coder)
      Analysis  → Gemma 4 31B IT      (Latest Google reasoning model)
      Narration → Llama 3.3 70B       (Reliable instruction following)
      Guardian  → Llama Guard 4 12B   (Next-gen safety architecture)
      Fallback  → Phi-4 Mini          (SOTA lightweight instruct)
    """

    CALL_TIMEOUT = 15  # seconds per individual LLM call

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.get()
        inf = self.config.inference

        self._nvidia_client = AsyncOpenAI(
            base_url=inf.nvidia_base_url,
            api_key=inf.nvidia_api_key,
            timeout=self.CALL_TIMEOUT,
        )
        self._openrouter_client = self._make_openrouter_client()
        self._keys_tried = 0
        logger.info(f"LLM Client initialized with {len(inf.openrouter_api_keys)} OpenRouter keys")

    def _make_openrouter_client(self) -> AsyncOpenAI:
        """Create a new OpenRouter client with the current key."""
        inf = self.config.inference
        return AsyncOpenAI(
            base_url=inf.openrouter_base_url,
            api_key=inf.openrouter_api_key,
            timeout=self.CALL_TIMEOUT,
            default_headers={
                "HTTP-Referer": "https://datacortex.ai",
                "X-Title": "DatacortexAI",
            },
        )

    @property
    def has_provider_credentials(self) -> bool:
        """True when at least one remote LLM provider key is configured."""
        inf = self.config.inference
        return bool(self.config.settings.enable_remote_llm and (inf.nvidia_api_key or inf.openrouter_api_keys))

    def _rotate_key(self):
        """Rotate to next OpenRouter key and rebuild client."""
        old_key = self.config.inference.openrouter_api_key[-8:]
        new_key_full = self.config.inference.rotate_openrouter_key()
        new_key = new_key_full[-8:]
        self._openrouter_client = self._make_openrouter_client()
        logger.info(f"Rotated OpenRouter key: ...{old_key} → ...{new_key}")

    # ─── Public API ───────────────────────────────────

    async def plan(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Use the planning model (Kimi-K2-Thinking) for DAG construction.
        Native high-depth reasoning for complex agentic plans.
        """
        return await self._call_with_fallback(
            primary_provider="nvidia",
            primary_model=self.config.inference.planning_model,
            prompt=prompt,
            system=system,
            enable_reasoning=True,
            **kwargs,
        )

    async def generate_code(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Use DeepSeek-V3.2 for code generation. 
        Agentic-ready 685B SOTA with thinking mode enabled.
        """
        return await self._call_with_fallback(
            primary_provider="nvidia",
            primary_model=self.config.inference.code_gen_model,
            prompt=prompt,
            system=system or "You are an expert Python data scientist. Output ONLY valid Python code.",
            enable_reasoning=True,
            **kwargs,
        )

    async def analyze(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Use the analysis model for general data analysis tasks."""
        return await self._call_with_fallback(
            primary_provider="nvidia",
            primary_model=self.config.inference.analysis_model,
            prompt=prompt,
            system=system,
            **kwargs,
        )

    async def narrate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Use the narration model for generating reports and narratives."""
        return await self._call_with_fallback(
            primary_provider="nvidia",
            primary_model=self.config.inference.narration_model,
            prompt=prompt,
            system=system or "You are a senior data science consultant writing executive-quality reports in Markdown.",
            **kwargs,
        )

    async def structured_output(
        self, prompt: str, system: str = "", schema_hint: str = "", **kwargs
    ) -> dict:
        """Get a JSON-structured response from the LLM."""
        full_system = (
            f"{system}\n\nYou MUST respond with valid JSON only. No markdown, no extra text."
            f"\n\nExpected schema:\n{schema_hint}" if schema_hint else
            f"{system}\n\nYou MUST respond with valid JSON only. No markdown, no extra text."
        )
        try:
            response = await self._call_with_fallback(
                primary_provider="nvidia",
                primary_model=self.config.inference.planning_model,
                prompt=prompt,
                system=full_system,
                **kwargs,
            )
            return self._parse_json(response.content)
        except Exception as e:
            logger.warning(f"structured_output failed: {e}")
            return {"parse_error": True, "error": str(e)}

    async def validate(self, action: str, reasoning: str, context: Optional[str] = None) -> bool:
        """
        Multi-Judge Guardian: Validates a proposed action via a secondary LLM.

        Uses the guardian_model (Llama Guard 4 12B) — a DIFFERENT model family
        from the planning_model (Kimi-K2) to ensure genuine multi-judge
        independence (Patent Claim 3).

        Returns True if the action is deemed safe and logically consistent.
        """
        prompt = f"""
Action Proposed: {action}
Reasoning: {reasoning}
Context: {context or 'N/A'}

As an AI Safety Guardian, evaluate this action for:
1. Safety (No destructive data loss unless explicitly requested for cleaning).
2. Logical Consistency (Does it follow from the reasoning?).
3. Transparency (Is it clear what this will do?).

Respond with a JSON object: {{"safe": true/false, "explanation": "..."}}
"""
        try:
            # Guardian uses a DIFFERENT model from primary planner (anti-monoculture)
            guardian_model = getattr(self.config.inference, "guardian_model",
                                     self.config.inference.fallback_model)
            resp = await self._call_with_fallback(
                primary_provider="nvidia",
                primary_model=guardian_model,
                prompt=prompt,
                system="You are an AI Safety Validator. Your ONLY job is to prevent incorrect or unsafe agent actions. Respond ONLY in valid JSON.",
                max_tokens=256,
            )
            parsed = self._parse_json(resp.content)
            safe = parsed.get("safe", False)
            if not safe:
                logger.warning(f"Guardian ({guardian_model}) rejected: {parsed.get('explanation')}")
            else:
                logger.info(f"Guardian approved: {action[:60]}...")
            return safe
        except Exception as e:
            logger.warning(f"Guardian validation failed ({e}), defaulting to False")
            return False

    # ─── Internal Machinery ──────────────────────────

    async def _call_with_fallback(
        self,
        primary_provider: str,
        primary_model: str,
        prompt: str,
        system: str = "",
        enable_reasoning: bool = False,
        tools: Optional[list[dict]] = None,
        max_retries: int = 1,
        **kwargs,
    ) -> LLMResponse:
        """Try primary provider, fall back to alternate on failure.
        On OpenRouter 429, rotate to next key before retrying."""
        # Allow overriding the model via kwargs
        model = kwargs.pop("model", primary_model)
        fallback_chain = self._build_fallback_chain(primary_provider, model)
        n_or_keys = len(self.config.inference.openrouter_api_keys)

        last_error = None
        for provider, model in fallback_chain:
            keys_to_try = n_or_keys if provider == "openrouter" else 1
            for key_attempt in range(keys_to_try):
                try:
                    return await asyncio.wait_for(
                        self._call_provider(
                            provider=provider,
                            model=model,
                            prompt=prompt,
                            system=system,
                            enable_reasoning=enable_reasoning,
                            tools=tools,
                            **kwargs,
                        ),
                        timeout=self.CALL_TIMEOUT + 5,
                    )
                except asyncio.TimeoutError:
                    last_error = TimeoutError(f"Timeout after {self.CALL_TIMEOUT}s calling {provider}/{model}")
                    logger.warning(f"LLM timeout ({provider}/{model})")
                    break  # No point retrying same provider on timeout
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    # Rotate key on 429 rate limit
                    if "429" in err_str and provider == "openrouter" and key_attempt < keys_to_try - 1:
                        self._rotate_key()
                        logger.info(f"Rate limited, trying next key for {model}")
                        continue
                    logger.warning(f"LLM call failed ({provider}/{model}): {e}")
                    break  # Move to next provider in chain

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    async def _call_provider(
        self,
        provider: str,
        model: str,
        prompt: str,
        system: str = "",
        enable_reasoning: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a single LLM call to a specific provider."""
        client = self._nvidia_client if provider == "nvidia" else self._openrouter_client

        # All providers are now NVIDIA NIM — no model ID remapping needed.
        # NVIDIA NIM accepts the full model IDs like 'meta/llama-3.3-70b-instruct',
        # 'deepseek-ai/deepseek-r1', 'nvidia/llama-3.3-nemotron-super-49b-v1' directly.

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4096),  # Reduced from 8192
        }

        if tools:
            call_kwargs["tools"] = tools

        # NVIDIA deepseek supports reasoning via extra_body
        if enable_reasoning and provider == "nvidia":
            call_kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}

        # OpenRouter supports reasoning via top-level param
        if enable_reasoning and provider == "openrouter":
            call_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        completion = await client.chat.completions.create(**call_kwargs)

        choice = completion.choices[0]
        content = choice.message.content or ""
        
        # Try to extract reasoning content (OAI, DeepSeek, or vendor-specific)
        reasoning = getattr(choice.message, "reasoning_content", None)
        if not reasoning and hasattr(choice.message, "reasoning"):
            reasoning = choice.message.reasoning
        
        tool_calls = getattr(choice.message, "tool_calls", None)

        if not content and not tool_calls:
            logger.warning(f"Model {model} returned empty content and no tool calls.")

        return LLMResponse(
            content=content,
            model_used=getattr(completion, "model", model),
            provider=provider,
            usage={
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0) if completion.usage else 0,
                "completion_tokens": getattr(completion.usage, "completion_tokens", 0) if completion.usage else 0,
            },
            reasoning=reasoning,
            tool_calls=tool_calls,
        )

    def _build_fallback_chain(
        self, primary_provider: str, primary_model: str
    ) -> list[tuple[str, str]]:
        """Build ordered list of (provider, model) pairs to try.

        Fallback order:
          1. Primary NVIDIA model (e.g. Kimi-K2 / DeepSeek-V3.2)
          2. Gemma 4 31B (NVIDIA Frontier fallback)
          3. Llama 3.3 70B (Reliability fallback)
          4. Phi-4 Mini (SOTA SLM fallback)
          5. OpenRouter (Last resort)
        """
        inf = self.config.inference
        chain = [(primary_provider, primary_model)]

        if primary_provider == "nvidia":
            if primary_model != inf.analysis_model:
                chain.append(("nvidia", inf.analysis_model))      # Gemma 4 31B
            if primary_model != inf.narration_model:
                chain.append(("nvidia", inf.narration_model))     # Llama 3.3 70B
            if primary_model != inf.fallback_model:
                chain.append(("nvidia", inf.fallback_model))      # Phi-4 Mini
            chain.append(("openrouter", inf.analysis_model))      # Last resort
        else:
            chain.append(("nvidia", inf.planning_model))
            chain.append(("nvidia", inf.fallback_model))

        return chain

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Robustly parse JSON from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {"raw_response": text, "parse_error": True}
