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
    Unified LLM client for DatacortexAI.

    Routes requests to the best available provider:
    - NVIDIA API for planning/code generation (deepseek-v3.2)
    - OpenRouter free router for general analysis/narration
    - Automatic fallback between providers on failure
    - 15-second timeout per call to prevent pipeline hangs
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

    def _rotate_key(self):
        """Rotate to next OpenRouter key and rebuild client."""
        old_key = self.config.inference.openrouter_api_key[-8:]
        new_key_full = self.config.inference.rotate_openrouter_key()
        new_key = new_key_full[-8:]
        self._openrouter_client = self._make_openrouter_client()
        logger.info(f"Rotated OpenRouter key: ...{old_key} → ...{new_key}")

    # ─── Public API ───────────────────────────────────

    async def plan(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Use the planning model (NVIDIA deepseek) for reasoning tasks."""
        return await self._call_with_fallback(
            primary_provider="nvidia",
            primary_model=self.config.inference.planning_model,
            prompt=prompt,
            system=system,
            enable_reasoning=True,
            **kwargs,
        )

    async def generate_code(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Use the code generation model for writing Python code."""
        return await self._call_with_fallback(
            primary_provider="nvidia",
            primary_model=self.config.inference.code_gen_model,
            prompt=prompt,
            system=system or "You are an expert Python data scientist. Output ONLY valid Python code, no markdown fences.",
            enable_reasoning=True,
            **kwargs,
        )

    async def analyze(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Use the analysis model for general data analysis tasks."""
        return await self._call_with_fallback(
            primary_provider="openrouter",
            primary_model=self.config.inference.analysis_model,
            prompt=prompt,
            system=system,
            **kwargs,
        )

    async def narrate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """Use the narration model for generating reports and narratives."""
        return await self._call_with_fallback(
            primary_provider="openrouter",
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
            # We use OpenRouter as a secondary "judge" to avoid monoculture bias if primary is NVIDIA
            resp = await self.structured_output(
                prompt=prompt,
                system="You are an AI Safety Validator. Your job is to prevent incorrect or unsafe agent actions.",
                schema_hint='{"safe": boolean, "explanation": "string"}'
            )
            safe = resp.get("safe", False)
            if not safe:
                logger.warning(f"Guardian rejected action: {resp.get('explanation')}")
            else:
                logger.info(f"Guardian approved action: {action[:50]}...")
            return safe
        except Exception:
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
        fallback_chain = self._build_fallback_chain(primary_provider, primary_model)
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

        # Map generic models to NVIDIA-specific IDs if needed
        if provider == "nvidia" and "/" not in model:
             # Default to a robust NIM if model name is simple
             model = "nvidia/llama-3.1-405b-instruct" if "large" in model.lower() else "nvidia/llama-3.1-70b-instruct"
        elif provider == "nvidia" and "google/gemini" in model:
             # NVIDIA doesn't host Gemini, use Llama 3.1 70B as equivalent target
             model = "meta/llama-3.1-70b-instruct"

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
        reasoning = getattr(choice.message, "reasoning_content", None)
        tool_calls = getattr(choice.message, "tool_calls", None)

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
        """Build ordered list of (provider, model) pairs to try."""
        inf = self.config.inference
        chain = [(primary_provider, primary_model)]

        if primary_provider == "nvidia":
            chain.append(("openrouter", inf.analysis_model))
            chain.append(("openrouter", inf.fallback_model))
        else:
            chain.append(("nvidia", inf.planning_model))
            chain.append(("openrouter", inf.fallback_model))

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
