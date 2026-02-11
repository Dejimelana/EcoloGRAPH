"""
LLM client for EcoloGRAPH.

Provides a unified interface for interacting with LLMs,
with support for local models via Ollama.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from .config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: dict[str, int] | None = None
    raw_response: Any = None


class LLMClient:
    """
    Unified LLM client with support for local models via Ollama.
    
    Features:
    - Structured output with Pydantic models
    - Retry logic
    - Timeout handling
    - Logging
    """
    
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        api_key: str | None = None
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model name (e.g., "qwen3:14b")
            base_url: API base URL (e.g., "http://localhost:11434/v1")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        settings = get_settings()
        
        self.model = model or settings.llm.model
        self.base_url = base_url or settings.llm.base_url
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.max_tokens = max_tokens or settings.llm.max_tokens
        self.timeout = timeout
        
        # API key: explicit arg > env var > None
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # HTTP client with optional auth
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.client = httpx.Client(timeout=timeout, headers=headers)
        
        # Auto-detect model if set to "auto"
        if self.model == "auto":
            detected = self._detect_loaded_model()
            if detected:
                self.model = detected
        
        logger.info(f"LLMClient initialized: model={self.model}, base_url={self.base_url}")
    
    def _detect_loaded_model(self) -> str | None:
        """Try to detect the loaded model from LM Studio."""
        try:
            # LM Studio's native API endpoint for models
            url = self.base_url.replace("/v1", "/api/v1/models")
            response = self.client.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                # LM Studio format: {"models": [...]}
                models = data.get("models", [])
                for model in models:
                    # Only check LLMs, not embeddings
                    if model.get("type") != "llm":
                        continue
                    # Check if model has loaded instances
                    loaded = model.get("loaded_instances", [])
                    if loaded:
                        # Return the display name or key
                        return model.get("display_name") or model.get("key")
        except Exception:
            pass
        return None
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """
        Generate text from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            LLMResponse with generated content
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self._chat_completion(
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )
    
    def generate_structured(
        self,
        prompt: str,
        output_schema: Type[T],
        system_prompt: str | None = None,
        temperature: float | None = None
    ) -> T | None:
        """
        Generate structured output matching a Pydantic schema.
        
        Args:
            prompt: User prompt
            output_schema: Pydantic model class for expected output
            system_prompt: Optional system prompt
            temperature: Override temperature
            
        Returns:
            Parsed Pydantic model or None if parsing failed
        """
        # Build prompt with schema
        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
        
        full_system_prompt = f"""You are a structured data extraction assistant.
You must respond ONLY with valid JSON that matches the following schema:

{schema_json}

Do not include any text before or after the JSON. Do not use markdown code blocks.
If you cannot extract the requested information, use null for optional fields."""
        
        if system_prompt:
            full_system_prompt = f"{system_prompt}\n\n{full_system_prompt}"
        
        response = self.generate(
            prompt=prompt,
            system_prompt=full_system_prompt,
            temperature=temperature or 0.1,  # Lower temperature for structured output
            max_tokens=self.max_tokens
        )
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response.content)
            data = json.loads(json_str)
            return output_schema.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to parse structured output: {e}")
            logger.debug(f"Raw response: {response.content[:500]}")
            return None
    
    def _chat_completion(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Make chat completion request to OpenAI-compatible API."""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            logger.debug(f"LLM request to {url}")
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage")
            # Get actual model name from response (LM Studio includes it)
            actual_model = data.get("model", self.model)
            
            return LLMResponse(
                content=content,
                model=actual_model,
                usage=usage,
                raw_response=data
            )
            
        except httpx.TimeoutException:
            logger.error(f"LLM request timed out after {self.timeout}s")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM request failed: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"LLM request error: {e}")
            raise
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might have markdown formatting."""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start != -1 and end > start:
            return text[start:end]
        
        # Try finding array
        start = text.find("[")
        end = text.rfind("]") + 1
        
        if start != -1 and end > start:
            return text[start:end]
        
        return text.strip()
    
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        try:
            url = f"{self.base_url}/models"
            response = self.client.get(url, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def __del__(self):
        """Close HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
