"""Anthropic client wrapper. Returns raw text content for structured parsing."""

from __future__ import annotations

import os
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5"
PROVIDER = "anthropic"


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "paste your key, or export ANTHROPIC_API_KEY in your shell."
            )
        self.client = anthropic.Anthropic()
        self.model = model
        self.provider = PROVIDER

    def call_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Call with `output_config.format` schema enforcement.

        Returns (raw_text, response_dict). Caller parses raw_text as JSON.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = _extract_text(response)
        return text, _response_summary(response)

    def call_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
    ) -> tuple[str, dict[str, Any]]:
        """Plain text call for reply / internal-note generation."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = _extract_text(response)
        return text, _response_summary(response)


def _extract_text(response: Any) -> str:
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def _response_summary(response: Any) -> dict[str, Any]:
    """Compact serializable summary of the response for log artifacts."""
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "stop_reason": getattr(response, "stop_reason", None),
        "usage": {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        if getattr(response, "usage", None)
        else None,
        "content": [
            {"type": b.type, "text": getattr(b, "text", None)}
            for b in response.content
        ],
    }
