"""Structured classification call + JSON recovery + one stricter retry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient
from .logging_utils import CallLogger

_DECODER = json.JSONDecoder()


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Find the first balanced JSON object in `text`, ignoring any prose around it.

    Uses ``json.JSONDecoder.raw_decode`` to consume a complete JSON value starting
    at each ``{`` candidate. Returns the first dict it can parse, or None.

    This is strictly better than a regex like ``\\{.*\\}``: the regex is greedy
    (matches the first ``{`` to the LAST ``}``) and fails on common LLM outputs
    where there are multiple ``{`` characters in surrounding prose.
    """
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = _DECODER.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            return obj
        idx = text.find("{", idx + 1)
    return None


CLASSIFY_SYSTEM = """You are a customer-support triage classifier.
Given a single customer ticket, output ONLY JSON matching the provided schema. \
Do not include explanations or prose outside the JSON object.

Rules:
- category MUST be one of the allowed categories.
- urgency MUST be one of the allowed urgency levels.
- confidence is a float in [0.0, 1.0] reflecting how certain you are about \
both category and urgency together. Be calibrated: if the message is ambiguous \
or mixes intents, lower the confidence below 0.65.
- reasoning_summary is a one-sentence justification (max 200 chars).
- needs_human_review is a hint; deterministic routing in code makes the final decision.
"""

STRICTER_RETRY_SYSTEM = (
    CLASSIFY_SYSTEM
    + "\nCRITICAL: your previous response could not be parsed. \
Return a single JSON object. No markdown, no code fences, no commentary."
)


def build_classification_schema(categories: list[str], urgency_levels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": list(categories)},
            "urgency": {"type": "string", "enum": list(urgency_levels)},
            "confidence": {"type": "number"},
            "reasoning_summary": {"type": "string"},
            "needs_human_review": {"type": "boolean"},
        },
        "required": [
            "category",
            "urgency",
            "confidence",
            "reasoning_summary",
            "needs_human_review",
        ],
        "additionalProperties": False,
    }


def build_user_prompt(
    cleaned_text: str, categories: list[str], urgency_levels: list[str]
) -> str:
    return (
        "Allowed categories: " + ", ".join(categories) + "\n"
        "Allowed urgency levels: " + ", ".join(urgency_levels) + "\n\n"
        "Customer ticket:\n" + cleaned_text
    )


@dataclass
class ParsedClassification:
    category: str
    urgency: str
    confidence: float
    reasoning_summary: str
    needs_human_review: bool
    parse_path: str  # "structured" | "regex_recovery" | "retry" | "failed"
    parse_error: str | None = None


def _validate_payload(
    payload: dict[str, Any], categories: list[str], urgency_levels: list[str]
) -> ParsedClassification | str:
    """Returns ParsedClassification on success, or an error string."""
    for key in ("category", "urgency", "confidence", "reasoning_summary", "needs_human_review"):
        if key not in payload:
            return f"missing field: {key}"
    if payload["category"] not in categories:
        return f"category {payload['category']!r} not in schema"
    if payload["urgency"] not in urgency_levels:
        return f"urgency {payload['urgency']!r} not in schema"
    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError):
        return f"confidence is not numeric: {payload['confidence']!r}"
    if not 0.0 <= confidence <= 1.0:
        return f"confidence out of range: {confidence}"
    if not isinstance(payload["reasoning_summary"], str):
        return "reasoning_summary must be a string"
    if not isinstance(payload["needs_human_review"], bool):
        return "needs_human_review must be a boolean"
    return ParsedClassification(
        category=payload["category"],
        urgency=payload["urgency"],
        confidence=confidence,
        reasoning_summary=payload["reasoning_summary"],
        needs_human_review=payload["needs_human_review"],
        parse_path="structured",
    )


def _try_parse(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Try strict JSON, then extract the first balanced JSON object. Returns (payload, error)."""
    try:
        return json.loads(raw_text), None
    except json.JSONDecodeError as e:
        strict_err = f"strict JSON parse failed: {e}"
    obj = _extract_first_json_object(raw_text)
    if obj is not None:
        return obj, None
    return None, strict_err + "; no parseable JSON object found in output"


def classify_ticket(
    *,
    client: LLMClient,
    logger: CallLogger,
    ticket_id: str,
    cleaned_text: str,
    categories: list[str],
    urgency_levels: list[str],
) -> ParsedClassification:
    schema = build_classification_schema(categories, urgency_levels)
    user = build_user_prompt(cleaned_text, categories, urgency_levels)

    raw_text, raw_response = client.call_structured(
        system=CLASSIFY_SYSTEM, user=user, schema=schema, max_tokens=1024
    )
    logger.log(
        stage="classification",
        ticket_id=ticket_id,
        provider=client.provider,
        model=client.model,
        prompt=CLASSIFY_SYSTEM + "\n---\n" + user,
        raw_output=raw_response,
    )

    payload, parse_err = _try_parse(raw_text)
    if payload is not None:
        result = _validate_payload(payload, categories, urgency_levels)
        if isinstance(result, ParsedClassification):
            # If we needed regex extraction (extra text around JSON), note it.
            try:
                json.loads(raw_text)
            except json.JSONDecodeError:
                result.parse_path = "regex_recovery"
            return result
        parse_err = f"schema validation failed: {result}"

    # Recovery: one stricter retry.
    raw_text2, raw_response2 = client.call_structured(
        system=STRICTER_RETRY_SYSTEM, user=user, schema=schema, max_tokens=1024
    )
    logger.log(
        stage="classification",
        ticket_id=ticket_id,
        provider=client.provider,
        model=client.model,
        prompt=STRICTER_RETRY_SYSTEM + "\n---\n" + user,
        raw_output=raw_response2,
        suffix="retry",
    )
    payload2, parse_err2 = _try_parse(raw_text2)
    if payload2 is not None:
        result2 = _validate_payload(payload2, categories, urgency_levels)
        if isinstance(result2, ParsedClassification):
            result2.parse_path = "retry"
            return result2
        parse_err2 = f"schema validation failed on retry: {result2}"

    return ParsedClassification(
        category="",
        urgency="",
        confidence=0.0,
        reasoning_summary="",
        needs_human_review=True,
        parse_path="failed",
        parse_error=f"{parse_err} | retry: {parse_err2}",
    )
