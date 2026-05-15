"""Deterministic text preprocessing."""

import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.])\1{2,}")


def clean_text(text: str) -> str:
    """Collapse whitespace, trim, and normalize runs of repeated terminal punctuation.

    Deterministic: same input -> same output.
    """
    if not isinstance(text, str):
        raise TypeError("customer_message must be a string")
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    normalized = _REPEATED_PUNCT_RE.sub(r"\1\1\1", collapsed)
    return normalized


def preprocess_ticket(ticket: dict[str, Any]) -> dict[str, Any]:
    original = ticket["customer_message"]
    cleaned = clean_text(original)
    return {
        "ticket_id": ticket["ticket_id"],
        "original_text": original,
        "cleaned_text": cleaned,
        "char_count": len(cleaned),
        "word_count": len(cleaned.split()) if cleaned else 0,
    }


def preprocess_all(tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [preprocess_ticket(t) for t in tickets]
