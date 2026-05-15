"""Deterministic confidence routing. Code-only — never trusts the model's flag."""

from dataclasses import dataclass

from .classify import ParsedClassification

DEFAULT_CONFIDENCE_THRESHOLD = 0.65


@dataclass
class RoutingDecision:
    ticket_id: str
    route: str  # "auto_triage" | "human_review"
    confidence: float
    routing_reason: str


def route(
    ticket_id: str,
    parsed: ParsedClassification,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> RoutingDecision:
    if parsed.parse_path == "failed":
        return RoutingDecision(
            ticket_id=ticket_id,
            route="human_review",
            confidence=0.0,
            routing_reason=f"classification_unparseable: {parsed.parse_error}",
        )
    if parsed.confidence < threshold:
        return RoutingDecision(
            ticket_id=ticket_id,
            route="human_review",
            confidence=parsed.confidence,
            routing_reason=f"confidence {parsed.confidence:.3f} below threshold {threshold}",
        )
    return RoutingDecision(
        ticket_id=ticket_id,
        route="auto_triage",
        confidence=parsed.confidence,
        routing_reason=f"confidence {parsed.confidence:.3f} at or above threshold {threshold}",
    )
