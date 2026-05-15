"""Routing threshold: 0.65 is the boundary; failures route to human_review."""

from pipeline.classify import ParsedClassification
from pipeline.routing import route


def _parsed(confidence: float, path: str = "structured") -> ParsedClassification:
    return ParsedClassification(
        category="billing",
        urgency="high",
        confidence=confidence,
        reasoning_summary="x",
        needs_human_review=False,
        parse_path=path,
    )


def test_below_threshold_routes_to_human_review():
    d = route("T1", _parsed(0.64), threshold=0.65)
    assert d.route == "human_review"
    assert d.confidence == 0.64
    assert "below threshold" in d.routing_reason


def test_at_threshold_routes_to_auto_triage():
    d = route("T1", _parsed(0.65), threshold=0.65)
    assert d.route == "auto_triage"
    assert "at or above" in d.routing_reason


def test_above_threshold_routes_to_auto_triage():
    d = route("T1", _parsed(0.95), threshold=0.65)
    assert d.route == "auto_triage"


def test_failed_parse_routes_to_human_review_regardless_of_confidence():
    parsed = ParsedClassification(
        category="",
        urgency="",
        confidence=0.99,  # would be auto_triage if it had parsed
        reasoning_summary="",
        needs_human_review=True,
        parse_path="failed",
        parse_error="schema validation failed",
    )
    d = route("T1", parsed, threshold=0.65)
    assert d.route == "human_review"
    assert d.confidence == 0.0
    assert "unparseable" in d.routing_reason


def test_routing_ignores_model_needs_human_review_flag():
    """Routing is deterministic in code — model's needs_human_review must not override."""
    parsed = ParsedClassification(
        category="billing",
        urgency="high",
        confidence=0.90,
        reasoning_summary="x",
        needs_human_review=True,  # model wants review
        parse_path="structured",
    )
    d = route("T1", parsed, threshold=0.65)
    assert d.route == "auto_triage"  # code overrides model
