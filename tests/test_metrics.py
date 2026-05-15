"""Evaluation metric computation against known fixtures."""

from pipeline.evaluate import confusion_summary, evaluation_report, per_ticket_comparison


def test_per_ticket_comparison_matches_expected():
    tickets = [
        {"ticket_id": "T1", "expected_category": "billing", "expected_urgency": "high"},
        {"ticket_id": "T2", "expected_category": "login_access", "expected_urgency": "medium"},
    ]
    predictions = {
        "T1": {"category": "billing", "urgency": "high"},        # both correct
        "T2": {"category": "billing", "urgency": "medium"},      # category wrong
    }
    comparison = per_ticket_comparison(tickets, predictions)
    by_id = {c["ticket_id"]: c for c in comparison}
    assert by_id["T1"]["category_match"] is True
    assert by_id["T1"]["urgency_match"] is True
    assert by_id["T2"]["category_match"] is False
    assert by_id["T2"]["urgency_match"] is True


def test_evaluation_report_computes_accuracy():
    tickets = [
        {"ticket_id": "T1", "expected_category": "billing", "expected_urgency": "high"},
        {"ticket_id": "T2", "expected_category": "verification", "expected_urgency": "medium"},
        {"ticket_id": "T3", "expected_category": "billing", "expected_urgency": "low"},
    ]
    predictions = {
        "T1": {"category": "billing", "urgency": "high"},        # correct/correct
        "T2": {"category": "billing", "urgency": "medium"},      # wrong/correct
        "T3": {"category": "billing", "urgency": "high"},        # correct/wrong
    }
    comparison = per_ticket_comparison(tickets, predictions)
    routing = [
        {"ticket_id": "T1", "route": "auto_triage", "confidence": 0.9, "routing_reason": ""},
        {"ticket_id": "T2", "route": "human_review", "confidence": 0.3, "routing_reason": ""},
        {"ticket_id": "T3", "route": "auto_triage", "confidence": 0.8, "routing_reason": ""},
    ]
    report = evaluation_report(comparison, routing, parse_failures=0)
    assert report["total_tickets"] == 3
    assert report["category_accuracy"] == 2 / 3
    assert report["urgency_accuracy"] == 2 / 3
    assert report["human_review_count"] == 1
    assert report["parse_validation_failures"] == 0


def test_evaluation_handles_missing_expected_fields():
    """Tickets without expected_* should not be counted in accuracy."""
    tickets = [
        {"ticket_id": "T1"},  # no expected fields
        {"ticket_id": "T2", "expected_category": "billing", "expected_urgency": "high"},
    ]
    predictions = {
        "T1": {"category": "other", "urgency": "low"},
        "T2": {"category": "billing", "urgency": "high"},
    }
    comparison = per_ticket_comparison(tickets, predictions)
    routing = [
        {"ticket_id": "T1", "route": "auto_triage", "confidence": 0.9, "routing_reason": ""},
        {"ticket_id": "T2", "route": "auto_triage", "confidence": 0.9, "routing_reason": ""},
    ]
    report = evaluation_report(comparison, routing, parse_failures=0)
    assert report["category_compared"] == 1
    assert report["category_accuracy"] == 1.0


def test_confusion_summary_groups_mispredictions():
    tickets = [
        {"ticket_id": "T1", "expected_category": "billing", "expected_urgency": "high"},
        {"ticket_id": "T2", "expected_category": "billing", "expected_urgency": "high"},
        {"ticket_id": "T3", "expected_category": "verification", "expected_urgency": "low"},
    ]
    predictions = {
        "T1": {"category": "billing", "urgency": "high"},
        "T2": {"category": "verification", "urgency": "high"},  # billing -> verification
        "T3": {"category": "other", "urgency": "low"},          # verification -> other
    }
    comparison = per_ticket_comparison(tickets, predictions)
    confusion = confusion_summary(comparison)
    assert confusion["matrix"]["billing"]["verification"] == 1
    assert confusion["matrix"]["billing"]["billing"] == 1
    assert confusion["matrix"]["verification"]["other"] == 1
    # All non-diagonal pairs surface in most_confused
    pairs = {(p["expected"], p["predicted"]) for p in confusion["most_confused_pairs"]}
    assert ("billing", "verification") in pairs
    assert ("verification", "other") in pairs
