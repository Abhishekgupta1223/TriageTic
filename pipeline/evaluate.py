"""Evaluation metrics and confusion summary."""

from collections import defaultdict
from typing import Any


def per_ticket_comparison(
    tickets: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One record per ticket, comparing expected vs predicted."""
    comparison = []
    for t in tickets:
        tid = t["ticket_id"]
        pred = predictions.get(tid, {})
        comparison.append(
            {
                "ticket_id": tid,
                "expected_category": t.get("expected_category"),
                "expected_urgency": t.get("expected_urgency"),
                "predicted_category": pred.get("category"),
                "predicted_urgency": pred.get("urgency"),
                "category_match": (
                    t.get("expected_category") == pred.get("category")
                    if t.get("expected_category") is not None
                    else None
                ),
                "urgency_match": (
                    t.get("expected_urgency") == pred.get("urgency")
                    if t.get("expected_urgency") is not None
                    else None
                ),
            }
        )
    return comparison


def evaluation_report(
    comparison: list[dict[str, Any]],
    routing_decisions: list[dict[str, Any]],
    parse_failures: int,
) -> dict[str, Any]:
    cat_compared = [c for c in comparison if c["category_match"] is not None]
    urg_compared = [c for c in comparison if c["urgency_match"] is not None]

    cat_correct = sum(1 for c in cat_compared if c["category_match"])
    urg_correct = sum(1 for c in urg_compared if c["urgency_match"])

    cat_accuracy = (cat_correct / len(cat_compared)) if cat_compared else None
    urg_accuracy = (urg_correct / len(urg_compared)) if urg_compared else None

    human_review_count = sum(1 for d in routing_decisions if d["route"] == "human_review")

    return {
        "total_tickets": len(comparison),
        "category_accuracy": cat_accuracy,
        "urgency_accuracy": urg_accuracy,
        "category_correct": cat_correct,
        "category_compared": len(cat_compared),
        "urgency_correct": urg_correct,
        "urgency_compared": len(urg_compared),
        "human_review_count": human_review_count,
        "parse_validation_failures": parse_failures,
    }


def confusion_summary(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts of (expected_category, predicted_category) pairs."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in comparison:
        exp = c.get("expected_category")
        pred = c.get("predicted_category")
        if exp is None:
            continue
        matrix[exp][pred or "<unparseable>"] += 1

    most_confused: list[dict[str, Any]] = []
    for exp, preds in matrix.items():
        for pred, count in preds.items():
            if exp != pred:
                most_confused.append({"expected": exp, "predicted": pred, "count": count})
    most_confused.sort(key=lambda r: (-r["count"], r["expected"], r["predicted"]))

    return {
        "matrix": {exp: dict(preds) for exp, preds in matrix.items()},
        "most_confused_pairs": most_confused,
    }
