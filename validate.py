"""Standalone validation. Checks every contract from the spec.

Usage:
    python validate.py
    python validate.py --out /path/to/outputs

Exits non-zero with a clear message on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_ARTIFACTS = [
    "tickets.json",
    "label_schema.json",
    "preprocessed_tickets.json",
    "routing_decisions.json",
    "triage_results.json",
    "prediction_comparison.json",
    "evaluation_report.json",
    "llm_calls.jsonl",
]

OPTIONAL_ARTIFACTS = ["confusion_summary.json"]

ALLOWED_ROUTES = {"auto_triage", "human_review"}


class ValidationError(Exception):
    pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValidationError(f"{path}: invalid JSON: {e}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValidationError(f"{path}:{lineno}: invalid JSON: {e}")
    return records


def validate(out_dir: Path) -> list[str]:
    """Run all checks. Returns a list of human-readable check results."""
    results: list[str] = []

    def ok(msg: str) -> None:
        results.append(f"  OK   {msg}")

    def fail(msg: str) -> None:
        raise ValidationError(msg)

    # 1. Required artifacts exist
    for name in REQUIRED_ARTIFACTS:
        if not (out_dir / name).is_file():
            fail(f"missing required artifact: {name}")
    ok("all required artifacts present")

    # 2. Inputs and schema
    schema = _read_json(out_dir / "label_schema.json")
    categories = set(schema["categories"])
    urgency_levels = set(schema["urgency_levels"])
    tickets = _read_json(out_dir / "tickets.json")
    ticket_ids = {t["ticket_id"] for t in tickets}
    ok(f"loaded {len(tickets)} tickets, {len(categories)} categories, {len(urgency_levels)} urgency levels")

    # 3. Preprocessed tickets have required fields
    preprocessed = _read_json(out_dir / "preprocessed_tickets.json")
    if {p["ticket_id"] for p in preprocessed} != ticket_ids:
        fail("preprocessed_tickets.json: ticket_id set does not match tickets.json")
    for p in preprocessed:
        for k in ("ticket_id", "original_text", "cleaned_text", "char_count", "word_count"):
            if k not in p:
                fail(f"preprocessed_tickets.json: ticket {p.get('ticket_id')} missing field {k}")
    ok("preprocessed_tickets.json schema is valid")

    # 4. All tickets received a routing decision; routes are valid
    routing = _read_json(out_dir / "routing_decisions.json")
    routed_ids = {r["ticket_id"] for r in routing}
    if routed_ids != ticket_ids:
        missing = ticket_ids - routed_ids
        extra = routed_ids - ticket_ids
        fail(f"routing_decisions.json: missing={missing}, extra={extra}")
    for r in routing:
        if r["route"] not in ALLOWED_ROUTES:
            fail(f"routing_decisions.json: invalid route {r['route']!r} for {r['ticket_id']}")
        for k in ("ticket_id", "route", "confidence", "routing_reason"):
            if k not in r:
                fail(f"routing_decisions.json: ticket {r.get('ticket_id')} missing field {k}")
    ok(f"all {len(routing)} tickets routed; routes ∈ {ALLOWED_ROUTES}")

    # 5. Triage results: route-specific output presence + label schema membership
    triage = _read_json(out_dir / "triage_results.json")
    if {t["ticket_id"] for t in triage} != ticket_ids:
        fail("triage_results.json: ticket_id set does not match tickets.json")
    for t in triage:
        tid = t["ticket_id"]
        route = t.get("route")
        if route not in ALLOWED_ROUTES:
            fail(f"triage_results.json[{tid}]: invalid route {route!r}")
        if route == "auto_triage":
            if not t.get("customer_reply"):
                fail(f"triage_results.json[{tid}]: auto_triage ticket missing customer_reply")
            if t.get("internal_note"):
                fail(f"triage_results.json[{tid}]: auto_triage ticket should not have internal_note")
        elif route == "human_review":
            if not t.get("internal_note"):
                fail(f"triage_results.json[{tid}]: human_review ticket missing internal_note")
            if t.get("customer_reply"):
                fail(f"triage_results.json[{tid}]: human_review ticket should not have customer_reply")
        # Labels must be in the schema (when present — failed-parse tickets may have null)
        cat = t.get("predicted_category")
        urg = t.get("predicted_urgency")
        if cat is not None and cat not in categories:
            fail(f"triage_results.json[{tid}]: predicted_category {cat!r} not in label schema")
        if urg is not None and urg not in urgency_levels:
            fail(f"triage_results.json[{tid}]: predicted_urgency {urg!r} not in label schema")
    ok("auto_triage tickets have customer_reply; human_review tickets have internal_note")
    ok("all predicted labels are in the allowed schema (or null for unparseable)")

    # 6. Prediction comparison
    comparison = _read_json(out_dir / "prediction_comparison.json")
    if {c["ticket_id"] for c in comparison} != ticket_ids:
        fail("prediction_comparison.json: ticket_id set does not match tickets.json")
    ok("prediction_comparison.json has one record per ticket")

    # 7. Evaluation metrics
    report = _read_json(out_dir / "evaluation_report.json")
    for k in (
        "total_tickets",
        "category_accuracy",
        "urgency_accuracy",
        "human_review_count",
        "parse_validation_failures",
    ):
        if k not in report:
            fail(f"evaluation_report.json: missing key {k}")
    if report["total_tickets"] != len(tickets):
        fail(
            f"evaluation_report.json: total_tickets={report['total_tickets']} "
            f"!= len(tickets)={len(tickets)}"
        )
    ok(
        "evaluation_report.json computed "
        f"(cat={report['category_accuracy']}, urg={report['urgency_accuracy']}, "
        f"review={report['human_review_count']}, parse_fail={report['parse_validation_failures']})"
    )

    # 8. LLM call log
    calls = _read_jsonl(out_dir / "llm_calls.jsonl")
    if not calls:
        fail("llm_calls.jsonl: no records (pipeline must log every LLM call)")
    for i, call in enumerate(calls):
        for k in (
            "stage",
            "ticket_id",
            "timestamp",
            "provider",
            "model",
            "prompt_hash",
            "output_artifact",
        ):
            if k not in call:
                fail(f"llm_calls.jsonl[{i}]: missing field {k}")
        # Resolve against out_dir so relative paths stored by main.py work
        # regardless of where validate.py is invoked from.
        artifact_str = call["output_artifact"]
        artifact = (out_dir / artifact_str) if not Path(artifact_str).is_absolute() else Path(artifact_str)
        if not artifact.is_file():
            fail(f"llm_calls.jsonl[{i}]: output_artifact does not exist: {artifact_str}")
    # Stage values must match the spec exactly: "classification | reply_generation".
    stages_seen = {c["stage"] for c in calls}
    allowed = {"classification", "reply_generation"}
    if not stages_seen.issubset(allowed):
        fail(f"llm_calls.jsonl: stages {stages_seen - allowed} are not in spec-allowed set {allowed}")
    if not allowed.issubset(stages_seen):
        fail(f"llm_calls.jsonl: expected to see all of {allowed}, missing {allowed - stages_seen}")
    ok(f"llm_calls.jsonl has {len(calls)} records with valid artifacts; stages seen={sorted(stages_seen)}")

    # 9. Optional: confusion summary
    confusion_path = out_dir / "confusion_summary.json"
    if confusion_path.is_file():
        confusion = _read_json(confusion_path)
        if "matrix" not in confusion or "most_confused_pairs" not in confusion:
            fail("confusion_summary.json: missing matrix or most_confused_pairs")
        ok("confusion_summary.json present and well-formed")
    else:
        results.append("  SKIP confusion_summary.json (optional)")

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    print(f"Validating outputs in: {args.out.resolve()}")
    try:
        for line in validate(args.out):
            print(line)
    except ValidationError as e:
        print(f"  FAIL {e}", file=sys.stderr)
        print("\nVALIDATION FAILED", file=sys.stderr)
        return 1
    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
