"""End-to-end ticket triage pipeline.

Stages (enforced by `pipeline.stages.StageTracker`):
  INIT -> INPUTS_LOADED -> TEXT_PREPROCESSED -> MODEL_PROMPTED
       -> STRUCTURED_OUTPUT_PARSED -> CONFIDENCE_CHECKED -> ROUTED
       -> RESPONSE_GENERATED -> RESULTS_SAVED -> EVALUATION_COMPUTED
       -> VALIDATION_COMPLETED
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from the project root before any module reads env vars.
load_dotenv()

from pipeline.classify import ParsedClassification, classify_ticket
from pipeline.evaluate import confusion_summary, evaluation_report, per_ticket_comparison
from pipeline.llm import DEFAULT_MODEL, LLMClient, LLMError
from pipeline.loader import load_schema, load_tickets
from pipeline.logging_utils import CallLogger
from pipeline.preprocess import preprocess_all
from pipeline.reply import generate_customer_reply, generate_internal_note
from pipeline.routing import DEFAULT_CONFIDENCE_THRESHOLD, route
from pipeline.stages import PipelineStage, StageTracker


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI ticket triage pipeline")
    p.add_argument("--tickets", type=Path, default=Path("tickets.json"))
    p.add_argument("--schema", type=Path, default=Path("label_schema.json"))
    p.add_argument("--out", type=Path, default=Path("."), help="Output directory")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Tickets below this confidence route to human_review",
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    tracker = StageTracker()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "llm_outputs"
    logger = CallLogger(out_dir / "llm_calls.jsonl", log_dir)

    # Stage: INPUTS_LOADED
    tickets = load_tickets(args.tickets)
    schema = load_schema(args.schema)
    tracker.advance(PipelineStage.INPUTS_LOADED, f"loaded {len(tickets)} tickets")
    print(f"[load] {len(tickets)} tickets, {len(schema['categories'])} categories", flush=True)

    # Stage: TEXT_PREPROCESSED
    preprocessed = preprocess_all(tickets)
    _write_json(out_dir / "preprocessed_tickets.json", preprocessed)
    tracker.advance(PipelineStage.TEXT_PREPROCESSED, "preprocessed_tickets.json written")
    print(f"[preprocess] wrote preprocessed_tickets.json", flush=True)

    # Stage: MODEL_PROMPTED + STRUCTURED_OUTPUT_PARSED + CONFIDENCE_CHECKED + ROUTED
    try:
        client = LLMClient(model=args.model)
    except LLMError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    classifications: dict[str, ParsedClassification] = {}
    routing_decisions: list[dict[str, Any]] = []
    parse_failures = 0

    text_by_id = {p["ticket_id"]: p["cleaned_text"] for p in preprocessed}

    for p in preprocessed:
        tid = p["ticket_id"]
        cleaned = p["cleaned_text"]
        print(f"[classify] {tid}", flush=True)
        parsed = classify_ticket(
            client=client,
            logger=logger,
            ticket_id=tid,
            cleaned_text=cleaned,
            categories=schema["categories"],
            urgency_levels=schema["urgency_levels"],
        )
        classifications[tid] = parsed
        if parsed.parse_path == "failed":
            parse_failures += 1
            print(f"[classify] {tid}: PARSE FAILED -> {parsed.parse_error}", flush=True)
        elif parsed.parse_path != "structured":
            print(f"[classify] {tid}: recovered via {parsed.parse_path}", flush=True)

    tracker.advance(PipelineStage.MODEL_PROMPTED, "classification calls complete")
    tracker.advance(PipelineStage.STRUCTURED_OUTPUT_PARSED, f"{parse_failures} parse failures")

    decisions = {}
    for tid, parsed in classifications.items():
        d = route(tid, parsed, threshold=args.confidence_threshold)
        decisions[tid] = d
        routing_decisions.append(asdict(d))

    _write_json(out_dir / "routing_decisions.json", routing_decisions)
    tracker.advance(PipelineStage.CONFIDENCE_CHECKED, "deterministic routing applied")
    tracker.advance(PipelineStage.ROUTED, "routing_decisions.json written")
    auto_count = sum(1 for r in routing_decisions if r["route"] == "auto_triage")
    review_count = len(routing_decisions) - auto_count
    print(f"[route] {auto_count} auto_triage, {review_count} human_review", flush=True)

    # Stage: RESPONSE_GENERATED
    triage_results: list[dict[str, Any]] = []
    for tid, parsed in classifications.items():
        decision = decisions[tid]
        cleaned = text_by_id[tid]
        customer_reply = None
        internal_note = None
        if decision.route == "auto_triage":
            print(f"[reply] {tid}: customer reply", flush=True)
            customer_reply = generate_customer_reply(
                client=client,
                logger=logger,
                ticket_id=tid,
                cleaned_text=cleaned,
                parsed=parsed,
            )
        else:
            print(f"[reply] {tid}: internal note", flush=True)
            internal_note = generate_internal_note(
                client=client,
                logger=logger,
                ticket_id=tid,
                cleaned_text=cleaned,
                parsed=parsed,
                decision=decision,
            )
        triage_results.append(
            {
                "ticket_id": tid,
                "predicted_category": parsed.category or None,
                "predicted_urgency": parsed.urgency or None,
                "confidence": decision.confidence,
                "route": decision.route,
                "customer_reply": customer_reply,
                "internal_note": internal_note,
            }
        )
    tracker.advance(PipelineStage.RESPONSE_GENERATED, "replies and notes generated")

    # Stage: RESULTS_SAVED
    _write_json(out_dir / "triage_results.json", triage_results)
    tracker.advance(PipelineStage.RESULTS_SAVED, "triage_results.json written")

    # Stage: EVALUATION_COMPUTED
    predictions = {
        tid: {"category": p.category, "urgency": p.urgency}
        for tid, p in classifications.items()
        if p.parse_path != "failed"
    }
    comparison = per_ticket_comparison(tickets, predictions)
    report = evaluation_report(comparison, routing_decisions, parse_failures)
    confusion = confusion_summary(comparison)
    _write_json(out_dir / "prediction_comparison.json", comparison)
    _write_json(out_dir / "evaluation_report.json", report)
    _write_json(out_dir / "confusion_summary.json", confusion)
    tracker.advance(PipelineStage.EVALUATION_COMPUTED, "metrics computed")
    print(
        f"[eval] category_accuracy={report['category_accuracy']} "
        f"urgency_accuracy={report['urgency_accuracy']} "
        f"human_review={report['human_review_count']} "
        f"parse_failures={report['parse_validation_failures']}",
        flush=True,
    )

    # Stage: VALIDATION_COMPLETED (pipeline-side sanity; full check lives in validate.py)
    tracker.advance(PipelineStage.VALIDATION_COMPLETED, "pipeline run complete")
    print("[done] all artifacts written. Run `python validate.py` to verify contracts.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as e:
        print(f"[fatal] {type(e).__name__}: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
