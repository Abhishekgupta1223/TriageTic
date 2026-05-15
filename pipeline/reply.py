"""Reply generation for auto_triage; internal-note generation for human_review."""

from __future__ import annotations

from .classify import ParsedClassification
from .llm import LLMClient
from .logging_utils import CallLogger
from .routing import RoutingDecision

REPLY_SYSTEM = """You draft short customer-support replies for an online financial-services company.

Requirements (all are hard constraints):
- 2 to 4 sentences only.
- Acknowledge the issue the customer described in their own terms.
- Be consistent with the predicted category and urgency.
- Do NOT invent account-specific facts (account balances, transaction IDs, names, dates not in the ticket).
- Do NOT promise specific actions, refunds, or timelines that are not stated in the ticket.
- Plain text only. No greeting boilerplate longer than one short opening sentence, no signatures.
"""

INTERNAL_NOTE_SYSTEM = """You write short internal notes for a human reviewer who will handle an escalated ticket.

Requirements:
- 2 to 4 sentences.
- State the predicted category and urgency, the model's confidence, and why the ticket was escalated.
- Highlight what the human should verify or ask the customer next.
- Plain text. No customer-facing language.
"""


def _reply_user_prompt(
    cleaned_text: str, parsed: ParsedClassification
) -> str:
    return (
        f"Predicted category: {parsed.category}\n"
        f"Predicted urgency: {parsed.urgency}\n"
        f"Model confidence: {parsed.confidence:.2f}\n\n"
        f"Customer ticket:\n{cleaned_text}\n\n"
        "Write the customer reply now."
    )


def _internal_note_user_prompt(
    cleaned_text: str, parsed: ParsedClassification, decision: RoutingDecision
) -> str:
    category = parsed.category or "(unparseable)"
    urgency = parsed.urgency or "(unparseable)"
    return (
        f"Predicted category: {category}\n"
        f"Predicted urgency: {urgency}\n"
        f"Model confidence: {decision.confidence:.2f}\n"
        f"Routing reason: {decision.routing_reason}\n\n"
        f"Customer ticket:\n{cleaned_text}\n\n"
        "Write the internal note now."
    )


def generate_customer_reply(
    *,
    client: LLMClient,
    logger: CallLogger,
    ticket_id: str,
    cleaned_text: str,
    parsed: ParsedClassification,
) -> str:
    user = _reply_user_prompt(cleaned_text, parsed)
    text, raw = client.call_text(system=REPLY_SYSTEM, user=user, max_tokens=400)
    logger.log(
        stage="reply_generation",
        ticket_id=ticket_id,
        provider=client.provider,
        model=client.model,
        prompt=REPLY_SYSTEM + "\n---\n" + user,
        raw_output=raw,
    )
    return text


def generate_internal_note(
    *,
    client: LLMClient,
    logger: CallLogger,
    ticket_id: str,
    cleaned_text: str,
    parsed: ParsedClassification,
    decision: RoutingDecision,
) -> str:
    user = _internal_note_user_prompt(cleaned_text, parsed, decision)
    text, raw = client.call_text(system=INTERNAL_NOTE_SYSTEM, user=user, max_tokens=400)
    logger.log(
        stage="reply_generation",
        ticket_id=ticket_id,
        provider=client.provider,
        model=client.model,
        prompt=INTERNAL_NOTE_SYSTEM + "\n---\n" + user,
        raw_output=raw,
    )
    return text
