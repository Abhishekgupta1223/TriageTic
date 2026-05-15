# Ticket Triage Pipeline

An AI-powered ticket triage pipeline that classifies customer support tickets, routes low-confidence cases to human review, and generates draft replies.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python main.py
python validate.py
pytest -q
```

Or via Make:

```bash
make install
make all   # run + validate + test
```

## CLI Flags

```bash
python main.py \
  --tickets tickets.json \
  --schema label_schema.json \
  --out . \
  --model claude-haiku-4-5 \
  --confidence-threshold 0.65
```

## Pipeline Stages

```
INIT → INPUTS_LOADED → TEXT_PREPROCESSED → MODEL_PROMPTED
     → STRUCTURED_OUTPUT_PARSED → CONFIDENCE_CHECKED → ROUTED
     → RESPONSE_GENERATED → RESULTS_SAVED → EVALUATION_COMPUTED
     → VALIDATION_COMPLETED
```

Each stage is enforced in code (`pipeline/stages.py`). State advances only after the previous stage completes.

## Generated Artifacts

| File | Stage |
|---|---|
| `preprocessed_tickets.json` | preprocessing |
| `llm_calls.jsonl` | every LLM call (append) |
| `llm_outputs/{stage}_{ticket_id}.json` | raw model outputs |
| `routing_decisions.json` | confidence routing |
| `triage_results.json` | final per-ticket outputs |
| `prediction_comparison.json` | per-ticket truth vs prediction |
| `evaluation_report.json` | accuracy metrics |
| `confusion_summary.json` | category confusion matrix |

## Design Decisions

**Structured output via Claude's `output_config.format`**: schema-enforced JSON eliminates most parse failures. A regex extraction + stricter-prompt retry path remains for defense-in-depth (and is exercised by tests).

**Deterministic routing**: `confidence < threshold` routes to `human_review` in code, never trusting the model's `needs_human_review` flag alone. Parse failures also route to review.

**Reproducible**: `temperature=0`, prompts hashed with sha256, all LLM calls logged with timestamps. Same inputs produce stable outputs across runs (modulo model nondeterminism).

**No fact invention in replies**: the reply-generation system prompt forbids inventing account-specific facts or promising actions not stated in the ticket.

## Validation

`validate.py` enforces every contract from the spec:

- All required artifacts exist and are valid JSON
- Every ticket received a routing decision
- Every `auto_triage` ticket has a customer reply
- Every `human_review` ticket has an internal note
- Predicted labels belong to the allowed schema
- Evaluation metrics are computable

Exits non-zero on any failure with a clear message.
