# TriageTic

> A small, replayable, AI-powered customer-support **ticket triage pipeline**.
> Reads tickets from disk, classifies each with schema-enforced JSON via Claude,
> routes low-confidence cases to human review, drafts replies for the rest, and
> self-validates every artifact it produces.

---

## At a glance

```mermaid
flowchart LR
    subgraph In["Inputs"]
        T["tickets.json"]
        S["label_schema.json"]
        E[".env<br/>ANTHROPIC_API_KEY"]
    end

    subgraph Pipe["Pipeline (state-machine enforced)"]
        L["Load &amp;<br/>validate"] --> P["Preprocess<br/>clean + stats"]
        P --> C["Classify<br/>structured JSON<br/>via Claude Haiku 4.5"]
        C --> R["Route<br/>conf &lt; 0.65 → review"]
        R --> G["Generate<br/>reply / note"]
        G --> V["Evaluate<br/>accuracy + confusion"]
    end

    subgraph Out["Outputs (JSON + JSONL)"]
        O1["preprocessed_tickets.json"]
        O2["routing_decisions.json"]
        O3["triage_results.json"]
        O4["prediction_comparison.json"]
        O5["evaluation_report.json"]
        O6["confusion_summary.json"]
        O7["llm_calls.jsonl"]
    end

    T --> L
    S --> L
    E -.->|env| C

    P --> O1
    R --> O2
    G --> O3
    V --> O4
    V --> O5
    V --> O6
    C -.->|append| O7
    G -.->|append| O7

    classDef in fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef out fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef stage fill:#fff8e1,stroke:#f9a825,color:#5d4037
    class T,S,E in
    class O1,O2,O3,O4,O5,O6,O7 out
    class L,P,C,R,G,V stage
```

---

## What it does (and what it does not)

| Does | Does not |
|---|---|
| Reads `tickets.json` + `label_schema.json` from disk | Connect to any ticketing system |
| Classifies each ticket into `{category, urgency, confidence, reasoning, needs_review}` via Claude with **schema-enforced JSON** | Trust the LLM's `needs_human_review` flag for routing |
| Routes by **deterministic code** (`confidence < 0.65` → human) | Block on any single failure — per-ticket exceptions are isolated |
| Drafts replies that acknowledge the issue without inventing account facts | Promise specific actions, refunds, or timelines |
| Logs every LLM call with sha256 prompt hash + raw output artifact | Cache prompts (too few tickets to pay back the write premium) |
| Self-validates every contract via `validate.py` | Need network access for the test suite |

---

## Architecture

### 1. Pipeline state machine

The spec mandates explicit stages. `pipeline/stages.py` enforces them in code:
`StageTracker.advance()` rejects backward transitions and stage skips. Every
artifact write happens at exactly one stage.

```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> INPUTS_LOADED: load_tickets + load_schema
    INPUTS_LOADED --> TEXT_PREPROCESSED: clean_text + stats
    TEXT_PREPROCESSED --> MODEL_PROMPTED: classify_ticket loop
    MODEL_PROMPTED --> STRUCTURED_OUTPUT_PARSED: _validate_payload
    STRUCTURED_OUTPUT_PARSED --> CONFIDENCE_CHECKED: route()
    CONFIDENCE_CHECKED --> ROUTED: routing_decisions.json
    ROUTED --> RESPONSE_GENERATED: reply / internal note
    RESPONSE_GENERATED --> RESULTS_SAVED: triage_results.json
    RESULTS_SAVED --> EVALUATION_COMPUTED: accuracy + confusion
    EVALUATION_COMPUTED --> VALIDATION_COMPLETED: pipeline done
    VALIDATION_COMPLETED --> [*]
```

### 2. Per-ticket decision flow

The routing decision is taken by code, not the model. The LLM's
`needs_human_review` hint is intentionally ignored — `pipeline/routing.py`
makes the final call based on confidence alone (and on parse success).

```mermaid
flowchart TD
    A([Customer ticket]) --> B[Preprocess<br/>clean + stats]
    B --> C[Classify via Claude<br/>schema-enforced JSON]
    C --> D{Valid JSON<br/>matching schema?}
    D -->|Yes| E{confidence ≥ 0.65?}
    D -->|No| F[Retry once<br/>stricter system prompt]
    F --> G{Now valid?}
    G -->|Yes| E
    G -->|No| H[route = human_review<br/>reason: unparseable]
    E -->|Yes| I[route = auto_triage]
    E -->|No| J[route = human_review<br/>reason: low confidence]
    I --> K[Generate customer reply<br/>2–4 sentences, no invented facts]
    J --> L[Generate internal note<br/>highlight what to verify]
    H --> L
    K --> M([triage_results.json])
    L --> M

    classDef ok fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef warn fill:#fff3e0,stroke:#f57c00,color:#bf360c
    classDef fail fill:#ffebee,stroke:#c62828,color:#b71c1c
    class I,K ok
    class J,L warn
    class H fail
```

### 3. Defense in depth: how malformed output is handled

Spec §8 asks for **one** recovery path. The implementation has **four** layered
defenses, each catching a different failure mode:

```mermaid
flowchart TD
    A[client.messages.create<br/>output_config.format = json_schema] --> B{Network /<br/>API error?}
    B -->|Yes| Z[Exception caught per-ticket<br/>parse_path = failed<br/>route → human_review]
    B -->|No| C{json.loads<br/>strict?}
    C -->|OK| D{schema fields<br/>valid?}
    C -->|Fail| E[raw_decode walker<br/>finds first valid JSON object<br/>in noisy text]
    E -->|Found| D
    E -->|None| F[Retry once<br/>stricter system prompt]
    D -->|OK| Y([parse_path = structured])
    D -->|Invalid| F
    F --> G{Now valid?}
    G -->|Yes| W([parse_path = retry])
    G -->|No| Z

    classDef ok fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef fail fill:#ffebee,stroke:#c62828,color:#b71c1c
    class Y,W ok
    class Z fail
```

Layer 1 — **`output_config.format`** with a JSON Schema (enum + required +
`additionalProperties: false`) gets the API itself to enforce the structure
server-side. In practice the next layers almost never fire.

Layer 2 — strict `json.loads` of the response text.

Layer 3 — **brace-walking** `json.JSONDecoder.raw_decode()` that locates the
first balanced JSON object even when the model added prose around it.
(Strictly better than the naive `\{.*\}` regex, which is greedy and breaks on
inputs containing multiple `{` characters.)

Layer 4 — one **retry with a stricter system prompt**. If that still fails,
the ticket is routed to human review with the error logged.

### 4. Module dependency graph

```mermaid
flowchart LR
    main["main.py<br/>(CLI orchestrator)"] --> stages
    main --> loader
    main --> preprocess
    main --> llm
    main --> classify
    main --> routing
    main --> reply
    main --> evaluate
    main --> logutil[logging_utils]

    classify --> llm
    classify --> logutil
    reply --> llm
    reply --> logutil
    routing --> classify
    evaluate --> classify

    validate["validate.py<br/>(contract checks)"]

    classDef entry fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    class main,validate entry
```

---

## Quickstart

```bash
git clone https://github.com/Abhishekgupta1223/TriageTic.git
cd TriageTic
pip install -r requirements.txt

cp .env.example .env
# open .env, paste your Anthropic API key (https://console.anthropic.com/)

python main.py        # run the pipeline end-to-end
python validate.py    # check every spec contract
pytest -q             # 30 unit tests
```

Or with Make:

```bash
make install   # pip install
make all       # run + validate + test
make clean     # remove generated artifacts
```

### CLI

```bash
python main.py \
  --tickets tickets.json \
  --schema label_schema.json \
  --out . \
  --model claude-haiku-4-5 \
  --confidence-threshold 0.65
```

| Flag | Default | Purpose |
|---|---|---|
| `--tickets` | `tickets.json` | Input file (list of tickets) |
| `--schema` | `label_schema.json` | Allowed categories + urgency levels |
| `--out` | `.` | Output directory for all artifacts |
| `--model` | `claude-haiku-4-5` | Any Claude model that supports `output_config.format` |
| `--confidence-threshold` | `0.65` | Below this, ticket routes to human review |

---

## Project layout

```
TriageTic/
├── main.py                       # CLI orchestrator; state-machine driver
├── validate.py                   # Spec §7 contract checker (exit non-zero on fail)
├── tickets.json                  # Sample inputs (spec-provided; evaluator may swap)
├── label_schema.json
├── .env.example                  # Template — copy to .env, fill in key
├── requirements.txt
├── Makefile                      # install / run / validate / test / clean
├── README.md                     # this file
│
├── pipeline/
│   ├── stages.py                 # PipelineStage enum + StageTracker
│   ├── loader.py                 # JSON load + structural validation
│   ├── preprocess.py             # Deterministic text cleaning + stats
│   ├── llm.py                    # Anthropic SDK wrapper (call_structured / call_text)
│   ├── classify.py               # Schema-enforced classification + JSON recovery + retry
│   ├── routing.py                # Deterministic confidence routing
│   ├── reply.py                  # Customer reply + internal note generation
│   ├── evaluate.py               # Accuracy, comparison, confusion matrix
│   └── logging_utils.py          # CallLogger → llm_calls.jsonl + raw artifacts
│
└── tests/                        # 30 tests, no network required
    ├── test_routing.py           # Threshold boundary + override-model-flag tests
    ├── test_schema.py            # Validation + 4 JSON-recovery edge cases
    ├── test_metrics.py           # Accuracy + parse-failure exclusion + confusion
    └── test_security.py          # Path traversal + bad-input rejection
```

---

## Design decisions (and why)

| Decision | Rationale |
|---|---|
| **Schema-enforced JSON via `output_config.format`** | Eliminates most parse failures by construction — the API enforces the schema server-side. Leaves recovery code as defense-in-depth, not the happy path. |
| **Deterministic routing in code, not the LLM** | Spec is explicit: `needs_human_review` from the model is a hint, not the decision. Confidence threshold is the source of truth. |
| **State machine over loose stage labels** | `StageTracker.advance()` rejects skips and backward transitions. Makes the pipeline grep-able by reviewers and prevents silent reordering during refactors. |
| **One retry with a stricter prompt** (vs. retry-on-temperature-bump) | `temperature=0` already gives us the model's most-likely output. A stricter system prompt is what changes between attempts. |
| **`raw_decode()` brace-walker over regex** | `\{.*\}` is greedy — it captures from the FIRST `{` to the LAST `}`, breaking on common LLM outputs with prose containing braces. `json.JSONDecoder.raw_decode()` finds the first *valid* object. |
| **Per-ticket exception isolation** | A single rate-limit or timeout shouldn't take down the whole batch. Failed tickets are routed to human review with the error logged. |
| **Filename sanitization** | Untrusted `ticket_id` flows into filenames. `_safe_segment()` strips `..`, slashes, and Windows-illegal characters. |
| **`temperature=0` + sha256 prompt hashing** | Same inputs → same hash → reproducible logs. Useful for diff-testing against future runs. |
| **No prompt caching** | Minimum cacheable prefix on Haiku 4.5 is 4096 tokens. Our system prompts don't reach that, and 10 tickets per run don't pay back the cache-write premium. |

---

## Generated artifacts (spec §"Required Artifacts")

| Artifact | Written by | Contents |
|---|---|---|
| `preprocessed_tickets.json` | `pipeline/preprocess.py` | `ticket_id, original_text, cleaned_text, char_count, word_count` |
| `routing_decisions.json` | `pipeline/routing.py` | `ticket_id, route, confidence, routing_reason` |
| `triage_results.json` | `main.py` | `ticket_id, predicted_*, confidence, route, customer_reply, internal_note` |
| `prediction_comparison.json` | `pipeline/evaluate.py` | Per-ticket expected vs predicted, with match flags |
| `evaluation_report.json` | `pipeline/evaluate.py` | `category_accuracy, urgency_accuracy, human_review_count, parse_validation_failures` |
| `confusion_summary.json` | `pipeline/evaluate.py` | Confusion matrix + `most_confused_pairs` (stretch) |
| `llm_calls.jsonl` | `pipeline/logging_utils.py` | One JSON record per LLM call (stage, ticket_id, timestamp, provider, model, prompt_hash, output_artifact) |
| `llm_outputs/{stage}_{ticket_id}[_retry].json` | `pipeline/logging_utils.py` | Raw model response for every call |

All artifacts are regenerated from scratch on every run. The evaluator can delete them and rerun.

---

## Spec compliance matrix

Mapping every "MUST" from the spec to where it lives in code.

| Spec requirement | Implementation |
|---|---|
| Load `tickets.json` + `label_schema.json` | `pipeline/loader.py` |
| Deterministic preprocessing (whitespace, length stats, punctuation) | `pipeline/preprocess.py:clean_text` |
| Write `preprocessed_tickets.json` with `ticket_id / original_text / cleaned_text / char_count / word_count` | `pipeline/preprocess.py:preprocess_ticket` + `main.py:run` |
| Prompt includes allowed labels from schema | `pipeline/classify.py:build_user_prompt` |
| Model returns valid JSON only | `pipeline/classify.py:CLASSIFY_SYSTEM` + `output_config.format` |
| Parse and validate response | `pipeline/classify.py:_try_parse` + `_validate_payload` |
| Invalid category/urgency rejected in code | `pipeline/classify.py:_validate_payload` |
| Routing is deterministic, not the LLM's flag | `pipeline/routing.py:route` (ignores `parsed.needs_human_review`) |
| `confidence < 0.65` → `human_review` | `pipeline/routing.py:DEFAULT_CONFIDENCE_THRESHOLD` + comparison |
| Invalid/unparsable output → `human_review` with reason | `pipeline/routing.py:route` (handles `parse_path == "failed"`) |
| Customer reply: 2–4 sentences, no invented facts, consistent with prediction | `pipeline/reply.py:REPLY_SYSTEM` |
| Internal note for `human_review` cases | `pipeline/reply.py:generate_internal_note` |
| Reply only for `auto_triage`; note only for `human_review` | `main.py:run` (route-dispatched generation) |
| Compute category accuracy, urgency accuracy, review count, parse failures | `pipeline/evaluate.py:evaluation_report` |
| Per-ticket comparison report | `pipeline/evaluate.py:per_ticket_comparison` |
| `llm_calls.jsonl` with all required fields per call | `pipeline/logging_utils.py:CallLogger.log` |
| Validation command | `validate.py` |
| Recovery path for malformed output (§8) | `pipeline/classify.py:_extract_first_json_object` + retry |
| Lightweight tests for routing, schema, metrics (§9) | `tests/test_routing.py`, `test_schema.py`, `test_metrics.py` |
| CLI with `--tickets / --schema` flags (§10) | `main.py:parse_args` |
| Confusion summary (§11, stretch) | `pipeline/evaluate.py:confusion_summary` |

---

## Testing

```bash
pytest -q
```

**30 tests, all pure-Python, no network required:**

| File | Covers |
|---|---|
| `tests/test_routing.py` | Threshold boundary (0.64 vs 0.65), failed-parse → review, model's `needs_human_review` flag does NOT override deterministic routing |
| `tests/test_schema.py` | Valid payloads accepted, invalid category/urgency/confidence rejected, JSON recovery handles strict / noisy / multi-object / nested-brace inputs |
| `tests/test_metrics.py` | Accuracy computation, parse failures excluded from accuracy denominator, confusion matrix correctness |
| `tests/test_security.py` | Path traversal stripped from filenames, illegal Windows chars stripped, empty/whitespace/non-string ticket_id rejected, duplicates rejected |

---

## Configuration

The `.env` file is the only place secrets live. Template:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
# ANTHROPIC_MODEL=claude-haiku-4-5     # optional override
# CONFIDENCE_THRESHOLD=0.65            # optional override
```

`.env` is gitignored. `.env.example` is the committed template.

If you prefer plain env vars:

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # bash / zsh
$env:ANTHROPIC_API_KEY = "sk-ant-..."  # PowerShell
```

---

## Cost transparency

Default model is `claude-haiku-4-5` at $1/M input, $5/M output tokens.
A typical run on the 6 sample tickets uses roughly:

- ~6 classification calls × ~600 input + ~150 output tokens
- ~6 reply / note calls × ~400 input + ~200 output tokens
- ≈ **6,000 input + 2,100 output tokens** total
- ≈ **$0.02 per full pipeline run**

(With ~10 tickets, expect ≲ $0.04.)

---

## Tech stack

- **Python ≥ 3.10** (uses PEP 604 union types, PEP 585 generics)
- **`anthropic` ≥ 0.40.0** — official Anthropic SDK
- **`python-dotenv`** — `.env` loader
- **`pytest`** — test runner (only required for development)

No web framework, no DB, no async — sequential per-ticket processing is the
right size for the 10-ticket batch the spec calls out.

---

<p align="center">
  made with ❤️ by <strong>Abhishek Gupta</strong>
</p>
