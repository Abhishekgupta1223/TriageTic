"""LLM call logging — emits one JSONL record per call."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]")


def prompt_hash(*parts: str) -> str:
    """Stable sha256 of one or more prompt segments, joined with a record separator."""
    joined = "\x1e".join(parts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_segment(s: str) -> str:
    """Replace anything that could escape an output directory with '_'.

    Prevents path-traversal via untrusted ticket_id (e.g., ``"../../etc/passwd"``)
    and avoids Windows-illegal filename characters (``: \\ / * ? " < > |``).
    Empty/whitespace input collapses to ``"_"``.
    """
    cleaned = _SAFE_FILENAME_RE.sub("_", s).strip("._")
    return cleaned or "_"


class CallLogger:
    """Append-only logger for `llm_calls.jsonl`."""

    def __init__(self, log_path: Path, output_dir: Path) -> None:
        self.log_path = log_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Truncate on init so each pipeline run starts clean.
        self.log_path.write_text("", encoding="utf-8")

    def log(
        self,
        *,
        stage: str,
        ticket_id: str,
        provider: str,
        model: str,
        prompt: str,
        raw_output: Any,
        suffix: str | None = None,
    ) -> Path:
        """Write the raw output to disk and append a JSONL record. Returns artifact path.

        ``suffix`` lets a caller disambiguate multiple artifacts for the same
        (stage, ticket_id) — for example, ``suffix="retry"`` for the second
        attempt of a classification — without changing the ``stage`` field,
        which must match the spec's allowed values.
        """
        safe_stage = _safe_segment(stage)
        safe_ticket = _safe_segment(ticket_id)
        safe_suffix = f"_{_safe_segment(suffix)}" if suffix else ""
        artifact_path = self.output_dir / f"{safe_stage}_{safe_ticket}{safe_suffix}.json"
        artifact_path.write_text(
            json.dumps(raw_output, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        record = {
            "stage": stage,
            "ticket_id": ticket_id,
            "timestamp": iso_now(),
            "provider": provider,
            "model": model,
            "prompt_hash": prompt_hash(prompt),
            "output_artifact": str(artifact_path.as_posix()),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return artifact_path
