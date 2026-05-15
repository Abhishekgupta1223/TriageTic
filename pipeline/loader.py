"""Load and validate input files."""

import json
from pathlib import Path
from typing import Any


def load_tickets(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of tickets")
    seen_ids: set[str] = set()
    for i, t in enumerate(data):
        if not isinstance(t, dict):
            raise ValueError(f"{path}[{i}]: ticket must be a JSON object")
        if "ticket_id" not in t or "customer_message" not in t:
            raise ValueError(f"{path}[{i}]: missing ticket_id or customer_message")
        if t["ticket_id"] in seen_ids:
            raise ValueError(f"{path}: duplicate ticket_id {t['ticket_id']!r}")
        seen_ids.add(t["ticket_id"])
    return data


def load_schema(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("categories", "urgency_levels"):
        if key not in data or not isinstance(data[key], list) or not data[key]:
            raise ValueError(f"{path}: {key} must be a non-empty list")
    return {"categories": list(data["categories"]), "urgency_levels": list(data["urgency_levels"])}
