"""Filename sanitization and input validation — defenses against malformed input."""

import json
from pathlib import Path

import pytest

from pipeline.loader import load_tickets
from pipeline.logging_utils import _safe_segment


def test_safe_segment_strips_path_traversal():
    assert _safe_segment("../../etc/passwd") == "etc_passwd"
    assert _safe_segment("..\\..\\windows\\system32") == "windows_system32"


def test_safe_segment_strips_illegal_windows_chars():
    # On Windows, : \ / * ? " < > | are forbidden in filenames.
    assert _safe_segment('T:1*?"<>|') == "T_1"
    assert _safe_segment("T<1>") == "T_1"


def test_safe_segment_empty_input_yields_underscore():
    assert _safe_segment("") == "_"
    assert _safe_segment("....") == "_"
    assert _safe_segment("..") == "_"


def test_safe_segment_preserves_normal_ids():
    assert _safe_segment("T1") == "T1"
    assert _safe_segment("ticket_abc-123") == "ticket_abc-123"


def test_loader_rejects_empty_ticket_id(tmp_path: Path):
    bad = tmp_path / "tickets.json"
    bad.write_text(json.dumps([{"ticket_id": "", "customer_message": "x"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="ticket_id must be a non-empty string"):
        load_tickets(bad)


def test_loader_rejects_whitespace_ticket_id(tmp_path: Path):
    bad = tmp_path / "tickets.json"
    bad.write_text(
        json.dumps([{"ticket_id": "   ", "customer_message": "x"}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ticket_id must be a non-empty string"):
        load_tickets(bad)


def test_loader_rejects_non_string_ticket_id(tmp_path: Path):
    bad = tmp_path / "tickets.json"
    bad.write_text(
        json.dumps([{"ticket_id": 42, "customer_message": "x"}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ticket_id must be a non-empty string"):
        load_tickets(bad)


def test_loader_rejects_non_string_customer_message(tmp_path: Path):
    bad = tmp_path / "tickets.json"
    bad.write_text(
        json.dumps([{"ticket_id": "T1", "customer_message": 123}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="customer_message must be a string"):
        load_tickets(bad)


def test_loader_rejects_duplicate_ticket_ids(tmp_path: Path):
    bad = tmp_path / "tickets.json"
    bad.write_text(
        json.dumps(
            [
                {"ticket_id": "T1", "customer_message": "first"},
                {"ticket_id": "T1", "customer_message": "second"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate ticket_id"):
        load_tickets(bad)
