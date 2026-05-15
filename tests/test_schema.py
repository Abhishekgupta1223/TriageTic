"""Schema and parse-recovery behavior in classify.py."""

from pipeline.classify import _try_parse, _validate_payload

CATEGORIES = ["billing", "verification", "login_access", "other"]
URGENCY = ["low", "medium", "high"]


def _good_payload():
    return {
        "category": "billing",
        "urgency": "high",
        "confidence": 0.8,
        "reasoning_summary": "ok",
        "needs_human_review": False,
    }


def test_valid_payload_returns_parsed_classification():
    result = _validate_payload(_good_payload(), CATEGORIES, URGENCY)
    assert not isinstance(result, str)
    assert result.category == "billing"
    assert result.confidence == 0.8


def test_invalid_category_rejected():
    p = _good_payload()
    p["category"] = "not_a_real_category"
    result = _validate_payload(p, CATEGORIES, URGENCY)
    assert isinstance(result, str)
    assert "category" in result


def test_invalid_urgency_rejected():
    p = _good_payload()
    p["urgency"] = "ultra_critical"
    result = _validate_payload(p, CATEGORIES, URGENCY)
    assert isinstance(result, str)
    assert "urgency" in result


def test_confidence_out_of_range_rejected():
    p = _good_payload()
    p["confidence"] = 1.5
    result = _validate_payload(p, CATEGORIES, URGENCY)
    assert isinstance(result, str)


def test_missing_field_rejected():
    p = _good_payload()
    del p["reasoning_summary"]
    result = _validate_payload(p, CATEGORIES, URGENCY)
    assert isinstance(result, str)
    assert "reasoning_summary" in result


def test_recovery_extracts_json_from_noisy_text():
    """The regex recovery path must pull the JSON out of surrounding prose."""
    noisy = (
        "Here is my classification:\n"
        '{"category": "billing", "urgency": "high", '
        '"confidence": 0.8, "reasoning_summary": "ok", "needs_human_review": false}\n'
        "Let me know if you need anything else."
    )
    payload, err = _try_parse(noisy)
    assert payload is not None, err
    assert payload["category"] == "billing"


def test_recovery_handles_strict_json():
    payload, err = _try_parse('{"category":"billing","urgency":"high","confidence":0.8,"reasoning_summary":"x","needs_human_review":false}')
    assert payload is not None
    assert payload["urgency"] == "high"


def test_recovery_returns_none_when_no_json():
    payload, err = _try_parse("there is no json in this response at all")
    assert payload is None
    assert err is not None
