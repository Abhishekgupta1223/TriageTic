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


def test_recovery_handles_multiple_json_candidates():
    """The OLD greedy regex `\\{.*\\}` would capture from the first `{` to the
    LAST `}` and produce an unparseable blob. The new raw_decode walker must
    return the first VALID JSON object regardless of surrounding noise.
    """
    # Two brace-prefixed strings; only the second is a valid JSON object.
    noisy = (
        "{this is not json}\n"
        '{"category": "billing", "urgency": "high", "confidence": 0.9, '
        '"reasoning_summary": "ok", "needs_human_review": false}\n'
        "{some trailing prose}"
    )
    payload, _ = _try_parse(noisy)
    assert payload is not None
    assert payload["category"] == "billing"
    assert payload["confidence"] == 0.9


def test_recovery_picks_first_valid_when_multiple_objects():
    """If multiple complete JSON objects appear, take the first."""
    noisy = (
        'prose {"category": "verification", "urgency": "low", "confidence": 0.5, '
        '"reasoning_summary": "first", "needs_human_review": true} '
        'more {"category": "billing", "urgency": "high", "confidence": 0.9, '
        '"reasoning_summary": "second", "needs_human_review": false}'
    )
    payload, _ = _try_parse(noisy)
    assert payload is not None
    assert payload["reasoning_summary"] == "first"


def test_recovery_handles_nested_braces():
    """A JSON object whose VALUE contains braces (e.g. nested object string)
    must still parse correctly — the old greedy regex could not distinguish
    inner from outer braces in noisy text.
    """
    obj_with_nested = (
        '{"category": "billing", "urgency": "high", "confidence": 0.8, '
        '"reasoning_summary": "Customer said: {urgent} please help", '
        '"needs_human_review": false}'
    )
    payload, _ = _try_parse(obj_with_nested)
    assert payload is not None
    assert "{urgent}" in payload["reasoning_summary"]
