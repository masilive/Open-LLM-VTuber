"""Tests for streaming MCP JSON object detection."""

from src.open_llm_vtuber.mcpp.json_detector import StreamJSONDetector


def test_json_detector_ignores_unmatched_brace_inside_string() -> None:
    """A brace in a JSON string must not terminate or corrupt object detection."""
    detector = StreamJSONDetector()
    text = 'pre {"tool":"search","arguments":{"query":"brace } inside"}} post'

    detected = detector.process_chunk(text)

    assert detected == [{"tool": "search", "arguments": {"query": "brace } inside"}}]


def test_json_detector_ignores_escaped_quotes_and_nested_objects() -> None:
    """Escaped quotes do not change string state and nested objects are preserved."""
    detector = StreamJSONDetector()
    text = '{"outer":{"quote":"escaped \\"} brace","value":1}}'

    assert detector.process_chunk(text) == [
        {"outer": {"quote": 'escaped "} brace', "value": 1}}
    ]


def test_json_detector_handles_chunk_boundaries_and_multiple_objects() -> None:
    """Objects split across chunks and adjacent objects are both detected."""
    detector = StreamJSONDetector()
    chunks = [
        'text {"one":{"x"',
        ':1}} middle {"two":',
        '{"y":2}} tail',
    ]

    detected = []
    for chunk in chunks:
        detected.extend(detector.process_chunk(chunk))

    assert detected == [{"one": {"x": 1}}, {"two": {"y": 2}}]
    assert detector.get_all_jsons() == detected


def test_json_detector_drops_balanced_invalid_json_and_recovers() -> None:
    """Malformed balanced candidates do not prevent later valid detection."""
    detector = StreamJSONDetector()

    assert detector.process_chunk("{not valid json}") == []
    assert detector.process_chunk('{"ok":true}') == [{"ok": True}]


def test_json_detector_reset_clears_incomplete_state() -> None:
    """Reset discards incomplete candidates and prior completed objects."""
    detector = StreamJSONDetector()
    detector.process_chunk('{"incomplete"')

    detector.reset()

    assert detector.buffer == ""
    assert detector.get_all_jsons() == []
    assert detector.process_chunk(':"late"}') == []
