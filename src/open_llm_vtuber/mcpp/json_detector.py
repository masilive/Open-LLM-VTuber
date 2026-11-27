"""Incremental detection of JSON objects embedded in streaming text."""

import json
from typing import Any

from loguru import logger


class StreamJSONDetector:
    """Detect complete JSON objects in streaming text in linear time.

    Braces inside JSON strings and escaped quotes do not affect object depth,
    so valid tool calls are detected even when string arguments contain braces.
    Only the current incomplete JSON candidate is retained in memory.
    """

    def __init__(self) -> None:
        """Initialize an empty streaming detector."""
        self.buffer = ""
        self.potential_jsons = []
        self.completed_jsons = []
        self.processed_ranges = []
        self._fragment: list[str] = []
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def process_chunk(self, chunk: str) -> list[dict[str, Any]]:
        """Process one text chunk and return newly completed JSON objects.

        Args:
            chunk: Received text chunk.

        Returns:
            Complete JSON objects that terminated while processing this chunk.
        """
        new_jsons = []
        for char in chunk:
            if self._depth == 0:
                if char == "{":
                    self._start_candidate(char)
                continue

            self._fragment.append(char)
            self._update_candidate_state(char)
            if self._depth == 0:
                parsed = self._finish_candidate()
                if parsed is not None:
                    new_jsons.append(parsed)

        self.buffer = "".join(self._fragment)
        return new_jsons

    def _start_candidate(self, char: str) -> None:
        """Start a candidate JSON object at an opening brace.

        Args:
            char: Opening brace character.
        """
        self._fragment = [char]
        self._depth = 1
        self._in_string = False
        self._escaped = False

    def _update_candidate_state(self, char: str) -> None:
        """Update JSON depth and string state for one character.

        Args:
            char: Character inside the current candidate object.
        """
        if self._escaped:
            self._escaped = False
            return

        if self._in_string:
            if char == "\\":
                self._escaped = True
            elif char == '"':
                self._in_string = False
            return

        if char == '"':
            self._in_string = True
        elif char == "{":
            self._depth += 1
        elif char == "}":
            self._depth -= 1

    def _finish_candidate(self) -> dict[str, Any] | None:
        """Parse and clear a complete candidate object.

        Returns:
            The parsed object, or ``None`` when the balanced candidate is not
            valid JSON.
        """
        json_text = "".join(self._fragment)
        self._fragment = []
        self._in_string = False
        self._escaped = False
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning(
                f"JSON structure found but parsing failed: {json_text[:50]}..."
            )
            return None

        if isinstance(parsed, dict):
            self.completed_jsons.append(parsed)
            return parsed
        logger.warning("Ignoring completed JSON value that is not an object")
        return None

    def get_all_jsons(self) -> list[dict[str, Any]]:
        """Get all JSON objects parsed so far.

        Returns:
            List of all parsed JSON objects.
        """
        return self.completed_jsons

    def reset(self) -> None:
        """Reset detector state before processing a new stream."""
        self.buffer = ""
        self.potential_jsons = []
        self.completed_jsons = []
        self.processed_ranges = []
        self._fragment = []
        self._depth = 0
        self._in_string = False
        self._escaped = False


# Usage example
if __name__ == "__main__":
    test_chunks = [
        "This is some plain text ",
        "Here comes JSON: {",
        '"name": "test", "values": [1, 2, ',
        '3]} This is text after JSON {"another": "json", ',
        '"nested": {"key": "value"}}',
    ]

    detector = StreamJSONDetector()

    for i, chunk in enumerate(test_chunks):
        logger.info(f"Processing chunk {i + 1}: {chunk}")
        new_jsons = detector.process_chunk(chunk)
        if new_jsons:
            logger.info(f"Complete JSON found in this chunk: {new_jsons}")

    logger.info(f"All detected JSONs: {detector.get_all_jsons()}")
