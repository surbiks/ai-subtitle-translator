"""Tests for the codex streaming-SSE parser (no network required)."""

import json
import sys
import unittest
from pathlib import Path

# Make the package importable when run as `python tests/test_codex_provider.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_subtitle_translator.translator import _extract_codex_text, _to_codex_input

FIXTURE = Path(__file__).resolve().parent.parent / "command.txt"


class ExtractCodexTextTest(unittest.TestCase):
    def test_matches_done_event_text(self):
        """Accumulated deltas must equal the response.output_text.done text."""
        lines = FIXTURE.read_text(encoding="utf-8").splitlines()

        expected = None
        for line in lines:
            if line.startswith("data:"):
                evt = json.loads(line[len("data:"):].strip())
                if evt.get("type") == "response.output_text.done":
                    expected = evt["text"]
                    break

        self.assertIsNotNone(expected, "fixture missing response.output_text.done")
        result = _extract_codex_text(lines)
        self.assertEqual(result, expected)
        self.assertIn("#!/usr/bin/env bash", result)

    def test_concatenates_deltas_and_stops_at_completed(self):
        lines = [
            'data: {"type":"response.output_text.delta","delta":"a"}',
            'data: {"type":"response.output_text.delta","delta":"b"}',
            'data: {"type":"response.completed","response":{"error":null}}',
            'data: {"type":"response.output_text.delta","delta":"c"}',  # ignored
        ]
        self.assertEqual(_extract_codex_text(lines), "ab")

    def test_ignores_unknown_events_and_falls_back_to_done(self):
        lines = [
            "event: codex.rate_limits",
            'data: {"type":"codex.rate_limits","plan_type":"free"}',
            'data: {"type":"response.output_text.done","text":"hello"}',
        ]
        self.assertEqual(_extract_codex_text(lines), "hello")

    def test_tolerates_done_sentinel_and_blank_lines(self):
        lines = [
            "",
            'data: {"type":"response.output_text.delta","delta":"hi"}',
            "data: [DONE]",
        ]
        self.assertEqual(_extract_codex_text(lines), "hi")

    def test_raises_on_error_event(self):
        with self.assertRaises(RuntimeError):
            _extract_codex_text(['data: {"type":"error","message":"boom"}'])


class ToCodexInputTest(unittest.TestCase):
    def test_role_to_content_type_mapping(self):
        items = _to_codex_input(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "prev"},
            ]
        )
        self.assertEqual(items[0]["content"][0]["type"], "input_text")
        self.assertEqual(items[0]["content"][0]["text"], "hi")
        self.assertEqual(items[1]["content"][0]["type"], "output_text")


if __name__ == "__main__":
    unittest.main()
