"""Tests for the web UI layer (webapp.server).

The translation provider is stubbed via ``translator._build_provider`` so no
network calls happen — a FakeProvider echoes a short canned translation for the
ids it is asked to translate. The job is driven to completion by reading its SSE
stream, which pumps the server's event loop.
"""

from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from ai_subtitle_translator import translator as _t
from webapp import server

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,000
Hello there.

2
00:00:04,500 --> 00:00:08,000
How are you today?

3
00:00:08,500 --> 00:00:12,000
I am doing great, thanks.
"""

# A translation distinct from the source and short enough to never trip the
# CPS-compression retry (visible length well under the per-line budget).
_FAKE_TRANSLATION = "ترجمه"


class FakeProvider:
    """Returns a valid JSON translation for every id in the request payload."""

    async def chat(self, system, messages, model, temperature):
        content = messages[-1]["content"]
        marker = "Translate the following:"
        arr = content[content.rfind(marker) + len(marker):]
        start, end = arr.find("["), arr.rfind("]")
        items = json.loads(arr[start : end + 1])
        return json.dumps(
            [{"id": it["id"], "text": _FAKE_TRANSLATION} for it in items],
            ensure_ascii=False,
        )


def _read_sse_until_terminal(client, job_id, *, max_lines=500):
    """Stream the job's SSE events, returning the list of parsed event dicts."""
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200, resp.status_code
        for i, line in enumerate(resp.iter_lines()):
            if i > max_lines:
                raise AssertionError("SSE stream did not terminate")
            if not line or not line.startswith("data:"):
                continue
            evt = json.loads(line[len("data:"):].strip())
            events.append(evt)
            if evt.get("type") in ("done", "error"):
                break
    return events


class WebTranslateTest(unittest.TestCase):
    def setUp(self):
        # Stub the provider for every Translator built during these tests.
        self._orig_build = _t._build_provider
        _t._build_provider = lambda cfg: FakeProvider()

    def tearDown(self):
        _t._build_provider = self._orig_build

    def _start_job(self, client, *, filename="sample.srt", body=SAMPLE_SRT):
        resp = client.post(
            "/api/translate",
            data={"target_language": "Persian (Farsi)", "model": "gpt-4o-mini"},
            files={"file": (filename, body.encode("utf-8"), "application/x-subrip")},
        )
        return resp

    def test_index_page_renders(self):
        with TestClient(server.app) as client:
            resp = client.get("/")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("AI Subtitle Translator", resp.text)

    def test_translate_job_completes_and_downloads(self):
        with TestClient(server.app) as client:
            resp = self._start_job(client)
            self.assertEqual(resp.status_code, 200)
            job_id = resp.json()["job_id"]

            events = _read_sse_until_terminal(client, job_id)
            done = events[-1]
            self.assertEqual(done["type"], "done")
            self.assertGreater(done["total"], 0)
            self.assertEqual(done["translated"], done["total"])
            self.assertEqual(done["failed"], 0)

            # Status snapshot agrees with the terminal event.
            status = client.get(f"/api/jobs/{job_id}").json()
            self.assertEqual(status["status"], "done")

            # Download returns the translated content with the .fa. filename.
            dl = client.get(f"/api/jobs/{job_id}/download")
            self.assertEqual(dl.status_code, 200)
            self.assertIn(_FAKE_TRANSLATION, dl.text)
            self.assertIn('filename="sample.fa.srt"', dl.headers["content-disposition"])

    def test_unsupported_extension_rejected(self):
        with TestClient(server.app) as client:
            resp = client.post(
                "/api/translate",
                data={"target_language": "Persian (Farsi)", "model": "gpt-4o-mini"},
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("Unsupported", resp.json()["detail"])

    def test_empty_subtitle_file_yields_error_event(self):
        with TestClient(server.app) as client:
            # Valid .srt extension but no parseable subtitle blocks.
            resp = self._start_job(client, filename="empty.srt", body="not a subtitle\n")
            self.assertEqual(resp.status_code, 200)
            job_id = resp.json()["job_id"]

            events = _read_sse_until_terminal(client, job_id)
            self.assertEqual(events[-1]["type"], "error")
            self.assertIn("No subtitles", events[-1]["message"])

    def test_unknown_job_returns_404(self):
        with TestClient(server.app) as client:
            self.assertEqual(client.get("/api/jobs/nope").status_code, 404)
            self.assertEqual(client.get("/api/jobs/nope/download").status_code, 404)


if __name__ == "__main__":
    unittest.main()
