"""In-memory job registry and translation orchestration for the web UI.

A ``Job`` tracks one translation run. ``run_translation_job`` drives the existing
pipeline (parse → chunk → translate → merge) and feeds per-chunk progress into
the job's ``asyncio.Queue`` so the SSE endpoint can stream it to the browser.

State is kept in memory for the lifetime of the process — sufficient for a local
single-process tool. Restarting the server clears all jobs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_subtitle_translator.chunker import build_context_window, chunk_subtitles
from ai_subtitle_translator.config import AppConfig
from ai_subtitle_translator.merger import format_ass, format_srt, merge_chunks
from ai_subtitle_translator.parser import parse_subtitle_file
from ai_subtitle_translator.translator import ChunkOutcome, Translator

logger = logging.getLogger(__name__)

# Job lifecycle states.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

# Terminal event types end the SSE stream.
_TERMINAL_EVENTS = ("done", "error")


@dataclass
class Job:
    """One translation run and its live progress state."""

    id: str
    filename: str
    out_format: str  # "srt" | "ass"
    status: str = STATUS_QUEUED
    total_chunks: int = 0
    done_chunks: int = 0
    failed_chunks: int = 0
    output_text: str | None = None
    error: str | None = None
    # Unbounded queue: progress events buffer here until the SSE consumer drains
    # them, so no events are lost even if the browser connects slightly late.
    events: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)

    @property
    def translated_chunks(self) -> int:
        return self.done_chunks - self.failed_chunks

    @property
    def output_filename(self) -> str:
        """Suggested download name: ``<stem>.fa.<ext>``."""
        stem = Path(self.filename).stem or "subtitles"
        return f"{stem}.fa.{self.out_format}"

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable status (for the polling-fallback endpoint)."""
        return {
            "job_id": self.id,
            "status": self.status,
            "filename": self.filename,
            "total_chunks": self.total_chunks,
            "done_chunks": self.done_chunks,
            "failed_chunks": self.failed_chunks,
            "translated_chunks": self.translated_chunks,
            "error": self.error,
            "download_url": (
                f"/api/jobs/{self.id}/download" if self.status == STATUS_DONE else None
            ),
        }


class JobRegistry:
    """In-memory registry of jobs keyed by id."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, filename: str, out_format: str) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename, out_format=out_format)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


async def run_translation_job(
    job: Job,
    raw_bytes: bytes,
    ext: str,
    app_cfg: AppConfig,
) -> None:
    """Translate an uploaded subtitle file, streaming progress into ``job.events``.

    Mirrors ``main.translate_file`` but trimmed for the web MVP: no on-disk resume
    or status files. Never raises — failures are reported as an ``error`` event so
    the SSE stream always terminates cleanly.
    """
    try:
        job.status = STATUS_RUNNING

        # parse_subtitle_file dispatches on extension and reads from a path (ASS
        # parsing in particular needs a real file), so round-trip through a temp
        # file, then drop it — the subtitles live in memory from here on.
        document = _parse_bytes(raw_bytes, ext)
        subtitles = document.subtitles
        if not subtitles:
            raise ValueError("No subtitles found in the uploaded file")

        chunks = chunk_subtitles(subtitles, app_cfg.chunk)
        contexts = build_context_window(chunks, app_cfg.chunk.context_lines)
        job.total_chunks = len(chunks)
        await job.events.put(
            {"type": "progress", "done": 0, "total": job.total_chunks,
             "failed": 0, "ok": True}
        )

        def on_chunk_done(outcome: ChunkOutcome) -> None:
            # Runs synchronously inside the translate loop's event loop, so
            # mutating the job and put_nowait are both safe here.
            job.done_chunks += 1
            if not outcome.ok:
                job.failed_chunks += 1
            job.events.put_nowait(
                {
                    "type": "progress",
                    "done": job.done_chunks,
                    "total": job.total_chunks,
                    "failed": job.failed_chunks,
                    "ok": outcome.ok,
                }
            )

        translator = Translator(app_cfg.translator)
        outcomes = await translator.translate_chunks_detailed(
            chunks, contexts, progress_callback=on_chunk_done,
        )

        merged = merge_chunks([o.subtitles for o in outcomes])
        if document.format == "ass":
            job.output_text = format_ass(document, merged)
        else:
            job.output_text = format_srt(merged)

        job.status = STATUS_DONE
        await job.events.put(
            {
                "type": "done",
                "total": job.total_chunks,
                "translated": job.translated_chunks,
                "failed": job.failed_chunks,
                "download_url": f"/api/jobs/{job.id}/download",
                "filename": job.output_filename,
            }
        )
        logger.info(
            "Job %s done: %d/%d chunks translated",
            job.id, job.translated_chunks, job.total_chunks,
        )

    except Exception as exc:  # noqa: BLE001 — report any failure to the client
        job.status = STATUS_ERROR
        job.error = str(exc)
        logger.exception("Job %s failed", job.id)
        await job.events.put({"type": "error", "message": str(exc)})


def _parse_bytes(raw_bytes: bytes, ext: str):
    """Write upload bytes to a temp file with ``ext`` and parse it, then clean up."""
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw_bytes)
        return parse_subtitle_file(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def is_terminal_event(event: dict[str, Any]) -> bool:
    return event.get("type") in _TERMINAL_EVENTS
