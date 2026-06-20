"""FastAPI app for the AI Subtitle Translator web UI.

Routes:
  GET  /                         -> upload page (form defaults from .env)
  POST /api/translate            -> accept upload + options, start a job
  GET  /api/jobs/{id}/events     -> SSE stream of live progress
  GET  /api/jobs/{id}            -> JSON status snapshot (polling fallback)
  GET  /api/jobs/{id}/download   -> translated subtitle file
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ai_subtitle_translator.config import AppConfig, ChunkConfig, TranslatorConfig
from webapp.jobs import (
    STATUS_DONE,
    STATUS_ERROR,
    JobRegistry,
    is_terminal_event,
    run_translation_job,
)
from webapp.models import TranslateResponse

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
SUPPORTED_EXTS = (".srt", ".ass")
# How long the SSE loop waits between keepalive comments when idle.
_SSE_KEEPALIVE_SECONDS = 15.0

app = FastAPI(title="AI Subtitle Translator")
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(_HERE / "templates"))
registry = JobRegistry()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the upload form, pre-filling defaults from .env via TranslatorConfig."""
    defaults = TranslatorConfig()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_language": defaults.target_language,
            "default_model": defaults.model,
            "default_provider": defaults.provider,
        },
    )


def _clean(value: str | None) -> str | None:
    """Treat blank form fields as unset."""
    return value.strip() if value and value.strip() else None


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(
    file: UploadFile,
    target_language: str = Form(...),
    model: str = Form(...),
    provider: str = Form("copilot"),
    api_key: str | None = Form(None),
    base_url: str | None = Form(None),
) -> TranslateResponse:
    """Accept an upload + options, start the translation job, return its id."""
    filename = file.filename or "subtitles.srt"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or '<none>'}'. Upload a .srt or .ass file.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    out_format = "ass" if ext == ".ass" else "srt"

    # Build config: ChunkConfig defaults + TranslatorConfig with form overrides.
    translator_cfg = TranslatorConfig()
    translator_cfg.target_language = target_language.strip() or translator_cfg.target_language
    translator_cfg.model = model.strip() or translator_cfg.model
    if _clean(provider):
        translator_cfg.provider = provider.strip()
    if _clean(api_key):
        translator_cfg.api_key = _clean(api_key)
    if _clean(base_url):
        translator_cfg.base_url = _clean(base_url)
    app_cfg = AppConfig(chunk=ChunkConfig(), translator=translator_cfg)

    job = registry.create(filename=filename, out_format=out_format)
    asyncio.create_task(run_translation_job(job, raw_bytes, ext, app_cfg))
    logger.info("Accepted job %s for %s", job.id, filename)
    return TranslateResponse(job_id=job.id)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    """Return a JSON status snapshot (polling fallback / late joiners)."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return JSONResponse(job.snapshot())


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Stream live progress as Server-Sent Events until the job finishes."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")

    async def event_stream():
        # Reconnect / fast-finish: the job already ended and its queue is drained.
        # Replay one terminal event from the snapshot and stop.
        if job.status in (STATUS_DONE, STATUS_ERROR) and job.events.empty():
            yield _sse(_terminal_from_snapshot(job))
            return

        while True:
            try:
                event = await asyncio.wait_for(
                    job.events.get(), timeout=_SSE_KEEPALIVE_SECONDS
                )
            except asyncio.TimeoutError:
                if job.status in (STATUS_DONE, STATUS_ERROR) and job.events.empty():
                    break
                yield ": keepalive\n\n"
                continue
            yield _sse(event)
            if is_terminal_event(event):
                break

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable proxy buffering so events flush live
    }
    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=headers
    )


@app.get("/api/jobs/{job_id}/download")
async def download(job_id: str) -> Response:
    """Return the translated subtitle file as an attachment."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    if job.status != STATUS_DONE or job.output_text is None:
        raise HTTPException(status_code=409, detail="Translation is not finished yet")

    return Response(
        content=job.output_text,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{job.output_filename}"'
        },
    )


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _terminal_from_snapshot(job) -> dict[str, Any]:
    """Build a terminal SSE event from a finished job's current state."""
    if job.status == STATUS_DONE:
        return {
            "type": "done",
            "total": job.total_chunks,
            "translated": job.translated_chunks,
            "failed": job.failed_chunks,
            "download_url": f"/api/jobs/{job.id}/download",
            "filename": job.output_filename,
        }
    return {"type": "error", "message": job.error or "Translation failed"}
