"""Response models for the web API."""

from __future__ import annotations

from pydantic import BaseModel


class TranslateResponse(BaseModel):
    """Returned by POST /api/translate once a job has been accepted."""

    job_id: str
