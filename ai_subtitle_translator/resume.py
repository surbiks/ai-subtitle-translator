"""Resume / retry-failed-chunks support.

Persists a per-file translation *status* describing which chunks translated
and which failed, so a later run on the same source file can retry only the
unfinished chunks instead of re-translating everything.

The status file is created only while there is unfinished work: once every
chunk is translated it is deleted automatically (see ``main.translate_file``).

Naming convention: ``<source_dir>/.translation-status/<short_hash>.<lang>.<fmt>.json``
The source-content hash in the name guarantees the status is matched to the
exact file content, regardless of the output file name.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_subtitle_translator.parser import Subtitle

logger = logging.getLogger(__name__)

# Bump when the chunking algorithm changes in a way that alters chunk
# boundaries, so old status files are not reused with new chunking.
CHUNKING_VERSION = "v1"
# Bump when the status-file JSON shape changes incompatibly.
STATUS_SCHEMA_VERSION = 1

STATUS_DIR_NAME = ".translation-status"

# Chunk status values.
STATUS_PENDING = "pending"
STATUS_TRANSLATED = "translated"
STATUS_FAILED = "failed"


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string, e.g. ``2026-06-18T10:00:00Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- Hashing & normalization --


def normalize_source_content(raw: str) -> str:
    """Normalize subtitle text before hashing so trivial differences (BOM,
    line endings) don't cause spurious status mismatches. Content is otherwise
    preserved — timing and dialogue are untouched."""
    if raw.startswith("﻿"):  # strip UTF-8 BOM
        raw = raw[1:]
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def calculate_source_file_hash(path: str | Path) -> str:
    """Stable SHA-256 of the normalized source subtitle file content."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    normalized = normalize_source_content(raw)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def calculate_chunk_hash(chunk: list[Subtitle]) -> str:
    """Stable SHA-256 of a chunk's source content (ids, timing, and text).

    Used to verify a stored chunk still matches the freshly re-chunked source
    before reusing its translation.
    """
    payload = [
        {"id": s.id, "start": s.start, "end": s.end, "text": s.text}
        for s in chunk
    ]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chunk_source_text(chunk: list[Subtitle]) -> str:
    """Human-readable joined source text for a chunk (for the status file)."""
    return "\n".join(s.text for s in chunk)


def subtitles_to_items(subs: list[Subtitle]) -> list[dict[str, Any]]:
    """Serialize a (translated) chunk as ``[{"id", "text"}, ...]`` so the final
    output can be rebuilt without re-translating."""
    return [{"id": s.id, "text": s.text} for s in subs]


def _slugify_language(language: str) -> str:
    """Filesystem-safe short slug for a target language name."""
    slug = "".join(c if c.isalnum() else "-" for c in language.lower())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "lang"


# -- Data model --


@dataclass
class ChunkRecord:
    chunk_id: int
    source_hash: str
    source_text: str
    translated_text: str | None = None
    # Per-subtitle translations ([{"id", "text"}, ...]); the source of truth
    # for rebuilding the final file. None until the chunk is translated.
    translated_items: list[dict[str, Any]] | None = None
    status: str = STATUS_PENDING
    attempts: int = 0
    last_error: str | None = None
    last_attempt_at: str | None = None

    @property
    def is_translated(self) -> bool:
        return self.status == STATUS_TRANSLATED and bool(self.translated_items)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        return cls(
            chunk_id=int(data["chunk_id"]),
            source_hash=str(data.get("source_hash", "")),
            source_text=str(data.get("source_text", "")),
            translated_text=data.get("translated_text"),
            translated_items=data.get("translated_items"),
            status=str(data.get("status", STATUS_PENDING)),
            attempts=int(data.get("attempts", 0)),
            last_error=data.get("last_error"),
            last_attempt_at=data.get("last_attempt_at"),
        )


@dataclass
class TranslationStatus:
    source_file_name: str
    source_file_hash: str
    target_language: str
    subtitle_format: str
    model: str
    total_chunks: int
    chunks: list[ChunkRecord]
    chunking_version: str = CHUNKING_VERSION
    schema_version: int = STATUS_SCHEMA_VERSION
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def pending_chunk_ids(self) -> list[int]:
        """Chunk ids that still need (re)translation: anything not translated."""
        return [c.chunk_id for c in self.chunks if not c.is_translated]

    def translated_count(self) -> int:
        return sum(1 for c in self.chunks if c.is_translated)

    def is_complete(self) -> bool:
        return all(c.is_translated for c in self.chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_file_name": self.source_file_name,
            "source_file_hash": self.source_file_hash,
            "target_language": self.target_language,
            "subtitle_format": self.subtitle_format,
            "chunking_version": self.chunking_version,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_chunks": self.total_chunks,
            "chunks": [asdict(c) for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslationStatus":
        return cls(
            source_file_name=str(data.get("source_file_name", "")),
            source_file_hash=str(data.get("source_file_hash", "")),
            target_language=str(data.get("target_language", "")),
            subtitle_format=str(data.get("subtitle_format", "")),
            model=str(data.get("model", "")),
            total_chunks=int(data.get("total_chunks", 0)),
            chunks=[ChunkRecord.from_dict(c) for c in data.get("chunks", [])],
            chunking_version=str(data.get("chunking_version", CHUNKING_VERSION)),
            schema_version=int(data.get("schema_version", STATUS_SCHEMA_VERSION)),
            created_at=str(data.get("created_at", now_iso())),
            updated_at=str(data.get("updated_at", now_iso())),
        )


def is_translation_complete(status: TranslationStatus) -> bool:
    """True when every chunk is translated with non-empty text."""
    return status.is_complete()


def create_initial_translation_status(
    source_path: str | Path,
    source_file_hash: str,
    target_language: str,
    subtitle_format: str,
    model: str,
    chunks: list[list[Subtitle]],
) -> TranslationStatus:
    """Build a fresh in-memory status with every chunk marked pending."""
    records = [
        ChunkRecord(
            chunk_id=i,
            source_hash=calculate_chunk_hash(chunk),
            source_text=chunk_source_text(chunk),
        )
        for i, chunk in enumerate(chunks)
    ]
    return TranslationStatus(
        source_file_name=Path(source_path).name,
        source_file_hash=source_file_hash,
        target_language=target_language,
        subtitle_format=subtitle_format,
        model=model,
        total_chunks=len(chunks),
        chunks=records,
    )


# -- Status-file path & persistence --


def get_status_file_path(
    source_path: str | Path,
    source_file_hash: str,
    target_language: str,
    subtitle_format: str,
    status_dir: str | Path | None = None,
) -> Path:
    """Deterministic status-file path for a (source content, language, format).

    Defaults to ``<source_dir>/.translation-status/`` so re-uploading the same
    file to the same place finds the same status file.
    """
    base = Path(status_dir) if status_dir else Path(source_path).resolve().parent / STATUS_DIR_NAME
    short_hash = source_file_hash[:16]
    lang = _slugify_language(target_language)
    fmt = subtitle_format.lower()
    return base / f"{short_hash}.{lang}.{fmt}.json"


def load_translation_status(path: str | Path) -> TranslationStatus | None:
    """Load a status file, or None if missing/corrupt."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return TranslationStatus.from_dict(data)
    except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
        logger.warning("Ignoring unreadable translation status %s: %s", p, exc)
        return None


def save_translation_status(status: TranslationStatus, path: str | Path) -> None:
    """Persist the status atomically (temp file + flush + rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    status.updated_at = now_iso()
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(status.to_dict(), fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def delete_translation_status(path: str | Path) -> None:
    """Remove the status file (and its dir if it becomes empty). Best-effort."""
    p = Path(path)
    try:
        p.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete translation status %s: %s", p, exc)
        return
    parent = p.parent
    if parent.name == STATUS_DIR_NAME:
        try:
            parent.rmdir()  # only succeeds when empty
        except OSError:
            pass


def find_existing_translation_status(
    source_path: str | Path,
    source_file_hash: str,
    target_language: str,
    subtitle_format: str,
    chunks: list[list[Subtitle]],
    status_dir: str | Path | None = None,
) -> tuple[TranslationStatus, Path] | None:
    """Locate and validate a resumable status for this exact job.

    Returns ``(status, path)`` when a compatible status is found, else None.
    Compatibility requires matching source hash, language, format, chunking
    version, chunk count, and per-chunk source hashes. A mismatch (e.g. changed
    chunking settings) is treated as "no resumable status" — a fresh run.
    """
    path = get_status_file_path(
        source_path, source_file_hash, target_language, subtitle_format, status_dir
    )
    status = load_translation_status(path)
    if status is None:
        return None

    reasons: list[str] = []
    if status.source_file_hash != source_file_hash:
        reasons.append("source hash")
    if status.target_language != target_language:
        reasons.append("target language")
    if status.subtitle_format != subtitle_format:
        reasons.append("subtitle format")
    if status.chunking_version != CHUNKING_VERSION:
        reasons.append("chunking version")
    if status.total_chunks != len(chunks):
        reasons.append("chunk count")

    if not reasons:
        for record, chunk in zip(status.chunks, chunks):
            if record.source_hash != calculate_chunk_hash(chunk):
                reasons.append(f"chunk {record.chunk_id} content")
                break

    if reasons:
        logger.warning(
            "Found a status file at %s but it doesn't match (%s) — starting fresh",
            path, ", ".join(reasons),
        )
        return None

    return status, path


# -- Final-output rebuild --


def _with_text(orig: Subtitle, text: str) -> Subtitle:
    return Subtitle(
        id=orig.id,
        start=orig.start,
        end=orig.end,
        text=text,
        speaker=orig.speaker,
        metadata=orig.metadata,
    )


def build_final_subtitle_from_status(
    chunks: list[list[Subtitle]],
    status: TranslationStatus,
) -> list[list[Subtitle]]:
    """Rebuild output chunks: translated text where available, else source text.

    The result is always complete and usable — untranslated chunks fall back to
    their original (source-language) text.
    """
    result: list[list[Subtitle]] = []
    for idx, chunk in enumerate(chunks):
        record = status.chunks[idx] if idx < len(status.chunks) else None
        if record is not None and record.is_translated and record.translated_items:
            by_id = {
                int(item["id"]): item["text"]
                for item in record.translated_items
                if "id" in item and "text" in item
            }
            result.append([_with_text(s, by_id.get(s.id, s.text)) for s in chunk])
        else:
            result.append([_with_text(s, s.text) for s in chunk])
    return result
