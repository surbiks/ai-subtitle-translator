"""Merge translated subtitle chunks back into a valid SRT file."""

from __future__ import annotations

from pathlib import Path

from ai_subtitle_translator.parser import Subtitle


def merge_chunks(chunks: list[list[Subtitle]]) -> list[Subtitle]:
    """Flatten translated chunks into a single ordered list, deduplicating by ID."""
    seen: set[int] = set()
    merged: list[Subtitle] = []

    for chunk in chunks:
        for sub in chunk:
            if sub.id not in seen:
                seen.add(sub.id)
                merged.append(sub)

    merged.sort(key=lambda s: s.id)
    return merged


def write_srt(subtitles: list[Subtitle], path: str | Path) -> None:
    """Write subtitles to an SRT file."""
    content = format_srt(subtitles)
    Path(path).write_text(content, encoding="utf-8")


def format_srt(subtitles: list[Subtitle]) -> str:
    """Format subtitles as an SRT string."""
    blocks: list[str] = []
    for sub in subtitles:
        block = f"{sub.id}\n{sub.start} --> {sub.end}\n{sub.text}"
        blocks.append(block)
    return "\n\n".join(blocks) + "\n"
