"""Merge translated subtitle chunks and write SRT or ASS output."""

from __future__ import annotations

from pathlib import Path

from ai_subtitle_translator.parser import Subtitle, SubtitleDocument


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


def write_subtitle_file(
    document: SubtitleDocument,
    subtitles: list[Subtitle],
    path: str | Path,
) -> None:
    """Write subtitles using the same format as the parsed input document."""
    if document.format == "ass":
        content = format_ass(document, subtitles)
    else:
        content = format_srt(subtitles)
    Path(path).write_text(content, encoding="utf-8")


def format_srt(subtitles: list[Subtitle]) -> str:
    """Format subtitles as an SRT string."""
    blocks: list[str] = []
    for sub in subtitles:
        block = f"{sub.id}\n{sub.start} --> {sub.end}\n{sub.text}"
        blocks.append(block)
    return "\n\n".join(blocks) + "\n"


def format_ass(document: SubtitleDocument, subtitles: list[Subtitle]) -> str:
    """Format translated subtitles back into the original ASS file structure."""
    if document.ass_lines is None:
        raise ValueError("ASS document is missing original file lines")

    lines = list(document.ass_lines)
    for subtitle in subtitles:
        metadata = subtitle.metadata or {}
        line_index = metadata.get("ass_line_index")
        prefix = metadata.get("ass_prefix")
        newline_token = metadata.get("ass_newline_token", "\\N")

        if not isinstance(line_index, int) or not isinstance(prefix, str):
            raise ValueError("ASS subtitle is missing reconstruction metadata")

        ass_text = subtitle.text.replace("\n", newline_token)
        lines[line_index] = prefix + ass_text

    return "\n".join(lines) + "\n"
