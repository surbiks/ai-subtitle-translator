"""Subtitle parsing for SRT and ASS formats."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Matches "HH:MM:SS,mmm --> HH:MM:SS,mmm"
_SRT_TIMESTAMP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)
_GENERIC_TIMESTAMP_RE = re.compile(
    r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<f>\d{2,3})"
)


@dataclass
class Subtitle:
    id: int
    start: str
    end: str
    text: str
    metadata: dict[str, Any] | None = None

    @property
    def start_ms(self) -> int:
        return _timestamp_to_ms(self.start)

    @property
    def end_ms(self) -> int:
        return _timestamp_to_ms(self.end)


@dataclass
class SubtitleDocument:
    format: str
    subtitles: list[Subtitle]
    ass_lines: list[str] | None = None


def _timestamp_to_ms(ts: str) -> int:
    """Convert SRT or ASS timestamps to total milliseconds."""
    match = _GENERIC_TIMESTAMP_RE.fullmatch(ts.strip())
    if not match:
        raise ValueError(f"Unsupported timestamp format: {ts}")

    fraction = match.group("f")
    if len(fraction) == 2:
        milliseconds = int(fraction) * 10
    else:
        milliseconds = int(fraction)

    return (
        int(match.group("h")) * 3_600_000
        + int(match.group("m")) * 60_000
        + int(match.group("s")) * 1_000
        + milliseconds
    )


def parse_subtitle_file(path: str | Path) -> SubtitleDocument:
    """Parse a subtitle file based on extension."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".srt":
        return SubtitleDocument(format="srt", subtitles=parse_srt(file_path))
    if suffix == ".ass":
        return parse_ass(file_path)

    raise ValueError(f"Unsupported subtitle format: {suffix or '<no extension>'}")


def parse_srt(path: str | Path) -> list[Subtitle]:
    """Parse an SRT file into a list of Subtitle objects."""
    content = Path(path).read_text(encoding="utf-8-sig")
    return parse_srt_content(content)


def parse_srt_content(content: str) -> list[Subtitle]:
    """Parse raw SRT content string into Subtitle objects."""
    blocks = re.split(r"\n\s*\n", content.strip())
    subtitles: list[Subtitle] = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        try:
            sub_id = int(lines[0].strip())
        except ValueError:
            continue

        ts_match = _SRT_TIMESTAMP_RE.search(lines[1])
        if not ts_match:
            continue

        text = "\n".join(lines[2:]).strip()
        subtitles.append(
            Subtitle(id=sub_id, start=ts_match.group(1), end=ts_match.group(2), text=text)
        )

    return subtitles


def parse_ass(path: str | Path) -> SubtitleDocument:
    """Parse an ASS file, extracting dialogue text while preserving the full file."""
    content = Path(path).read_text(encoding="utf-8-sig")
    lines = content.splitlines()
    subtitles: list[Subtitle] = []

    in_events = False
    format_fields: list[str] | None = None
    dialogue_id = 1

    for line_index, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()

        if lower == "[events]":
            in_events = True
            format_fields = None
            continue

        if stripped.startswith("[") and lower != "[events]":
            in_events = False
            format_fields = None
            continue

        if not in_events:
            continue

        if lower.startswith("format:"):
            format_fields = [
                field.strip() for field in stripped.split(":", 1)[1].split(",")
            ]
            continue

        if not lower.startswith("dialogue:"):
            continue

        subtitle = _parse_ass_dialogue_line(
            line,
            line_index=line_index,
            format_fields=format_fields,
            dialogue_id=dialogue_id,
        )
        if subtitle is None:
            continue

        subtitles.append(subtitle)
        dialogue_id += 1

    return SubtitleDocument(format="ass", subtitles=subtitles, ass_lines=lines)


def _parse_ass_dialogue_line(
    line: str,
    *,
    line_index: int,
    format_fields: list[str] | None,
    dialogue_id: int,
) -> Subtitle | None:
    payload = line.split(":", 1)[1]
    fields = format_fields or _default_ass_format_fields()

    if not fields:
        return None

    parts = payload.split(",", len(fields) - 1)
    if len(parts) != len(fields):
        return None

    data = {field.lower(): value for field, value in zip(fields, parts)}
    start = data.get("start", "").strip()
    end = data.get("end", "").strip()
    text = data.get("text")
    if not start or not end or text is None:
        return None

    text_field_index = next(
        (index for index, field in enumerate(fields) if field.lower() == "text"),
        len(fields) - 1,
    )
    prefix_parts = parts[:text_field_index]
    prefix = "Dialogue:" + ",".join(prefix_parts) + ","

    return Subtitle(
        id=dialogue_id,
        start=start,
        end=end,
        text=_decode_ass_text(text),
        metadata={
            "ass_line_index": line_index,
            "ass_prefix": prefix,
            "ass_newline_token": _detect_ass_newline_token(text),
        },
    )


def _default_ass_format_fields() -> list[str]:
    return [
        "Layer",
        "Start",
        "End",
        "Style",
        "Name",
        "MarginL",
        "MarginR",
        "MarginV",
        "Effect",
        "Text",
    ]


def _detect_ass_newline_token(text: str) -> str:
    if "\\N" in text:
        return "\\N"
    if "\\n" in text:
        return "\\n"
    return "\\N"


def _decode_ass_text(text: str) -> str:
    return text.replace("\\N", "\n").replace("\\n", "\n")

