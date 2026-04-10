"""SRT subtitle parser and data model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Matches "HH:MM:SS,mmm --> HH:MM:SS,mmm"
_TIMESTAMP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)


@dataclass
class Subtitle:
    id: int
    start: str  # raw timestamp string "HH:MM:SS,mmm"
    end: str
    text: str

    @property
    def start_ms(self) -> int:
        return _timestamp_to_ms(self.start)

    @property
    def end_ms(self) -> int:
        return _timestamp_to_ms(self.end)


def _timestamp_to_ms(ts: str) -> int:
    """Convert 'HH:MM:SS,mmm' to total milliseconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


def parse_srt(path: str | Path) -> list[Subtitle]:
    """Parse an SRT file into a list of Subtitle objects."""
    content = Path(path).read_text(encoding="utf-8-sig")  # handles BOM
    return parse_srt_content(content)


def parse_srt_content(content: str) -> list[Subtitle]:
    """Parse raw SRT content string into Subtitle objects."""
    blocks = re.split(r"\n\s*\n", content.strip())
    subtitles: list[Subtitle] = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        # First line: numeric ID
        try:
            sub_id = int(lines[0].strip())
        except ValueError:
            continue

        # Second line: timestamps
        ts_match = _TIMESTAMP_RE.search(lines[1])
        if not ts_match:
            continue

        # Remaining lines: subtitle text (may be multi-line)
        text = "\n".join(lines[2:]).strip()

        subtitles.append(
            Subtitle(id=sub_id, start=ts_match.group(1), end=ts_match.group(2), text=text)
        )

    return subtitles
