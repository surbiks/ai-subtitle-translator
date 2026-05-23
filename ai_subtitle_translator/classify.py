"""Classify subtitle lines into dialog, sound cues, stage directions, screen text, or lyrics."""

from __future__ import annotations

import re
from enum import Enum


class LineKind(str, Enum):
    DIALOG = "dialog"
    SOUND_CUE = "sound_cue"      # [music], [door slams]
    STAGE_DIR = "stage_dir"      # (whispering), (laughing)
    SCREEN_TEXT = "screen_text"  # ALL CAPS titles, signs
    LYRICS = "lyrics"            # ♪ ... ♪


_SOUND_RE = re.compile(r"^\s*\[[^\[\]]+\]\s*$", re.DOTALL)
_STAGE_RE = re.compile(r"^\s*\([^()]+\)\s*$", re.DOTALL)
_LYRICS_RE = re.compile(r"^\s*[♪#].*?[♪#]?\s*$|^\s*♪")
_CAPS_RE = re.compile(r"^[A-Z0-9\s\.\,\!\?\-']{4,}$")


def classify(text: str) -> LineKind:
    """Best-effort classification of a single subtitle's text."""
    stripped = text.strip()
    if not stripped:
        return LineKind.DIALOG

    if _LYRICS_RE.match(stripped):
        return LineKind.LYRICS
    if _SOUND_RE.match(stripped):
        return LineKind.SOUND_CUE
    if _STAGE_RE.match(stripped):
        return LineKind.STAGE_DIR
    if _CAPS_RE.match(stripped) and any(c.isalpha() for c in stripped):
        return LineKind.SCREEN_TEXT
    return LineKind.DIALOG
