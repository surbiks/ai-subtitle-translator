"""Glossary system for consistent translation of names and terms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Glossary:
    """Loads and manages a term→translation mapping for prompt injection."""

    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self._entries: dict[str, str] = entries or {}

    @classmethod
    def from_file(cls, path: str | Path) -> Glossary:
        """Load glossary from a JSON file: {"term": "translation", ...}."""
        p = Path(path)
        if not p.exists():
            logger.warning("Glossary file not found: %s — using empty glossary", p)
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Glossary must be a JSON object, got {type(data).__name__}")
        logger.info("Loaded glossary with %d entries from %s", len(data), p)
        return cls(entries=data)

    @property
    def entries(self) -> dict[str, str]:
        return self._entries

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def build_prompt_section(self) -> str:
        """Build the glossary block to inject into the translation prompt."""
        if self.is_empty:
            return ""
        lines = [f'  "{k}": "{v}"' for k, v in self._entries.items()]
        glossary_block = "{\n" + ",\n".join(lines) + "\n}"
        return (
            "\n\nGLOSSARY (use these translations strictly — do not deviate):\n"
            + glossary_block
        )

    def find_relevant(self, text: str) -> dict[str, str]:
        """Return only glossary entries whose source term appears in the text."""
        return {k: v for k, v in self._entries.items() if k in text}
