"""Simple in-memory translation cache with persistence support."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TranslationCache:
    """
    Maps source text → translated text.
    Avoids re-translating identical subtitle lines across chunks.
    Optionally persists to a JSON file between runs.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._hits = 0
        self._misses = 0

    def get(self, source: str) -> str | None:
        result = self._store.get(source)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, source: str, translated: str) -> None:
        self._store[source] = translated

    def has(self, source: str) -> bool:
        return source in self._store

    def bulk_lookup(self, sources: list[str]) -> dict[str, str]:
        """Look up multiple keys at once. Returns only those found."""
        return {s: self._store[s] for s in sources if s in self._store}

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> dict[str, int]:
        return {"size": self.size, "hits": self._hits, "misses": self._misses}

    def load_from_file(self, path: str | Path) -> None:
        """Load cached translations from a JSON file."""
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._store.update(data)
                logger.info("Loaded %d cached translations from %s", len(data), p)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load cache file %s: %s", p, exc)

    def save_to_file(self, path: str | Path) -> None:
        """Persist cache to a JSON file."""
        p = Path(path)
        p.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d cached translations to %s", self.size, p)
