"""Smart + adaptive chunking: groups subtitles into translation-friendly chunks."""

from __future__ import annotations

import logging

from ai_subtitle_translator.config import ChunkConfig
from ai_subtitle_translator.parser import Subtitle

logger = logging.getLogger(__name__)


def chunk_subtitles(
    subtitles: list[Subtitle],
    cfg: ChunkConfig | None = None,
) -> list[list[Subtitle]]:
    """
    Split subtitles into chunks using a hybrid strategy:
      1. Time-gap based splitting (dialogue boundaries)
      2. Adaptive size limits based on text density
      3. Context overlap for continuity (NOT translated, only used for reference)
    """
    if not subtitles:
        return []

    cfg = cfg or ChunkConfig()

    # Step 1: split into dialogue groups by time gap
    dialogue_groups = _split_by_time_gap(subtitles, cfg.time_gap_threshold_ms)

    # Step 2: enforce adaptive size limits within each group
    sized_chunks: list[list[Subtitle]] = []
    for group in dialogue_groups:
        adapted_max_lines, adapted_max_chars = _adaptive_limits(group, cfg)
        sized_chunks.extend(_split_by_size(group, adapted_max_lines, adapted_max_chars))

    logger.info(
        "Chunked %d subtitles → %d dialogue groups → %d sized chunks",
        len(subtitles), len(dialogue_groups), len(sized_chunks),
    )

    return sized_chunks


def build_context_window(
    chunks: list[list[Subtitle]],
    context_size: int,
) -> list[list[Subtitle] | None]:
    """
    For each chunk, return the last N subtitles of the previous chunk
    to be used as read-only context in the prompt (not translated).
    Returns None for the first chunk.
    """
    if context_size <= 0:
        return [None] * len(chunks)

    contexts: list[list[Subtitle] | None] = [None]
    for i in range(1, len(chunks)):
        contexts.append(chunks[i - 1][-context_size:])
    return contexts


# -- Internal helpers --


def _adaptive_limits(
    group: list[Subtitle], cfg: ChunkConfig
) -> tuple[int, int]:
    """
    Dynamically adjust chunk size based on text density.
    Short lines → allow more lines per chunk.
    Long lines → use fewer lines per chunk.
    """
    if not group:
        return cfg.max_lines, cfg.max_chars

    avg_chars = sum(len(s.text) for s in group) / len(group)

    if avg_chars < 30:
        # Short lines (e.g. single words, exclamations) — pack more
        scale = 1.4
    elif avg_chars < 60:
        # Normal subtitle length — use defaults
        scale = 1.0
    else:
        # Long/dense lines — use smaller chunks for quality
        scale = 0.7

    adapted_lines = max(5, int(cfg.max_lines * scale))
    adapted_chars = max(400, int(cfg.max_chars * scale))

    return adapted_lines, adapted_chars


def _split_by_time_gap(
    subtitles: list[Subtitle], threshold_ms: int
) -> list[list[Subtitle]]:
    """Start a new group whenever the gap between consecutive subtitles exceeds threshold."""
    groups: list[list[Subtitle]] = [[subtitles[0]]]

    for prev, curr in zip(subtitles, subtitles[1:]):
        gap = curr.start_ms - prev.end_ms
        if gap > threshold_ms:
            groups.append([curr])
        else:
            groups[-1].append(curr)

    return groups


def _split_by_size(
    group: list[Subtitle], max_lines: int, max_chars: int
) -> list[list[Subtitle]]:
    """Break a group further if it exceeds line or character limits."""
    chunks: list[list[Subtitle]] = []
    current: list[Subtitle] = []
    current_chars = 0

    for sub in group:
        sub_chars = len(sub.text)

        would_exceed_lines = len(current) >= max_lines
        would_exceed_chars = current_chars + sub_chars > max_chars and current

        if would_exceed_lines or would_exceed_chars:
            chunks.append(current)
            current = [sub]
            current_chars = sub_chars
        else:
            current.append(sub)
            current_chars += sub_chars

    if current:
        chunks.append(current)

    return chunks
